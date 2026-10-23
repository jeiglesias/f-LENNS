"""Core f-LENNS inference: super-resolve an fMRI volume to 1 mm isotropic.

Each frame of a 3D/4D NIfTI (or MGZ) is processed independently, as described
in the paper, so the tool is compatible with any TR and does not introduce
temporal dependencies.
"""

import numpy as np
import torch

from flenns.NAFNet import NAFNet
from flenns.model_zoo import get_model_path
from flenns.utils import (
    align_volume_to_ref,
    load_volume,
    myzoom_torch,
    save_volume,
)

#: Target (output) resolution in mm, isotropic.
REF_RES = 1.0

#: Architecture of the distributed model (Methods, "Architecture").
ARCHITECTURE = {
    "img_channel": 1,
    "width": 128,
    "enc_blk_nums": [2, 2, 56],
    "middle_blk_num": 2,
    "dec_blk_nums": [2, 2, 2],
}

#: Spatial dimensions are zero-padded to a multiple of this before the forward
#: pass, so that the encoder/decoder down- and up-sampling line up. The padding
#: is cropped off again before writing the output.
PAD_MULTIPLE = 16


def build_model(device="cpu"):
    """Instantiate the f-LENNS network (random weights)."""
    return NAFNet(**ARCHITECTURE).to(device)


def load_model(model_path=None, device="cpu", verbose=True):
    """Build the network and load the distributed weights into it.

    Parameters
    ----------
    model_path : str, optional
        Path to a checkpoint. If None, the distributed model is located
        automatically (and downloaded on first use).
    device : str
        'cpu' or 'cuda'.
    """
    model_path = get_model_path(model_path)
    if verbose:
        print("Preparing model and loading weights")
        print("  model file: " + model_path)
    model = build_model(device)
    try:
        # torch >= 2.6 defaults to weights_only=True, which cannot unpickle the
        # training checkpoint (it carries more than tensors).
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) \
        if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def tiled_forward(model, input_tensor, patch_size=128, overlap=0.25):
    """
    Run a model on a 3-D volume using overlapping cubic patches, blending
    results with a 3-D Hann window so that central voxels of every patch
    carry more weight than edge voxels.

    Args:
        model        : PyTorch model with signature pred = model(x)
                       where x has shape (1, 1, D, H, W).
        input_tensor : 3-D input tensor of shape (D, H, W).
        patch_size   : Side length of each cubic patch (must be a multiple
                       of 16).  Defaults to 128.
        overlap      : Fraction of a patch by which consecutive patches
                       overlap, in [0, 1).  E.g. 0.5 -> 50 % overlap.
                       Defaults to 0.25.
                       NOTE: overlap=0 is not recommended with a Hann window
                       because the window is exactly 0 at patch boundaries.

    Returns:
        Tensor of shape (D, H, W) containing the weighted-average predictions
        over all patches.
    """
    assert patch_size % 16 == 0, "patch_size must be a multiple of 16"
    assert 0.0 <= overlap < 1.0, "overlap must be in [0, 1)"

    device = input_tensor.device
    D, H, W = input_tensor.shape
    ps = patch_size

    stride = max(1, round(ps * (1.0 - overlap)))

    # 3-D Hann window: outer product of three 1-D windows
    w1d = torch.hann_window(ps, periodic=False, device=device)
    win3d = w1d.view(-1, 1, 1) * w1d.view(1, -1, 1) * w1d.view(1, 1, -1)

    pred_sum = torch.zeros(D, H, W, device=device)
    weight_sum = torch.zeros(D, H, W, device=device)

    def tile_positions(size):
        positions = list(range(0, size - ps + 1, stride))
        last = size - ps
        if not positions or positions[-1] < last:
            positions.append(last)
        return positions

    d_starts = tile_positions(D)
    h_starts = tile_positions(H)
    w_starts = tile_positions(W)

    for d in d_starts:
        for h in h_starts:
            for w in w_starts:
                patch = input_tensor[d:d + ps, h:h + ps, w:w + ps]
                out = model(patch[None, None, ...])  # (1, 1, ps, ps, ps)
                out = out.squeeze()                  # (ps, ps, ps)
                pred_sum[d:d + ps, h:h + ps, w:w + ps] += out * win3d
                weight_sum[d:d + ps, h:h + ps, w:w + ps] += win3d

    return pred_sum / weight_sum.clamp(min=1e-8)


def super_resolve(input_file, output_file, model=None, model_path=None,
                  device="cpu", upscaled_file=None, n_frames=None,
                  tiled_inference=False, patch_size=128, overlap=0.25,
                  verbose=True):
    """Super-resolve `input_file` to 1 mm isotropic and write `output_file`.

    Parameters
    ----------
    input_file : str
        3D or 4D NIfTI / MGZ to super-resolve.
    output_file : str
        Where to write the 1 mm isotropic result.
    model : torch.nn.Module, optional
        Preloaded model (e.g. when looping over many scans). If None, the
        distributed model is loaded.
    model_path : str, optional
        Explicit checkpoint path; only used when `model` is None.
    device : str
        'cpu' or 'cuda'.
    upscaled_file : str, optional
        Also write the linearly upsampled input, for comparison.
    n_frames : int, optional
        Process only the first `n_frames` frames (debugging).
    tiled_inference : bool
        Process each frame in overlapping tiles to save GPU memory (slower).
    """
    if model is None:
        model = load_model(model_path=model_path, device=device, verbose=verbose)

    if verbose:
        print("Loading input volume and normalizing to [0,1]")
    image, aff, data_type = load_volume(input_file, return_data_type=True)
    image = image.astype(float)
    image2, aff2 = align_volume_to_ref(
        image, aff, aff_ref=np.eye(4), return_aff=True, n_dims=3
    )
    was_3d = len(image2.shape) == 3
    if was_3d:
        image2 = image2[..., np.newaxis]
    nc = image2.shape[3]
    if n_frames is not None and nc > n_frames:
        nc = n_frames
        image2 = image2[..., :nc]

    image_torch = torch.tensor(
        np.ascontiguousarray(image2), dtype=torch.float32, device=device
    )
    maxis = torch.zeros(nc)
    for c in range(nc):
        maxis[c] = torch.max(image_torch[:, :, :, c])
        image_torch[:, :, :, c] = image_torch[:, :, :, c] / maxis[c]

    if verbose:
        print("Upscaling to target resolution")
    voxsize = np.sqrt(np.sum(aff2 ** 2, axis=0))[:-1]
    factors = voxsize / REF_RES
    upscaled = myzoom_torch(image_torch, factors, device=device)
    if len(upscaled.shape) == 3:
        upscaled = upscaled[..., np.newaxis]
    siz1 = upscaled.shape[:-1]
    siz2 = (np.ceil(np.array(siz1) / PAD_MULTIPLE) * PAD_MULTIPLE).astype(int)
    upscaled2 = torch.zeros([*siz2, upscaled.shape[-1]], device=device)
    upscaled2[:siz1[0], :siz1[1], :siz1[2], :] = upscaled
    upscaled = upscaled2
    aff_upscaled = aff2.copy()
    for j in range(3):
        aff_upscaled[:-1, j] = aff_upscaled[:-1, j] / factors[j]
    aff_upscaled[:-1, -1] = aff_upscaled[:-1, -1] - np.matmul(
        aff_upscaled[:-1, :-1], 0.5 * (factors - 1)
    )

    if verbose:
        print("Pushing data through the CNN")
    with torch.no_grad():
        sr = torch.zeros_like(upscaled)
        for c in range(nc):
            if verbose:
                print("   Frame " + str(1 + c) + " of " + str(nc), end="\r")
            if tiled_inference:
                pred = tiled_forward(
                    model, upscaled[:, :, :, c],
                    patch_size=patch_size, overlap=overlap,
                )
            else:
                pred = model(upscaled[:, :, :, c][None, None, ...])
            sr[:, :, :, c] = maxis[c] * torch.squeeze(pred)
            upscaled[:, :, :, c] = upscaled[:, :, :, c] * maxis[c]

    if verbose:
        print("\nSaving to disk")
    # Crop off the padding that was added to reach a multiple of PAD_MULTIPLE,
    # and drop the trailing singleton axis if the input was 3D.
    crop = (slice(0, siz1[0]), slice(0, siz1[1]), slice(0, siz1[2]), slice(0, nc))

    def _to_numpy(tensor):
        array = tensor[crop].detach().cpu().numpy()
        return array[..., 0] if was_3d else array

    save_volume(_to_numpy(sr), aff_upscaled, output_file, data_type=data_type)
    if upscaled_file is not None:
        save_volume(_to_numpy(upscaled), aff_upscaled, upscaled_file,
                    data_type=data_type)

    if verbose:
        print("All done")
        if upscaled_file is None:
            print("freeview " + input_file + " " + output_file)
        else:
            print("freeview " + input_file + " " + upscaled_file + " "
                  + output_file)
    return output_file
