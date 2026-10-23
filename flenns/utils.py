# Volume I/O and resampling helpers.
#
# These functions are taken from Benjamin Billot's lab2im package
# (https://github.com/BBillot/lab2im); only the handful that f-LENNS needs at
# inference time are kept here, so that the package stays light to import.
# See THIRD_PARTY_NOTICES.md.
import nibabel as nib
import numpy as np
import torch


# Load nifti or mgz file
def load_volume(path_volume, return_data_type=False):

    assert path_volume.endswith(('.nii', '.nii.gz', '.mgz')), 'Unknown data file: %s' % path_volume

    x = nib.load(path_volume)
    volume = x.get_fdata()
    aff = x.affine

    if return_data_type:
        return volume, aff, x.header.get_data_dtype()
    else:
        return volume, aff


# Save nifti or mgz file
def save_volume(volume, aff, path, data_type=None):

    header = nib.Nifti1Header()

    if aff is None:
        aff = np.eye(4)

    if data_type is not None:
        header.set_data_dtype(data_type)
        nifti = nib.Nifti1Image(volume.astype(data_type), aff, header)
    else:
        nifti = nib.Nifti1Image(volume, aff, header)

    nib.save(nifti, path)


def myzoom_torch(X, factor, device='cpu'):

    if len(X.shape)==3:
        X = X[..., None]

    delta = (1.0 - factor) / (2.0 * factor)
    newsize = np.round(X.shape[:-1] * factor).astype(int)

    vx = torch.arange(delta[0], delta[0] + newsize[0] / factor[0], 1 / factor[0], device=device)
    vy = torch.arange(delta[1], delta[1] + newsize[1] / factor[1], 1 / factor[1], device=device)
    vz = torch.arange(delta[2], delta[2] + newsize[2] / factor[2], 1 / factor[2], device=device)

    vx[vx < 0] = 0
    vy[vy < 0] = 0
    vz[vz < 0] = 0
    vx[vx > (X.shape[0]-1)] = (X.shape[0]-1)
    vy[vy > (X.shape[1] - 1)] = (X.shape[1] - 1)
    vz[vz > (X.shape[2] - 1)] = (X.shape[2] - 1)

    fx = torch.floor(vx).int()
    cx = fx + 1
    cx[cx > (X.shape[0]-1)] = (X.shape[0]-1)
    wcx = vx - fx
    wfx = 1 - wcx

    fy = torch.floor(vy).int()
    cy = fy + 1
    cy[cy > (X.shape[1]-1)] = (X.shape[1]-1)
    wcy = vy - fy
    wfy = 1 - wcy

    fz = torch.floor(vz).int()
    cz = fz + 1
    cz[cz > (X.shape[2]-1)] = (X.shape[2]-1)
    wcz = vz - fz
    wfz = 1 - wcz

    Y = torch.zeros([newsize[0], newsize[1], newsize[2], X.shape[3]], device=device)

    for channel in range(X.shape[3]):
        Xc = X[:,:,:,channel]

        tmp1 = torch.zeros([newsize[0], Xc.shape[1], Xc.shape[2]], device=device)
        for i in range(newsize[0]):
            tmp1[i, :, :] = wfx[i] * Xc[fx[i], :, :] +  wcx[i] * Xc[cx[i], :, :]
        tmp2 = torch.zeros([newsize[0], newsize[1], Xc.shape[2]], device=device)
        for j in range(newsize[1]):
            tmp2[:, j, :] = wfy[j] * tmp1[:, fy[j], :] +  wcy[j] * tmp1[:, cy[j], :]
        for k in range(newsize[2]):
            Y[:, :, k, channel] = wfz[k] * tmp2[:, :, fz[k]] +  wcz[k] * tmp2[:, :, cz[k]]

    if Y.shape[3] == 1:
        Y = Y[:,:,:, 0]

    return Y


#
#
#
#
def get_dims(shape, max_channels=10):
    """Get the number of dimensions and channels from the shape of an array.
    The number of dimensions is assumed to be the length of the shape, as long as the shape of the last dimension is
    inferior or equal to max_channels (default 3).
    :param shape: shape of an array. Can be a sequence or a 1d numpy array.
    :param max_channels: maximum possible number of channels.
    :return: the number of dimensions and channels associated with the provided shape.
    example 1: get_dims([150, 150, 150], max_channels=10) = (3, 1)
    example 2: get_dims([150, 150, 150, 3], max_channels=10) = (3, 3)
    example 3: get_dims([150, 150, 150, 15], max_channels=10) = (4, 1), because 5>3"""
    if shape[-1] <= max_channels:
        n_dims = len(shape) - 1
        n_channels = shape[-1]
    else:
        n_dims = len(shape)
        n_channels = 1
    return n_dims, n_channels


def align_volume_to_ref(volume, aff, aff_ref=None, return_aff=False, n_dims=None):
    """This function aligns a volume to a reference orientation (axis and direction) specified by an affine matrix.
    :param volume: a numpy array
    :param aff: affine matrix of the floating volume
    :param aff_ref: (optional) affine matrix of the target orientation. Default is identity matrix.
    :param return_aff: (optional) whether to return the affine matrix of the aligned volume
    :param n_dims: number of dimensions (excluding channels) of the volume corresponding to the provided affine matrix.
    :return: aligned volume, with corresponding affine matrix if return_aff is True.
    """

    # work on copy
    aff_flo = aff.copy()

    # default value for aff_ref
    if aff_ref is None:
        aff_ref = np.eye(4)

    # extract ras axes
    if n_dims is None:
        n_dims, _ = get_dims(volume.shape)
    ras_axes_ref = get_ras_axes(aff_ref, n_dims=n_dims)
    ras_axes_flo = get_ras_axes(aff_flo, n_dims=n_dims)

    # align axes
    aff_flo[:, ras_axes_ref] = aff_flo[:, ras_axes_flo]
    for i in range(n_dims):
        if ras_axes_flo[i] != ras_axes_ref[i]:
            volume = np.swapaxes(volume, ras_axes_flo[i], ras_axes_ref[i])
            swapped_axis_idx = np.where(ras_axes_flo == ras_axes_ref[i])
            ras_axes_flo[swapped_axis_idx], ras_axes_flo[i] = ras_axes_flo[i], ras_axes_flo[swapped_axis_idx]

    # align directions
    dot_products = np.sum(aff_flo[:3, :3] * aff_ref[:3, :3], axis=0)
    for i in range(n_dims):
        if dot_products[i] < 0:
            volume = np.flip(volume, axis=i)
            aff_flo[:, i] = - aff_flo[:, i]
            aff_flo[:3, 3] = aff_flo[:3, 3] - aff_flo[:3, i] * (volume.shape[i] - 1)

    if return_aff:
        return volume, aff_flo
    else:
        return volume


def get_ras_axes(aff, n_dims=3):
    # Exhaustive search is fine, only 6 possible permutations...
    candidates = [[0,1,2], [0,2,1], [1,0,2], [2,0,1], [1,2,0], [2,1,0]]
    best_score = -100000
    best_candidate = None
    for c in candidates:
        score = np.abs(aff[0, c[0]] / np.linalg.norm(aff[:, c[0]])) \
              + np.abs(aff[1, c[1]] / np.linalg.norm(aff[:, c[1]])) \
              + np.abs(aff[2, c[2]] / np.linalg.norm(aff[:, c[2]]))
        if score>best_score:
            best_score = score
            best_candidate = c
    return np.array(best_candidate)
