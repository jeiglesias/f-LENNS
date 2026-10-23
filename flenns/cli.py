"""Command-line interface for f-LENNS."""

import argparse
import os
import sys

basepath = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if basepath not in sys.path:
    sys.path.append(basepath)

from flenns import __version__
from flenns.inference import super_resolve
from flenns.model_zoo import MODEL_FILENAME, get_model_path


DESCRIPTION = (
    "f-LENNS: super-resolve functional MRI to 1 mm isotropic resolution. "
    "Takes a 3D or 4D NIfTI/MGZ and writes a 1 mm isotropic NIfTI/MGZ; each "
    "frame is processed independently, so any TR is supported. The pretrained "
    "model is located automatically (and downloaded on first use)."
)

EPILOG = (
    "examples:\n"
    "  flenns --i bold.nii.gz --o bold_sr.nii.gz\n"
    "  flenns --i bold.nii.gz --o bold_sr.nii.gz --device cuda\n"
    "  flenns --i bold.nii.gz --o bold_sr.nii.gz --device cuda --tiled_inference\n"
)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="flenns",
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--i", "--input", dest="input", metavar="FILE",
                        help="Image to super-resolve (3D or 4D nifti/mgz).")
    parser.add_argument("--o", "--output", dest="output", metavar="FILE",
                        help="Output image (3D/4D nifti/mgz at 1 mm isotropic).")
    parser.add_argument("--device", default="cpu", metavar="DEV",
                        help="Device: cpu (default) or cuda. cuda is strongly "
                             "recommended for 4D data.")
    parser.add_argument("--upscaled", default=None, metavar="FILE",
                        help="(optional) Also write the linearly upsampled "
                             "input, for comparison.")
    parser.add_argument("--frames", type=int, default=None, metavar="N",
                        help="(optional) Process only the first N frames "
                             "(useful for a quick test).")
    parser.add_argument("--tiled_inference", action="store_true",
                        help="Process each frame in overlapping tiles. Slower, "
                             "but uses much less (GPU) memory.")
    parser.add_argument("--patch_size", type=int, default=128, metavar="N",
                        help="Tile size for --tiled_inference (multiple of 16; "
                             "default: 128).")
    parser.add_argument("--overlap", type=float, default=0.25, metavar="F",
                        help="Tile overlap fraction for --tiled_inference, in "
                             "[0, 1) (default: 0.25).")
    parser.add_argument("--model", default=None, metavar="FILE",
                        help="(advanced) Use a specific checkpoint instead of "
                             "the distributed model (" + MODEL_FILENAME + ").")
    parser.add_argument("--download_only", action="store_true",
                        help="Fetch the model weights and exit, without "
                             "processing any data. Useful on login nodes of "
                             "clusters whose compute nodes have no internet "
                             "access.")
    parser.add_argument("--version", action="version",
                        version="f-LENNS " + __version__)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.download_only:
        try:
            print(get_model_path(args.model))
        except (RuntimeError, FileNotFoundError) as err:
            sys.stderr.write("ERROR: " + str(err) + "\n")
            return 1
        return 0

    missing = [flag for flag, value in (("--i", args.input), ("--o", args.output))
               if value is None]
    if missing:
        parser.error("the following arguments are required: " + ", ".join(missing))

    if not os.path.isfile(args.input):
        parser.error("input file not found: " + args.input)

    if args.device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            parser.error("--device cuda was requested but no CUDA device is "
                         "visible to PyTorch. Use --device cpu, or check your "
                         "installation.")

    try:
        super_resolve(
            input_file=args.input,
            output_file=args.output,
            model_path=args.model,
            device=args.device,
            upscaled_file=args.upscaled,
            n_frames=args.frames,
            tiled_inference=args.tiled_inference,
            patch_size=args.patch_size,
            overlap=args.overlap,
        )
    except (RuntimeError, FileNotFoundError) as err:
        # Model missing / download failed / out of memory: report the problem
        # itself rather than a stack trace.
        sys.stderr.write("ERROR: " + str(err) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
