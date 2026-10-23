#!/bin/bash
# f-LENNS launcher for users who have FreeSurfer installed.
#
# Runs scripts/inference.py with FreeSurfer's bundled Python interpreter
# (fspython), which already provides torch, numpy, scipy and nibabel, so no
# dependencies need to be installed.
#
# Usage:
#   bash run_flenns.sh --i <input_nifti> --o <output_nifti> [options]
#   bash run_flenns.sh --help
#
# (or ./run_flenns.sh directly, if the executable bit is set on your copy)

set -e

# Absolute path of the directory containing this script (portable: macOS has
# no `readlink -f`).
SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Make fspython available if it is not on the PATH yet.
if ! command -v fspython >/dev/null 2>&1; then
    if [ -n "$FREESURFER_HOME" ] && [ -f "$FREESURFER_HOME/SetUpFreeSurfer.sh" ]; then
        source "$FREESURFER_HOME/SetUpFreeSurfer.sh" >/dev/null
    fi
fi

if ! command -v fspython >/dev/null 2>&1; then
    echo "ERROR: fspython was not found." >&2
    echo "Set FREESURFER_HOME and source FreeSurfer first, e.g.:" >&2
    echo "    export FREESURFER_HOME=/usr/local/freesurfer/7.4.1" >&2
    echo "    source \$FREESURFER_HOME/SetUpFreeSurfer.sh" >&2
    echo "If you do not have FreeSurfer, install f-LENNS with pip instead" >&2
    echo "(see the README) and use the 'flenns' command." >&2
    exit 1
fi

exec fspython "$SCRIPTPATH/scripts/inference.py" "$@"
