"""Locate (and, if needed, fetch) the f-LENNS model weights.

f-LENNS distributes a single pretrained model, so users never have to point the
code at a checkpoint. The file is searched for in a few standard locations and,
if absent from all of them, downloaded once from the public model repository
and cached for subsequent runs.

Only the Python standard library is used for the download, so this works inside
FreeSurfer's ``fspython`` without installing anything.

Environment variables
---------------------
FLENNS_MODEL      Full path to a checkpoint; bypasses the search entirely.
FLENNS_MODEL_URL  Alternative download URL (e.g. a local mirror).
FLENNS_HOME       Directory used to cache the downloaded weights.
"""

import os
import sys
import urllib.error
import urllib.request

#: Name of the distributed checkpoint.
MODEL_FILENAME = "f-LENNS.pth"

#: Public copy of the weights.
DEFAULT_MODEL_URL = (
    "https://huggingface.co/jeiglesias/f-LENNS/resolve/main/" + MODEL_FILENAME
)

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_PACKAGE_DIR)


def model_url():
    """URL the weights are downloaded from."""
    return os.environ.get("FLENNS_MODEL_URL", DEFAULT_MODEL_URL)


def cache_dir():
    """Directory used to cache the weights when they have to be downloaded."""
    if os.environ.get("FLENNS_HOME"):
        return os.environ["FLENNS_HOME"]
    xdg = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    return os.path.join(xdg, "flenns")


def search_paths():
    """Candidate locations for the checkpoint, in order of precedence."""
    paths = []
    if os.environ.get("FLENNS_MODEL"):
        paths.append(os.environ["FLENNS_MODEL"])
    if os.environ.get("FLENNS_HOME"):
        paths.append(os.path.join(os.environ["FLENNS_HOME"], MODEL_FILENAME))
    # Alongside the code: this is where the download lands for a git checkout,
    # and where a FreeSurfer-bundled copy would live.
    paths.append(os.path.join(_REPO_DIR, "models", MODEL_FILENAME))
    paths.append(os.path.join(_PACKAGE_DIR, "models", MODEL_FILENAME))
    if os.environ.get("FREESURFER_HOME"):
        paths.append(
            os.path.join(os.environ["FREESURFER_HOME"], "models", MODEL_FILENAME)
        )
    paths.append(os.path.join(cache_dir(), MODEL_FILENAME))
    return paths


def find_model():
    """Return the first existing checkpoint among :func:`search_paths`, else None."""
    for path in search_paths():
        if os.path.isfile(path):
            return path
    return None


def _download_target():
    """Pick a writable directory for the download (repo first, then cache)."""
    repo_models = os.path.join(_REPO_DIR, "models")
    if os.path.isdir(repo_models) and os.access(repo_models, os.W_OK):
        return os.path.join(repo_models, MODEL_FILENAME)
    target_dir = cache_dir()
    os.makedirs(target_dir, exist_ok=True)
    return os.path.join(target_dir, MODEL_FILENAME)


def manual_instructions(destination=None):
    """Message explaining how to install the weights by hand."""
    destination = destination or _download_target()
    return (
        "Download the model manually with either of:\n"
        "    curl -L -o " + destination + " " + model_url() + "\n"
        "    wget -O " + destination + " " + model_url() + "\n"
        "(or download it in a browser and move it to that path). You can also "
        "keep it anywhere you like and point f-LENNS at it with\n"
        "    export FLENNS_MODEL=/path/to/" + MODEL_FILENAME
    )


def download_model(destination=None, url=None):
    """Download the weights to `destination`, atomically. Returns the path."""
    destination = destination or _download_target()
    url = url or model_url()

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    partial = destination + ".part"
    sys.stderr.write("Downloading f-LENNS weights (this happens only once)\n")
    sys.stderr.write("  from: " + url + "\n")
    sys.stderr.write("    to: " + destination + "\n")

    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            total = response.headers.get("Content-Length")
            total = int(total) if total else None
            done = 0
            next_report = 0
            with open(partial, "wb") as out:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if done >= next_report:
                        if total:
                            msg = "  %.0f/%.0f MB (%.0f%%)" % (
                                done / 1e6, total / 1e6, 100.0 * done / total)
                        else:
                            msg = "  %.0f MB" % (done / 1e6,)
                        sys.stderr.write("\r" + msg + "    ")
                        sys.stderr.flush()
                        next_report = done + (8 << 20)
        sys.stderr.write("\n")
    except (urllib.error.URLError, OSError) as err:
        if os.path.exists(partial):
            os.remove(partial)
        raise RuntimeError(
            "Could not download the f-LENNS weights (" + str(err) + ").\n\n"
            + manual_instructions(destination)
        )

    os.replace(partial, destination)
    return destination


def get_model_path(explicit=None, allow_download=True):
    """Resolve the checkpoint to use.

    Parameters
    ----------
    explicit : str, optional
        An explicit path (from ``--model``). Used as-is if it exists.
    allow_download : bool
        If True (default), fetch the weights when they are not found locally.
    """
    if explicit is not None:
        if not os.path.isfile(explicit):
            raise FileNotFoundError("No such model file: " + explicit)
        return explicit

    path = find_model()
    if path is not None:
        return path

    if not allow_download:
        raise FileNotFoundError(
            "f-LENNS weights not found. Looked in:\n  "
            + "\n  ".join(search_paths())
            + "\n\n" + manual_instructions()
        )

    return download_model()
