# Maintaining f-LENNS

## Current state

| Item | Value |
|---|---|
| Weights | https://huggingface.co/jeiglesias/f-LENNS/resolve/main/f-LENNS.pth |
| Parameters | 136.7 M (loads with `strict=True` against `flenns.inference.ARCHITECTURE`) |

The URL is wired into `flenns/model_zoo.py`; users do not have to configure
anything. The downloader writes to a `.part` file and renames it only once the
transfer completes, so an interrupted download cannot leave a truncated
checkpoint behind. It does not verify a checksum: if you ever suspect a corrupt
file, delete `models/f-LENNS.pth` and let it download again.

The loader accepts either a dict containing `model_state_dict` or a bare state
dict, so you can re-upload the weights in either form without touching the code,
as long as the file keeps the name `f-LENNS.pth`.

## Placeholders still to fill in before going public

| Placeholder | Files |
|---|---|
| `XXXX.XXXXX` (arXiv ID) | `README.md`, `CITATION.cff`, `pyproject.toml` |

```bash
grep -rn "XXXX.XXXXX\|TODO" --exclude-dir=.git .
```

## Checking a new upload

After replacing the file on Hugging Face, confirm the path users take:

```bash
rm -f models/f-LENNS.pth
flenns --download_only
```

## Hugging Face model card

Set `license: cc-by-nc-sa-4.0` in the model card front matter and link back to
this repository and the paper. Users download the single file directly over
HTTPS, so no `huggingface_hub` dependency is imposed on them.

## Nice-to-haves, in rough order of value

- A tiny example dataset (one or two frames of a de-identified low-resolution
  run) so users can verify their installation in under a minute.
- A GitHub Actions workflow that imports the package and runs `flenns --help`
  on each push, to catch dependency breakage.
- A Zenodo deposit of the same checkpoint for a citable DOI, documented as a
  mirror via `FLENNS_MODEL_URL`.
