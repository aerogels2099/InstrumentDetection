"""Download the PANNs CNN14 pre-trained checkpoint from Zenodo (~120 MB)."""

import os
import urllib.request

URL  = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
DEST = os.path.join(os.path.dirname(__file__), "Cnn14_mAP=0.431.pth")


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        mb  = downloaded / 1024 / 1024
        print(f"\r  {pct:.1f}%  ({mb:.1f} MB)", end="", flush=True)


if __name__ == "__main__":
    if os.path.exists(DEST):
        print(f"Checkpoint already exists: {DEST}")
    else:
        print(f"Downloading PANNs CNN14 checkpoint to:\n  {DEST}")
        urllib.request.urlretrieve(URL, DEST, reporthook=_progress)
        print(f"\nDone.")
