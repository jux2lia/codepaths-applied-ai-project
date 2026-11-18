"""Download the song catalog to data/raw/spotify_songs.csv and verify its checksum.

Dataset: "30000 Spotify Songs" (TidyTuesday, 2020-01-21), collected with the
spotifyr R package. Also mirrored on Kaggle. Not committed to git (see .gitignore).
Source: https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-01-21

Usage:  python data/download.py
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = (
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/"
    "data/2020/2020-01-21/spotify_songs.csv"
)
SHA256 = "9902920cfe4940f0914ebb6716801a42f7f19ac88fbd6693eaae7e2822b16845"
DEST = Path(__file__).resolve().parent / "raw" / "spotify_songs.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if DEST.exists():
        if sha256(DEST) == SHA256:
            print(f"{DEST} already present, checksum OK.")
            return
        sys.exit(
            f"{DEST} exists but its checksum doesn't match the expected file.\n"
            "Delete it and run this script again."
        )

    DEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = DEST.with_suffix(".part")
    print(f"Downloading {URL} ...")
    urllib.request.urlretrieve(URL, tmp)
    if sha256(tmp) != SHA256:
        tmp.unlink()
        sys.exit("Downloaded file has an unexpected checksum; the source may have changed.")
    tmp.replace(DEST)
    print(f"Saved to {DEST}, checksum OK.")


if __name__ == "__main__":
    main()
