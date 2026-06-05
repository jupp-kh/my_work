from __future__ import annotations

import argparse
import mimetypes
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "downloaded_images"
IMAGE_CONTENT_TYPES = {
    "image/bmp",
    "image/gif",
    "image/jpeg",
    "image/png",
    "image/tiff",
    "image/webp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download an image URL into the local data folder.")
    parser.add_argument("url", help="Image URL to download.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Folder where the image is saved.")
    parser.add_argument("--filename", type=str, default=None, help="Optional output filename.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Download timeout in seconds.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file if it already exists.")
    return parser.parse_args()


def safe_filename(value: str) -> str:
    value = urllib.parse.unquote(value).strip()
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = value.strip("._")
    return value or "downloaded_image"


def filename_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    name = Path(parsed.path).name
    return safe_filename(name) if name else "downloaded_image"


def extension_from_content_type(content_type: str | None) -> str:
    if not content_type:
        return ""
    content_type = content_type.split(";", 1)[0].strip().lower()
    return mimetypes.guess_extension(content_type) or ""


def unique_path(path: Path, overwrite: bool) -> Path:
    if overwrite or not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    index = 1
    while True:
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def download_image(url: str, output_dir: Path, filename: str | None, timeout: float, overwrite: bool) -> Path:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; visual-computing-lab/1.0)"},
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type and content_type not in IMAGE_CONTENT_TYPES:
            raise ValueError(f"URL did not return an image content type: {content_type}")

        data = response.read()
        if not data:
            raise ValueError("Downloaded file is empty.")

    output_dir.mkdir(parents=True, exist_ok=True)
    name = safe_filename(filename) if filename else filename_from_url(url)
    path = output_dir / name

    if not path.suffix:
        path = path.with_suffix(extension_from_content_type(content_type) or ".jpg")

    path = unique_path(path, overwrite)
    path.write_bytes(data)
    return path


def main() -> None:
    args = parse_args()
    try:
        path = download_image(args.url, args.output_dir, args.filename, args.timeout, args.overwrite)
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise SystemExit(f"Download failed: {exc}") from exc

    print(f"Saved image to {path}")


if __name__ == "__main__":
    main()
