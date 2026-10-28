
"""Independent real-photograph count benchmark from PixMo-Count.

Dataset metadata: ODC-BY-1.0.
Source: https://huggingface.co/datasets/allenai/pixmo-count

PixMo-Count contains real photographs, not geometric-shape instances.
This is a cross-domain counting experiment, not evidence of in-domain
shape-counting generalization. Original image URLs may have separate
copyright and access restrictions.
"""

import hashlib
import json
import random
import time
from io import BytesIO
from pathlib import Path
from urllib.parse import urlencode

import requests
from PIL import Image, UnidentifiedImageError

API = "https://datasets-server.huggingface.co/rows"
DATASET = "allenai/pixmo-count"
CONFIG = "default"

PAGE_SIZE = 100
MAX_SOURCE_ROWS = 100_000
MAX_API_ATTEMPTS = 7
MAX_IMAGE_ATTEMPTS = 3


def _get_json_with_retry(url, params=None):
    """Fetch JSON with bounded retries for transient API errors."""
    last_error = None

    for attempt in range(MAX_API_ATTEMPTS):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=(15, 60),
                headers={"User-Agent": "lab12-research/1.0"},
            )

            if response.status_code in (403, 404):
                response.raise_for_status()

            if response.status_code == 429 or response.status_code >= 500:
                response.raise_for_status()

            response.raise_for_status()
            return response.json()

        except Exception as exc:
            last_error = exc
            if attempt + 1 == MAX_API_ATTEMPTS:
                break

            delay = min(2 ** attempt + random.random(), 60)
            print(
                f"API error ({attempt + 1}/{MAX_API_ATTEMPTS}): "
                f"{exc}. Retry in {delay:.1f}s"
            )
            time.sleep(delay)

    raise RuntimeError(
        f"Hugging Face API unavailable after {MAX_API_ATTEMPTS} attempts: "
        f"{last_error}"
    )


def _download_image(url):
    """Return validated image bytes, or None if the URL stays unavailable."""
    for attempt in range(MAX_IMAGE_ATTEMPTS):
        try:
            response = requests.get(
                url,
                timeout=(15, 45),
                headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) "
                                  "AppleWebKit/537.36 Chrome/124 Safari/537.36",
                    "Referer": "https://www.flickr.com/",
                },
            )
            response.raise_for_status()

            data = response.content
            if not data:
                raise ValueError("Empty response")

            with Image.open(BytesIO(data)) as image:
                image.verify()

            return data

        except Exception as exc:
            print(
                f"Image unavailable ({attempt + 1}/{MAX_IMAGE_ATTEMPTS}): "
                f"{url} — {exc}"
            )
            if attempt + 1 < MAX_IMAGE_ATTEMPTS:
                time.sleep(min(2 ** attempt + random.random(), 15))

    return None


def _get_source_rows(source_split, root):
    """Read all available metadata pages, retrying transient API failures."""
    rows = []
    offset = 0

    while offset < MAX_SOURCE_ROWS:
        length = min(PAGE_SIZE, MAX_SOURCE_ROWS - offset)
        params = {
            "dataset": DATASET,
            "config": CONFIG,
            "split": source_split,
            "offset": offset,
            "length": length,
        }

        try:
            payload = _get_json_with_retry(API, params=params)
        except RuntimeError as exc:
            # Don't silently claim the source is exhausted if the API failed.
            raise RuntimeError(
                f"Could not read PixMo-Count split={source_split}, "
                f"offset={offset}. Partial metadata rows: {len(rows)}. "
                f"Original error: {exc}"
            ) from exc

        page = payload.get("rows", [])
        if not page:
            break

        rows.extend(page)
        print(
            f"{source_split}: metadata rows fetched "
            f"{len(rows)} (offset {offset})"
        )

        offset += len(page)
        total = payload.get("num_rows_total")

        if total is not None and offset >= int(total):
            break
        if len(page) < length:
            break

    return rows


def load_real_scenes(
    root="data/lab12",
    train_size=280,
    test_size=140,
    seed=42,
):
    """Load real photos from validation -> train and official test -> test.

    Retries transient errors, skips persistently unavailable image URLs,
    caches successful downloads, and scans all available metadata rows
    until each split has enough examples from both count classes.
    Raises a diagnostic error if the source cannot provide enough data.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    split_specs = [
        ("train", "validation", train_size),
        ("test", "test", test_size),
    ]

    out = []
    source_hashes_by_split = {"train": set(), "test": set()}
    local_hashes_by_split = {"train": set(), "test": set()}

    for split, source_split, limit in split_specs:
        if limit <= 0:
            raise ValueError(f"{split}_size must be positive, got {limit}")

        print(f"\nLoading {split} from PixMo-Count {source_split}...")
        entries = _get_source_rows(source_split, root)

        # Shuffle deterministically to avoid depending on the first rows.
        rng.shuffle(entries)

        # Balance classes as closely as possible.
        target_per_class = limit // 2
        if limit % 2:
            target_per_class += 1

        chosen = []
        counts = {2: 0, 3: 0}
        seen_urls = set()
        failures = 0

        for entry in entries:
            if len(chosen) >= limit:
                break

            r = entry.get("row", {})
            try:
                count = int(r.get("count"))
            except (TypeError, ValueError):
                continue

            if count not in (2, 3):
                continue
            if counts[count] >= target_per_class:
                # For odd limits, fill any remaining slot from either class.
                if len(chosen) < limit and (
                    counts[2] < target_per_class
                    or counts[3] < target_per_class
                ):
                    pass
                else:
                    continue

            url = r.get("image_url")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            source_row = entry.get("row_idx")
            if source_row is None:
                continue

            path = root / f"pixmo-count-{split}-{source_row}.jpg"
            data = None

            # Reuse a valid cached file.
            if path.exists():
                try:
                    with Image.open(path) as image:
                        image.verify()
                    data = path.read_bytes()
                except Exception:
                    path.unlink(missing_ok=True)

            if data is None:
                data = _download_image(url)

            if data is None:
                failures += 1
                continue

            try:
                with Image.open(BytesIO(data)) as image:
                    image.verify()

                # Normalize the saved format to JPEG.
                with Image.open(BytesIO(data)) as image:
                    image.convert("RGB").save(path, format="JPEG", quality=95)
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                print(f"Invalid image {url}: {exc}")
                path.unlink(missing_ok=True)
                failures += 1
                continue

            local_digest = hashlib.sha256(path.read_bytes()).hexdigest()
            source_digest = str(r.get("image_sha256", "")).lower()

            # Avoid exact duplicates within a split.
            if local_digest in local_hashes_by_split[split]:
                path.unlink(missing_ok=True)
                continue

            if source_digest and source_digest in source_hashes_by_split[split]:
                path.unlink(missing_ok=True)
                continue

            # Check that count labels are supported by the source.
            label = r.get("label", "")
            record = {
                "id": f"pixmo-count-{split}-{source_row}",
                "split": split,
                "image": str(path.resolve()),
                "answer": str(count),
                "label": label,
                "question": (
                    f"How many {label} are visible? "
                    "Answer with one integer."
                ),
                "source": (
                    "https://huggingface.co/datasets/allenai/pixmo-count"
                ),
                "source_row": source_row,
                "source_split": source_split,
                "image_url": url,
                "image_sha256": source_digest,
                "downloaded_sha256": local_digest,
                "checksum_matches": (
                    local_digest == source_digest if source_digest else None
                ),
                "dataset_license": (
                    "Metadata: ODC-BY-1.0; image rights may differ"
                ),
            }

            chosen.append(record)
            counts[count] += 1
            local_hashes_by_split[split].add(local_digest)
            if source_digest:
                source_hashes_by_split[split].add(source_digest)

            print(
                f"  accepted {len(chosen)}/{limit}: "
                f"count={count}, class totals={counts}"
            )

        if len(chosen) < limit or counts[2] == 0 or counts[3] == 0:
            raise RuntimeError(
                f"{split}: obtained {len(chosen)}/{limit} images; "
                f"class counts={counts}; unavailable image URLs={failures}; "
                f"metadata rows scanned={len(entries)}. "
                "The loader tried all retrieved metadata rows. "
                "Do not substitute test images into training. "
                "Reduce requested sizes or use another approved source."
            )

        out.extend(chosen)

    # Prevent overlap across splits using source and downloaded hashes.
    train_records = [r for r in out if r["split"] == "train"]
    test_records = [r for r in out if r["split"] == "test"]

    train_source = {r["image_sha256"] for r in train_records if r["image_sha256"]}
    test_source = {r["image_sha256"] for r in test_records if r["image_sha256"]}
    if train_source & test_source:
        raise ValueError("Real train/test overlap in source image hashes")

    train_local = {r["downloaded_sha256"] for r in train_records}
    test_local = {r["downloaded_sha256"] for r in test_records}
    if train_local & test_local:
        raise ValueError("Real train/test overlap in local file hashes")

    manifest = root / "real_manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out),
        encoding="utf-8",
    )

    print("\nReal-image loading complete.")
    print(f"Train: {len(train_records)}")
    print(f"Test: {len(test_records)}")
    print(f"Manifest: {manifest.resolve()}")

    return out
