#!/usr/bin/env python3
"""Download Football-Data.co.uk CSVs for the top two divisions in five leagues.

Keeps the publisher's CSVs unchanged and writes them under:
  data/football_data/raw/<season>/<league_code>.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_URL = "https://www.football-data.co.uk/mmz4281"
# Football-Data.co.uk division codes: England E, Germany D, Italy I,
# Spain SP, France F. 0/1 means first/second division.
LEAGUES = {
    "England": ("E0", "E1"),
    "Germany": ("D1", "D2"),
    "Italy": ("I1", "I2"),
    "Spain": ("SP1", "SP2"),
    "France": ("F1", "F2"),
}
SEASONS = ("1920", "2021", "2122", "2223", "2324", "2425")


def is_csv_payload(payload: bytes) -> bool:
    """Reject HTML error pages and malformed/empty downloads."""
    if not payload.strip():
        return False
    sample = payload[:8192].decode("utf-8-sig", errors="replace")
    if "<html" in sample.lower() or "<!doctype" in sample.lower():
        return False
    try:
        rows = csv.reader(io.StringIO(sample))
        header = next(rows)
    except (csv.Error, StopIteration):
        return False
    normalized = {cell.strip().lstrip("\ufeff").lower() for cell in header}
    return "hometeam" in normalized and "awayteam" in normalized


def download(url: str, timeout: int, attempts: int = 3) -> tuple[bytes | None, str]:
    request = Request(url, headers={"User-Agent": "Bookmakersodds-research/1.0"})
    for attempt in range(1, attempts + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = response.read()
            if is_csv_payload(payload):
                return payload, "downloaded"
            return None, "response was not a valid match CSV"
        except HTTPError as exc:
            if exc.code == 404:
                return None, "not found (HTTP 404)"
            reason = f"HTTP {exc.code}"
        except (URLError, TimeoutError, OSError) as exc:
            reason = str(exc)
        if attempt < attempts:
            time.sleep(attempt)
    return None, reason


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/football_data/raw"),
        help="folder for downloaded data (default: data/football_data/raw)",
    )
    parser.add_argument("--timeout", type=int, default=30, help="request timeout in seconds")
    parser.add_argument("--overwrite", action="store_true", help="download again over existing CSVs")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": "https://www.football-data.co.uk/",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "seasons": list(SEASONS),
        "leagues": {name: list(codes) for name, codes in LEAGUES.items()},
        "files": [],
    }
    failures = 0
    total = len(SEASONS) * sum(map(len, LEAGUES.values()))

    for season in SEASONS:
        season_dir = args.output_dir / season
        season_dir.mkdir(parents=True, exist_ok=True)
        for country, codes in LEAGUES.items():
            for code in codes:
                destination = season_dir / f"{code}.csv"
                url = f"{BASE_URL}/{season}/{code}.csv"
                if destination.exists() and not args.overwrite:
                    status = "already exists (skipped)"
                    print(f"[SKIP] {season}/{code}.csv")
                else:
                    payload, status = download(url, args.timeout)
                    if payload is not None:
                        destination.write_bytes(payload)
                        print(f"[OK]   {season}/{code}.csv ({len(payload):,} bytes)")
                    else:
                        failures += 1
                        print(f"[FAIL] {season}/{code}.csv: {status} — {url}")
                manifest["files"].append({
                    "season": season,
                    "season_label": f"20{season[:2]}/20{season[2:]}",
                    "country": country,
                    "division_code": code,
                    "url": url,
                    "path": str(destination),
                    "status": status,
                })

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nFinished: {total - failures}/{total} files available.")
    print(f"Manifest: {manifest_path}")
    if failures:
        print("Some files could not be downloaded. Review manifest.json and retry later.")
        return 1
    print("All requested league/season files are available.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
