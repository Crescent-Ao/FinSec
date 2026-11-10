"""Download the official checkpoint at a fixed revision and verify its files."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
REPO = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
DEST = ROOT / "models" / "Qwen3.5-9B"


def main():
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    print(f"Downloading {REPO}@{info.sha} to {DEST}", flush=True)
    snapshot_download(REPO, revision=info.sha, local_dir=DEST, max_workers=2)
    records = []
    for entry in info.siblings:
        path = DEST / entry.rfilename
        if not path.is_file() or path.stat().st_size != entry.size:
            raise RuntimeError(f"Missing or incomplete file: {entry.rfilename}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        checksum = digest.hexdigest()
        if entry.lfs and checksum != entry.lfs.sha256:
            raise RuntimeError(f"Checksum mismatch: {entry.rfilename}")
        records.append({"file": entry.rfilename, "bytes": entry.size, "sha256": checksum,
                        "verified_against_upstream_sha256": bool(entry.lfs)})
        print(f"Verified {entry.rfilename}: {entry.size} bytes", flush=True)
    report = {"repo": REPO, "revision": info.sha, "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
              "total_bytes": sum(row["bytes"] for row in records), "files": records}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "model_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Download and integrity verification complete.", flush=True)


if __name__ == "__main__":
    main()
