from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from contextlib import contextmanager
from pathlib import Path

from .errors import RNovAError
from .validation import REPO_ROOT, checkpoint_paths


CHECKPOINT_RECORD_URL = "https://zenodo.org/records/18352464"
CHECKPOINT_DOWNLOAD_URL = (
    "https://zenodo.org/api/records/18352464/files/RNovA_Checkpoint.zip/content"
)
CHECKPOINT_ARCHIVE_NAME = "RNovA_Checkpoint.zip"
CHECKPOINT_MD5 = "e114813f043dbc93b800aa48d4797ada"
CHECKPOINT_SHA256: str | None = None
CHECKPOINT_SIZE = 497_609_275
MAX_CHECKPOINT_ARCHIVE_SIZE = 2_000_000_000
MAX_CHECKPOINT_MEMBER_SIZE = 2_000_000_000
DOWNLOAD_TIMEOUT_SECONDS = 60
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
PROGRESS_INTERVAL_BYTES = 50 * 1024 * 1024

CHECKPOINT_MEMBER_CANDIDATES = {
    "pathsearcher": (
        "RNovA_PathSearcher_Inference/save/rnova.pt",
        "PathSearcher/save/rnova.pt",
        "RNovA_PathSearcher/save/rnova.pt",
        "RNovA_Checkpoint/RNovA_PathSearcher_Inference/save/rnova.pt",
        "RNovA_Checkpoint/PathSearcher/save/rnova.pt",
        "RNovA_Checkpoint/RNovA_PathSearcher/save/rnova.pt",
    ),
    "seqfiller": (
        "RNovA_SeqFiller_Inference/save/rnova.pt",
        "SeqFiller/save/rnova.pt",
        "RNovA_SeqFiller/save/rnova.pt",
        "RNovA_Checkpoint/RNovA_SeqFiller_Inference/save/rnova.pt",
        "RNovA_Checkpoint/SeqFiller/save/rnova.pt",
        "RNovA_Checkpoint/RNovA_SeqFiller/save/rnova.pt",
    ),
}


def download_checkpoints(
    *,
    repo_root: Path = REPO_ROOT,
    archive: str | Path | None = None,
    force: bool = False,
    allow_unverified_archive: bool = False,
) -> list[Path]:
    archive_path = Path(archive).expanduser().resolve() if archive else _download_archive(repo_root, force)

    if not archive_path.exists():
        raise RNovAError(f"Checkpoint archive not found: {archive_path}")

    if allow_unverified_archive:
        print(
            "WARNING: installing checkpoint archive without checksum verification.",
            file=sys.stderr,
        )
    else:
        verify_archive(archive_path)

    return extract_checkpoints(archive_path, repo_root=repo_root, force=force)


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: Path) -> None:
    size = path.stat().st_size
    if size > MAX_CHECKPOINT_ARCHIVE_SIZE:
        raise RNovAError(
            f"Checkpoint archive is unexpectedly large: {size} bytes "
            f"(limit {MAX_CHECKPOINT_ARCHIVE_SIZE})"
        )
    if size != CHECKPOINT_SIZE:
        raise RNovAError(
            f"Checkpoint archive size mismatch for {path}\n"
            f"Expected: {CHECKPOINT_SIZE}\n"
            f"Actual:   {size}"
        )

    if CHECKPOINT_SHA256:
        actual_sha256 = file_sha256(path)
        if actual_sha256 != CHECKPOINT_SHA256:
            raise RNovAError(
                f"Checkpoint archive sha256 mismatch for {path}\n"
                f"Expected: {CHECKPOINT_SHA256}\n"
                f"Actual:   {actual_sha256}"
            )
        return

    actual_md5 = file_md5(path)
    if actual_md5 != CHECKPOINT_MD5:
        raise RNovAError(
            f"Checkpoint archive md5 mismatch for {path}\n"
            f"Expected: {CHECKPOINT_MD5}\n"
            f"Actual:   {actual_md5}"
        )


def extract_checkpoints(archive_path: Path, *, repo_root: Path = REPO_ROOT, force: bool = False) -> list[Path]:
    with zipfile.ZipFile(archive_path) as zf:
        pathsearcher_member, seqfiller_member = _checkpoint_members(zf, archive_path)

        destinations = checkpoint_paths(repo_root)
        selected = (pathsearcher_member, seqfiller_member)
        placed: list[Path] = []
        for member, destination in zip(selected, destinations, strict=True):
            if destination.exists() and not force:
                placed.append(destination)
                continue
            if member.file_size > MAX_CHECKPOINT_MEMBER_SIZE:
                raise RNovAError(
                    f"Checkpoint member is unexpectedly large: {member.filename} "
                    f"({member.file_size} bytes)"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, _named_temp_file(destination.parent) as tmp_path:
                with tmp_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                tmp_path.replace(destination)
            placed.append(destination)

    return placed


def _download_archive(repo_root: Path, force: bool) -> Path:
    cache_dir = repo_root / ".cache" / "rnova"
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / CHECKPOINT_ARCHIVE_NAME
    if archive_path.exists() and not force:
        return archive_path

    print(f"Downloading checkpoints from {CHECKPOINT_RECORD_URL}")
    with _named_temp_file(cache_dir) as tmp_path:
        downloaded = 0
        next_progress = PROGRESS_INTERVAL_BYTES
        with urllib.request.urlopen(CHECKPOINT_DOWNLOAD_URL, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            with tmp_path.open("wb") as out:
                while True:
                    chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > MAX_CHECKPOINT_ARCHIVE_SIZE:
                        raise RNovAError(
                            "Checkpoint archive download exceeded expected size limit "
                            f"({MAX_CHECKPOINT_ARCHIVE_SIZE} bytes)"
                        )
                    out.write(chunk)
                    if downloaded >= next_progress:
                        print(f"Downloaded {downloaded // (1024 * 1024)} MiB...")
                        next_progress += PROGRESS_INTERVAL_BYTES

        size = tmp_path.stat().st_size
        if size != CHECKPOINT_SIZE:
            raise RNovAError(
                f"Checkpoint archive size mismatch for {tmp_path}: "
                f"expected {CHECKPOINT_SIZE}, got {size}"
            )
        verify_archive(tmp_path)
        tmp_path.replace(archive_path)

    return archive_path


def _checkpoint_members(
    zf: zipfile.ZipFile,
    archive_path: Path,
) -> tuple[zipfile.ZipInfo, zipfile.ZipInfo]:
    manifest_members = [m for m in zf.infolist() if Path(m.filename).name == "rnova-checkpoints.json"]
    if manifest_members:
        return _checkpoint_members_from_manifest(zf, manifest_members[0], archive_path)

    members_by_name = {_normalize_zip_name(member.filename): member for member in zf.infolist()}
    selected: list[zipfile.ZipInfo] = []
    missing: list[str] = []
    for checkpoint_name in ("pathsearcher", "seqfiller"):
        matches = []
        for candidate in CHECKPOINT_MEMBER_CANDIDATES[checkpoint_name]:
            member = members_by_name.get(_normalize_zip_name(candidate))
            if member is not None:
                matches.append(member)
        if not matches:
            missing.append(checkpoint_name)
        elif len(matches) > 1:
            raise RNovAError(
                "Checkpoint archive is ambiguous: multiple exact candidates match "
                f"{checkpoint_name}: " + ", ".join(member.filename for member in matches)
            )
        else:
            selected.append(matches[0])

    if missing:
        exact_candidates = "\n".join(
            f"  {candidate}"
            for candidates in CHECKPOINT_MEMBER_CANDIDATES.values()
            for candidate in candidates
        )
        found = "\n".join(f"  {member.filename}" for member in zf.infolist())
        raise RNovAError(
            "Could not identify checkpoint files by exact archive path.\n"
            f"Missing: {', '.join(missing)}\n"
            f"Expected one of:\n{exact_candidates}\n"
            f"Archive contains:\n{found}"
        )

    return selected[0], selected[1]


def _checkpoint_members_from_manifest(
    zf: zipfile.ZipFile,
    manifest_member: zipfile.ZipInfo,
    archive_path: Path,
) -> tuple[zipfile.ZipInfo, zipfile.ZipInfo]:
    try:
        with zf.open(manifest_member) as handle:
            manifest = json.loads(handle.read().decode("utf-8"))
        pathsearcher = manifest["pathsearcher"]
        seqfiller = manifest["seqfiller"]
    except (KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
        raise RNovAError(
            f"Invalid checkpoint manifest {manifest_member.filename} in {archive_path}"
        ) from exc

    members_by_name = {_normalize_zip_name(member.filename): member for member in zf.infolist()}
    try:
        return members_by_name[_normalize_zip_name(pathsearcher)], members_by_name[_normalize_zip_name(seqfiller)]
    except KeyError as exc:
        raise RNovAError(
            f"Checkpoint manifest references missing archive member: {exc}"
        ) from exc


def _normalize_zip_name(name: str) -> str:
    return str(Path(name.replace("\\", "/")).as_posix()).lstrip("./")


@contextmanager
def _named_temp_file(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=".rnova-",
        suffix=".tmp",
        dir=directory,
        delete=False,
    )
    temp_path = Path(handle.name)
    handle.close()
    try:
        yield temp_path
    finally:
        if temp_path.exists():
            temp_path.unlink()
