from __future__ import annotations

import hashlib
import zipfile

import pytest

from rnova import checkpoints
from rnova.errors import RNovAError


def _write_checkpoint_zip(path, entries: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in entries.items():
            zf.writestr(name, content)


def test_extract_checkpoints_requires_exact_unambiguous_paths(tmp_path) -> None:
    archive = tmp_path / "RNovA_Checkpoint.zip"
    _write_checkpoint_zip(
        archive,
        {
            "RNovA_PathSearcher_Inference/save/rnova.pt": b"path",
            "PathSearcher/save/rnova.pt": b"ambiguous",
            "RNovA_SeqFiller_Inference/save/rnova.pt": b"seq",
        },
    )

    with pytest.raises(RNovAError, match="ambiguous"):
        checkpoints.extract_checkpoints(archive, repo_root=tmp_path / "repo")


def test_extract_checkpoints_places_exact_members_atomically(tmp_path) -> None:
    archive = tmp_path / "RNovA_Checkpoint.zip"
    repo_root = tmp_path / "repo"
    _write_checkpoint_zip(
        archive,
        {
            "RNovA_PathSearcher_Inference/save/rnova.pt": b"path",
            "RNovA_SeqFiller_Inference/save/rnova.pt": b"seq",
        },
    )

    placed = checkpoints.extract_checkpoints(archive, repo_root=repo_root)

    assert len(placed) == 2
    assert (repo_root / "RNovA_PathSearcher_Inference/save/rnova.pt").read_bytes() == b"path"
    assert (repo_root / "RNovA_SeqFiller_Inference/save/rnova.pt").read_bytes() == b"seq"
    assert not list(repo_root.rglob(".rnova-*.tmp"))


def test_download_checkpoints_refuses_unverified_local_archive(tmp_path) -> None:
    archive = tmp_path / "RNovA_Checkpoint.zip"
    _write_checkpoint_zip(
        archive,
        {
            "RNovA_PathSearcher_Inference/save/rnova.pt": b"path",
            "RNovA_SeqFiller_Inference/save/rnova.pt": b"seq",
        },
    )

    with pytest.raises(RNovAError, match="size mismatch"):
        checkpoints.download_checkpoints(
            repo_root=tmp_path / "repo",
            archive=archive,
        )


def test_download_checkpoints_allows_explicit_unverified_archive(tmp_path) -> None:
    archive = tmp_path / "RNovA_Checkpoint.zip"
    repo_root = tmp_path / "repo"
    _write_checkpoint_zip(
        archive,
        {
            "RNovA_PathSearcher_Inference/save/rnova.pt": b"path",
            "RNovA_SeqFiller_Inference/save/rnova.pt": b"seq",
        },
    )

    placed = checkpoints.download_checkpoints(
        repo_root=repo_root,
        archive=archive,
        allow_unverified_archive=True,
    )

    assert placed == (
        [
            repo_root / "RNovA_PathSearcher_Inference/save/rnova.pt",
            repo_root / "RNovA_SeqFiller_Inference/save/rnova.pt",
        ]
    )


def test_verify_archive_uses_checksum_after_size_match(tmp_path, monkeypatch) -> None:
    archive = tmp_path / "RNovA_Checkpoint.zip"
    archive.write_bytes(b"not a real checkpoint archive")
    monkeypatch.setattr(checkpoints, "CHECKPOINT_SIZE", archive.stat().st_size)
    monkeypatch.setattr(checkpoints, "CHECKPOINT_MD5", hashlib.md5(b"different").hexdigest())
    monkeypatch.setattr(checkpoints, "CHECKPOINT_SHA256", None)

    with pytest.raises(RNovAError, match="md5 mismatch"):
        checkpoints.verify_archive(archive)
