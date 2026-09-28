from pathlib import Path

from autofish.artifacts.provenance import (
    ManifestEntry,
    build_local_manifest,
    code_digest,
    data_digest,
    manifest_digest,
    run_fingerprint,
    write_manifest,
)


def test_manifest_digest_is_order_independent() -> None:
    entries = [
        ManifestEntry(path="b/file", size_bytes=2, sha256="b" * 64),
        ManifestEntry(path="a/file", size_bytes=1, sha256="a" * 64),
    ]
    assert manifest_digest(entries) == manifest_digest(reversed(entries))


def test_local_manifest_uses_relative_paths_and_content_hashes(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "two.txt").write_text("two")
    (tmp_path / "one.txt").write_text("one")
    entries = build_local_manifest(tmp_path)
    assert [entry.path for entry in entries] == ["nested/two.txt", "one.txt"]
    output = tmp_path / "manifest.jsonl"
    digest = write_manifest(entries, output)
    assert len(digest) == 64
    assert output.read_text().endswith("\n")


def test_run_fingerprint_is_stable_and_separates_both_halves() -> None:
    base = run_fingerprint(data_digest="a" * 64, code_digest="c3ed1a9")
    assert base == run_fingerprint(data_digest="a" * 64, code_digest="c3ed1a9")
    assert len(base) == 12
    # Either half changing must change the id -- that is the whole point.
    assert base != run_fingerprint(data_digest="b" * 64, code_digest="c3ed1a9")
    assert base != run_fingerprint(data_digest="a" * 64, code_digest="c3ed1a9-dirty")


def test_data_digest_hashes_the_manifest_and_is_none_when_absent(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("AUTOFISH_MANIFEST_PATH", raising=False)
    assert data_digest(tmp_path / "missing.jsonl") is None
    manifest = tmp_path / "manifest.jsonl"
    entries = [ManifestEntry(path="a/file", size_bytes=1, sha256="a" * 64)]
    written = write_manifest(entries, manifest)
    # The digest write_manifest returns is the digest of the bytes on disk.
    assert data_digest(manifest) == written == manifest_digest(entries)


def test_code_digest_prefers_baked_git_commit(monkeypatch) -> None:
    monkeypatch.setenv("GIT_COMMIT", "c3ed1a9-dirty")
    assert code_digest() == "c3ed1a9-dirty"
