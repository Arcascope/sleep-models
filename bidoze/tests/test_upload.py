import pytest

from bidoze.artifacts.upload import MANIFEST_NAME, UploadTarget, upload_run


class _FakeS3:
    """Records calls; enough to assert on keys, ordering and overwrite checks."""

    def __init__(self, existing=()):
        self.existing = list(existing)
        self.uploaded = []
        self.objects = []

    def list_objects_v2(self, Bucket, Prefix, MaxKeys=1):
        hits = [{"Key": k} for k in self.existing if k.startswith(Prefix)][:MaxKeys]
        return {"Contents": hits} if hits else {}

    def upload_file(self, filename, bucket, key):
        self.uploaded.append(key)

    def put_object(self, Bucket, Key, Body, ContentType=None):
        self.objects.append((Key, Body))


@pytest.fixture
def run_dir(tmp_path):
    root = tmp_path / "run"
    (root / "training" / "fold0" / "model_outputs").mkdir(parents=True)
    (root / "training" / "fold0" / "training.log").write_text("log\n")
    (root / "training" / "fold0" / "model_outputs" / "m.pisces").write_bytes(b"weights")
    return root


def test_target_parses_bucket_and_prefix():
    t = UploadTarget.from_uri("s3://bucket/a/b/")
    assert (t.bucket, t.prefix) == ("bucket", "a/b")
    assert t.uri_for("x.json") == "s3://bucket/a/b/x.json"


@pytest.mark.parametrize("uri", ["/local/path", "s3://", "https://example.com/x"])
def test_target_rejects_non_s3_uris(uri):
    with pytest.raises(ValueError):
        UploadTarget.from_uri(uri)


def test_upload_sends_every_file_with_relative_keys(run_dir):
    client = _FakeS3()
    written = upload_run(run_dir, "s3://b/out/run1", client=client)
    assert sorted(client.uploaded) == [
        "out/run1/training/fold0/model_outputs/m.pisces",
        "out/run1/training/fold0/training.log",
    ]
    assert written[-1].endswith(MANIFEST_NAME)


def test_manifest_is_written_last_so_it_means_the_rest_arrived(run_dir):
    client = _FakeS3()
    upload_run(run_dir, "s3://b/out/run1", client=client)
    key, body = client.objects[0]
    assert key == f"out/run1/{MANIFEST_NAME}"
    # Every uploaded file is accounted for in the manifest it leaves behind.
    for uploaded in client.uploaded:
        assert uploaded.split("out/run1/")[1].encode() in body


def test_upload_refuses_a_destination_that_already_has_objects(run_dir):
    client = _FakeS3(existing=["out/run1/training/fold0/training.log"])
    with pytest.raises(FileExistsError):
        upload_run(run_dir, "s3://b/out/run1", client=client)
    assert client.uploaded == []  # nothing sent before the refusal


def test_overwrite_allows_replacing_an_existing_destination(run_dir):
    client = _FakeS3(existing=["out/run1/training/fold0/training.log"])
    upload_run(run_dir, "s3://b/out/run1", client=client, overwrite=True)
    assert len(client.uploaded) == 2


def test_dry_run_sends_nothing_but_reports_the_keys(run_dir):
    written = upload_run(run_dir, "s3://b/out/run1", dry_run=True)
    assert len(written) == 3 and all(w.startswith("s3://b/out/run1/") for w in written)


def test_missing_or_empty_run_directory_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        upload_run(tmp_path / "nope", "s3://b/out/run1", dry_run=True)
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        upload_run(tmp_path / "empty", "s3://b/out/run1", dry_run=True)
