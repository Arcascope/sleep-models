"""Image-copy auth boundaries, exercised without cloud access or Docker."""

import base64
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DIGEST = "sha256:" + "a" * 64
ECR = "111122223333.dkr.ecr.us-east-1.amazonaws.com"
AR = "us-central1-docker.pkg.dev"
JWT = "test-google-identity"
ECR_PASSWORD = "test-ecr-password"
GOOGLE_TOKEN = "test-google-token"
SECRETS = [
    JWT,
    ECR_PASSWORD,
    GOOGLE_TOKEN,
    "test-access-key",
    "test-secret-key",
    "test-session-token",
]

MOCK = r"""
import base64, json, os, pathlib, stat, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
root = pathlib.Path(os.environ["MOCK_ROOT"])
with (root / "calls.jsonl").open("a") as f:
    f.write(json.dumps([name, args]) + "\n")
digest = "sha256:" + "a" * 64
direction = os.environ.get("DIRECTION", "ecr-to-ar")
if name == "curl":
    url = next(a for a in args if a.startswith("http://"))
    if "/attributes/" in url:
        print(json.loads(os.environ["METADATA"])[url.rsplit("/", 1)[-1]], end="")
    elif "/identity" in url:
        print("test-google-identity", end="")
    elif "/token" in url:
        print(json.dumps({"access_token": "test-google-token"}))
    else:
        raise AssertionError(args)
elif name == "docker":
    assert "--log-driver=none" in args, "credential stdout must not enter daemon logs"
    mount = args[args.index("--volume") + 1].split(":")[0]
    auth = pathlib.Path(mount)
    assert stat.S_IMODE(auth.stat().st_mode) == 0o700
    for p in auth.iterdir():
        assert stat.S_IMODE(p.stat().st_mode) == 0o600
    if "assume-role-with-web-identity" in args:
        assert (auth / "google-identity.jwt").read_text() == "test-google-identity"
        assert "--web-identity-token" in args
        if os.environ.get("FAIL_STAGE") == "sts":
            sys.exit(13)
        print("test-access-key\ttest-secret-key\ttest-session-token")
    elif "get-login-password" in args:
        credentials = (auth / "aws-credentials").read_text()
        assert "aws_session_token = test-session-token" in credentials
        if os.environ.get("FAIL_STAGE") == "ecr":
            sys.exit(14)
        print("test-ecr-password")
    elif "copy" in args:
        assert "--authfile" in args and "--src-creds" not in args and "--dest-creds" not in args
        assert sorted(p.name for p in auth.iterdir()) == ["registry-auth.json"]
        config = json.loads((auth / "registry-auth.json").read_text())
        decoded = {k: base64.b64decode(v["auth"]).decode() for k, v in config["auths"].items()}
        (root / "copied-auth.json").write_text(json.dumps(decoded))
        if os.environ.get("FAIL_STAGE") == "copy":
            sys.exit(15)
    else:
        raise AssertionError(args)
elif name == "sleep":
    pass
elif name in ("aws", "gcloud"):
    if name == "aws":
        assert args[:2] == ["ecr", "describe-images"], "launcher must not mint registry credentials"
        assert "--registry-id" in args
        source = args[args.index("--image-ids") + 1] in ("imageTag=src", "imageDigest=" + digest)
        assert ":" not in args[args.index("--repository-name") + 1]
    elif args[:4] == ["artifacts", "docker", "images", "describe"]:
        source = args[4].endswith(":src") or "@" in args[4]
        if "@" in args[4]:
            assert ":" not in args[4].split("@")[0]
    elif args[:3] == ["compute", "instances", "create"]:
        assert any(a.startswith("--service-account=") for a in args)
        assert not any("password=" in a or "token=" in a for a in args)
        if os.environ.get("HOST_FAILURE") == "create":
            sys.exit(9)
        sys.exit(0)
    elif args[:3] == ["compute", "instances", "get-serial-port-output"]:
        (root / "copied").touch()
        code = "17" if os.environ.get("HOST_FAILURE") == "copy" else "0"
        print("copy-image: start\ncopy-image: exit " + code)
        sys.exit(0)
    elif args[:3] == ["compute", "instances", "delete"]:
        counter = root / "delete-count"
        count = int(counter.read_text()) + 1 if counter.exists() else 1
        counter.write_text(str(count))
        if os.environ.get("DELETE_ALREADY_GONE"):
            print("The resource was not found", file=sys.stderr)
            sys.exit(1)
        failed = count <= int(os.environ.get("DELETE_FAILURES", "0"))
        if os.environ.get("HOST_FAILURE") == "cleanup" or failed:
            print("Transient deletion error", file=sys.stderr)
            sys.exit(1)
        if count > int(os.environ.get("DELETE_NOOPS", "0")):
            (root / "deleted").touch()
        sys.exit(0)
    elif args[:3] == ["compute", "instances", "describe"]:
        if os.environ.get("VERIFY_DENIED"):
            print("Permission denied verifying VM", file=sys.stderr)
            sys.exit(1)
        if (root / "deleted").exists() or os.environ.get("DELETE_ALREADY_GONE"):
            print("The resource was not found", file=sys.stderr)
            sys.exit(1)
        print("RUNNING")
        sys.exit(0)
    else:
        raise AssertionError(args)
    if os.environ.get("LOOKUP_DENIED"):
        print("AccessDeniedException: test lookup denied", file=sys.stderr)
        sys.exit(1)
    if source or (root / "copied").exists() or os.environ.get("EXISTING"):
        if not source and os.environ.get("EXISTING") == "different":
            print("sha256:" + "b" * 64)
        else:
            print(digest)
    else:
        print("ImageNotFoundException" if name == "aws" else "NOT_FOUND", file=sys.stderr)
        sys.exit(1)
else:
    raise AssertionError(name)
"""


@pytest.fixture
def harness(tmp_path):
    binary = tmp_path / "bin"
    binary.mkdir()
    for name in ("aws", "gcloud", "curl", "docker"):
        tool = binary / name
        tool.write_text(f"#!{sys.executable}\n" + MOCK)
        tool.chmod(0o755)
    runtime = tmp_path / "run"
    runtime.mkdir()
    env = {
        **os.environ,
        "PATH": str(binary) + os.pathsep + os.environ["PATH"],
        "MOCK_ROOT": str(tmp_path),
        "COPY_RUNTIME_DIR": str(runtime),
        "COPY_SERIAL_DEVICE": str(tmp_path / "serial.log"),
        "ARCASLEEP_PROJECT": "example-project",
        "COPY_SERVICE_ACCOUNT": "copy-ecr-to-ar@example-project.iam.gserviceaccount.com",
        "COPY_AWS_ROLE_ARN": "arn:aws:iam::111122223333:role/image-copy-ecr-pull",
    }
    return tmp_path, env


def images(direction):
    ecr = ECR + "/training/data"
    ar = AR + "/example-project/training/data"
    return (
        (ecr + ":src", ar + ":dest") if direction == "ecr-to-ar" else (ar + ":src", ecr + ":dest")
    )


def calls(root):
    return [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("direction", ["ecr-to-ar", "ar-to-ecr"])
def test_startup_keeps_credentials_out_of_metadata_arguments_and_logs(harness, direction):
    root, env = harness
    source, destination = images(direction)
    env["METADATA"] = json.dumps(
        {
            "src": source,
            "dst": destination,
            "src-kind": "ecr" if direction == "ecr-to-ar" else "ar",
            "aws-role": env["COPY_AWS_ROLE_ARN"],
            "aws-audience": "test-image-copy",
            "ecr-region": "us-east-1",
            "awscli": "test-awscli",
            "skopeo": "test-skopeo",
        }
    )
    result = subprocess.run(
        ["bash", str(ROOT / "deploy/gcp/copy-image-startup.sh")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    decoded = json.loads((root / "copied-auth.json").read_text())
    assert decoded == {ECR: "AWS:" + ECR_PASSWORD, AR: "oauth2accesstoken:" + GOOGLE_TOKEN}
    exposed = result.stdout + result.stderr + (root / "serial.log").read_text()
    exposed += json.dumps(calls(root)) + env["METADATA"]
    for secret in SECRETS:
        assert secret not in exposed
        assert base64.b64encode(secret.encode()).decode() not in exposed
    assert not list((root / "run").iterdir())
    assert (root / "serial.log").read_text().endswith("copy-image: exit 0\n")


@pytest.mark.parametrize("stage,code", [("sts", 13), ("ecr", 14), ("copy", 15)])
def test_startup_failures_report_failure_and_erase_credentials(harness, stage, code):
    root, env = harness
    source, destination = images("ecr-to-ar")
    env["FAIL_STAGE"] = stage
    env["METADATA"] = json.dumps(
        {
            "src": source,
            "dst": destination,
            "src-kind": "ecr",
            "aws-role": env["COPY_AWS_ROLE_ARN"],
            "aws-audience": "test-image-copy",
            "ecr-region": "us-east-1",
            "awscli": "test-awscli",
            "skopeo": "test-skopeo",
        }
    )
    result = subprocess.run(
        ["bash", str(ROOT / "deploy/gcp/copy-image-startup.sh")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == code
    assert (root / "serial.log").read_text().endswith(f"copy-image: exit {code}\n")
    assert not list((root / "run").iterdir())
    if stage != "copy":
        assert not (root / "copied-auth.json").exists()
    for secret in SECRETS:
        assert secret not in result.stdout + result.stderr + json.dumps(calls(root))


@pytest.mark.parametrize("direction", ["ecr-to-ar", "ar-to-ecr"])
@pytest.mark.parametrize("reference", ["tag", "digest", "tag-and-digest"])
def test_launcher_pins_source_and_attaches_explicit_identity(harness, direction, reference):
    root, env = harness
    env["DIRECTION"] = direction
    source, destination = images(direction)
    source_repo = source.rsplit(":", 1)[0]
    if reference == "digest":
        source = source_repo + "@" + DIGEST
    elif reference == "tag-and-digest":
        source += "@" + DIGEST
    result = subprocess.run(
        ["bash", str(ROOT / "deploy/scripts/copy_image.sh"), source, destination],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    events = calls(root)
    creates = [args for name, args in events if args[:3] == ["compute", "instances", "create"]]
    assert len(creates) == 1
    create = creates[0]
    assert "--service-account=" + env["COPY_SERVICE_ACCOUNT"] in create
    metadata = next(a for a in create if a.startswith("--metadata="))
    assert "src=" + source_repo + "@" + DIGEST in metadata
    assert "aws-role=" + env["COPY_AWS_ROLE_ARN"] in metadata
    assert next(a for a in create if a.startswith("--metadata-from-file=")).endswith(
        "copy-image-startup.sh"
    )
    assert any(args[:3] == ["compute", "instances", "delete"] for _, args in events)


def test_dry_run_never_launches_or_obtains_credentials(harness):
    root, env = harness
    result = subprocess.run(
        ["bash", str(ROOT / "deploy/scripts/copy_image.sh"), "--dry-run", *images("ecr-to-ar")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert all(args[:3] != ["compute", "instances", "create"] for _, args in calls(root))
    assert all(name != "docker" and "get-login-password" not in args for name, args in calls(root))


def test_lookup_permission_error_aborts_before_vm_creation(harness):
    root, env = harness
    env["LOOKUP_DENIED"] = "1"
    result = subprocess.run(
        ["bash", str(ROOT / "deploy/scripts/copy_image.sh"), *images("ecr-to-ar")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "AccessDeniedException" in result.stderr
    assert not any(args[:3] == ["compute", "instances", "create"] for _, args in calls(root))


@pytest.mark.parametrize("value,code", [("same", 0), ("different", 1)])
def test_existing_destination_is_never_overwritten(harness, value, code):
    root, env = harness
    env["EXISTING"] = value
    result = subprocess.run(
        ["bash", str(ROOT / "deploy/scripts/copy_image.sh"), *images("ecr-to-ar")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == code
    assert not any(args[:3] == ["compute", "instances", "create"] for _, args in calls(root))


@pytest.mark.parametrize("key", ["COPY_SERVICE_ACCOUNT", "COPY_AWS_ROLE_ARN", "ARCASLEEP_PROJECT"])
def test_missing_identity_configuration_fails_before_cloud_calls(harness, key):
    root, env = harness
    env.pop(key)
    result = subprocess.run(
        ["bash", str(ROOT / "deploy/scripts/copy_image.sh"), "--dry-run", *images("ecr-to-ar")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert not (root / "calls.jsonl").exists()


def test_iam_template_is_directional_and_repository_scoped():
    template = json.loads((ROOT / "deploy/aws/image-copy-auth.json").read_text())
    resources = template["Resources"]
    assert set(resources) == {"EcrPullRole", "EcrPushRole"}
    pattern = template["Parameters"]["EcrRepositoryArns"]["AllowedPattern"]
    assert re.fullmatch(pattern, "arn:aws:ecr:us-east-1:111122223333:repository/training/data")
    assert not re.fullmatch(pattern, "arn:aws:ecr:us-east-1:111122223333:repository/*")
    for name, subject in [
        ("EcrPullRole", "GoogleEcrToArSubject"),
        ("EcrPushRole", "GoogleArToEcrSubject"),
    ]:
        role = resources[name]["Properties"]
        trust = role["AssumeRolePolicyDocument"]["Statement"][0]
        assert trust["Principal"] == {"Federated": "accounts.google.com"}
        assert trust["Action"] == "sts:AssumeRoleWithWebIdentity"
        assert trust["Condition"]["StringEquals"] == {
            "accounts.google.com:aud": {"Ref": subject},
            "accounts.google.com:sub": {"Ref": subject},
            "accounts.google.com:oaud": {"Ref": "Audience"},
        }
        statements = role["Policies"][0]["PolicyDocument"]["Statement"]
        for statement in statements:
            if statement["Resource"] == "*":
                assert statement["Action"] == ["ecr:GetAuthorizationToken"]
            else:
                assert statement["Resource"] == {"Ref": "EcrRepositoryArns"}
            assert all(a.startswith("ecr:") and "Delete" not in a for a in statement["Action"])
        actions = {a for s in statements for a in s["Action"]}
        if name == "EcrPullRole":
            assert not any(a in actions for a in ["ecr:PutImage", "ecr:UploadLayerPart"])
        else:
            assert "ecr:PutImage" in actions
            assert "ecr:GetDownloadUrlForLayer" not in actions


@pytest.mark.parametrize("stage", ["create", "copy", "cleanup"])
def test_launcher_cleans_up_its_vm_and_reports_failures(harness, stage):
    root, env = harness
    env["HOST_FAILURE"] = stage
    result = subprocess.run(
        ["bash", str(ROOT / "deploy/scripts/copy_image.sh"), *images("ecr-to-ar")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert any(args[:3] == ["compute", "instances", "delete"] for _, args in calls(root))


def test_wrong_aws_account_fails_before_cloud_calls(harness):
    root, env = harness
    env["COPY_AWS_ROLE_ARN"] = "arn:aws:iam::444455556666:role/image-copy-ecr-pull"
    result = subprocess.run(
        ["bash", str(ROOT / "deploy/scripts/copy_image.sh"), *images("ecr-to-ar")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert not (root / "calls.jsonl").exists()


@pytest.mark.parametrize(
    "setting,value,attempts",
    [
        ("DELETE_FAILURES", "1", 2),
        ("DELETE_NOOPS", "1", 2),
        ("DELETE_ALREADY_GONE", "1", 1),
    ],
)
def test_cleanup_retries_and_confirms_only_its_own_vm(harness, setting, value, attempts):
    root, env = harness
    env[setting] = value
    result = subprocess.run(
        ["bash", str(ROOT / "deploy/scripts/copy_image.sh"), *images("ecr-to-ar")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    events = calls(root)
    created = next(args[3] for _, args in events if args[:3] == ["compute", "instances", "create"])
    deletions = [args for _, args in events if args[:3] == ["compute", "instances", "delete"]]
    descriptions = [args for _, args in events if args[:3] == ["compute", "instances", "describe"]]
    assert len(deletions) == len(descriptions) == attempts
    assert all(args[3] == created for args in deletions + descriptions)
    assert not any(args[:3] == ["compute", "instances", "list"] for _, args in events)


def test_cleanup_fails_when_deletion_cannot_be_verified(harness):
    root, env = harness
    env["VERIFY_DENIED"] = "1"
    result = subprocess.run(
        ["bash", str(ROOT / "deploy/scripts/copy_image.sh"), *images("ecr-to-ar")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "VM cleanup failed after 3 attempts" in result.stderr
    assert sum(args[:3] == ["compute", "instances", "describe"] for _, args in calls(root)) == 3
