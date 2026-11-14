"""Verify the complete Newton snapshot and its explicitly recorded local patch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "third_party" / "newton"
COMMIT = "f89e82fed043fe78d1844b26d3f289a25db03810"


def verify(root: Path = ROOT, *, pristine: bool = False) -> int:
    root = root.resolve()
    manifest = json.loads((root / "UPSTREAM.json").read_text(encoding="utf-8"))
    if manifest["commit"] != COMMIT:
        raise ValueError("Unexpected Newton revision")
    expected = manifest["files"]
    fork_path = root / "FORK.json"
    fork = json.loads(fork_path.read_text(encoding="utf-8")) if fork_path.exists() else {"files": {}}
    local_changes = fork["files"]
    if local_changes and fork.get("base_commit") != COMMIT:
        raise ValueError("Fork metadata references an unexpected upstream revision")
    if pristine and local_changes:
        raise ValueError("This is an explicitly patched upstream solver; see FORK.json")
    for name, record in local_changes.items():
        if name not in expected or record["upstream_sha256"] != expected[name]:
            raise ValueError(f"Fork baseline mismatch: {name}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.relative_to(root).parts
        and path.suffix not in (".pyc", ".pyo")
        and path.name not in ("UPSTREAM.json", "FORK.json")
    }
    errors = [f"unrecorded: {name}" for name in sorted(actual - expected.keys())]
    for name, original_digest in expected.items():
        digest = local_changes.get(name, {}).get("sha256", original_digest)
        path = root / name
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            errors.append(f"invalid source path: {name}")
        elif name not in actual:
            errors.append(f"missing: {name}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            errors.append(f"modified: {name}")
    package_count = sum(name.startswith("newton/") for name in expected)
    if package_count != manifest["package_files"] or package_count != 1108:
        errors.append("Incomplete Newton Python package inventory")
    if errors:
        raise ValueError("Newton source verification failed:\n" + "\n".join(errors))
    return len(expected)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", action="store_true", help="verify the exact unpatched reference snapshot")
    options = parser.parse_args()
    selected = ROOT.with_name("newton_reference") if options.reference else ROOT
    try:
        count = verify(selected, pristine=options.reference)
    except (OSError, KeyError, ValueError) as error:
        raise SystemExit(str(error)) from error
    fork = selected / "FORK.json"
    changes = len(json.loads(fork.read_text())["files"]) if fork.exists() else 0
    print(f"PASS: {count} source files at {COMMIT}; {changes} explicitly recorded local patches")
