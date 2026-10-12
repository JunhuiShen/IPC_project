#!/usr/bin/env python3
"""Record an isolated source snapshot, execute explicit tests, then seal evidence.

This helper does not infer success from existing logs. `run` executes the
command after `--` and records its exit code and immutable inputs; `seal`
requires successful receipts for the same candidate/source snapshot. No build
or test is chosen automatically. Run the parent test chain only when idle.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import subprocess
import time

from profile_contact_round2_full import write_json
from profile_example1_full import sha256
from validate_optimized_example1_full import valid_sha
from wait_example1_comparison import stop_our_process

EXCLUDED = {".git", ".hg", ".svn", "__pycache__", "build", "_deps", "outputs"}


def source_files(root: Path) -> dict:
    files = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in EXCLUDED for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError("Source snapshots must copy contents, not symlink sources: " + str(path))
        if path.is_file():
            files[str(relative)] = dict(bytes=path.stat().st_size, sha256=sha256(path))
    if not {"CMakeLists.txt", "gpu/CMakeLists.txt"} <= set(files):
        raise ValueError("Source snapshot requires CPU and GPU CMakeLists.txt")
    return files


def verify_source(path: Path) -> dict:
    document = json.loads(path.read_text())
    root = Path(document["root"]).resolve(strict=True)
    if not Path(document["root"]).is_absolute() or document.get("files") != source_files(root):
        raise ValueError("Isolated source snapshot changed, gained/lost a file, or has an invalid root")
    return document


def fresh(path: Path) -> Path:
    path = path.resolve()
    if path.exists():
        raise ValueError("Refusing to replace an existing evidence file: " + str(path))
    if not path.parent.is_dir():
        raise ValueError("Evidence parent directory must already exist: " + str(path.parent))
    return path


def manifest(source_root: Path, output: Path) -> None:
    root = source_root.resolve(strict=True);output = fresh(output)
    if output.is_relative_to(root):
        raise ValueError("Put manifest outside the isolated source tree")
    write_json(output, dict(kind="isolated working-source snapshot", root=str(root), files=source_files(root),
        excluded_components=sorted(EXCLUDED), created_unix=time.time(),
        note="Working contents, including dirty changes. Build directory must be outside this tree; no claim of Git cleanliness or baseline-binary provenance."))


def run_test(candidate: Path, source_manifest: Path, log: Path, receipt: Path,
             argv: list[str], cwd: Path | None = None, timeout: float = 7200) -> int:
    if not argv:
        raise ValueError("An explicit command after -- is required; existing logs never imply success")
    candidate = candidate.resolve(strict=True);source_manifest = source_manifest.resolve(strict=True)
    log, receipt = fresh(log), fresh(receipt)
    if log == receipt:
        raise ValueError("Log and receipt must be different files")
    snapshot = verify_source(source_manifest)
    if log.is_relative_to(Path(snapshot["root"])) or receipt.is_relative_to(Path(snapshot["root"])):
        raise ValueError("Keep test artifacts outside the frozen source tree")
    pinned = {candidate: sha256(candidate), source_manifest: sha256(source_manifest)}
    result = dict(kind="executed validation command", status="running", command=argv,
        cwd=str((cwd or Path.cwd()).resolve()), candidate=str(candidate), candidate_sha256=pinned[candidate],
        source_manifest=dict(path=str(source_manifest), sha256=pinned[source_manifest]), log=str(log),
        started_unix=time.time(), returncode=None)
    write_json(receipt, result)
    process = None;started = time.monotonic()

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Termination signal {signum}")

    previous_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        with log.open("w") as stream:
            process = subprocess.Popen(argv, cwd=result["cwd"], stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            result["child_pid"] = process.pid;write_json(receipt, result)
            while process.poll() is None:
                if time.monotonic() - started > timeout:
                    raise TimeoutError("Validation command exceeded its deadline")
                try:
                    process.wait(timeout=min(10, max(.01, timeout - (time.monotonic() - started))))
                except subprocess.TimeoutExpired:
                    pass
        result["returncode"] = process.returncode
        verify_source(source_manifest)
        if any(sha256(path) != digest for path, digest in pinned.items()):
            raise RuntimeError("Candidate/source manifest changed while tests ran")
        result["status"] = "passed" if process.returncode == 0 else "failed"
    except BaseException as error:
        if process is not None:
            stop_our_process(process)
        result.update(status="failed", error=f"{type(error).__name__}: {error}",
                      returncode=process.returncode if process is not None else None)
        raise
    finally:
        if process is not None:
            stop_our_process(process)
        result.update(finished_unix=time.time(), elapsed_s=time.monotonic() - started,
                      log_sha256=sha256(log) if log.is_file() else None)
        write_json(receipt, result)
        signal.signal(signal.SIGTERM, previous_handler)
    return result["returncode"]


def seal(candidate: Path, source_manifest: Path, receipts: list[Path], output: Path) -> None:
    if not receipts:
        raise ValueError("At least one explicitly executed successful test command is required")
    output = fresh(output);candidate = candidate.resolve(strict=True);source_manifest = source_manifest.resolve(strict=True)
    verify_source(source_manifest)
    candidate_hash, source_hash = sha256(candidate), sha256(source_manifest)
    evidence, records = {}, []
    for path in receipts:
        path = path.resolve(strict=True);record = json.loads(path.read_text())
        if (record.get("kind") != "executed validation command" or record.get("status") != "passed"
                or type(record.get("returncode")) is not int or record["returncode"] != 0
                or record.get("candidate_sha256") != candidate_hash or not record.get("command")
                or record.get("source_manifest") != dict(path=str(source_manifest), sha256=source_hash)
                or not record.get("finished_unix")):
            raise ValueError("Receipt does not certify successful tests for these exact inputs: " + str(path))
        log = Path(record["log"])
        if (not log.is_absolute() or not log.is_file() or not log.stat().st_size
                or not valid_sha(record.get("log_sha256")) or sha256(log) != record["log_sha256"]):
            raise ValueError("Missing/empty/changed validation log: " + str(log))
        if str(log) in evidence:
            raise ValueError("Duplicate validation log in receipts")
        evidence[str(log)] = record["log_sha256"]
        records.append(dict(path=str(path), sha256=sha256(path), command=record["command"]))
    write_json(output, dict(validation_chain_exit_code=0, candidate_sha256=candidate_hash,
        source_manifest=dict(path=str(source_manifest), sha256=source_hash), evidence=evidence,
        command_receipts=records, sealed_unix=time.time(),
        note="Every listed command was explicitly executed and exited0 on this candidate/source snapshot. The caller chooses sufficient regression/sanitizer coverage; no tests are inferred from unrelated old logs."))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    sub = parser.add_subparsers(dest="action")
    make = sub.add_parser("manifest");make.add_argument("--source-root", type=Path, required=True);make.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run")
    finish = sub.add_parser("seal")
    for child in (run, finish):
        child.add_argument("--candidate", type=Path, required=True)
        child.add_argument("--source-manifest", type=Path, required=True)
    run.add_argument("--log", type=Path, required=True);run.add_argument("--receipt", type=Path, required=True)
    run.add_argument("--cwd", type=Path);run.add_argument("--timeout", type=float, default=7200)
    run.add_argument("command", nargs=argparse.REMAINDER)
    finish.add_argument("--receipt", type=Path, action="append", required=True);finish.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.self_test:
        self_test();return
    if args.action == "manifest":
        manifest(args.source_root, args.output)
    elif args.action == "run":
        if not 0 < args.timeout < float("inf"):
            parser.error("Positive finite timeout required")
        argv = args.command[1:] if args.command[:1] == ["--"] else args.command
        raise SystemExit(run_test(args.candidate, args.source_manifest, args.log, args.receipt, argv, args.cwd, args.timeout))
    elif args.action == "seal":
        seal(args.candidate, args.source_manifest, args.receipt, args.output)
    else:
        parser.error("Select manifest, run or seal")


def self_test() -> None:
    import sys
    import tempfile
    from validate_contact_round3 import validate_marker
    with tempfile.TemporaryDirectory(prefix="record_gpu_validation_") as name:
        root = Path(name);source = root / "source";source.mkdir();(source / "gpu").mkdir()
        for relative in ("CMakeLists.txt", "gpu/CMakeLists.txt"):
            (source / relative).write_text("test source\n")
        binary = root / "candidate";binary.write_text("pinned binary placeholder")
        inputs = root / "source.json";manifest(source, inputs)
        good_log, good_receipt = root / "good.log", root / "good.json"
        assert run_test(binary, inputs, good_log, good_receipt, [sys.executable, "-c", "print('test passed')"]) == 0
        marker = root / "validation_exit.json";seal(binary, inputs, [good_receipt], marker)
        assert validate_marker(marker, sha256(binary))[0]["validation_chain_exit_code"] == 0
        bad_log, bad_receipt = root / "bad.log", root / "bad.json"
        assert run_test(binary, inputs, bad_log, bad_receipt, [sys.executable, "-c", "print('test failed');raise SystemExit(2)"]) == 2
        try:
            seal(binary, inputs, [bad_receipt], root / "must_not_exist.json")
        except ValueError:
            pass
        else:
            raise AssertionError("Failed command accepted")
        assert not (root / "must_not_exist.json").exists()
        (source / "gpu/CMakeLists.txt").write_text("changed\n")
        try:
            seal(binary, inputs, [good_receipt], root / "changed.json")
        except ValueError:
            pass
        else:
            raise AssertionError("Changed source accepted")
    print("Validation evidence recorder self-test PASS (executed success/failure, sealed exact inputs, rejected changed sources)")


if __name__ == "__main__":
    main()
