#!/usr/bin/env python3
"""Wait for terminal GPU optimization status, then download reports and GEOs.

Binary checkpoints/executables remain on the server. No simulation is started
or stopped. A held comparison lock prevents our transfer from overlapping a
benchmark using the same lock; a separate activity check also watches other
simulations, GPU processes, compilers and profilers. Suspended jobs are ignored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import signal
import subprocess
import time

from example1_comparison import parse_trial

HOST = "zhhzhao@biot.math.ucdavis.edu"
REMOTE_GPU = PurePosixPath("/homes/home01/z/zhhzhao/Desktop/IPC_project/3D_IPC/gpu")

# Kept inline so the downloader requires no additional server installation.
# Reading /proc and nvidia-smi does not modify remote simulation state.
REMOTE_PROBE = r'''
import json,pathlib,pwd,os,socket,subprocess,sys
root=pathlib.Path(sys.argv[1]); allowed=pathlib.Path(sys.argv[2])
if root.resolve()==allowed.resolve() or allowed.resolve() not in root.resolve().parents:
    raise RuntimeError("remote task directory escapes the GPU workspace")
user=pwd.getpwuid(os.getuid()).pw_name
hostname=socket.gethostname()
if user!="zhhzhao" or hostname.split(".")[0]!="biot":
    raise RuntimeError("SSH connection is not the expected zhhzhao@biot server")
status=json.loads((root/"status.json").read_text())
active=[]
for path in pathlib.Path("/proc").iterdir():
    if not path.name.isdigit(): continue
    try:
        raw=(path/"stat").read_text(); name,tail=raw[raw.index("(")+1:].rsplit(")",1)
        state=tail.split()[0]
        if state in "TtZ": continue
        if name.startswith(("3D_sim","cap_sim","compute-saniti")) or name in ("nvcc","ptxas","ncu","nsys"):
            active.append({"pid":int(path.name),"name":name,"state":state})
    except (FileNotFoundError,ProcessLookupError,PermissionError): pass
pids=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True,timeout=10)
suspended=set()
for line in pids.splitlines():
    if not line.strip(): continue
    pid=int(line.strip())
    try:
        raw=pathlib.Path(f"/proc/{pid}/stat").read_text(); state=raw[raw.rfind(")")+1:].split()[0]
        if state in "TtZ": suspended.add(pid)
    except (FileNotFoundError,ProcessLookupError,PermissionError): pass
gpu_pids=[int(line.strip()) for line in pids.splitlines() if line.strip() and int(line.strip()) not in suspended]
print(json.dumps({"status":status,"active_jobs":active,"active_gpu_pids":gpu_pids,"user":user,"hostname":hostname}))
'''

# Acquire the EXISTING benchmark lock read-only in the wrapper process. Its
# descriptor stays open while rsync runs, even if rsync closes inherited FDs.
REMOTE_RSYNC = r'''
import fcntl,os,subprocess,sys
fd=os.open(sys.argv[1],os.O_RDONLY)
fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
sys.exit(subprocess.call(["rsync",*sys.argv[2:]]))
'''


def validate_remote(value: str) -> PurePosixPath:
    remote = PurePosixPath(value)
    if (not remote.is_absolute() or ".." in remote.parts or remote == REMOTE_GPU
            or REMOTE_GPU not in remote.parents):
        raise ValueError("remote-root must identify a task directory inside the server GPU workspace")
    return remote


def full_relative(remote: PurePosixPath, state: dict) -> PurePosixPath:
    value = state.get("full_directory", str(remote / "full_validation"))
    if not isinstance(value, str) or not value:
        raise ValueError("Invalid full_directory in supervisor status")
    full = PurePosixPath(value)
    if not full.is_absolute():
        full = remote / full
    if ".." in full.parts or remote not in full.parents or full == remote:
        raise ValueError("full_directory must stay beneath the optimization task directory")
    return full.relative_to(remote)


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify_geos(full: Path, count: int) -> dict:
    manifest_path = full / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = {f"{prefix}{i:04d}{suffix}" for i in range(count + 1)
                for prefix, suffix in (("frame_", ".geo"), ("state_", ".bin"))}
    files = manifest.get("files", {})
    if manifest.get("frames") != count or set(files) != expected:
        raise ValueError("Manifest does not contain exactly every GEO/binary checkpoint")
    actual_geos = {path.name for path in (full / "frames").glob("frame_*.geo")}
    if actual_geos != {name for name in expected if name.endswith(".geo")}:
        raise ValueError("Downloaded GEO set differs from the complete manifest")
    for name, item in files.items():
        if (not isinstance(item, dict) or type(item.get("bytes")) is not int or item["bytes"] <= 0
                or not isinstance(item.get("sha256"), str) or len(item["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in item["sha256"])):
            raise ValueError("Invalid checkpoint/GEO manifest entry: " + name)
        if name.endswith(".geo"):
            path = full / "frames" / name
            if path.is_symlink() or path.stat().st_size != item["bytes"] or file_hash(path) != item["sha256"]:
                raise ValueError("Downloaded GEO differs from validated server output: " + name)
    return dict(validated_geo_frames=count + 1, binary_checkpoints="Retained on server; manifest downloaded but binary bytes not downloaded",
                checkpoint_manifest=str(manifest_path))


def transfer_arguments(ssh: list[str], remote: PurePosixPath, local: Path) -> list[str]:
    remote_rsync = shlex.join(["python3", "-c", REMOTE_RSYNC, str(REMOTE_GPU / ".example1_comparison.lock")])
    return ["rsync", "-av", "--prune-empty-dirs", "--exclude=*scratch*/", "--exclude=build*/", "--exclude=deps/",
            "--include=*/", "--include=*.geo", "--include=*.csv", "--include=*.json", "--include=*.txt", "--include=*.log",
            "--exclude=*", "--rsync-path=" + remote_rsync, "-e", shlex.join(ssh), f"{HOST}:{remote}/", str(local) + "/"]


def stop_transfer(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if os.getpgid(process.pid) != process.pid:
            raise RuntimeError("Download process group changed; refusing a broad kill")
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait();return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            if os.getpgid(process.pid) == process.pid:
                os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-root")
    parser.add_argument("--local-root", type=Path)
    parser.add_argument("--socket")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test();return
    if not args.remote_root or not args.local_root or not args.socket:
        parser.error("--remote-root, --local-root and --socket are required")
    try:
        remote = validate_remote(args.remote_root)
    except ValueError as error:
        parser.error(str(error))
    local = args.local_root.resolve();local.mkdir(parents=True, exist_ok=True)
    existing = local / "download_status.json"
    if existing.is_file() and json.loads(existing.read_text()).get("remote_root") != str(remote):
        raise ValueError("Local destination already belongs to a different remote task")
    ssh = ["ssh", "-S", args.socket, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
    read_status = shlex.join(["python3", "-c", REMOTE_PROBE, str(remote), str(REMOTE_GPU)])

    def save(status, **fields):
        data = dict(status=status, updated_unix=time.time(), remote_root=str(remote), local_root=str(local), **fields)
        temporary = local / "download_status.tmp"
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        temporary.replace(local / "download_status.json")

    def probe():
        response = subprocess.run(ssh + [HOST, read_status], check=True, capture_output=True, text=True, timeout=30)
        return json.loads(response.stdout)

    def interrupted(signum, frame):
        save("stopped", error=f"Downloader termination signal {signum}")
        # The transfer's finally block stops its own dedicated process group.
        raise KeyboardInterrupt(f"Downloader termination signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    deadline = time.monotonic() + 48 * 3600
    previous = None
    while time.monotonic() < deadline:
        try:
            observation = probe();state = observation["status"]
            key = (state["status"], state.get("phase"), state.get("completed_frames", 0))
            if key != previous:
                print("GPU optimization progress:", key, flush=True);previous = key
            save("waiting", remote_status=state["status"], phase=state.get("phase"),
                 completed_frames=state.get("completed_frames", 0))
            if state["status"] not in ("complete", "stopped"):
                time.sleep(30);continue
            if observation["active_jobs"] or observation["active_gpu_pids"]:
                save("waiting_for_server_idle", remote_status=state["status"],
                    active_jobs=observation["active_jobs"], active_gpu_pids=observation["active_gpu_pids"])
                time.sleep(30);continue
            if shutil.disk_usage(local).free < 2 * 1024**3:
                raise RuntimeError("Less than 2 GiB local free space; transfer deferred")
            relative = full_relative(remote, state)
            save("downloading", remote_status=state["status"], full_directory=str(local / relative))
            transfer = subprocess.Popen(transfer_arguments(ssh, remote, local), start_new_session=True)
            transfer_started = time.monotonic()
            try:
                while transfer.poll() is None:
                    time.sleep(5)
                    current = probe()
                    if current["status"]["status"] not in ("complete", "stopped"):
                        raise RuntimeError("Supervisor resumed work; pausing our transfer")
                    if current["active_jobs"] or current["active_gpu_pids"]:
                        raise RuntimeError("Server work started; pausing our transfer")
                    if time.monotonic() - transfer_started > 3600:
                        raise RuntimeError("Transfer attempt exceeded one hour")
                if transfer.returncode:
                    raise subprocess.CalledProcessError(transfer.returncode, transfer.args)
            finally:
                stop_transfer(transfer)
            downloaded_state = json.loads((local / "status.json").read_text())
            if downloaded_state["status"] != state["status"]:
                raise ValueError("Supervisor status changed during transfer")
            if state["status"] == "stopped":
                save("partial_download", remote_status="stopped", phase=state.get("phase"), error=state.get("error"))
                print("Partial artifacts downloaded; no completed optimization claim.", flush=True)
                return
            full = local / relative
            full_status = json.loads((full / "status.json").read_text())
            count = full_status.get("frames")
            if full_status.get("status") != "complete" or type(count) is not int or count != 240:
                raise ValueError("Full validation did not complete all 240 frames")
            if full_status.get("byte_identical_to_reference") is not True:
                raise ValueError("Server full validation did not certify byte identity")
            validated = verify_geos(full, count)
            parsed = parse_trial(full / "run.log", full / "frames", count, "gpu", readme_command=True)
            validation = dict(parsed=parsed, **validated, server_status=str(full / "status.json"),
                local_visualization_directory=str(full / "frames"),
                note="Local GEO bytes verified against server manifest. Full binary checkpoint identity was validated on the server; binaries remain there.")
            (local / "local_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
            save("complete", frames=count, **validated, visualization_directory=str(full / "frames"),
                 report=str(local / "final_report.txt"), clean_report=str(full / "clean_report.txt"))
            print("Downloaded and validated optimized GPU simulation:", full / "frames", flush=True)
            return
        except (OSError, ValueError, KeyError, subprocess.SubprocessError, RuntimeError) as error:
            save("retrying", error=str(error));print("Retrying optimization download:", error, flush=True)
            time.sleep(30)
    save("stopped", error="48-hour download wait limit reached")


def self_test() -> None:
    import tempfile
    import unittest

    class Tests(unittest.TestCase):
        def test_paths_and_terminal_directory(self):
            remote = validate_remote(str(REMOTE_GPU / "exact_contact_tuning_test/validated_optimization"))
            self.assertEqual(full_relative(remote, {}), PurePosixPath("full_validation"))
            self.assertEqual(full_relative(remote, {"full_directory": str(remote / "full_validation")}), PurePosixPath("full_validation"))
            for path in (str(REMOTE_GPU), "/tmp/unrelated", str(REMOTE_GPU / "task/../escape")):
                with self.assertRaises(ValueError):
                    validate_remote(path)
            with self.assertRaises(ValueError):
                full_relative(remote, {"full_directory": "/tmp/elsewhere"})

        def test_geo_hashes_and_missing_checkpoint_metadata(self):
            with tempfile.TemporaryDirectory(prefix="gpu_download_optimization_test_") as name:
                root = Path(name);(root / "frames").mkdir();files = {}
                for frame in range(2):
                    filename = f"frame_{frame:04d}.geo";path = root / "frames" / filename
                    path.write_text(filename)
                    files[filename] = dict(bytes=path.stat().st_size, sha256=file_hash(path))
                    files[f"state_{frame:04d}.bin"] = dict(bytes=1, sha256="1" * 64)
                manifest = root / "checkpoint_manifest.json"
                manifest.write_text(json.dumps(dict(frames=1, files=files)))
                self.assertEqual(verify_geos(root, 1)["validated_geo_frames"], 2)
                (root / "frames/frame_0001.geo").write_text("corrupt")
                with self.assertRaises(ValueError):
                    verify_geos(root, 1)
                del files["state_0001.bin"]
                manifest.write_text(json.dumps(dict(frames=1, files=files)))
                with self.assertRaises(ValueError):
                    verify_geos(root, 1)

        def test_transfer_filters_and_lock(self):
            compile(REMOTE_PROBE, "remote_probe", "exec")
            compile(REMOTE_RSYNC, "remote_rsync_lock", "exec")
            argv = transfer_arguments(["ssh", "-S", "/tmp/test.sock"], REMOTE_GPU / "task", Path("/tmp/local"))
            self.assertIn("--include=*.geo", argv)
            self.assertNotIn("--include=*.bin", argv)
            self.assertIn("--exclude=*scratch*/", argv)
            self.assertTrue(any(value.startswith("--rsync-path=") and ".example1_comparison.lock" in value for value in argv))

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
