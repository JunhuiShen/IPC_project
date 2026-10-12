#!/usr/bin/env python3
"""Wait conservatively for biot, then benchmark only this task's CPU/GPU jobs.

No CPU/GPU reservation is implied. External work causes our attempt to stop;
partial output is retained and never counted as a timing result.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import time

from example1_comparison import command, environment, parse_trial, summarize, verify_readme_command

EXPECTED = {
    "cpu": ("build/3D_sim", "25a3a8a6b06312bb141f7bc940a9981d7f3243837f6dff8c798db14c89335a0e"),
    "gpu": ("gpu/build/3D_sim_gpu_resident", "c22aef94a473414d86c23382fdd393ec0486335d31cabba47ebf655b6b4256d4"),
}
HZ = os.sysconf("SC_CLK_TCK")
MONITORING_NOTE = (
    "Idle checks reject system I/O wait; during an owned run it is recorded, not alone a contamination trigger, "
    "because GEO/checkpoint writes can cause it. Foreign transfers, simulations/GPU jobs, CPU load and swapping "
    "still block. System I/O-wait counters cannot attribute I/O to a process; acceptance does not prove zero external I/O."
)


def process_stat(pid):
    raw = Path(f"/proc/{pid}/stat").read_text()
    name, tail = raw[raw.index("(")+1:].rsplit(")", 1)
    fields = tail.split()
    return {"pid": int(pid), "name": name, "state": fields[0],
            "group": int(fields[2]), "ticks": int(fields[11])+int(fields[12]),
            "start": int(fields[19])}


def sample(group=None):
    processes = {}
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            try:
                p = process_stat(path.name)
                processes[(p["pid"], p["start"])] = p
            except (FileNotFoundError, ProcessLookupError):
                pass
    stat = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
    ticks = list(map(int, stat))
    vm = dict(line.split() for line in Path("/proc/vmstat").read_text().splitlines())
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    own_cpu = own.ru_utime+own.ru_stime+children.ru_utime+children.ru_stime
    own_cpu += sum(p["ticks"]/HZ for p in processes.values() if group and p["group"] == group)
    return {"time": time.monotonic(), "processes": processes,
            "busy": sum(ticks[i] for i in (0, 1, 2, 5, 6, 7)),
            "iowait": ticks[4], "swaps": int(vm["pswpin"])+int(vm["pswpout"]),
            "own_cpu": own_cpu}


def gpu_state():
    def query(text):
        return subprocess.check_output(["nvidia-smi", text, "--format=csv,noheader,nounits"],
                                       text=True, timeout=10).strip().splitlines()
    pids = {int(row.strip()) for row in query("--query-compute-apps=pid") if row.strip()}
    rows = [list(map(float, row.split(","))) for row in query("--query-gpu=utilization.gpu,memory.free")]
    if len(rows) != 1:
        raise RuntimeError("Expected the previously verified single GPU")
    return pids, rows[0][0], rows[0][1]


def activity(a, b, watched, group=None, max_background_cores=.5):
    dt = b["time"]-a["time"]
    external = max(0., (b["busy"]-a["busy"])/HZ/dt-(b["own_cpu"]-a["own_cpu"])/dt)
    own_pids = {p["pid"] for p in b["processes"].values() if group and p["group"] == group}
    if group:
        own_pids.add(group)
    reasons = []
    for key, p in b["processes"].items():
        if p["pid"] == os.getpid() or p["pid"] in own_pids or p["state"] in "TtZ":
            continue
        if p["name"].startswith(("3D_sim", "cap_sim", "compute-saniti")) or p["name"] in (
                "nvcc", "ptxas", "ncu", "nsys", "rsync", "scp", "sftp", "sftp-server"):
            reasons.append(f"active job {p['pid']} ({p['name']})")
    for key in watched:
        p = b["processes"].get(key)
        if p and p["state"] != "Z":
            reasons.append(f"waiting for study process {p['pid']}")
    if external > max_background_cores:
        reasons.append(f"external CPU load {external:.2f} core equivalents")
    # System counters cannot distinguish this run's GEO/checkpoint output from
    # another process's I/O. Require a quiet disk before launch, but do not
    # repeatedly kill a valid run just for its own output traffic. All other
    # contamination checks remain active, including explicit foreign transfers.
    if not group and (b["iowait"]-a["iowait"])/HZ/dt > .25:
        reasons.append("I/O wait")
    if b["swaps"] != a["swaps"]:
        reasons.append("swap activity")
    pids, utilization, free_mb = gpu_state()
    if pids-own_pids:
        reasons.append(f"external GPU processes {sorted(pids-own_pids)}")
    if not group and utilization > 5:
        reasons.append(f"GPU utilization {utilization}%")
    if not group and free_mb < 8192:
        reasons.append("less than 8 GiB free VRAM")
    mem = {line.split(":")[0]: int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()}
    if mem["MemAvailable"] < 8*1024*1024:
        reasons.append("less than 8 GiB available RAM")
    return reasons, external


def record_iowait(attempt, a, b):
    """Record system-wide wait observations, not additive solver wall time."""
    seconds = max(0., (b["iowait"] - a["iowait"]) / HZ)
    cores = seconds / (b["time"] - a["time"])
    attempt["max_observed_system_iowait_cores"] = max(
        attempt.get("max_observed_system_iowait_cores", 0.), cores)
    attempt["observed_system_iowait_cpu_s"] = attempt.get("observed_system_iowait_cpu_s", 0.) + seconds


def self_test():
    import unittest
    from unittest.mock import patch

    class MonitorTests(unittest.TestCase):
        def samples(self):
            a = dict(time=1., processes={}, busy=0, iowait=0, swaps=0, own_cpu=0.)
            b = dict(time=11., processes={}, busy=0, iowait=10 * HZ, swaps=0, own_cpu=0.)
            return a, b

        def activity(self, a, b, group=None):
            with patch(__name__ + ".gpu_state", return_value=(set(), 0., 48000.)), \
                    patch.object(Path, "read_text", return_value="MemAvailable: 100000000 kB\n"):
                return activity(a, b, [], group=group)

        def test_iowait_is_idle_veto_not_run_veto(self):
            a, b = self.samples()
            self.assertIn("I/O wait", self.activity(a, b)[0])
            self.assertEqual(self.activity(a, b, group=123)[0], [])
            attempt = {}
            record_iowait(attempt, a, b)
            record_iowait(attempt, a, b)
            self.assertEqual(attempt["max_observed_system_iowait_cores"], 1.)
            self.assertEqual(attempt["observed_system_iowait_cpu_s"], 20.)

        def test_external_transfers_always_block(self):
            a, b = self.samples()
            for name in ("rsync", "scp", "sftp", "sftp-server", "3D_sim", "nvcc"):
                b["processes"] = {(456, 0): dict(pid=456, group=456, state="S", name=name)}
                self.assertTrue(any("active job 456" in r for r in self.activity(a, b, group=123)[0]))
                b["processes"][(456, 0)]["group"] = 123
                self.assertEqual(self.activity(a, b, group=123)[0], [])

        def test_other_contamination_still_blocks(self):
            a, b = self.samples()
            b["swaps"] = 1
            self.assertIn("swap activity", self.activity(a, b, group=123)[0])
            b["swaps"] = 0
            b["busy"] = 10 * HZ
            self.assertTrue(any("external CPU load" in r for r in self.activity(a, b, group=123)[0]))

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(MonitorTests)
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise SystemExit(1)


def physical_cpus():
    lines = subprocess.check_output(["lscpu", "-p=CPU,CORE,SOCKET,ONLINE"], text=True).splitlines()
    allowed = os.sched_getaffinity(0)
    cores = {}
    for row in lines:
        if row.startswith("#"):
            continue
        cpu, core, socket, online = row.split(",")
        if online == "Y" and int(cpu) in allowed:
            cores.setdefault((int(socket), int(core)), int(cpu))
    if len(cores) < 64:
        raise RuntimeError("Fewer than 64 allowed physical cores; refusing oversubscription")
    return [cores[k] for k in sorted(cores)[:64]]


def verify_binaries(repo):
    for relative, expected in EXPECTED.values():
        digest = hashlib.sha256((repo/relative).read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(f"Executable changed while queued: {relative}; refusing mixed-version results")


def stop_our_process(process):
    # start_new_session=True makes this child's PID its dedicated PGID.
    if process.poll() is None:
        try:
            if os.getpgid(process.pid) != process.pid:
                raise RuntimeError("Benchmark process group changed; refusing a broad kill")
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            process.wait()
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                if os.getpgid(process.pid) == process.pid:
                    os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true", help="Test monitoring without server/hardware access")
    parser.add_argument("--watch-pid", type=int, action="append", default=[])
    parser.add_argument("--frames", type=int, help="Run this many frames instead of the adaptive short benchmark")
    parser.add_argument("--repeats", type=int, default=1, help="Paired repetitions for --frames")
    parser.add_argument("--cpu-first", action="store_true", help="Start fixed-frame pairs with CPU")
    parser.add_argument("--readme-command", action="store_true",
                        help="Use README Example 1 argv/defaults and GEO; requires explicit --frames")
    parser.add_argument("--trial-timeout", type=float, default=600, help="Maximum elapsed seconds per attempt")
    parser.add_argument("--max-background-cores", type=float, default=.5,
                        help="Aggregate background CPU ceiling; active simulations/GPU jobs always block")
    parser.add_argument("--cpu-sha256", help="Explicitly pin a different CPU executable hash")
    parser.add_argument("--gpu-sha256", help="Explicitly pin a different GPU executable hash")
    parser.add_argument("--monitor-interval", type=float, default=3,
                        help="Seconds between external-load observations")
    parser.add_argument("--max-attempts", type=int, default=20,
                        help="Maximum attempts per CPU/GPU run after contamination")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.output is None:
        parser.error("--output is required unless --self-test is selected")
    if (args.frames is not None and args.frames < 1) or args.repeats < 1 or not 0 < args.trial_timeout < float("inf"):
        parser.error("frames, repeats and trial-timeout must be positive and finite")
    if not 0 <= args.max_background_cores < float("inf"):
        parser.error("max-background-cores must be nonnegative and finite")
    if not 0 < args.monitor_interval <= 60 or args.max_attempts < 1:
        parser.error("monitor-interval must be in (0,60], max-attempts positive")
    if args.readme_command and args.frames is None:
        parser.error("--readme-command requires --frames (240 for the full documented run)")
    for mode in ("cpu", "gpu"):
        expected = getattr(args, mode + "_sha256")
        if expected is not None:
            if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
                parser.error(mode + "-sha256 must contain 64 lowercase hexadecimal digits")
            EXPECTED[mode] = (EXPECTED[mode][0], expected)
    repo = Path(__file__).resolve().parent.parent
    if args.readme_command:
        verify_readme_command(repo)
    root = args.output.resolve()
    root.mkdir(exist_ok=True)
    if any(path.name != "queue.log" for path in root.iterdir()):
        raise RuntimeError("Choose a fresh output directory; existing results will not be replaced")
    def terminate(signum, frame):
        raise KeyboardInterrupt(f"Termination signal {signum}")
    signal.signal(signal.SIGTERM, terminate)
    lock = (repo/"gpu/.example1_comparison.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cpus = physical_cpus()
    watched = []
    for pid in args.watch_pid:
        try:
            p = process_stat(pid)
            watched.append((pid, p["start"]))
        except FileNotFoundError:
            pass
    state = {"status": "waiting_for_idle", "pid": os.getpid(), "watched": watched,
             "directory": str(root), "cpu_affinity": cpus, "expected_binaries": EXPECTED,
             "max_background_cores": args.max_background_cores,
             "monitor_interval_s": args.monitor_interval,
             "max_attempts_per_run": args.max_attempts,
             "monitoring_policy": MONITORING_NOTE,
             "readme_command": args.readme_command,
             "mesh_format": "geo" if args.readme_command else "obj",
             "readme_sha256": hashlib.sha256((repo / "README.md").read_bytes()).hexdigest() if args.readme_command else None,
             "runs": [], "attempts": [],
             "scope": f"Example 1 application comparison, {args.frames} frames per run" if args.frames else
                      "short Example 1 application comparison, not full 240 frames"}

    def save(status, **fields):
        state.update(status=status, updated_unix=time.time(), **fields)
        temporary = root/"status.tmp"
        temporary.write_text(json.dumps(state, indent=2)+"\n")
        temporary.replace(root/"status.json")

    def wait_idle():
        consecutive = 0
        while consecutive < 3:
            try:
                a = sample()
                time.sleep(10)
                b = sample()
                reasons, load = activity(a, b, watched, max_background_cores=args.max_background_cores)
                if shutil.disk_usage(root).free < 2*1024**3:
                    reasons.append("less than 2 GiB available disk space")
                consecutive = 0 if reasons else consecutive+1
                save("waiting_for_idle", idle_checks=consecutive, waiting_reasons=reasons,
                     external_cpu_cores=load)
            except (OSError, ValueError, subprocess.SubprocessError, RuntimeError) as error:
                consecutive = 0
                save("waiting_for_idle", idle_checks=0, waiting_reasons=[str(error)])
            if not consecutive:
                time.sleep(20)

    def trial(mode, frames):
        for retry in range(args.max_attempts):
            wait_idle()
            verify_binaries(repo)
            if args.readme_command:
                verify_readme_command(repo)
            name = f"{len(state['attempts']):03d}_{mode}_{frames}frames"
            outdir, log = root/name, root/(name+".log")
            cmd = command(repo, outdir, mode, frames, readme_command=args.readme_command)
            affinity = cpus if mode == "cpu" or args.readme_command else cpus[:1]
            cmd = ["taskset", "--cpu-list", ",".join(map(str, affinity)), *cmd]
            # Keep progress/error output visible even if a trial hits its cap.
            if shutil.which("stdbuf"):
                cmd = ["stdbuf", "-oL", "-eL", *cmd]
            env = environment(mode, readme_command=args.readme_command)
            attempt = {"mode": mode, "frames": frames, "command": cmd, "log": str(log),
                       "outdir": str(outdir), "started_unix": time.time(), "status": "running",
                       "readme_command": args.readme_command, "mesh_format": state["mesh_format"],
                       "omp": {k: v for k, v in env.items() if k.startswith("OMP_")}}
            state["attempts"].append(attempt)
            save("running", current=name, waiting_reasons=[])
            start = time.monotonic()
            # Sample before launch, so even a trial shorter than the poll
            # interval has aggregate external-load accounting. Reaped child
            # rusage retains our CPU time after /proc/PID disappears.
            previous = sample()
            process = None
            reasons = []
            try:
                with log.open("w") as output:
                    process = subprocess.Popen(cmd, cwd=repo, env=env, stdout=output,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                    attempt["pid"] = process.pid
                    save("running", current=name)
                    while True:
                        time.sleep(args.monitor_interval)
                        finished = process.poll() is not None
                        current = sample(process.pid)
                        record_iowait(attempt, previous, current)
                        load = None
                        try:
                            reasons, load = activity(previous, current, watched, process.pid,
                                                     args.max_background_cores)
                        except (OSError, ValueError, subprocess.SubprocessError, RuntimeError) as error:
                            reasons = [f"monitor unavailable: {error}"]
                        previous = current
                        if load is not None:
                            attempt["max_observed_background_cores"] = max(
                                attempt.get("max_observed_background_cores", 0.), load)
                        if reasons:
                            stop_our_process(process)
                            break
                        if not finished and time.monotonic()-start > args.trial_timeout:
                            stop_our_process(process)
                            raise RuntimeError(f"{name}: {args.trial_timeout:g}-second runtime cap exceeded; not a timing result")
                        if finished:
                            break
                attempt["process_monitor_elapsed_s"] = time.monotonic()-start
                if reasons:
                    attempt.update(status="contaminated", reasons=reasons)
                    save("waiting_for_idle", waiting_reasons=reasons)
                    continue
                if process.returncode:
                    raise RuntimeError(f"{name} exited {process.returncode}; inspect {log}")
                verify_binaries(repo)
                result = parse_trial(log, outdir, frames, mode, readme_command=args.readme_command)
                result.update(command=cmd, omp=attempt["omp"], attempt=name)
                result.update({key: attempt[key] for key in (
                    "max_observed_system_iowait_cores", "observed_system_iowait_cpu_s") if key in attempt})
                state["runs"].append(result)
                attempt["status"] = "complete"
                save("between_trials")
                return result
            finally:
                if process is not None:
                    stop_our_process(process)
        raise RuntimeError(f"{args.max_attempts} contaminated attempts; pausing for manual review")

    try:
        verify_binaries(repo)
        save("waiting_for_idle")
        print(f"Waiting job PID {os.getpid()}; status: {root/'status.json'}", flush=True)
        if args.frames is not None:
            for repeat in range(args.repeats):
                order = ("cpu", "gpu") if args.cpu_first else ("gpu", "cpu")
                for mode in (order if repeat % 2 == 0 else order[::-1]):
                    trial(mode, args.frames)
        else:
            gpu, cpu = trial("gpu", 2), trial("cpu", 2)
            # Grow only after both full-resolution startup trials finish cheaply.
            if max(cpu["total_solver_s"], gpu["total_solver_s"]) <= 30:
                for repeat in range(3):
                    for mode in (("gpu", "cpu") if repeat % 2 == 0 else ("cpu", "gpu")):
                        trial(mode, 10)
            else:
                for repeat in range(2):
                    for mode in (("gpu", "cpu") if repeat == 0 else ("cpu", "gpu")):
                        trial(mode, 2)
        report = summarize(state["runs"])
        report.update(expected_binaries=EXPECTED, cpu_affinity=cpus,
                      readme_sha256=state["readme_sha256"],
                      monitoring_policy=MONITORING_NOTE,
                      max_observed_system_iowait_cores=max(
                          (a.get("max_observed_system_iowait_cores", 0.) for a in state["attempts"]
                           if a["status"] == "complete"), default=0.),
                      max_background_cores=args.max_background_cores,
                      max_observed_background_cores=max(
                          (a.get("max_observed_background_cores", 0.) for a in state["attempts"]
                           if a["status"] == "complete"), default=0.))
        report["notes"].extend([
            "Idle polling is not an exclusive reservation; contaminated attempts are excluded.",
            MONITORING_NOTE,
            "Application comparison, not isolated hardware throughput; numerical schedules may differ.",
            (f"Fixed {args.frames}-frame runs; {args.repeats} paired repetition(s)." if args.frames else
             "Early frames only; no contact-heavy or full-240-frame performance claim.")])
        (root/"report.json").write_text(json.dumps(report, indent=2)+"\n")
        save("complete", report=str(root/"report.json"))
    except BaseException as error:
        save("stopped", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
