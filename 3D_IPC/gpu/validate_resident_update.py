#!/usr/bin/env python3
"""Correctness-only validation of the resident port after a CPU source update.

Run on the CUDA server after building gpu_resident_validation and
gpu_rigid_kernel_test. Uses unique smoke output paths, never benchmarks or
changes production source/binaries. Sanitizer absence/failure is an error.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent.parent
    root = args.output.resolve()
    root.mkdir(exist_ok=True)
    report_path = root / "report.json"
    if report_path.exists():
        raise RuntimeError("Existing validation report will not be replaced")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OMP_", "GOMP_", "KMP_"))}
    env.update(OMP_NUM_THREADS="1", OMP_DYNAMIC="FALSE", OMP_WAIT_POLICY="PASSIVE")
    report = {"purpose": "correctness validation, not a performance comparison",
              "cpu_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
              "executables": {}, "checks": [], "smokes": []}
    for relative in ("build/3D_sim", "gpu/build/3D_sim_gpu_resident"):
        report["executables"][relative] = hashlib.sha256((repo / relative).read_bytes()).hexdigest()

    def save():
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    def run(name, command, timeout=600):
        log = root / (name + ".log")
        entry = {"name": name, "command": [str(x) for x in command], "log": str(log), "passed": False}
        report["checks"].append(entry)
        start = time.monotonic()
        print("Running " + name, flush=True)
        try:
            with log.open("x") as output:
                process = subprocess.Popen(command, cwd=repo, env=env, stdout=output,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = process.wait(timeout=timeout)
                except BaseException:
                    # Stop this check's private session, including sanitizer
                    # children, never other jobs with the same executable name.
                    if process.poll() is None:
                        try:
                            if os.getpgid(process.pid) == process.pid:
                                os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait()
                    raise
            entry["exit_code"] = code
            if code:
                raise RuntimeError(name + " failed; inspect " + str(log))
            entry["passed"] = True
        finally:
            entry["validation_elapsed_s"] = time.monotonic() - start
            save()
        print("Passed " + name, flush=True)

    try:
        run("resident_suites", ["ctest", "--test-dir", "gpu/build", "-R", "^gpu_resident_", "--output-on-failure", "-j", "1"])
        run("rigid_kernel", ["./gpu/build/gpu_rigid_kernel_test"])
        sanitizer = shutil.which("compute-sanitizer") or "/usr/local/cuda-12.6/bin/compute-sanitizer"
        if not Path(sanitizer).is_file():
            raise RuntimeError("CUDA compute-sanitizer unavailable")
        for target in ("contact", "contact_pipeline", "linear_pipeline", "solver"):
            run("memcheck_" + target, [sanitizer, "--tool", "memcheck", "--error-exitcode", "99",
                                      "./gpu/build/gpu_resident_" + target + "_test"])
        run("initcheck_solver", [sanitizer, "--tool", "initcheck", "--error-exitcode", "99", "./gpu/build/gpu_resident_solver_test"])
        run("initcheck_contact_pipeline", [sanitizer, "--tool", "initcheck", "--error-exitcode", "99", "./gpu/build/gpu_resident_contact_pipeline_test"])

        common = ["./gpu/build/3D_sim_gpu_resident", "--fps", "30", "--fixed_iters",
                  "--max_substep_iters", "2", "--node_box_update_count", "1",
                  "--friction_coefficient", "0", "--use_ccd", "true", "--use_ticcd", "false",
                  "--exact_computation_fallback", "true", "--use_ccd_guess", "false",
                  "--use_colored_ccd_guess", "true", "--colored_ccd_guess_iters", "2",
                  "--write_substeps", "false", "--format", "obj"]
        fixtures = [
            ("gpu_cloth_smoke", 2, 100, 162,
             ["--example", "1", "--twist_nx", "9", "--twist_ny", "9", "--substeps", "5",
              "--E", "115000", "--nu", "0.25", "--kB", "0.009", "--kpin", "1e9",
              "--twist_rate", "0.5", "--d_hat", "0.005", "--k_barrier", "100"]),
            ("gpu_scene15_small_smoke", 1, 18, 16,
             ["--example", "15", "--drop_stack_count", "2", "--drop_cloth_nx", "2",
              "--drop_cloth_ny", "2", "--drop_cloth_w", "0.1", "--drop_cloth_h", "0.1", "--substeps", "2"]),
        ]
        for name, frames, vertices, triangles, flags in fixtures:
            outdir = root / name
            if outdir.exists():
                raise RuntimeError("Existing smoke directory will not be replaced: " + str(outdir))
            run(name, [*common, *flags, "--num_frames", str(frames), "--outdir", str(outdir)])
            log = (root / (name + ".log")).read_text()
            if "Simulation finished." not in log or len(re.findall(r"^Frame\s+\d+\s*\|", log, re.M)) != frames:
                raise RuntimeError(name + ": missing completed frames")
            for frame in range(frames + 1):
                path = outdir / f"frame_{frame:04d}.obj"
                positions, faces = [], 0
                for line in path.read_text().splitlines():
                    fields = line.split()
                    if fields and fields[0] == "v":
                        positions.append([float(x) for x in fields[1:]])
                    elif fields and fields[0] == "f":
                        faces += 1
                if len(positions) != vertices or faces != triangles or any(
                        len(p) != 3 or not all(map(math.isfinite, p)) for p in positions):
                    raise RuntimeError(name + ": invalid output mesh " + str(path))
            report["smokes"].append({"name": name, "frames": frames, "vertices": vertices, "triangles": triangles, "finite_meshes": True})
        report["passed"] = True
    except BaseException as error:
        report.update(passed=False, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
