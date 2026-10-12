#!/usr/bin/env python3
"""Short, non-destructive CPU/CUDA integration tests; never a speedup claim."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time


def positions(path):
    result = []
    with path.open() as source:
        for line in source:
            if line.startswith("v "):
                result.append(tuple(map(float, line.split()[1:4])))
    if not result or any(not math.isfinite(x) for v in result for x in v):
        raise RuntimeError(f"Missing/nonfinite positions in {path}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, default=Path(__file__).resolve().parent / "build")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--full-size", action="store_true", help="Also test two frames with 10,000 cloth vertices")
    parser.add_argument("--scene-smoke", action="store_true", help="Also test one frame of the actual mixed Example 10 and rigid Example 9 drivers")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent.parent
    root = Path(tempfile.mkdtemp(prefix="validation_", dir=Path(__file__).resolve().parent))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OMP_", "GOMP_", "KMP_"))}
    env.update(OMP_NUM_THREADS=str(args.threads), OMP_DYNAMIC="FALSE", OMP_WAIT_POLICY="PASSIVE")
    report = {"status": "running", "directory": str(root), "threads": args.threads, "runs": []}
    print(root, flush=True)

    def save():
        (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def run(name, executable, flags, required=("point_records", "triangle_records")):
        directory = root / name
        command = [str(args.build.resolve() / executable), *flags, "--outdir", str(directory)]
        started = time.monotonic()
        with (root / f"{name}.log").open("w") as log:
            result = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600)
        text = (root / f"{name}.log").read_text()
        if result.returncode:
            raise RuntimeError(f"{name} exited {result.returncode}: {text[-1500:]}")
        frames = re.findall(r"Frame\s+(\d+)\s*\|.*solver_time\s*=\s*([0-9.]+) ms", text)
        total = re.search(r"Total solver time:\s*([0-9.]+) ms", text)
        record = {"name": name, "command": command, "process_seconds": time.monotonic()-started,
                  "frames": len(frames), "total_solver_seconds": float(total[1])/1000 if total else None}
        if executable == "3D_sim_gpu":
            counters = re.search(r"CUDA calls=\d+ launches=\d+[^\n]+", text)
            values = dict((key,int(value)) for key,value in re.findall(r"(\w+)=(\d+)", counters[0])) if counters else {}
            if any(values.get(key,0) <= 0 for key in ("calls", "launches", *required)):
                raise RuntimeError(f"{name} did not execute all required GPU kernels")
            record["cuda_counters"] = values
        if not total or not frames:
            raise RuntimeError(f"Missing completion/timing in {name}")
        report["runs"].append(record)
        save()
        return directory, record

    common = ["--example", "1", "--num_frames", "3", "--fps", "30", "--twist_nx", "12", "--twist_ny", "12",
              "--E", "115000", "--nu", "0.25", "--kB", "0.009", "--kpin", "1e9", "--twist_rate", "0.5",
              "--fixed_iters", "--max_substep_iters", "4", "--substeps", "2", "--node_box_update_count", "2",
              "--use_parallel", "true", "--use_basic_experimental", "true", "--use_simd", "true",
              "--write_substeps", "false", "--format", "obj"]
    cases = [
        ("collision_off", ["--d_hat", "0", "--k_barrier", "0", "--use_ccd", "false", "--use_ccd_guess", "false", "--friction_coefficient", "0"]),
        ("collision_on", ["--d_hat", "0.005", "--k_barrier", "100", "--use_ccd", "true", "--friction_coefficient", "0"]),
        ("friction_on", ["--d_hat", "0.005", "--k_barrier", "100", "--use_ccd", "true", "--friction_coefficient", "0.1"]),
        ("no_bending", ["--d_hat", "0", "--use_ccd", "false", "--use_ccd_guess", "false", "--kB", "0"]),
    ]
    if args.full_size:
        size_flags = ["--twist_nx", "99", "--twist_ny", "99", "--num_frames", "2",
                      "--substeps", "5", "--max_substep_iters", "10", "--node_box_update_count", "10"]
        cases += [("full_size_off", size_flags + cases[0][1]), ("full_size_on", size_flags + cases[1][1])]
    try:
        sanitizer = Path("/usr/local/cuda-12.6/bin/compute-sanitizer")
        for executable in ("gpu_kernel_test", "gpu_solid_kernel_test", "gpu_rigid_kernel_test"):
            test = subprocess.run([str(args.build.resolve()/executable)], capture_output=True,
                                  text=True, timeout=180, env=env)
            (root/f"{executable}.log").write_text(test.stdout+test.stderr)
            if test.returncode:
                raise RuntimeError(f"CUDA kernel correctness test failed: {executable}")
            if sanitizer.exists():
                checked = subprocess.run([str(sanitizer),"--tool","memcheck","--error-exitcode","3",
                                          str(args.build.resolve()/executable)],
                                         capture_output=True,text=True,timeout=300,env=env)
                (root/f"{executable}_memcheck.log").write_text(checked.stdout+checked.stderr)
                if checked.returncode: raise RuntimeError(f"CUDA memcheck failed: {executable}")
                report.setdefault("cuda_memcheck",{})[executable] = "passed"
        for name, flags in cases:
            cpu, _ = run(name+"_cpu", "3D_sim_cpu_control", common+flags)
            gpu, record = run(name+"_gpu", "3D_sim_gpu", common+flags)
            cpu_files = sorted(cpu.glob("frame_*.obj"))
            gpu_files = sorted(gpu.glob("frame_*.obj"))
            expected_frames = 3 if name.startswith("full_size") else 4
            if len(cpu_files)!=expected_frames or [p.name for p in cpu_files]!=[p.name for p in gpu_files]:
                raise RuntimeError(f"Missing output frames for {name}")
            worst = 0.0
            for a,b in zip(cpu_files,gpu_files):
                x,y = positions(a), positions(b)
                if len(x)!=len(y): raise RuntimeError("Vertex counts differ")
                worst=max(worst,max(abs(u-v) for p,q in zip(x,y) for u,v in zip(p,q)))
            record["max_position_component_error"] = worst
            record["position_tolerance"] = 1e-7
            if worst>1e-7: raise RuntimeError(f"{name}: trajectory error {worst} exceeds 1e-7")
            print(f"PASS {name}: max coordinate error={worst:.3g}",flush=True)
            save()
        for friction in ("0", "0.1"):
            results = []
            for mode in ("cpu", "gpu"):
                command = [str(args.build.resolve()/f"{mode}_contact_fixture"),friction]
                test = subprocess.run(command,capture_output=True,text=True,env=env,cwd=repo,timeout=120)
                (root/f"contact_{friction}_{mode}.log").write_text(test.stdout+test.stderr)
                if test.returncode: raise RuntimeError(f"Contact fixture {mode}/{friction}: {test.stderr}")
                results.append(json.loads(test.stdout))
            a,b = results
            if a["vertices"]!=b["vertices"]: raise RuntimeError("Contact fixture vertex count differs")
            error=max(abs(x-y) for p,q in zip(a["positions"],b["positions"]) for x,y in zip(p,q))
            report.setdefault("active_contact_fixtures",[]).append({
                "friction":float(friction), "max_position_component_error":error,
                "cuda_launches":b.get("cuda_launches",0),
                "initial_contacts":b["initial_contacts"], "contacts_after_substeps":b["contacts_after_substeps"]})
            if b.get("cuda_launches",0)<=0: raise RuntimeError("Contact fixture did not execute GPU kernels")
            if error>1e-7: raise RuntimeError(f"Contact fixture friction={friction}: error {error} exceeds 1e-7")
            print(f"PASS active contact friction={friction}: max coordinate error={error:.3g}",flush=True)
            save()
        for friction in ("0", "0.1"):
            results = []
            for mode in ("cpu", "gpu"):
                command = [str(args.build.resolve()/f"{mode}_mixed_fixture"),friction]
                test = subprocess.run(command,capture_output=True,text=True,env=env,cwd=repo,timeout=180)
                (root/f"mixed_{friction}_{mode}.log").write_text(test.stdout+test.stderr)
                if test.returncode: raise RuntimeError(f"Mixed fixture {mode}/{friction}: {test.stderr}")
                results.append(json.loads(test.stdout))
            cpu, gpu = results
            if cpu["vertices"] != gpu["vertices"]: raise RuntimeError("Mixed fixture vertex counts differ")
            errors = {}
            for key in ("positions", "velocities", "x_coms", "v_coms", "orientations", "omega"):
                if len(cpu[key]) != len(gpu[key]): raise RuntimeError(f"Mixed {key} sizes differ")
                for values in (cpu[key], gpu[key]):
                    if any(not math.isfinite(x) for v in values for x in v):
                        raise RuntimeError(f"Mixed {key} contains nonfinite values")
                errors[key] = max(abs(x-y) for p,q in zip(cpu[key],gpu[key]) for x,y in zip(p,q))
                tolerance = 1e-7 if key in ("positions", "x_coms", "orientations") else 1e-5
                if errors[key] > tolerance:
                    raise RuntimeError(f"Mixed fixture friction={friction}: {key} error {errors[key]} > {tolerance}")
            counters = {key: gpu[key] for key in ("cuda_launches", "cuda_points", "cuda_triangles",
                        "cuda_hinges", "cuda_solids", "cuda_rigid_contacts", "cuda_rigid_inertia")}
            if min(counters.values()) <= 0: raise RuntimeError("Mixed fixture missed a GPU family")
            report.setdefault("mixed_contact_fixtures",[]).append({
                "friction": float(friction), "max_component_errors": errors, "cuda_counters": counters,
                "initial_contacts": gpu["initial_contacts"], "contacts_after_substeps":gpu["contacts_after_substeps"]})
            print(f"PASS mixed cloth/solid/rigid friction={friction}: errors={errors}",flush=True)
            save()
        if args.scene_smoke:
            base = ["--num_frames","1","--fps","30","--fixed_iters","--max_substep_iters","2",
                    "--use_parallel","true","--use_basic_experimental","true","--use_simd","true",
                    "--write_substeps","false","--format","obj"]
            scenes = [
                ("example10_mixed",["--example","10","--substeps","15","--E","1.25e9","--nu","0.25",
                    "--thickness","0.001","--solid_E","1.25e5","--solid_nu","0.25","--d_hat","0.019",
                    "--k_barrier","1000","--friction_coefficient","0.1","--use_ccd","true",
                    "--node_box_update_count","10","--use_colored_ccd_guess","true","--colored_ccd_guess_iters","10"],
                    ("point_records","triangle_records","solid_records","rigid_inertia_records")),
                ("example9_rigid",["--example","9","--substeps","10"],("rigid_inertia_records",))]
            for name, flags, required in scenes:
                cpu,_ = run(name+"_cpu", "3D_sim_cpu_control", base+flags, required)
                gpu,record = run(name+"_gpu", "3D_sim_gpu", base+flags, required)
                cpu_files,gpu_files = sorted(cpu.glob("frame_*.obj")),sorted(gpu.glob("frame_*.obj"))
                if len(cpu_files)!=2 or [p.name for p in cpu_files]!=[p.name for p in gpu_files]:
                    raise RuntimeError(f"Missing scene smoke outputs: {name}")
                error=0.0
                for a,b in zip(cpu_files,gpu_files):
                    x,y=positions(a),positions(b)
                    if len(x)!=len(y): raise RuntimeError(f"Scene smoke vertex counts differ: {name}")
                    error=max(error,max(abs(u-v) for p,q in zip(x,y) for u,v in zip(p,q)))
                record["max_position_component_error"]=error
                record["position_tolerance"]=1e-7
                if error>1e-7: raise RuntimeError(f"{name} position error {error} exceeds 1e-7")
                print(f"PASS {name}: max coordinate error={error:.3g}",flush=True)
                save()
        for label, flags in [("missing_gpu_mode",[]),("grid_mode",["--example","1","--use_parallel","true","--use_basic_experimental","true","--use_simd","true","--use_cloth_grid","true"])]:
            rejected=subprocess.run([str(args.build.resolve()/"3D_sim_gpu"),*flags,"--num_frames","0",
                                     "--outdir",str(root/label)],capture_output=True,text=True,cwd=repo,env=env,timeout=120)
            (root/f"{label}.log").write_text(rejected.stdout+rejected.stderr)
            if rejected.returncode!=2 or "GPU prototype requires" not in rejected.stderr:
                raise RuntimeError(f"Unsupported GPU mode was not explicitly rejected: {label}")
        report["unsupported_mode_guards"]="passed"
        report["status"] = "passed"
        report["limitations"] = "Short cloth and mixed cloth/solid/rigid active-contact runs, not full-scene long-trajectory qualification or a performance benchmark."
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
