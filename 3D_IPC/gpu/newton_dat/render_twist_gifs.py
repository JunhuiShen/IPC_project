"""Render matched DAT twist trajectories with Pillow and a fixed common camera.

The input NPZ files come from ``export_twist_trajectory.py``. Every polygon
uses the exported simulation vertices and triangles. No simulated geometry is
reconstructed or changed by this renderer.

GIF delays have 10 ms precision. A repeated 30/30/40 ms schedule preserves
the 30 fps average and the exact eight-second playback duration.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 1000, 800
PANEL = (28, 171, 972, 656)
BACKGROUND = (245, 244, 248)
TEXT = (119, 19, 81)
DIVIDER = (208, 159, 190)
CLOTH_DARK = (132, 13, 72)
CLOTH_LIGHT = (212, 43, 134)


@dataclass
class Trajectory:
    path: Path
    positions: np.ndarray
    initial: np.ndarray
    faces: np.ndarray
    frames: np.ndarray
    times: np.ndarray
    seconds: np.ndarray
    metadata: dict


def load_trajectory(path: Path, variant: str) -> Trajectory:
    with np.load(path, allow_pickle=False) as archive:
        result = Trajectory(
            path=path, positions=archive["positions"].copy(),
            initial=archive["initial_positions"].copy(), faces=archive["faces"].copy(),
            frames=archive["frame_indices"].copy(), times=archive["times"].copy(),
            seconds=archive["step_seconds"].copy(),
            metadata=json.loads(str(archive["metadata_json"].item())),
        )
    expected_frames = np.arange(2, 481, 2, dtype=np.int32)
    if result.positions.shape != (240, 2500, 3) or result.initial.shape != (2500, 3):
        raise ValueError(f"Unexpected trajectory dimensions in {path}")
    if result.faces.shape != (4802, 3) or result.faces.dtype.kind not in "iu":
        raise ValueError(f"Unexpected triangle topology in {path}")
    if result.faces.min() < 0 or result.faces.max() >= 2500:
        raise ValueError(f"Invalid triangle vertex in {path}")
    if not np.array_equal(result.frames, expected_frames):
        raise ValueError(f"Expected samples at frames 2,4,...,480 in {path}")
    if result.times.shape != (240,) or not np.allclose(result.times, expected_frames / 60.0, rtol=0, atol=1e-12):
        raise ValueError(f"Sample times do not match 60 Hz physics in {path}")
    if result.seconds.shape != (480,) or not np.isfinite(result.seconds).all() or np.any(result.seconds <= 0):
        raise ValueError(f"Expected 480 positive measured step durations in {path}")
    if not np.isfinite(result.positions).all() or not np.isfinite(result.initial).all():
        raise ValueError(f"Nonfinite simulated coordinates in {path}")
    required = {
        "variant": variant, "frames": 480, "simulation_fps": 60, "samples": 240,
        "sample_fps": 30, "substeps": 10, "iterations_per_substep": 4,
        "particles": 2500, "triangles": 4802,
    }
    for name, expected in required.items():
        if result.metadata.get(name) != expected:
            raise ValueError(f"Unexpected {name} in {path}: {result.metadata.get(name)!r}")
    if not math.isclose(result.metadata["angular_velocity_per_endpoint_radians_per_second"], math.pi / 2,
                        rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"The requested endpoint rotation rate is not present in {path}")
    if not math.isclose(float(result.seconds.sum()), result.metadata["measured_step_seconds_sum"],
                        rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"Measured step duration metadata disagrees with the samples in {path}")
    return result


def _normalize(value):
    value = np.asarray(value, np.float64)
    length = np.linalg.norm(value)
    if not np.isfinite(length) or length <= 0:
        raise ValueError("A camera vector must have positive finite length")
    return value / length


def make_camera(trajectories, direction=(1.0, .25, .18), clockwise_roll=12.0):
    view = _normalize(direction)
    up_hint = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(view, up_hint)) > .99:
        up_hint = np.array([0.0, 1.0, 0.0])
    right = _normalize(np.cross(up_hint, view))
    up = _normalize(np.cross(view, right))
    angle = math.radians(clockwise_roll)
    screen_right = math.cos(angle) * right + math.sin(angle) * up
    screen_up = -math.sin(angle) * right + math.cos(angle) * up
    basis = np.column_stack((screen_right, screen_up, view))
    lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
    world_lo, world_hi = np.full(3, np.inf), np.full(3, -np.inf)
    for trajectory in trajectories:
        for vertices in (trajectory.initial, trajectory.positions.reshape(-1, 3)):
            projected = vertices.astype(np.float64) @ basis[:, :2]
            lo = np.minimum(lo, projected.min(axis=0))
            hi = np.maximum(hi, projected.max(axis=0))
            world_lo = np.minimum(world_lo, vertices.min(axis=0))
            world_hi = np.maximum(world_hi, vertices.max(axis=0))
    extent = hi - lo
    if np.any(extent <= 0):
        raise ValueError("The exported trajectories have a zero projected extent")
    panel_width, panel_height = PANEL[2] - PANEL[0], PANEL[3] - PANEL[1]
    padding = 20
    scale = min((panel_width - 2 * padding) / extent[0],
                (panel_height - 2 * padding) / extent[1])
    light = _normalize(.35 * screen_right + .65 * screen_up + 1.2 * view)
    return {
        "basis": basis, "offset": .5 * (lo + hi), "scale": float(scale), "light": light,
        "direction": view.tolist(), "clockwise_roll_degrees": clockwise_roll,
        "projection_min": lo.tolist(), "projection_max": hi.tolist(),
        "world_min": world_lo.tolist(), "world_max": world_hi.tolist(),
    }


def _font(size, bold=False):
    choices = (["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"] if bold else
               ["/System/Library/Fonts/Supplemental/Arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"])
    for choice in choices:
        if Path(choice).is_file():
            return ImageFont.truetype(choice, size)
    # Pillow bundles this scalable fallback in current versions.
    return ImageFont.load_default(size=size)


def make_palette():
    colors = [BACKGROUND, TEXT, DIVIDER]
    for fraction in np.linspace(0, 1, 128):
        colors.append(tuple(round((1 - fraction) * a + fraction * b)
                            for a, b in zip(CLOTH_DARK, CLOTH_LIGHT)))
    for fraction in np.linspace(0, 1, 96):
        colors.append(tuple(round((1 - fraction) * a + fraction * b)
                            for a, b in zip(BACKGROUND, TEXT)))
    while len(colors) < 256:
        colors.append(BACKGROUND)
    palette = Image.new("P", (1, 1))
    palette.putpalette([channel for color in colors for channel in color])
    return palette, colors[3:131]


def make_base(trajectory):
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    variant = trajectory.metadata["variant"]
    label = {"reference": "Reference DAT", "own": "Our energies DAT", "gs": "Our energies GS"}[variant]
    draw.text((32, 16), f"Newton cloth_twist | {label} | CUDA", fill=TEXT, font=_font(30, True))
    draw.text((32, 62), "2,500 vertices  |  4,802 triangles  |  10 substeps × 4 iterations",
              fill=TEXT, font=_font(21))
    gpu = trajectory.metadata.get("gpu", "CUDA GPU")
    if "RTX 6000 Ada" in gpu:
        gpu = "RTX 6000 Ada"
    draw.text((32, 91), f"{gpu}  |  480 simulation frames at 60 Hz", fill=TEXT, font=_font(20))
    draw.text((32, 120), "Opposing edges: π/2 rad/s each  |  Relative: 0.5 turn/s",
              fill=TEXT, font=_font(20))
    draw.line((28, 154, 972, 154), fill=DIVIDER, width=1)
    elapsed = float(trajectory.seconds.sum())
    draw.text((32, 721), f"Simulation runtime: {elapsed:.3f} s  |  {1000 * elapsed / 480:.3f} ms/frame (480 frames)",
              fill=TEXT, font=_font(19))
    footer = ("Original Newton energies" if variant == "reference" else
              "Our energies + colored CCD guess" if variant == "gs" else
              "Our energies + full curvature Hessian")
    draw.text((32, 759), f"{footer}  |  fixed view", fill=TEXT, font=_font(18))
    return image


def render_frame(trajectory, sample, camera, base, cloth_colors, supersample=2):
    image = base.copy()
    vertices = trajectory.positions[sample].astype(np.float64)
    faces = trajectory.faces
    projected = vertices @ camera["basis"]
    xy = (projected[:, :2] - camera["offset"]) * camera["scale"]
    panel_width, panel_height = PANEL[2] - PANEL[0], PANEL[3] - PANEL[1]
    xy[:, 0] += panel_width / 2
    xy[:, 1] = panel_height / 2 - xy[:, 1]
    if np.any(xy < -.01) or np.any(xy[:, 0] > panel_width + .01) or np.any(xy[:, 1] > panel_height + .01):
        raise RuntimeError("The fixed camera clipped simulated vertices")
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 0
    illumination = np.zeros(len(faces), np.float64)
    illumination[valid] = np.abs(normals[valid] @ camera["light"]) / lengths[valid]
    # Two-sided cloth uses the same magenta material on both faces.
    shades = np.clip(np.rint((.20 + .80 * illumination) * (len(cloth_colors) - 1)),
                     0, len(cloth_colors) - 1).astype(np.int32)
    order = np.argsort(projected[faces, 2].mean(axis=1), kind="stable")
    panel = Image.new("RGB", (panel_width * supersample, panel_height * supersample), BACKGROUND)
    draw = ImageDraw.Draw(panel)
    corners = np.rint(xy[faces] * supersample).astype(np.int32)
    for index in order:
        if valid[index]:
            draw.polygon([tuple(point) for point in corners[index]], fill=cloth_colors[shades[index]])
    if supersample > 1:
        panel = panel.resize((panel_width, panel_height), Image.Resampling.LANCZOS)
    image.paste(panel, PANEL[:2])
    draw = ImageDraw.Draw(image)
    draw.text((32, 684),
              f"t = {trajectory.times[sample]:.3f} s  |  frame {trajectory.frames[sample]} / 480"
              "  |  Playback 1×  |  sample 30 fps",
              fill=TEXT, font=_font(21))
    return image


def verify_gif(path):
    with Image.open(path) as gif:
        durations = []
        if gif.size != (WIDTH, HEIGHT) or gif.n_frames != 240:
            raise RuntimeError(f"Unexpected GIF dimensions or frame count in {path}")
        loop = gif.info.get("loop")
        for index in range(gif.n_frames):
            gif.seek(index)
            durations.append(gif.info.get("duration", 0))
    if loop != 0 or durations != [30, 30, 40] * 80 or sum(durations) != 8000:
        raise RuntimeError(f"Unexpected GIF timing in {path}: {sum(durations)} ms")
    return {"frames": 240, "width": WIDTH, "height": HEIGHT,
            "duration_ms": sum(durations), "loop": loop,
            "frame_duration_pattern_ms": [30, 30, 40], "average_sample_fps": 30}


def render_pair(reference_path, own_path, output_dir, *, direction=(1.0, .25, .18),
                clockwise_roll=12.0, supersample=2):
    reference = load_trajectory(Path(reference_path), "reference")
    own = load_trajectory(Path(own_path), "own")
    if not np.array_equal(reference.faces, own.faces) or not np.array_equal(reference.initial, own.initial):
        raise ValueError("The two inputs do not have identical initial geometry and topology")
    camera = make_camera([reference, own], direction, clockwise_roll)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    palette, cloth_colors = make_palette()
    previews = []
    receipt = {"camera": {key: value for key, value in camera.items()
                           if key not in ("basis", "offset", "light")},
               "background_rgb": BACKGROUND, "orthographic": True,
               "shared_palette": True, "supersample": supersample, "outputs": {}}
    for trajectory, stem in ((reference, "reference_dat"), (own, "our_energies_dat")):
        base = make_base(trajectory)
        frames = []
        saved_previews = []
        for sample in range(240):
            frame = render_frame(trajectory, sample, camera, base, cloth_colors, supersample)
            if sample in (0, 119, 239):
                saved_previews.append(frame.copy())
            frames.append(frame.quantize(palette=palette, dither=Image.Dither.NONE))
        target = output_dir / f"{stem}.gif"
        frames[0].save(target, save_all=True, append_images=frames[1:],
                       duration=[30, 30, 40] * 80, loop=0, disposal=1,
                       optimize=False, interlace=False, palette=bytes(palette.getpalette()))
        verification = verify_gif(target)
        previews.append(saved_previews)
        for sample, preview in zip((2, 240, 480), saved_previews):
            preview.save(output_dir / f"{stem}_frame_{sample:03d}.png")
        receipt["outputs"][stem] = {
            **verification, "path": str(target),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "input_npz": str(trajectory.path),
            "input_sha256": hashlib.sha256(trajectory.path.read_bytes()).hexdigest(),
            "measured_step_seconds_sum": float(trajectory.seconds.sum()),
            "measured_step_ms_mean": float(trajectory.seconds.mean() * 1000),
            "input_metadata": trajectory.metadata,
        }
        print(json.dumps({"output": str(target), **verification}), flush=True)
        del frames
    sheet = Image.new("RGB", (1500, 800), BACKGROUND)
    for row, row_previews in enumerate(previews):
        for column, preview in enumerate(row_previews):
            sheet.paste(preview.resize((500, 400), Image.Resampling.LANCZOS), (column * 500, row * 400))
    sheet.save(output_dir / "preview_contactsheet.png")
    (output_dir / "render_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--own", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(tempfile.gettempdir()) / "dat_twist_gifs")
    parser.add_argument("--camera-direction", type=float, nargs=3, default=(1.0, .25, .18))
    parser.add_argument("--clockwise-roll", type=float, default=12.0)
    parser.add_argument("--supersample", type=int, choices=(1, 2, 3), default=2)
    args = parser.parse_args(argv)
    render_pair(args.reference, args.own, args.output_dir,
                direction=args.camera_direction, clockwise_roll=args.clockwise_roll,
                supersample=args.supersample)


if __name__ == "__main__":
    main()
