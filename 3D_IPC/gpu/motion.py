"""Authored native pin and SDF motion executed inside the CUDA frame graph."""

from __future__ import annotations

import math

import numpy as np
import warp as wp


@wp.func
def _trapezoid(s: float, omega: float, ramp: float, steady: float):
    result = float(0.0)
    if s > 0.0:
        if s <= ramp:
            result = 0.5 * omega * s * s / ramp
        else:
            s1 = s - ramp
            if s1 <= steady:
                result = 0.5 * omega * ramp + omega * s1
            else:
                s2 = s1 - steady
                if s2 < ramp:
                    result = 0.5 * omega * ramp + omega * steady + omega * s2 - 0.5 * omega * s2 * s2 / ramp
                else:
                    result = omega * (ramp + steady)
    return result


@wp.func
def effective_theta(omega: float, time: float, settle: float, ramp: float, maximum: float, reverse: float, hold: float):
    result = float(0.0)
    if time > settle:
        speed = wp.abs(omega)
        sign = float(1.0)
        if omega < 0.0:
            sign = -1.0
        s = time - settle
        if maximum <= 0.0 or speed <= 0.0:
            if ramp <= 0.0:
                result = omega * s
            elif s >= ramp:
                result = omega * (s - 0.5 * ramp)
            else:
                result = 0.5 * omega * s * s / ramp
        else:
            steady = wp.max(0.0, maximum / speed - ramp)
            forward = 2.0 * ramp + steady
            if s <= forward:
                result = sign * _trapezoid(s, speed, ramp, steady)
            elif reverse == 0.0:
                result = sign * maximum
            else:
                s2 = s - forward
                if s2 <= hold:
                    result = sign * maximum
                else:
                    s3 = s2 - hold
                    if s3 <= forward:
                        result = sign * (maximum - _trapezoid(s3, speed, ramp, steady))
    return result


@wp.func
def _rotate(initial: wp.vec3, center: wp.vec3, axis: wp.vec3, theta: float):
    c, s = wp.cos(theta), wp.sin(theta)
    offset = initial - center
    result = center + c * offset + s * wp.cross(axis, offset) + (1.0 - c) * wp.dot(axis, offset) * axis
    if axis[0] == 1.0 and axis[1] == 0.0 and axis[2] == 0.0:
        result = wp.vec3(initial[0], center[1] + c * offset[1] - s * offset[2], center[2] + s * offset[1] + c * offset[2])
    elif axis[0] == 0.0 and axis[1] == 1.0 and axis[2] == 0.0:
        result = wp.vec3(center[0] + c * offset[0] + s * offset[2], initial[1], center[2] - s * offset[0] + c * offset[2])
    return result


@wp.kernel
def _advance(counter: wp.array[wp.int64]):
    counter[0] += wp.int64(1)


@wp.kernel
def _pins(counter: wp.array[wp.int64], dt: float, nodes: wp.array[int], data: wp.array2d[float], targets: wp.array[wp.vec3]):
    i = wp.tid()
    time = float(counter[0]) * dt
    initial = wp.vec3(data[i, 0], data[i, 1], data[i, 2])
    center = wp.vec3(data[i, 3], data[i, 4], data[i, 5])
    axis = wp.vec3(data[i, 6], data[i, 7], data[i, 8])
    if data[i, 15] != 0.0:
        targets[nodes[i]] = initial + data[i, 15] * wp.sin(6.283185307179586 * data[i, 16] * time) * axis
    else:
        theta = effective_theta(data[i, 9], time, data[i, 10], data[i, 11], data[i, 12], data[i, 13], data[i, 14])
        targets[nodes[i]] = _rotate(initial, center, axis, theta)


@wp.kernel
def _colliders(counter: wp.array[wp.int64], dt: float, indices: wp.array[int], data: wp.array2d[float], colliders: wp.array2d[float]):
    i = wp.tid()
    row = indices[i]
    time = float(counter[0]) * dt
    center = wp.vec3(data[i, 0], data[i, 1], data[i, 2])
    theta = effective_theta(data[i, 3], time, data[i, 4], data[i, 5], data[i, 6], data[i, 7], data[i, 8])
    previous_theta = effective_theta(data[i, 3], time - dt, data[i, 4], data[i, 5], data[i, 6], data[i, 7], data[i, 8])
    colliders[row, 1] = center[0]
    colliders[row, 2] = center[1]
    colliders[row, 3] = center[2]
    colliders[row, 4] = wp.cos(theta)
    colliders[row, 5] = 0.0
    colliders[row, 6] = -wp.sin(theta)
    for pose in range(2):
        angle = previous_theta
        begin = int(8)
        translation = int(17)
        if pose != 0:
            angle = theta
            begin = 20
            translation = 29
        c, s = wp.cos(angle), wp.sin(angle)
        colliders[row, begin] = c
        colliders[row, begin + 1] = 0.0
        colliders[row, begin + 2] = s
        colliders[row, begin + 3] = 0.0
        colliders[row, begin + 4] = 1.0
        colliders[row, begin + 5] = 0.0
        colliders[row, begin + 6] = -s
        colliders[row, begin + 7] = 0.0
        colliders[row, begin + 8] = c
        colliders[row, translation] = center[0] - (c * center[0] + s * center[2])
        colliders[row, translation + 1] = 0.0
        colliders[row, translation + 2] = center[2] - (-s * center[0] + c * center[2])


@wp.kernel
def _rigid_pins(counter: wp.array[wp.int64], dt: float, moving_clock: int, authored_time: float,
                data: wp.array2d[float], targets: wp.array[wp.vec3]):
    i = wp.tid()
    time = authored_time
    if moving_clock != 0:
        time = float(counter[0]) * dt
    initial = wp.vec3(data[i, 0], data[i, 1], data[i, 2])
    center = wp.vec3(data[i, 3], data[i, 4], data[i, 5])
    axis = wp.vec3(data[i, 6], data[i, 7], data[i, 8])
    target = initial
    speed, ramp = data[i, 9], data[i, 10]
    if speed != 0.0 and not (ramp > 0.0 and time <= 0.0):
        rotation_time = time
        if ramp > 0.0:
            if time >= ramp:
                rotation_time = time - 0.5 * ramp
            else:
                u = time / ramp
                rotation_time = ramp * u * u * u * u * (2.5 + u * (-3.0 + u))
        target = _rotate(initial, center, axis, speed * rotation_time)
    translation = wp.vec3(data[i, 11], data[i, 12], data[i, 13])
    start, duration = data[i, 14], data[i, 15]
    if (translation[0] != 0.0 or translation[1] != 0.0 or translation[2] != 0.0) and time > start:
        u = wp.min(1.0, (time - start) / duration)
        weight = u * u * u * (10.0 + u * (-15.0 + 6.0 * u))
        target += weight * translation
    targets[i] = target


class BoundaryMotion:
    """Graph-safe authored targets; no per-step CPU exporter roundtrip."""
    def __init__(self, model):
        self.model, self.device = model, model.device
        self.counter = wp.zeros(1, dtype=wp.int64, device=self.device)
        records = model.ipc.payload.get("motion", {})
        pins = records.get("pins", [])
        pin_data = np.zeros((len(pins), 17), dtype=np.float32)
        for i, p in enumerate(pins):
            pin_data[i, :3] = p["initial"]
            pin_data[i, 3:6] = p["center"]
            pin_data[i, 6:9] = p["axis"]
            pin_data[i, 9:] = [p["speed"], p["settle"], p["ramp"], p["maximum"], p["reverse"], p["hold"], p["amplitude"], p["frequency"]]
        self.pin_nodes = wp.array([p["node"] for p in pins], dtype=int, device=self.device)
        self.pin_data = wp.array(pin_data, dtype=float, device=self.device)
        colliders = records.get("colliders", [])
        collider_data = np.asarray([[*p["center"], p["speed"], p["settle"], p["ramp"], p["maximum"], p["reverse"], p["hold"]] for p in colliders], dtype=np.float32).reshape((-1, 9))
        self.collider_nodes = wp.array([p["index"] for p in colliders], dtype=int, device=self.device)
        self.collider_data = wp.array(collider_data, dtype=float, device=self.device)
        rigid = model.ipc.payload.get("rigid_pins", [])
        rigid_data = np.zeros((len(rigid), 16), dtype=np.float32)
        for i, p in enumerate(rigid):
            axis = np.asarray(p["rotation_axis"], dtype=np.float64)
            if float(p["angular_velocity_z"]) != 0:
                scale = np.max(np.abs(axis))
                if scale == 0 or not np.isfinite(axis).all():
                    raise ValueError("A moving rigid pin requires a finite nonzero axis")
                axis = axis / scale
                axis /= np.linalg.norm(axis)
            rigid_data[i] = [*p["initial_target"], *p["rotation_center"], *axis,
                p["angular_velocity_z"], p["angular_velocity_ramp_duration"], *p["target_translation"],
                p["translation_start_time"], p["translation_duration"]]
            if np.any(np.asarray(p["target_translation"]) != 0) and p["translation_duration"] <= 0:
                raise ValueError("Translated rigid pin requires a positive translation duration")
        self.rigid_data = wp.array(rigid_data, dtype=float, device=self.device)
        if not np.isfinite(pin_data).all() or not np.isfinite(collider_data).all() or not np.isfinite(rigid_data).all():
            raise ValueError("Authored motion is not representable in float32")

    def reset(self, substep=0):
        if isinstance(substep, bool) or not isinstance(substep, int) or substep < 0:
            raise ValueError("Substep counter must be a nonnegative integer")
        self.counter.assign(np.asarray([substep], dtype=np.int64))

    def apply(self, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Motion timestep must be finite and positive")
        ipc = self.model.ipc
        if self.pin_nodes.size:
            wp.launch(_pins, self.pin_nodes.size, [self.counter, float(dt), self.pin_nodes, self.pin_data, ipc.pin_targets], device=self.device)
        if self.collider_nodes.size:
            wp.launch(_colliders, self.collider_nodes.size, [self.counter, float(dt), self.collider_nodes, self.collider_data, ipc.collider_data], device=self.device)
        if self.rigid_data.shape[0]:
            moving = self.model.body_count > 0 and not ipc.payload.get("uses_general_solver", False)
            wp.launch(_rigid_pins, self.rigid_data.shape[0], [self.counter, float(dt), int(moving), float(ipc.params["rigid_pin_time"]), self.rigid_data, ipc.rigid_pin_targets], device=self.device)

    def step(self, dt):
        wp.launch(_advance, 1, [self.counter], device=self.device)
        self.apply(dt)


wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})
