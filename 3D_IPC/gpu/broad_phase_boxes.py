"""Project node/edge/triangle bounds and BVHs, without collision queries.

This sidecar deliberately owns no contact buffers and never changes positions.
Blue node boxes use the GS movement-radius policy. Green triangles are unions
of blue boxes expanded by d_hat; red edges are unexpanded endpoint unions, with
separate green expanded edge boxes. The three BVHs store blue nodes, green
triangles and red edges, as in BroadPhase::initialize_from_vertex_boxes.

With collision processing off these are checkpoint bounds, not guarantees on
the unrestricted motion that can follow a checkpoint.
"""

import math

import warp as wp


@wp.kernel
def _advance_ccd_epoch(epoch: wp.array[wp.uint64]):
    epoch[0] += wp.uint64(1)


@wp.func_native(r"""
    if(amount==0.0)return center;
    const double raw=upper!=0?double(center)+amount:double(center)-amount;
    // One outward binary64 ULP before converting to binary32. Do not rely on
    // libc nextafter: Warp's CPU native snippets do not include its declaration.
    union DoubleBits { double d; uint64_t u; } wide;
    wide.d=raw;
    if(raw==0.0)wide.u=upper!=0?uint64_t(1):uint64_t(0x8000000000000001ULL);
    else if((raw>0.0)==(upper!=0))++wide.u;
    else --wide.u;
    const double bound=wide.d;
    float value=static_cast<float>(bound);
    union FloatBits { float f; uint32_t u; } bits;
    bits.f=value;
    if(upper!=0 && double(value)<bound) {
        if(value==0.0f)bits.u=1u;
        else if(value>0.0f)++bits.u;else --bits.u;
    } else if(upper==0 && double(value)>bound) {
        if(value==0.0f)bits.u=0x80000001u;
        else if(value>0.0f)--bits.u;else ++bits.u;
    }
    return bits.f;
""")
def outward_bound(center: float, amount: wp.float64, upper: int) -> float: ...


@wp.kernel
def _begin_radii(velocity: wp.array[wp.vec3], previous: wp.array[wp.float64],
                 dt: wp.float64, minimum: wp.float64, maximum: wp.float64, padding: wp.float64,
                 radii: wp.array[wp.float64], error: wp.array[int]):
    i = wp.tid()
    v = wp.vec3d(velocity[i])
    distance = dt * wp.length(v)
    if wp.isfinite(distance) and wp.isfinite(previous[i]):
        radii[i] = wp.clamp(padding * wp.max(previous[i], distance), minimum, maximum)
    else:
        radii[i] = maximum
        wp.atomic_max(error, 0, 202)


@wp.kernel
def _finish_history(positions: wp.array[wp.vec3], start: wp.array[wp.vec3],
                    previous: wp.array[wp.float64], error: wp.array[int]):
    i = wp.tid()
    distance = wp.length(wp.vec3d(positions[i]) - wp.vec3d(start[i]))
    if wp.isfinite(distance):
        previous[i] = distance
    else:
        wp.atomic_max(error, 0, 202)


@wp.kernel
def _reset_history(positions: wp.array[wp.vec3], worlds: wp.array[int], world_count: int,
                   mask: wp.array[wp.bool], all_worlds: bool, maximum: wp.float64,
                   start: wp.array[wp.vec3], previous: wp.array[wp.float64], radii: wp.array[wp.float64]):
    i = wp.tid()
    world = worlds[i]
    selected = all_worlds
    if not all_worlds:
        if world == -1:
            world = world_count
        selected = mask[world]
    if selected:
        start[i] = positions[i]
        previous[i] = maximum
        radii[i] = maximum


@wp.kernel
def _node_bounds(positions: wp.array[wp.vec3], radii: wp.array[wp.float64],
                 lower: wp.array[wp.vec3], upper: wp.array[wp.vec3], error: wp.array[int]):
    i = wp.tid()
    point = positions[i]
    lo = wp.vec3(0.0)
    hi = wp.vec3(0.0)
    valid = wp.isfinite(radii[i]) and radii[i] > wp.float64(0.0)
    for axis in range(3):
        lo[axis] = outward_bound(point[axis], radii[i], 0)
        hi[axis] = outward_bound(point[axis], radii[i], 1)
        valid = valid and wp.isfinite(point[axis]) and wp.isfinite(lo[axis]) and wp.isfinite(hi[axis])
    if valid:
        lower[i] = lo
        upper[i] = hi
    else:
        # Keep NaN/Inf out of BVH construction; status remains latched.
        lower[i] = wp.vec3(0.0)
        upper[i] = wp.vec3(0.0)
        wp.atomic_max(error, 0, 202)


@wp.kernel
def _triangle_bounds(triangles: wp.array2d[int], node_lower: wp.array[wp.vec3],
                     node_upper: wp.array[wp.vec3], reach: wp.float64,
                     lower: wp.array[wp.vec3], upper: wp.array[wp.vec3], error: wp.array[int]):
    i = wp.tid()
    lo = wp.min(node_lower[triangles[i, 0]], wp.min(node_lower[triangles[i, 1]], node_lower[triangles[i, 2]]))
    hi = wp.max(node_upper[triangles[i, 0]], wp.max(node_upper[triangles[i, 1]], node_upper[triangles[i, 2]]))
    valid = bool(True)
    for axis in range(3):
        lo[axis] = outward_bound(lo[axis], reach, 0)
        hi[axis] = outward_bound(hi[axis], reach, 1)
        valid = valid and wp.isfinite(lo[axis]) and wp.isfinite(hi[axis])
    if not valid:
        lo = wp.vec3(0.0)
        hi = wp.vec3(0.0)
        wp.atomic_max(error, 0, 202)
    lower[i] = lo
    upper[i] = hi


@wp.kernel
def _edge_bounds(edges: wp.array2d[int], node_lower: wp.array[wp.vec3],
                 node_upper: wp.array[wp.vec3], reach: wp.float64,
                 lower: wp.array[wp.vec3], upper: wp.array[wp.vec3],
                 query_lower: wp.array[wp.vec3], query_upper: wp.array[wp.vec3], error: wp.array[int]):
    i = wp.tid()
    # Newton edge slots 0,1 are opposite hinge vertices (possibly -1).
    lo = wp.min(node_lower[edges[i, 2]], node_lower[edges[i, 3]])
    hi = wp.max(node_upper[edges[i, 2]], node_upper[edges[i, 3]])
    lower[i] = lo
    upper[i] = hi
    valid = bool(True)
    for axis in range(3):
        lo[axis] = outward_bound(lo[axis], reach, 0)
        hi[axis] = outward_bound(hi[axis], reach, 1)
        valid = valid and wp.isfinite(lo[axis]) and wp.isfinite(hi[axis])
    if not valid:
        lo = wp.vec3(0.0)
        hi = wp.vec3(0.0)
        wp.atomic_max(error, 0, 202)
    query_lower[i] = lo
    query_upper[i] = hi


class BroadPhaseBoxes:
    """Preallocated bounds and three hierarchies; no detector or pair storage."""

    def __init__(self, model, error, *, minimum=.001, maximum=.01, padding=1.2,
                 d_hat=.002, update_interval=10):
        self.minimum, self.maximum = float(minimum), float(maximum)
        self.padding, self.d_hat = float(padding), float(d_hat)
        if not (math.isfinite(self.minimum) and math.isfinite(self.maximum)
                and 1.e-10 < self.minimum <= self.maximum):
            raise ValueError('Node-box radii must be finite and 1e-10 < min <= max')
        if not math.isfinite(self.padding) or self.padding < 1.:
            raise ValueError('Node-box padding must be finite and at least 1')
        if not math.isfinite(self.d_hat) or self.d_hat < 0.:
            raise ValueError('d_hat must be finite and nonnegative')
        if isinstance(update_interval, bool) or int(update_interval) != update_interval or update_interval <= 0:
            raise ValueError('node_box_update_count must be a positive integer')
        self.update_interval = int(update_interval)
        self.model, self.device, self.error = model, model.device, error
        # Allocated only by the opt-in color CCD consumer. Existing ablations
        # retain their original launches and arithmetic.
        self.ccd_epoch = None
        self.ccd_centers = None
        self.ccd_radii = None
        self.radii = wp.full(model.particle_count, self.maximum, wp.float64, device=self.device)
        self.previous_displacement = wp.full(model.particle_count, self.maximum, wp.float64, device=self.device)
        self.start = wp.clone(model.particle_q)
        for prefix, count in (('node', model.particle_count), ('triangle', model.tri_count),
                              ('edge', model.edge_count), ('edge_query', model.edge_count)):
            setattr(self, prefix+'_lower', wp.empty(count, wp.vec3, device=self.device))
            setattr(self, prefix+'_upper', wp.empty(count, wp.vec3, device=self.device))
        self._compute_bounds(model.particle_q)
        # Construct once before graph capture; update/refit/rebuild reuse storage.
        self.node_bvh = self._make_bvh(self.node_lower, self.node_upper)
        self.triangle_bvh = self._make_bvh(self.triangle_lower, self.triangle_upper)
        self.edge_bvh = self._make_bvh(self.edge_lower, self.edge_upper)

    @staticmethod
    def _make_bvh(lower, upper):
        return wp.Bvh(lower, upper) if lower.shape[0] else None

    def _compute_bounds(self, positions):
        if positions.dtype != wp.vec3 or positions.shape != (self.model.particle_count,) or positions.device != self.device:
            raise TypeError('Box positions must match the model particle array')
        m = self.model
        if m.particle_count:
            wp.launch(_node_bounds, m.particle_count,
                      [positions, self.radii, self.node_lower, self.node_upper, self.error], device=self.device)
        if m.tri_count:
            wp.launch(_triangle_bounds, m.tri_count,
                      [m.tri_indices, self.node_lower, self.node_upper, self.d_hat,
                       self.triangle_lower, self.triangle_upper, self.error], device=self.device)
        if m.edge_count:
            wp.launch(_edge_bounds, m.edge_count,
                      [m.edge_indices, self.node_lower, self.node_upper, self.d_hat,
                       self.edge_lower, self.edge_upper, self.edge_query_lower, self.edge_query_upper,
                       self.error], device=self.device)
        if self.ccd_epoch is not None:
            wp.copy(self.ccd_centers, positions)
            wp.copy(self.ccd_radii, self.radii)
            wp.launch(_advance_ccd_epoch, 1, [self.ccd_epoch], device=self.device)

    def begin(self, velocity, dt):
        if velocity.dtype != wp.vec3 or velocity.shape != (self.model.particle_count,) or velocity.device != self.device:
            raise TypeError('Box velocities must match the model particle array')
        if not math.isfinite(dt) or dt <= 0.:
            raise ValueError('Box timestep must be positive and finite')
        if self.model.particle_count:
            wp.launch(_begin_radii, self.model.particle_count,
                      [velocity, self.previous_displacement, float(dt), self.minimum, self.maximum,
                       self.padding, self.radii, self.error], device=self.device)

    def anchor(self, positions):
        wp.copy(self.start, positions)

    def update(self, positions, *, rebuild=False):
        self._compute_bounds(positions)
        for tree in (self.node_bvh, self.triangle_bvh, self.edge_bvh):
            if tree is not None:
                if rebuild:
                    tree.rebuild()
                else:
                    tree.refit()

    def finish(self, positions):
        if self.model.particle_count:
            wp.launch(_finish_history, self.model.particle_count,
                      [positions, self.start, self.previous_displacement, self.error], device=self.device)

    def reset(self, positions, world_mask=None):
        if self.model.particle_count:
            wp.launch(_reset_history, self.model.particle_count,
                      [self.start if positions is None else positions,
                       self.model.particle_world, self.model.world_count, world_mask,
                       world_mask is None, self.maximum, self.start, self.previous_displacement, self.radii],
                      device=self.device)
        # Newton permits partial-state resets with no particle position array.
        # The next initialization will anchor/refit to its actual input state.
        if positions is not None:
            self.update(positions)
