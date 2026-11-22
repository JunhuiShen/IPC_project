"""Feasible boundary initialization for the stock GS cloth twist.

The hard pins reach their prescribed represented positions together. If their
move alone is obstructed, offer the nearby free corners of those constraints
a rotation interpolated between the driven edges. This changes only the starting
iterate: forward_step has already saved the physical history and inertia.
"""
import warp as wp

from newton._src.geometry.tri_mesh_collision import TriMeshCollisionInfo
from . import ccd
from .ccd_defaults import LINEAR_CCD_GAP_FLOOR
from .contact import decode_candidate
from .warp_build import configure_module


@wp.kernel
def _targets(indices: wp.array[int], axes: wp.array[wp.vec3], roots: wp.array[wp.vec3],
             offsets: wp.array[wp.vec3], clock: wp.array[float], speed: float,
             end_time: float, target: wp.array[wp.vec3]):
    i = wp.tid()
    if clock[0] <= end_time:
        # Both prescribed axes are unit Y directions in the stock scene.
        theta = axes[i][1] * clock[0] * speed
        c = wp.cos(theta)
        s = wp.sin(theta)
        v = offsets[i]
        target[indices[i]] = roots[i] + wp.vec3(c*v[0] + s*v[2], v[1], c*v[2] - s*v[0])


@wp.kernel
def _extend(q: wp.array[wp.vec3], target: wp.array[wp.vec3], indices: wp.array[int],
            flags: wp.array[int], extended: wp.array[wp.vec3], error: wp.array[int]):
    i = wp.tid()
    x = q[i]
    if flags[i] & 1 == 0:
        extended[i] = target[i]
    else:
        # The outer corner of each driven row has a nonzero rotation radius.
        first = indices[0]
        second = indices[indices.shape[0] // 2]
        p0 = q[first]
        t0 = target[first]
        first_angle = wp.atan2(t0[0]*p0[2] - t0[2]*p0[0], p0[0]*t0[0] + p0[2]*t0[2])
        p1 = q[second]
        t1 = target[second]
        second_angle = wp.atan2(t1[0]*p1[2] - t1[2]*p1[0], p1[0]*t1[0] + p1[2]*t1[2])
        w = wp.clamp((x[1] - q[first][1]) / (q[second][1] - q[first][1]), 0.0, 1.0)
        w = w*w*(3.0 - 2.0*w)
        angle = (1.0-w)*first_angle + w*second_angle
        c = wp.cos(angle)
        s = wp.sin(angle)
        extended[i] = wp.vec3(c*x[0] + s*x[2], x[1], c*x[2] - s*x[0])
    for k in range(3):
        if not wp.isfinite(extended[i][k]) or not wp.isfinite(x[k]):
            wp.atomic_max(error, 0, 202)


@wp.func_native(ccd._INPUT + r"""
    using namespace ipc_gpu::ccd32direct::detail;
    if (!valid(query)) { result[1]=2.0f; return result; }
    Vec n0, n1;
    float h0, h1;
    if (prepare_axis(query,n0,h0)) {
        if (certify_axis(query,n0,h0)) { result[2]=1.0f; return result; }
        auto finish=query;
        for (int i=0;i<4;++i) for (int k=0;k<3;++k)
            finish.start[i][k]=query.end[i][k];
        if (prepare_axis(finish,n1,h1)) {
            // With affine corner paths and n(t)=(1-t)n0+t*n1 each
            // projected gap is quadratic. Its three Bernstein coefficients
            // bound it throughout the step. ||n(t)|| <= max(||n0||,||n1||).
            const float h=maximum(h0,h1);
            const int split=query.vertex_face ? 1 : 2;
            bool safe=true;
            for (int i=0;i<split && safe;++i) for (int j=split;j<4 && safe;++j) {
                const auto start=projected_gap(query.start[i],query.start[j],n0);
                const auto end=projected_gap(query.end[i],query.end[j],n1);
                const auto middle_sum=add(projected_gap(query.start[i],query.start[j],n1),
                                          projected_gap(query.end[i],query.end[j],n0));
                safe=bounded(start) && bounded(end) && bounded(middle_sum)
                    && start.lo>h && end.lo>h && middle_sum.lo>multiply_bound(2.0f,h,true);
            }
            if (safe) { result[2]=1.0f; return result; }
        }
    }
    if (ipc_gpu::twist_ccd::swept_certificate(query.start, query.end, query.vertex_face, query.gap_floor)) {
        result[2]=1.0f; return result;
    }
    result[2]=ipc_gpu::ccd32::validate_linear(query)==ipc_gpu::ccd32::Status::Safe ? 1.0f : 0.0f;
    return result;
""")
def _certify(positions: wp.array[wp.vec3], proposed: wp.array[wp.vec3],
             nodes: wp.vec4i, kind: int) -> ccd.PairResult: ...


@wp.kernel(module="unique", enable_backward=False)
def _check(q: wp.array[wp.vec3], trial: wp.array[wp.vec3], extended: wp.array[wp.vec3],
           ids: wp.array[int], count: wp.array[int], triangles: wp.array2d[int],
           edges: wp.array2d[int], info: TriMeshCollisionInfo, workers: int,
           promote: wp.array[int], blocked: wp.array[int], error: wp.array[int]):
    if error[0] == 0:
        for j in range(wp.tid(), count[0], workers):
            nodes, kind = decode_candidate(ids[j], triangles, edges, info)
            changed = False
            for k in range(4):
                changed = changed or not (q[nodes[k]] == trial[nodes[k]])
            if changed:
                result = _certify(q, trial, nodes, kind)
                if result[1] != 0.0:
                    wp.atomic_max(error, 0, 202)
                elif result[2] == 0.0:
                    wp.atomic_max(blocked, 0, 1)
                    for k in range(4):
                        v = nodes[k]
                        if not (extended[v] == trial[v]):
                            wp.atomic_max(promote, v, 1)


configure_module(_check, ("ccd_direct_fp32.cuh", "twist_ccd.cuh"))


@wp.kernel
def _include_neighbors(trial: wp.array[wp.vec3], extended: wp.array[wp.vec3],
                       promote: wp.array[int], changed: wp.array[int]):
    v = wp.tid()
    if promote[v]:
        trial[v] = extended[v]
        promote[v] = 0
        wp.atomic_max(changed, 0, 1)


@wp.kernel
def _next_round(changed: wp.array[int], condition: wp.array[int], error: wp.array[int]):
    # Each promotion round adds at least one new corner. A terminal blocker
    # goes through the hard-floor repair before publication.
    condition[0] = int(changed[0] != 0 and error[0] == 0)


@wp.kernel(module="unique", enable_backward=False)
def _first_obstruction(q: wp.array[wp.vec3], trial: wp.array[wp.vec3], ids: wp.array[int],
                       count: wp.array[int], triangles: wp.array2d[int], edges: wp.array2d[int],
                       info: TriMeshCollisionInfo, workers: int, index: wp.array[int], error: wp.array[int]):
    if error[0] == 0:
        for j in range(wp.tid(), count[0], workers):
            nodes, kind = decode_candidate(ids[j], triangles, edges, info)
            changed = False
            for k in range(4):
                changed = changed or not (q[nodes[k]] == trial[nodes[k]])
            if changed:
                result = _certify(q, trial, nodes, kind)
                if result[1] != 0.0:
                    wp.atomic_max(error, 0, 202)
                elif result[2] == 0.0:
                    wp.atomic_min(index, 0, j)


configure_module(_first_obstruction, ("ccd_direct_fp32.cuh", "twist_ccd.cuh"))
Offsets = wp.types.vector(length=12, dtype=wp.float32)


@wp.func_native(r"""
    wp::vec_t<12,float> result{};
    float start[4][3];
    double end[4][3], scale=0.0;
    bool free_corner[4];
    for (int i=0; i<4; ++i) {
        free_corner[i]=(flags.data[nodes[i]] & 1)!=0;
        for (int k=0; k<3; ++k) {
            start[i][k]=q.data[nodes[i]][k];
            end[i][k]=double(trial.data[nodes[i]][k]);
            scale=fmax(scale,fabs(end[i][k]));
        }
    }
    const auto n=ipc_gpu::ccd_clearance::best_axis(start,kind==0);
    const double target_gap=fmax(2.0*floor,double(floor)+16.0*1.1920928955078125e-7*scale);
    const int split=kind==0 ? 1 : 2;
    double offset[4]={};
    for (int a=0; a<split; ++a) for (int b=split; b<4; ++b) {
        double projected=0.;
        for (int k=0; k<3; ++k) projected+=(end[a][k]-end[b][k])*n.v[k];
        const int movers=static_cast<int>(free_corner[a])+static_cast<int>(free_corner[b]);
        if (movers && projected<target_gap) {
            const double amount=(target_gap-projected)/movers;
            if (free_corner[a]) offset[a]=fmax(offset[a],amount);
            if (free_corner[b]) offset[b]=fmin(offset[b],-amount);
        }
    }
    for (int i=0; i<4; ++i) for (int k=0; k<3; ++k)
        result[3*i+k]=static_cast<float>(offset[i]*n.v[k]);
    return result;
""")
def _repair_offsets(q: wp.array[wp.vec3], trial: wp.array[wp.vec3], nodes: wp.vec4i,
                    kind: int, flags: wp.array[int], floor: float) -> Offsets: ...


@wp.kernel(module="unique", enable_backward=False)
def _correct_obstruction(q: wp.array[wp.vec3], trial: wp.array[wp.vec3], center: wp.array[wp.vec3],
                         flags: wp.array[int], floor: float, max_change: float, ids: wp.array[int],
                         triangles: wp.array2d[int], edges: wp.array2d[int], info: TriMeshCollisionInfo,
                         index: wp.array[int], attempts: wp.array[int], condition: wp.array[int],
                         error: wp.array[int]):
    condition[0] = 0
    if error[0] == 0 and index[0] != 2147483647:
        nodes, kind = decode_candidate(ids[index[0]], triangles, edges, info)
        offsets = _repair_offsets(q, trial, nodes, kind, flags, floor)
        changed = False
        for i in range(4):
            v = nodes[i]
            if flags[v] & 1 != 0:
                p = trial[v] + wp.vec3(offsets[3*i], offsets[3*i+1], offsets[3*i+2])
                p = wp.min(wp.max(p, center[v] - wp.vec3(max_change)), center[v] + wp.vec3(max_change))
                changed = changed or not (p == trial[v])
                trial[v] = p
        attempts[0] += 1
        if changed and attempts[0] <= 64:
            condition[0] = 1
        else:
            wp.atomic_max(error, 0, 806)


configure_module(_correct_obstruction, ("ccd_clearance.cuh",))


@wp.kernel
def _publish(q: wp.array[wp.vec3], q_out: wp.array[wp.vec3], trial: wp.array[wp.vec3],
             flags: wp.array[int], error: wp.array[int]):
    v = wp.tid()
    if error[0] == 0:
        q[v] = trial[v]
        if flags[v] & 1 == 0:
            q_out[v] = trial[v]


@wp.kernel
def _advance_clock(clock: wp.array[float], dt: float, end_time: float, error: wp.array[int]):
    if error[0] == 0 and clock[0] <= end_time:
        clock[0] += dt


class TwistBoundary:
    def __init__(self, example):
        self.indices = example.rot_point_indices
        self.axes, self.roots, self.offsets = example.rot_axes, example.roots, example.roots_to_ps
        self.clock, self.speed, self.end_time = example.t, example.rot_angular_velocity, example.rot_end_time
        model = example.model
        self.device = model.device
        self.extended = wp.empty_like(model.particle_q)
        self.promote = wp.zeros(model.particle_count, int, device=self.device)
        self.changed = wp.zeros(1, int, device=self.device)
        self.blocked = wp.zeros_like(self.changed)
        self.condition = wp.zeros_like(self.changed)
        self.repair_center = wp.empty_like(model.particle_q)
        self.repair_index = wp.empty_like(self.changed)
        self.repair_attempts = wp.zeros_like(self.changed)

    def advance(self, solver, q, q_out, dt):
        c = solver.contact_backend
        trial = solver.ccd_backend.candidate
        wp.copy(trial, q)
        wp.launch(_targets, self.indices.shape[0],
                  [self.indices, self.axes, self.roots, self.offsets, self.clock,
                   self.speed, self.end_time, trial], device=self.device)
        wp.launch(_extend, q.shape[0], [q, trial, self.indices, c.model.particle_flags,
                  self.extended, c.error], device=self.device)
        # Every trial corner is either q or extended. The BVHs enclose all
        # these choices, including newly moving free vertices on later rounds.
        c.detect_swept(solver.trimesh_collision_detector, q, self.extended)
        self.condition.fill_(1)

        def check_motion():
            self.blocked.zero_()
            self.changed.zero_()
            if c.capacity:
                wp.launch(_check, c.fast_workers, [q, trial, self.extended, c.candidate_ids,
                          c.candidate_count, c.model.tri_indices, c.model.edge_indices, c.info,
                          c.fast_workers, self.promote, self.blocked, c.error],
                          device=self.device, block_dim=64)
            wp.launch(_include_neighbors, q.shape[0],
                      [trial, self.extended, self.promote, self.changed], device=self.device)
            wp.launch(_next_round, 1, [self.changed, self.condition, c.error], device=self.device)

        wp.capture_while(self.condition, check_motion)

        def repair_motion():
            wp.copy(self.repair_center, trial)
            self.condition.fill_(1)
            self.repair_attempts.zero_()

            def repair_round():
                # Correcting even a single free corner changes the swept
                # region. Check against a fresh BVH from the original state.
                c.detect_swept(solver.trimesh_collision_detector, q, trial)
                self.repair_index.fill_(2147483647)
                if c.capacity:
                    wp.launch(_first_obstruction, c.fast_workers, [q, trial, c.candidate_ids,
                              c.candidate_count, c.model.tri_indices, c.model.edge_indices, c.info,
                              c.fast_workers, self.repair_index, c.error], device=self.device, block_dim=64)
                wp.launch(_correct_obstruction, 1, [q, trial, self.repair_center, c.model.particle_flags,
                          LINEAR_CCD_GAP_FLOOR, float(solver.params["node_box_max"]), c.candidate_ids,
                          c.model.tri_indices, c.model.edge_indices, c.info, self.repair_index,
                          self.repair_attempts, self.condition, c.error], device=self.device)

            wp.capture_while(self.condition, repair_round)

        wp.capture_if(self.blocked, on_true=repair_motion)
        wp.launch(_publish, q.shape[0], [q, q_out, trial, c.model.particle_flags, c.error], device=self.device)
        wp.launch(_advance_clock, 1, [self.clock, dt, self.end_time, c.error], device=self.device)


def simulate_gs_twist(example):
    """The GS initialization stages and certifies the driven positions."""
    example.collision_pipeline.collide(example.state_0, example.contacts)
    example.solver.rebuild_bvh(example.state_0)
    for _ in range(example.sim_substeps):
        example.state_0.clear_forces()
        example.viewer.apply_forces(example.state_0)
        example.solver.step(example.state_0, example.state_1, example.control, example.contacts, example.sim_dt)
        example.state_0, example.state_1 = example.state_1, example.state_0
