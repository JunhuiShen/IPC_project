"""Independent frozen-contact rigid pullbacks and production precision dispatch."""

from functools import cache
import unittest

import numpy as np
import warp as wp

from gpu import contact, rigid, dat_energies
from gpu.gs_contact_energies import make_full_evaluator
from gpu.scene import create_model
from gpu.solver import SolverIPC
from .test_body_solver import body_payload
from .test_rigid import fd_gradient, fd_hessian, matrix, orientation


@cache
def packet_kernel(precision):
    evaluate = make_full_evaluator(precision)

    @wp.kernel(module="unique", enable_backward=False)
    def probe(q: wp.array[wp.vec3], previous: wp.array[wp.vec3], kind: int,
              params: contact.Parameters, packets: wp.array[contact.FullPacket]):
        packets[0] = evaluate(q, previous, wp.vec4i(0, 1, 2, 3), kind, params)

    return probe


@wp.kernel(module="unique", enable_backward=False)
def pullback(packets: wp.array[contact.FullPacket], owners: wp.array[int], references: wp.array[wp.vec3],
             q: wp.quat, omega: wp.vec3, dt: float, kind: int, rotational: int,
             scale: float, approximate: int, gradients: wp.array[wp.vec3], hessians: wp.array[wp.mat33]):
    gradient, hessian = rigid.contact_terms(packets[0], wp.vec4i(0, 1, 2, 3), owners, references,
        0, q, omega, dt, kind, rotational, scale, approximate)
    gradients[0] = gradient
    hessians[0] = hessian


@wp.kernel(module="unique", enable_backward=False)
def production_system(data: rigid.BodyData, rotational: int,
                      gradients: wp.array[wp.vec3], hessians: wp.array[wp.mat33]):
    gradient, hessian = rigid._system(data, 0, rotational, 0)
    gradients[0] = gradient
    hessians[0] = hessian


@wp.kernel(module="unique", enable_backward=False)
def production_selected_body_system(data: rigid.BodyData, body: int, rotational: int,
                                    gradients: wp.array[wp.vec3], hessians: wp.array[wp.mat33]):
    gradient, hessian = rigid._system(data, body, rotational, 0)
    gradients[0] = gradient
    hessians[0] = hessian


@cache
def native_directed_edge_kernel(precision):
    edge_energy = dat_energies._energy_wrappers(precision)[0]
    full_packet = make_full_evaluator(precision)

    @wp.kernel(module="unique", enable_backward=False)
    def probe(data: rigid.BodyData, body: int, rotational: int,
              source_edge: int, target_edge: int, ctx: dat_energies.EnergyContext,
              active: wp.array[int], forces: wp.array[wp.vec3], diagonal: wp.array[wp.mat33],
              weights: wp.array[wp.vec2], inertia_g: wp.array[wp.vec3], inertia_h: wp.array[wp.mat33],
              canonical_g: wp.array[wp.vec3], canonical_h: wp.array[wp.mat33]):
        p = data.contact_parameters
        yes, f0, f1, h0, h1 = edge_energy(source_edge, target_edge, data.q, data.previous,
            data.edges, float(p.d_hat), 0.0, 0.0, float(p.friction), float(p.eps_v), data.dt,
            1.0e-5, True, ctx)
        active[0] = int(yes)
        forces[0] = f0
        forces[1] = f1
        diagonal[0] = h0
        diagonal[1] = h1
        a = data.edges[source_edge, 2]
        b = data.edges[source_edge, 3]
        c = data.edges[target_edge, 2]
        d = data.edges[target_edge, 3]
        closest = wp.closest_point_edge_edge(data.q[a], data.q[b], data.q[c], data.q[d], 1.0e-5)
        weights[0] = wp.vec2(1.0 - closest[0], closest[0])
        old = data.previous_pose[body]
        old_q = wp.transform_get_rotation(old)
        velocity = data.previous_velocity[body]
        g = wp.vec3(0.0)
        h = wp.mat33(0.0)
        if rotational != 0:
            g, h = rigid.rotation_inertia(old_q, wp.vec3(velocity[3], velocity[4], velocity[5]),
                data.omega[body], data.dt, data.second_moment[body])
        else:
            g, h = rigid.translation_inertia(wp.transform_get_translation(data.poses[body]),
                wp.transform_get_translation(old), wp.vec3(velocity[0], velocity[1], velocity[2]),
                data.mass[body], data.gravity_y, data.dt)
        inertia_g[0] = g
        inertia_h[0] = h
        # The former implementation used this body-0-first frame for both
        # bodies. Preserve it solely to prove this fixture detects that bug.
        packet = full_packet(data.q, data.previous, wp.vec4i(0, 1, 3, 4), 1, p)
        legacy_g, legacy_h = rigid.contact_terms(packet, wp.vec4i(0, 1, 3, 4), data.owners,
            data.reference, body, old_q, data.omega[body], data.dt, 1, rotational,
            wp.float32(p.k_barrier) * data.dt * data.dt, 1)
        canonical_g[0] = legacy_g
        canonical_h[0] = legacy_h

    rigid.configure_body_module(probe)
    return probe


@cache
def selected_system_kernel(precision):
    evaluate = make_full_evaluator(precision)

    @wp.kernel(module="unique", enable_backward=False)
    def direct(data: rigid.BodyData, rotational: int, packets: wp.array[contact.FullPacket],
               gradients: wp.array[wp.vec3], hessians: wp.array[wp.mat33]):
        # Deliberately select the evaluator outside BodyData's dispatch branch.
        candidate = data.candidates[data.incidence[data.offsets[0]]]
        nodes, kind = contact.decode_candidate(candidate, data.triangles, data.edges, data.info)
        packet = evaluate(data.q, data.previous, nodes, kind, data.contact_parameters)
        packets[0] = packet
        old = data.previous_pose[0]
        old_q = wp.transform_get_rotation(old)
        velocity = data.previous_velocity[0]
        gradient = wp.vec3(0.0)
        hessian = wp.mat33(0.0)
        if rotational != 0:
            gradient, hessian = rigid.rotation_inertia(old_q, wp.vec3(velocity[3], velocity[4], velocity[5]),
                data.omega[0], data.dt, data.second_moment[0])
        else:
            gradient, hessian = rigid.translation_inertia(wp.transform_get_translation(data.poses[0]),
                wp.transform_get_translation(old), wp.vec3(velocity[0], velocity[1], velocity[2]),
                data.mass[0], data.gravity_y, data.dt)
        contact_g, contact_h = rigid.contact_terms(packet, nodes, data.owners, data.reference, 0,
            old_q, data.omega[0], data.dt, kind, rotational,
            wp.float32(data.contact_parameters.k_barrier) * data.dt * data.dt, 1)
        gradients[0] = gradient + contact_g
        hessians[0] = hessian + contact_h

    rigid.configure_body_module(direct)
    return direct


rigid.configure_body_module(pullback)
rigid.configure_body_module(production_system)
rigid.configure_body_module(production_selected_body_system)


def independent_jacobians(references, old_xyzw, omega, dt, rotational):
    if not rotational:
        return np.repeat(np.eye(3)[None], len(references), axis=0), np.zeros((len(references), 3, 3, 3))
    quaternion = old_xyzw[[3, 0, 1, 2]].astype(np.float64)
    quaternion /= np.linalg.norm(quaternion)
    jacobians, curvature = [], []
    for reference in references:
        def position(value):
            return matrix(orientation(quaternion, value, dt)) @ reference
        jacobians.append(np.array([fd_gradient(lambda value: position(value)[axis], omega)
                                   for axis in range(3)]))
        curvature.append(np.array([fd_hessian(lambda value: position(value)[axis], omega)
                                   for axis in range(3)]))
    return np.asarray(jacobians), np.asarray(curvature)


def numpy_pullback(packet, owners, references, old_xyzw, omega, dt, rotational, scale):
    jacobians, curvature = independent_jacobians(references, old_xyzw, omega, dt, rotational)
    chosen = np.flatnonzero(np.asarray(owners) == 0)
    gradients = packet[8:20].reshape(4, 3)
    blocks = packet[20:164].reshape(4, 4, 3, 3)
    friction_g, friction_h = packet[164:167], packet[167:176].reshape(3, 3)
    weighted = sum((packet[1 + i] * jacobians[i] for i in chosen), np.zeros((3, 3)))
    gradient = sum((jacobians[i].T @ gradients[i] for i in chosen), np.zeros(3))
    hessian = np.zeros((3, 3))
    for i in chosen:
        for j in chosen:
            hessian += jacobians[i].T @ blocks[i, j] @ jacobians[j]
    curved = sum((np.einsum("a,aij->ij", gradients[i], curvature[i]) for i in chosen), np.zeros((3, 3)))
    return (scale * gradient + weighted.T @ friction_g,
            scale * hessian + weighted.T @ friction_h @ weighted,
            scale * curved)


class GsRigidEnergyParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.devices = [wp.get_device("cpu"), *wp.get_cuda_devices()]

    def fixture(self, device, kind, precision, slip):
        if kind == 0:
            q = np.array([[.25, .25, .024], [0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
        else:
            q = np.array([[0, 0, 0], [1, 0, 0], [.5, -.5, .026], [.5, .5, .026]], np.float32)
        previous = q.copy()
        previous[0] -= np.array([slip, .4 * slip, 0.], np.float32)
        params = contact.parameters(d_hat=.1, k_barrier=300., dt=.075, friction=.35, eps_v=.02)
        packet = wp.empty(1, contact.FullPacket, device=device)
        wp.launch(packet_kernel(precision), 1,
            [wp.array(q, wp.vec3, device=device), wp.array(previous, wp.vec3, device=device), kind, params, packet], device=device)
        raw = packet.numpy()[0]
        self.assertEqual(raw[176], 0)
        self.assertEqual(raw[177], 1)
        return q, previous, params, packet, raw

    def assert_packet_units(self, q, previous, params, raw):
        distance, weights, normal = raw[0], raw[1:5], raw[5:8]
        radius = float(np.float32(params.d_hat))
        gap = distance - radius
        # Direct derivatives of -k*(d-r)^2*log(d/r), without DAT callbacks.
        first = -2. * gap * np.log(distance / radius) - gap * gap / distance
        second = -2. * np.log(distance / radius) - 4. * gap / distance + gap * gap / distance**2
        expected_g = weights[:, None] * first * normal
        expected_h = np.einsum("i,j,a,b->ijab", weights, weights, normal, normal) * second
        np.testing.assert_allclose(raw[8:20].reshape(4, 3), expected_g, rtol=3e-5, atol=2e-7)
        np.testing.assert_allclose(raw[20:164].reshape(4, 4, 3, 3), expected_h, rtol=3e-5, atol=2e-7)
        motion = np.sum(weights[:, None] * (q.astype(np.float64) - previous), axis=0)
        tangent = np.eye(3) - np.outer(normal, normal)
        slip = tangent @ motion
        length = np.linalg.norm(slip)
        expected_fg, expected_fh = np.zeros(3), np.zeros((3, 3))
        if length > 0:
            threshold = float(np.float32(params.eps_v)) * float(np.float32(params.dt))
            factor = 1. / length if length > threshold else (-length / threshold + 2.) / threshold
            normal_force = -float(np.float32(params.k_barrier)) * first
            weight = float(np.float32(params.friction)) * normal_force * factor
            expected_fg = weight * slip * params.dt**2
            expected_fh = weight * tangent * params.dt**2
        np.testing.assert_allclose(raw[164:167], expected_fg, rtol=8e-5, atol=2e-8)
        np.testing.assert_allclose(raw[167:176].reshape(3, 3), expected_fh, rtol=8e-5, atol=2e-8)

    def test_native_packet_normal_and_friction_units_include_zero_smooth_and_sliding_slip(self):
        for device in self.devices:
            for precision in ("float32", "float64"):
                for kind in (0, 1):
                    for slip in (0., .0005, .008):
                        with self.subTest(device=str(device), precision=precision, kind=kind, slip=slip):
                            q, previous, params, packet, raw = self.fixture(device, kind, precision, slip)
                            self.assert_packet_units(q, previous, params, raw)

    def test_mixed_ownership_psd_pullback_matches_numpy_and_keeps_gradient(self):
        old = np.array([.12, -.17, .08, .97], np.float32)
        omega = np.array([.7, -.4, .3], np.float32)
        dt = float(np.float32(.075))
        scale = float(np.float32(np.float32(300.) * np.float32(dt) * np.float32(dt)))
        saw_curvature = False
        for device in self.devices:
            for precision in ("float32", "float64"):
                for kind in (0, 1):
                    q, previous, params, packet, raw = self.fixture(device, kind, precision, .008)
                    owners_cases = ([0, -1, -1, -1], [-1, 0, 0, 0]) if kind == 0 else ([0, 0, 1, 1], [1, 1, 0, 0])
                    references = q + np.array([.13, -.11, .19], np.float32)
                    for owners in owners_cases:
                        for rotational in (0, 1):
                            with self.subTest(device=str(device), precision=precision, kind=kind,
                                              owners=owners, rotational=rotational):
                                expected_g, expected_h, curvature = numpy_pullback(raw, owners, references,
                                    old, omega.astype(np.float64), dt, rotational, scale)
                                outputs = []
                                for approximate in (0, 1):
                                    g = wp.empty(1, wp.vec3, device=device)
                                    h = wp.empty(1, wp.mat33, device=device)
                                    wp.launch(pullback, 1,
                                        [packet, wp.array(owners, int, device=device), wp.array(references, wp.vec3, device=device),
                                         wp.quat(*old), wp.vec3(omega), dt, kind, rotational, scale, approximate, g, h], device=device)
                                    outputs.append((g.numpy()[0], h.numpy()[0]))
                                np.testing.assert_allclose(outputs[1][0], expected_g, rtol=3e-5, atol=3e-7)
                                np.testing.assert_allclose(outputs[1][1], expected_h, rtol=3e-5, atol=3e-7)
                                np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
                                np.testing.assert_allclose(outputs[0][1] - outputs[1][1], curvature, rtol=5e-3, atol=3e-7)
                                self.assertGreaterEqual(np.linalg.eigvalsh(outputs[1][1].astype(np.float64)).min(), -1e-6)
                                saw_curvature |= rotational != 0 and np.max(np.abs(curvature)) > 1e-5
        self.assertTrue(saw_curvature, "Nontrivial rotation must distinguish signed curvature from J^T H J")

    def test_body_solver_uses_selected_native_scalar_precision_in_actual_system(self):
        for device in self.devices:
            for precision in ("float32", "float64"):
                with self.subTest(device=str(device), precision=precision):
                    payload = body_payload(centers=((0., 0., 0.),), rotate=True)
                    payload["positions"].append([.013, .021, .00231])
                    payload["velocities"].append([0, 0, 0])
                    payload["mass"].append(1.)
                    payload["node_to_rb"].append(-1)
                    payload["pin_targets"].append([0, 0, 0])
                    payload["surface_nodes"].append(3)
                    payload["params"].update(gs_contact_precision=precision, friction_coefficient=.37,
                        friction_velocity_epsilon=.013, gravity=[0., 0., 0.])
                    model = create_model(payload, device)
                    solver = SolverIPC(model, allow_cpu=True)
                    state = model.state()
                    body, backend = solver._rigid, solver.contact_backend
                    dt = payload["params"]["dt"]
                    body.begin(state, dt)
                    body.omega.assign([[.7, -.4, .3]])
                    previous = state.particle_q.numpy()
                    previous[3] -= [.0011, -.0007, 0.]
                    solver.particle_q_prev.assign(previous)
                    pairs = backend.info.vertex_colliding_triangles.numpy()
                    pairs[:2] = [3, 0]
                    backend.info.vertex_colliding_triangles.assign(pairs)
                    ids = np.full(backend.candidate_ids.shape[0], -1, np.int32)
                    ids[0] = 0
                    backend.candidate_ids.assign(ids)
                    backend.candidate_count.assign([1])
                    offsets = np.zeros(model.body_count + 1, np.int32)
                    offsets[1] = 1
                    body.offsets.assign(offsets)
                    incidence = np.zeros(body.incidence.shape[0], np.int32)
                    body.incidence.assign(incidence)
                    data = body._data(state, dt)
                    self.assertEqual(data.use_dat_energies, 2 if precision == "float64" else 1)
                    self.assertEqual(backend.energy_model, "dat")
                    packets = []
                    for rotational in (0, 1):
                        actual_g = wp.empty(1, wp.vec3, device=device)
                        actual_h = wp.empty(1, wp.mat33, device=device)
                        wp.launch(production_system, 1, [data, rotational, actual_g, actual_h], device=device)
                        expected_g = wp.empty(1, wp.vec3, device=device)
                        expected_h = wp.empty(1, wp.mat33, device=device)
                        for chosen in ("float32", "float64"):
                            packet = wp.empty(1, contact.FullPacket, device=device)
                            wp.launch(selected_system_kernel(chosen), 1,
                                [data, rotational, packet, expected_g, expected_h], device=device)
                            if chosen == precision:
                                np.testing.assert_array_equal(actual_g.numpy(), expected_g.numpy())
                                np.testing.assert_array_equal(actual_h.numpy(), expected_h.numpy())
                            if rotational == 0:
                                packets.append(packet.numpy()[0])
                    self.assertTrue(np.any(packets[0] != packets[1]), "Fixture must distinguish scalar precision")
                    self.assertEqual(int(solver.error.numpy()[0]), 0)

    def test_actual_body_system_uses_owning_edge_first_for_antiparallel_contacts(self):
        # Both directions have an interval of equally close points. Native
        # source-edge evaluation picks x=.2 in one direction and x=.8 in the
        # other; their angular pullbacks and friction must retain that choice.
        q = np.array([[0., 0., 0.], [1., 0., 0.], [.1, -.3, .2],
                      [.8, .00231, 0.], [.2, .00231, 0.], [.45, .25, .1]], np.float32)
        for device in self.devices:
            for precision in ("float32", "float64"):
                payload = body_payload(centers=((0., 0., 0.), (0., 0., 0.)), rotate=True)
                payload["positions"] = q.tolist()
                for body in range(2):
                    payload["bodies"][body]["reference_positions"] = q[3*body:3*body+3].tolist()
                payload["params"].update(gs_contact_precision=precision, friction_coefficient=.37,
                    friction_velocity_epsilon=.013, gravity=[0., 0., 0.])
                model = create_model(payload, device)
                solver = SolverIPC(model, allow_cpu=True)
                state = model.state()
                body_solver, backend = solver._rigid, solver.contact_backend
                dt = float(np.float32(payload["params"]["dt"]))
                body_solver.begin(state, dt)
                body_solver.omega.assign([[.7, -.4, .3], [-.2, .6, -.5]])
                previous = q.copy()
                previous[0] -= [.0013, 0., .0004]
                previous[3] -= [-.0007, 0., .0011]
                solver.particle_q_prev.assign(previous)
                ee = backend.info.edge_colliding_edges.numpy()
                ee[:2] = [0, 1]
                backend.info.edge_colliding_edges.assign(ee)
                candidates = np.zeros(backend.candidate_ids.shape[0], np.int32)
                candidates[0] = -1
                backend.candidate_ids.assign(candidates)
                backend.candidate_count.assign([1])
                body_solver.offsets.assign([0, 1, 2])
                body_solver.incidence.zero_()
                data = body_solver._data(state, dt)
                # Use two explicit collision edges while retaining the real
                # BodySolver's state, incidence, owners and energy selection.
                data.edges = wp.array([[-1, -1, 0, 1], [-1, -1, 3, 4]], int, device=device)
                workspace = dat_energies.EnergyWorkspace(backend.info, device=device,
                    k_barrier=backend.options["k_barrier"], precision=precision, error=solver.error)
                context = workspace.begin(dt, backend.options["d_hat"], backend.options["friction"],
                                          backend.options["eps_v"])
                own_points = []
                distinguishes_canonical = False
                for body in range(2):
                    source_nodes = np.array([0, 1]) if body == 0 else np.array([3, 4])
                    for rotational in (0, 1):
                        with self.subTest(device=str(device), precision=precision, body=body, rotational=rotational):
                            active = wp.empty(1, int, device=device)
                            force = wp.empty(2, wp.vec3, device=device)
                            diagonal = wp.empty(2, wp.mat33, device=device)
                            weights = wp.empty(1, wp.vec2, device=device)
                            inertia_g = wp.empty(1, wp.vec3, device=device)
                            inertia_h = wp.empty(1, wp.mat33, device=device)
                            legacy_g = wp.empty(1, wp.vec3, device=device)
                            legacy_h = wp.empty(1, wp.mat33, device=device)
                            wp.launch(native_directed_edge_kernel(precision), 1,
                                [data, body, rotational, body, 1-body, context, active, force, diagonal,
                                 weights, inertia_g, inertia_h, legacy_g, legacy_h], device=device)
                            self.assertEqual(int(active.numpy()[0]), 1)
                            w = weights.numpy()[0].astype(np.float64)
                            jacobians, _ = independent_jacobians(model.ipc.body_references.numpy()[source_nodes],
                                body_solver.previous_pose.numpy()[body, 3:],
                                body_solver.omega.numpy()[body].astype(np.float64), dt, rotational)
                            # Native rows return physical force and the two
                            # weighted diagonal blocks. Recover their common
                            # frozen world block and contract the full body J.
                            pivot = int(np.argmax(np.abs(w)))
                            world_h = diagonal.numpy()[pivot].astype(np.float64) / w[pivot]**2
                            weighted_j = np.einsum("i,ijk->jk", w, jacobians)
                            expected_g = -dt**2 * sum((jacobians[i].T @ force.numpy()[i]
                                for i in range(2)), np.zeros(3))
                            expected_h = dt**2 * weighted_j.T @ world_h @ weighted_j
                            actual_g = wp.empty(1, wp.vec3, device=device)
                            actual_h = wp.empty(1, wp.mat33, device=device)
                            wp.launch(production_selected_body_system, 1,
                                [data, body, rotational, actual_g, actual_h], device=device)
                            np.testing.assert_allclose(actual_g.numpy()[0], inertia_g.numpy()[0] + expected_g,
                                                       rtol=5e-5, atol=3e-7)
                            np.testing.assert_allclose(actual_h.numpy()[0], inertia_h.numpy()[0] + expected_h,
                                                       rtol=5e-5, atol=3e-7)
                            if rotational == 0:
                                own_points.append(np.sum(w[:, None] * q[source_nodes], axis=0))
                            if body == 1 and rotational == 1:
                                distinguishes_canonical = (np.max(np.abs(expected_g - legacy_g.numpy()[0])) > 1e-6 or
                                    np.max(np.abs(expected_h - legacy_h.numpy()[0])) > 1e-6)
                self.assertGreater(np.linalg.norm(own_points[0] - own_points[1]), .3)
                self.assertTrue(distinguishes_canonical, "Fixture must catch use of the opposite edge's frozen frame")
                self.assertEqual(int(solver.error.numpy()[0]), 0)


if __name__ == "__main__":
    unittest.main()
