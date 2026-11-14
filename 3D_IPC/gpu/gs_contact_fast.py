"""Compact float32 contact energies for CPU-style GPU GS color spans.

Reuse DAT's complete geometric diagonal Hessians and frozen PSD friction
law. Float32 uncertainty is replayed with the existing ordinary binary64
evaluator; only its robust geometry requests enter the exact worker queue.
The shared contact backend and fixed-color GS evaluator remain unchanged.
"""

from functools import cache

import warp as wp

from newton._src.geometry.tri_mesh_collision import TriMeshCollisionInfo

from . import contact, dat_contact_fast


@wp.func
def _scatter_role(packet: contact.Result, nodes: wp.vec4i, colors: wp.array[int],
                   flags: wp.array[int], mass: wp.array[float], color: int, role: int,
                   forces: wp.array[wp.vec3], hessians: wp.array[wp.mat33]):
    vertex = nodes[role]
    if colors[vertex] == color and flags[vertex] & 1 != 0 and mass[vertex] > 0.0:
        force = wp.vec3(packet[12 * role], packet[12 * role + 1], packet[12 * role + 2])
        hessian = wp.mat33(packet[12 * role + 3], packet[12 * role + 4], packet[12 * role + 5],
            packet[12 * role + 6], packet[12 * role + 7], packet[12 * role + 8],
            packet[12 * role + 9], packet[12 * role + 10], packet[12 * role + 11])
        wp.atomic_add(forces, vertex, force)
        wp.atomic_add(hessians, vertex, hessian)


@wp.func
def _scatter(packet: contact.Result, nodes: wp.vec4i, colors: wp.array[int],
              flags: wp.array[int], mass: wp.array[float], color: int,
              forces: wp.array[wp.vec3], hessians: wp.array[wp.mat33]):
    # Literal role arguments avoid dynamically indexed packet copies, as in
    # DAT's compact result extraction. Warp 1.19 cannot static-unroll range.
    _scatter_role(packet, nodes, colors, flags, mass, color, 0, forces, hessians)
    _scatter_role(packet, nodes, colors, flags, mass, color, 1, forces, hessians)
    _scatter_role(packet, nodes, colors, flags, mass, color, 2, forces, hessians)
    _scatter_role(packet, nodes, colors, flags, mass, color, 3, forces, hessians)


@cache
def make_fast():
    @wp.kernel(module="unique", enable_backward=False)
    def evaluate(positions: wp.array[wp.vec3], previous: wp.array[wp.vec3],
                  triangles: wp.array2d[int], edges: wp.array2d[int], info: TriMeshCollisionInfo,
                  ids: wp.array[int], count: wp.array[int], colors: wp.array[int], flags: wp.array[int],
                  mass: wp.array[float], current_color: wp.array[int], offsets: wp.array[int], use_view: int,
                  view_epoch: wp.array[wp.uint64], query_epoch: wp.array[wp.uint64],
                  params: contact.Parameters, workers: int, ordinary_pending: wp.array[int],
                  ordinary_count: wp.array[int], forces: wp.array[wp.vec3],
                  hessians: wp.array[wp.mat33], error: wp.array[int]):
        worker = wp.tid()
        color = current_color[0]
        valid = color >= 0 and color < colors.shape[0]
        begin = int(0)
        end = count[0]
        if use_view != 0 and valid:
            valid = view_epoch[0] == query_epoch[0]
            begin = offsets[color]
            end = offsets[color + 1]
        valid = valid and begin >= 0 and end >= begin and end <= ids.shape[0]
        if not valid:
            wp.atomic_max(error, 0, 11)
        if error[0] == 0 and valid:
            for at in range(begin + worker, end, workers):
                nodes, kind = contact.decode_candidate(ids[at], triangles, edges, info)
                touches = False
                for role in range(4):
                    node = nodes[role]
                    if colors[node] == color and flags[node] & 1 != 0 and mass[node] > 0.0:
                        touches = True
                if touches:
                    packet = dat_contact_fast.evaluate_self(positions, previous, nodes, kind, params)
                    status = int(packet[48])
                    if status == 6:
                        slot = wp.atomic_add(ordinary_count, 0, 1)
                        if slot < ordinary_pending.shape[0]:
                            ordinary_pending[slot] = at
                        else:
                            wp.atomic_max(error, 0, 12)
                    elif status != 0:
                        wp.atomic_max(error, 0, 100 + status)
                    elif packet[49] != 0.0:
                        _scatter(packet, nodes, colors, flags, mass, color, forces, hessians)

    dat_contact_fast.configure_module(evaluate)
    return evaluate


@cache
def make_ordinary(distribute_blocks=False):
    """Keep sparse queues spread across 64-thread CUDA blocks.

    The distributed variant requires a 64-thread block and a worker count
    divisible by 64. This permutation changes ownership only; grid-stride
    iteration, per-pair arithmetic and exact replay routing stay identical.
    """
    @wp.kernel(module="unique", enable_backward=False)
    def resolve(positions: wp.array[wp.vec3], previous: wp.array[wp.vec3],
                 triangles: wp.array2d[int], edges: wp.array2d[int], info: TriMeshCollisionInfo,
                 ids: wp.array[int], colors: wp.array[int], flags: wp.array[int], mass: wp.array[float],
                 current_color: wp.array[int], params: contact.Parameters, workers: int,
                 ordinary_pending: wp.array[int], ordinary_count: wp.array[int],
                 exact_pending: wp.array[int], exact_count: wp.array[int],
                 forces: wp.array[wp.vec3], hessians: wp.array[wp.mat33], error: wp.array[int]):
        worker = wp.tid()
        if wp.static(distribute_blocks):
            worker = (wp.tid() % 64) * (workers // 64) + wp.tid() // 64
        if error[0] == 0:
            color = current_color[0]
            for index in range(worker, wp.min(ordinary_count[0], ordinary_pending.shape[0]), workers):
                at = ordinary_pending[index]
                if at < 0 or at >= ids.shape[0]:
                    wp.atomic_max(error, 0, 11)
                else:
                    nodes, kind = contact.decode_candidate(ids[at], triangles, edges, info)
                    packet = contact.evaluate_self(positions, previous, nodes, kind, params)
                    status = int(packet[48])
                    if status == 2 and params.exact_geometry != 0:
                        slot = wp.atomic_add(exact_count, 0, 1)
                        if slot < exact_pending.shape[0]:
                            exact_pending[slot] = at
                        else:
                            wp.atomic_max(error, 0, 12)
                    elif status != 0:
                        wp.atomic_max(error, 0, 100 + status)
                    elif packet[49] != 0.0:
                        _scatter(packet, nodes, colors, flags, mass, color, forces, hessians)

    contact.configure_module(resolve)
    return resolve


class GsContactFast:
    """Persistent float32 → ordinary → exact replay queues for grouped GS."""

    def __init__(self, contacts):
        self.contacts = contacts
        self.device = contacts.device
        self.ordinary_pending = wp.empty(contacts.capacity, int, device=self.device)
        self.ordinary_count = wp.zeros(1, int, device=self.device)
        self.ordinary_workers = max(1, self.device.sm_count * 64) if self.device.is_cuda else 1
        self.ordinary_block_dim = 64 if self.device.is_cuda else 128
        self._fast = make_fast()
        self._ordinary = make_ordinary(distribute_blocks=self.device.is_cuda)

    def accumulate_grouped(self, positions, previous, color, forces, hessians, dt, *,
                           colors, current_color, query_view=None):
        c, model = self.contacts, self.contacts.model
        n = model.particle_count
        if colors is None or colors.dtype != wp.int32 or colors.shape != (n,) or colors.device != self.device:
            raise TypeError("Dynamic contact colors must match the model device int array")
        if (current_color is None or current_color.dtype != wp.int32 or current_color.shape != (1,)
                or current_color.device != self.device):
            raise TypeError("Dynamic contact selection requires one device int")
        if not c.track_query_epoch:
            raise ValueError("Dynamic GS contact assembly requires tracked query epochs")
        ids, offsets, epoch = c.candidate_ids, c.candidate_count, c.query_epoch
        selected_colors = colors
        if query_view is not None:
            query_view.validate(c)
            if query_view.current_group is not current_color:
                raise ValueError("Contact query span and current_color must share their device selector")
            ids, offsets, epoch = query_view.ids, query_view.offsets, query_view.epoch
            selected_colors = query_view.node_groups
        params = contact.parameters(dt=float(dt), **c.options)
        self.ordinary_count.zero_()
        c.pending_count.zero_()
        if c.capacity:
            wp.launch(self._fast, c.fast_workers,
                [positions, previous, model.tri_indices, model.edge_indices, c.info,
                 ids, c.candidate_count, selected_colors, model.particle_flags, model.particle_mass,
                 current_color, offsets, int(query_view is not None), epoch, c.query_epoch,
                 params, c.fast_workers, self.ordinary_pending, self.ordinary_count,
                 forces, hessians, c.error], device=self.device, block_dim=128)
            wp.launch(self._ordinary, self.ordinary_workers,
                [positions, previous, model.tri_indices, model.edge_indices, c.info,
                 ids, selected_colors, model.particle_flags, model.particle_mass, current_color,
                 params, self.ordinary_workers, self.ordinary_pending, self.ordinary_count,
                 c.pending, c.pending_count, forces, hessians, c.error], device=self.device, block_dim=self.ordinary_block_dim)
            wp.launch(c._resolve_grouped, c.workers,
                [positions, previous, model.tri_indices, model.edge_indices, c.info,
                 ids, selected_colors, model.particle_flags, model.particle_mass,
                 current_color, params, c.words, c.slots, c.workers, c.pending,
                 c.pending_count, forces, hessians, c.error], device=self.device, block_dim=1)
