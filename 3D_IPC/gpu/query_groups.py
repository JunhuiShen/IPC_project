"""Complete candidate spans for one immutable predictor group-map epoch.

Each candidate is copied once to every distinct nonnegative corner group. Therefore
one group owns at most the original candidate count, and all groups together
need at most four copies per candidate. Values retain Newton's original
encoded candidate IDs; the existing CCD exact resolver can decode them without
any numerical or kernel change. No force, geometry or accepted fraction is
cached here.
"""

from dataclasses import dataclass

import warp as wp

from newton._src.geometry.tri_mesh_collision import TriMeshCollisionInfo

from .contact import decode_candidate, _same_body
from .ccd import round_inside, pair_certificate, configure_module as _configure_ccd


@wp.kernel
def compact_rows_epoch(info: TriMeshCollisionInfo, triangles: wp.array2d[int], edges: wp.array2d[int],
                  owners: wp.array[int], ids: wp.array[int], count: wp.array[int], error: wp.array[int],
                  query_epoch: wp.array[wp.uint64]):
    tid = wp.tid()
    if tid == 0:
        query_epoch[0] += wp.uint64(1)
    row = tid // 4
    lane = tid % 4
    nv = info.vertex_colliding_triangles_count.shape[0]
    ne = info.edge_colliding_edges_count.shape[0]
    if error[0] == 0:
        if row < nv:
            found = info.vertex_colliding_triangles_count[row]
            capacity = info.vertex_colliding_triangles_buffer_sizes[row]
            if found < 0 or found > capacity:
                wp.atomic_max(error, 0, 10)
            for local in range(lane, wp.min(found, capacity), 4):
                slot = info.vertex_colliding_triangles_offsets[row] + local
                vertex = info.vertex_colliding_triangles[2 * slot]
                tri = info.vertex_colliding_triangles[2 * slot + 1]
                if vertex < 0 or vertex >= owners.shape[0] or tri < 0 or tri >= triangles.shape[0]:
                    wp.atomic_max(error, 0, 11)
                else:
                    nodes = wp.vec4i(vertex, triangles[tri, 0], triangles[tri, 1], triangles[tri, 2])
                    if vertex != nodes[1] and vertex != nodes[2] and vertex != nodes[3] and not _same_body(nodes, owners):
                        at = wp.atomic_add(count, 0, 1)
                        if at < ids.shape[0]:
                            ids[at] = slot
                        else:
                            wp.atomic_max(error, 0, 12)
        elif row - nv < ne:
            edge_row = row - nv
            found = info.edge_colliding_edges_count[edge_row]
            capacity = info.edge_colliding_edges_buffer_sizes[edge_row]
            if found < 0 or found > capacity:
                wp.atomic_max(error, 0, 10)
            for local in range(lane, wp.min(found, capacity), 4):
                slot = info.edge_colliding_edges_offsets[edge_row] + local
                first = info.edge_colliding_edges[2 * slot]
                second = info.edge_colliding_edges[2 * slot + 1]
                if first < 0 or first >= edges.shape[0] or second < 0 or second >= edges.shape[0]:
                    wp.atomic_max(error, 0, 11)
                elif first < second:
                    nodes = wp.vec4i(edges[first, 2], edges[first, 3], edges[second, 2], edges[second, 3])
                    shared = nodes[0] == nodes[2] or nodes[0] == nodes[3] or nodes[1] == nodes[2] or nodes[1] == nodes[3]
                    if not shared and not _same_body(nodes, owners):
                        at = wp.atomic_add(count, 0, 1)
                        if at < ids.shape[0]:
                            ids[at] = -slot - 1
                        else:
                            wp.atomic_max(error, 0, 12)



@wp.kernel
def prepare_grouped(q: wp.array[wp.vec3], proposed: wp.array[wp.vec3], center: wp.array[wp.vec3],
              radius: wp.float64, radii: wp.array[wp.float64], use_radii: int,
              bounded: int, colors: wp.array[int], flags: wp.array[int], mass: wp.array[float],
              color: int, view_groups: wp.array[int], current_group: wp.array[int], use_view: int, select_group: int,
              view_epoch: wp.array[wp.uint64], current_epoch: wp.array[wp.uint64],
              candidate: wp.array[wp.vec3], error: wp.array[int]):
    i = wp.tid()
    if use_view != 0 and i == 0:
        if view_epoch[0] != current_epoch[0]:
            wp.atomic_max(error, 0, 11)
    if error[0] == 0:
        point = q[i]
        selected = color < 0 or colors[i] == color
        if select_group != 0:
            selected = selected and view_groups[i] == current_group[0]
        if flags[i] & 1 != 0 and mass[i] > 0.0 and selected:
            point = proposed[i]
        changed = point[0] != q[i][0] or point[1] != q[i][1] or point[2] != q[i][2]
        if bounded != 0 and changed:
            budget = radius
            if use_radii != 0:
                budget = radii[i]
            point = round_inside(q[i], point, center[i], budget)
        for k in range(3):
            if not wp.isfinite(q[i][k]) or not wp.isfinite(point[k]):
                wp.atomic_max(error, 0, 202)
        if use_view != 0:
            actual_change = point[0] != q[i][0] or point[1] != q[i][1] or point[2] != q[i][2]
            if actual_change and view_groups[i] != current_group[0]:
                wp.atomic_max(error, 0, 202)
        candidate[i] = point



@wp.kernel
def certify_grouped(q: wp.array[wp.vec3], candidate: wp.array[wp.vec3], ids: wp.array[int], count: wp.array[int],
              triangles: wp.array2d[int], edges: wp.array2d[int], info: TriMeshCollisionInfo,
              offsets: wp.array[int], current_group: wp.array[int], use_view: int,
              exact_geometry: int, workers: int, pending: wp.array[int], pending_count: wp.array[int], error: wp.array[int]):
    worker = wp.tid()
    if error[0] == 0:
        begin = int(0)
        end = count[0]
        valid = bool(True)
        if use_view != 0:
            group = current_group[0]
            if group < 0 or group + 1 >= offsets.shape[0]:
                wp.atomic_max(error, 0, 11)
                valid = False
            else:
                begin = offsets[group]
                end = offsets[group + 1]
                if begin < 0 or end < begin or end > offsets[offsets.shape[0] - 1] or end > ids.shape[0]:
                    wp.atomic_max(error, 0, 12)
                    valid = False
        if valid:
            for at in range(begin + worker, end, workers):
                nodes, kind = decode_candidate(ids[at], triangles, edges, info)
                result = pair_certificate(q, candidate, nodes, kind, exact_geometry)
                if int(result[1]) == 5:
                    slot = wp.atomic_add(pending_count, 0, 1)
                    if slot < pending.shape[0]:
                        pending[slot] = at
                    else:
                        wp.atomic_max(error, 0, 12)
                elif int(result[1]) != 0:
                    wp.atomic_max(error, 0, 200 + int(result[1]))


@wp.func
def _unique_groups(nodes: wp.vec4i, groups: wp.array[int], error: wp.array[int]):
    keys = wp.vec4i(-1)
    for role in range(4):
        node = nodes[role]
        if node < 0 or node >= groups.shape[0]:
            wp.atomic_max(error, 0, 11)
        else:
            group = groups[node]
            if group < -1 or group >= groups.shape[0]:
                wp.atomic_max(error, 0, 11)
            elif group >= 0:
                duplicate = False
                for prior in range(role):
                    duplicate = duplicate or keys[prior] == group
                if not duplicate:
                    keys[role] = group
    return keys


@wp.kernel
def _count(ids: wp.array[int], count: wp.array[int], triangles: wp.array2d[int],
            edges: wp.array2d[int], info: TriMeshCollisionInfo, groups: wp.array[int],
            workers: int,
            counts: wp.array[int], error: wp.array[int]):
    worker = wp.tid()
    if error[0] == 0:
        if count[0] < 0 or count[0] > ids.shape[0]:
            wp.atomic_max(error, 0, 12)
        else:
            for index in range(worker, count[0], workers):
                nodes, kind = decode_candidate(ids[index], triangles, edges, info)
                keys = _unique_groups(nodes, groups, error)
                for role in range(4):
                    if keys[role] >= 0:
                        wp.atomic_add(counts, keys[role], 1)


@wp.kernel
def _fill(ids: wp.array[int], count: wp.array[int], triangles: wp.array2d[int],
           edges: wp.array2d[int], info: TriMeshCollisionInfo, groups: wp.array[int],
           workers: int,
           offsets: wp.array[int], cursors: wp.array[int], grouped: wp.array[int], error: wp.array[int]):
    worker = wp.tid()
    if error[0] == 0:
        for index in range(worker, count[0], workers):
            encoded = ids[index]
            nodes, kind = decode_candidate(encoded, triangles, edges, info)
            keys = _unique_groups(nodes, groups, error)
            for role in range(4):
                group = keys[role]
                if group >= 0:
                    slot = offsets[group] + wp.atomic_add(cursors, group, 1)
                    if slot < offsets[group] or slot >= offsets[group + 1] or slot >= grouped.shape[0]:
                        wp.atomic_max(error, 0, 12)
                    else:
                        grouped[slot] = encoded


@dataclass(frozen=True)
class QueryView:
    """Read-only host description of the latest bank and a GPU group counter."""

    owner: "QueryGroups"
    current_group: wp.array
    version: int
    source_generation: int | None

    @property
    def ids(self):
        return self.owner.ids

    @property
    def offsets(self):
        return self.owner.offsets

    @property
    def node_groups(self):
        return self.owner.node_groups

    @property
    def epoch(self):
        return self.owner.epoch

    def validate(self, contacts):
        if contacts is not self.owner.contacts or self.version != self.owner.version:
            raise ValueError("The grouped CCD view belongs to another candidate/group epoch")
        if self.source_generation != getattr(contacts, "query_generation", None):
            raise ValueError("Rebuild grouped CCD queries after collision detection")
        if self.current_group.dtype != wp.int32 or self.current_group.shape != (1,):
            raise TypeError("The current query group must be a one-element device int array")
        if self.current_group.device != self.owner.device:
            raise ValueError("The current query group is on another device")


class QueryGroups:
    def __init__(self, contacts):
        self.contacts, self.device = contacts, contacts.device
        self.n = contacts.model.particle_count
        # Exact-resolve pending indices remain int32 and index this bank.
        # Very large storage configurations retain the unchanged all-pairs
        # oracle rather than introducing a new index-limit solver failure.
        self.enabled = bool(getattr(contacts, "track_query_epoch", False)) and 4 * contacts.capacity < 2**31
        self.ids = wp.empty(4 * contacts.capacity if self.enabled else 0, int, device=self.device)
        self.counts = wp.zeros(self.n + 1, int, device=self.device)
        self.offsets = wp.zeros(self.n + 1, int, device=self.device)
        self.cursors = wp.zeros(self.n + 1, int, device=self.device)
        self.node_groups = wp.empty(self.n, int, device=self.device)
        self.epoch = wp.zeros(1, wp.uint64, device=self.device)
        self.version = 0
        self.source_generation = None

    def build(self, groups):
        if groups.dtype != wp.int32 or groups.shape != (self.n,) or groups.device != self.device:
            raise TypeError("Candidate query groups must match the model's device int layout")
        self.version += 1
        self.source_generation = getattr(self.contacts, "query_generation", None)
        if not self.enabled:
            return
        c, model = self.contacts, self.contacts.model
        wp.copy(self.epoch, c.query_epoch)
        wp.copy(self.node_groups, groups)
        self.counts.zero_()
        self.cursors.zero_()
        if c.capacity:
            arguments = [c.candidate_ids, c.candidate_count, model.tri_indices, model.edge_indices,
                         c.info, self.node_groups, c.fast_workers]
            wp.launch(_count, c.fast_workers, [*arguments, self.counts, c.error], device=self.device)
        wp.utils.array_scan(self.counts, self.offsets, inclusive=False)
        if c.capacity:
            wp.launch(_fill, c.fast_workers,
                [*arguments, self.offsets, self.cursors, self.ids, c.error], device=self.device)

    def view(self, current_group):
        if not self.enabled:
            return None
        if self.version == 0:
            raise ValueError("Build candidate query groups before creating a view")
        result = QueryView(self, current_group, self.version, self.source_generation)
        result.validate(self.contacts)
        return result


_configure_ccd(__name__, include_filter=True)
wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})
