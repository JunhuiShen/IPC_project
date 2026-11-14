"""Small device kernels for rigid contact incidence, coloring and state IO.

These kernels are separate from the block-size-one exact rigid phase so
ordinary parallel preprocessing does not compile the cold phase again.
"""

import warp as wp

from newton._src.geometry.tri_mesh_collision import TriMeshCollisionInfo

from .contact import decode_candidate


@wp.kernel
def begin(poses: wp.array[wp.transform], velocity: wp.array[wp.spatial_vector],
           gravity: wp.array[wp.vec3], body_world: wp.array[int], dt: float,
           previous_pose: wp.array[wp.transform], previous_velocity: wp.array[wp.spatial_vector],
           omega: wp.array[wp.vec3]):
    b = wp.tid()
    previous_pose[b] = poses[b]
    old = velocity[b]
    # Linear components encode the inertial velocity used by the existing
    # translation system: v_hat = v_old + dt*g. Angular components remain the
    # frozen physical step-start omega used by rotational inertia.
    linear = wp.vec3(old[0], old[1], old[2]) + dt * gravity[body_world[b]]
    previous_velocity[b] = wp.spatial_vector(linear[0], linear[1], linear[2], old[3], old[4], old[5])
    # The CPU rotational unknown is the step's angular increment / dt, and
    # starts at zero. Previous physical angular velocity enters inertia only.
    omega[b] = wp.vec3(0.0)


@wp.func
def primitive_owners(nodes: wp.vec4i, kind: int, owners: wp.array[int]):
    first = owners[nodes[0]]
    second = owners[nodes[1]]
    valid = True
    if kind == 0:
        valid = second == owners[nodes[2]] and second == owners[nodes[3]]
    else:
        second = owners[nodes[2]]
        valid = first == owners[nodes[1]] and second == owners[nodes[3]]
    return first, second, valid


@wp.kernel
def count_incidence(ids: wp.array[int], count: wp.array[int], triangles: wp.array2d[int],
                      edges: wp.array2d[int], info: TriMeshCollisionInfo, owners: wp.array[int],
                      counts: wp.array[int], conflicts: wp.array2d[wp.uint32], error: wp.array[int]):
    index = wp.tid()
    if index < count[0] and error[0] == 0:
        nodes, kind = decode_candidate(ids[index], triangles, edges, info)
        first, second, valid = primitive_owners(nodes, kind, owners)
        if not valid:
            wp.atomic_max(error, 0, 602)
        else:
            if first >= 0:
                wp.atomic_add(counts, first, 1)
            if second >= 0 and second != first:
                wp.atomic_add(counts, second, 1)
            if first >= 0 and second >= 0 and first != second:
                wp.atomic_or(conflicts, first, second // 32, wp.uint32(1) << wp.uint32(second % 32))
                wp.atomic_or(conflicts, second, first // 32, wp.uint32(1) << wp.uint32(first % 32))


@wp.kernel
def prefix_incidence(counts: wp.array[int], offsets: wp.array[int]):
    total = int(0)
    for body in range(counts.shape[0]):
        offsets[body] = total
        total += counts[body]


@wp.kernel
def fill_incidence(ids: wp.array[int], count: wp.array[int], triangles: wp.array2d[int],
                     edges: wp.array2d[int], info: TriMeshCollisionInfo, owners: wp.array[int],
                     offsets: wp.array[int], cursors: wp.array[int], incidence: wp.array[int], error: wp.array[int]):
    index = wp.tid()
    if index < count[0] and error[0] == 0:
        nodes, kind = decode_candidate(ids[index], triangles, edges, info)
        first, second, valid = primitive_owners(nodes, kind, owners)
        if valid:
            if first >= 0:
                slot = offsets[first] + wp.atomic_add(cursors, first, 1)
                incidence[slot] = index
            if second >= 0 and second != first:
                slot = offsets[second] + wp.atomic_add(cursors, second, 1)
                incidence[slot] = index


@wp.kernel
def color_bodies(conflicts: wp.array2d[wp.uint32], colors: wp.array[int], marks: wp.array[int],
                    count: wp.array[int], error: wp.array[int]):
    if error[0] == 0:
        maximum = int(0)
        for b in range(colors.shape[0]):
            for a in range(b):
                adjacent = (conflicts[b, a // 32] & (wp.uint32(1) << wp.uint32(a % 32))) != wp.uint32(0)
                if adjacent:
                    marks[colors[a]] = b + 1
            color = int(0)
            while marks[color] == b + 1:
                color += 1
            colors[b] = color
            maximum = wp.max(maximum, color + 1)
        count[0] = maximum


@wp.kernel
def start_colors(color: wp.array[int], count: wp.array[int], condition: wp.array[int], error: wp.array[int]):
    color[0] = 0
    condition[0] = int(count[0] > 0 and error[0] == 0)


@wp.kernel
def next_color(color: wp.array[int], count: wp.array[int], condition: wp.array[int], error: wp.array[int]):
    color[0] += 1
    condition[0] = int(color[0] < count[0] and error[0] == 0)


@wp.kernel
def finish(poses: wp.array[wp.transform], previous: wp.array[wp.transform], omega: wp.array[wp.vec3],
            translate: wp.array[int], rotate: wp.array[int], target_pose: wp.array[wp.transform],
            target_velocity: wp.array[wp.spatial_vector], dt: float, error: wp.array[int]):
    b = wp.tid()
    if error[0] == 0:
        velocity = wp.vec3(0.0)
        angular = wp.vec3(0.0)
        if translate[b] != 0:
            velocity = (wp.transform_get_translation(poses[b]) - wp.transform_get_translation(previous[b])) / dt
        if rotate[b] != 0:
            angular = omega[b]
        target_pose[b] = poses[b]
        target_velocity[b] = wp.spatial_vector(velocity[0], velocity[1], velocity[2], angular[0], angular[1], angular[2])


wp.set_module_options({"enable_backward": False})
