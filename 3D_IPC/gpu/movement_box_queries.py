"""Outward represented bounds for one immutable GS movement-box epoch.

The radii are binary64 and the checkpoint coordinates are represented
binary32.  Round the arithmetic outward before storing binary32 BVH bounds;
a nearest-rounded ``center +/- float(radius)`` can omit allowed motion.
"""

import math
from collections.abc import Mapping

import warp as wp

from newton._src.geometry.tri_mesh_collision import TriMeshCollisionInfo

from .warp_build import configure_module


@wp.func_native(r"""
    if(amount==0.0)return center;
    const double raw=upper!=0?double(center)+amount:double(center)-amount;
    const double bound=nextafter(raw,upper!=0?double(INFINITY):-double(INFINITY));
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
def outward_endpoint(center: float, amount: wp.float64, upper: int) -> float: ...


@wp.kernel
def build_nodes(centers: wp.array[wp.vec3], radii: wp.array[wp.float64], reach: wp.float64,
                lower: wp.array[wp.vec3], upper: wp.array[wp.vec3],
                query_lower: wp.array[wp.vec3], query_upper: wp.array[wp.vec3], error: wp.array[int]):
    i = wp.tid()
    point = centers[i]
    radius = radii[i]
    valid = wp.isfinite(radius) and radius > wp.float64(1.0e-10)
    valid = valid and wp.isfinite(reach) and reach >= wp.float64(0.0)
    lo = wp.vec3(0.0)
    hi = wp.vec3(0.0)
    qlo = wp.vec3(0.0)
    qhi = wp.vec3(0.0)
    for k in range(3):
        valid = valid and wp.isfinite(point[k])
        lo[k] = outward_endpoint(point[k], radius, 0)
        hi[k] = outward_endpoint(point[k], radius, 1)
        qlo[k] = outward_endpoint(lo[k], reach, 0)
        qhi[k] = outward_endpoint(hi[k], reach, 1)
        valid = valid and wp.isfinite(lo[k]) and wp.isfinite(hi[k])
        valid = valid and wp.isfinite(qlo[k]) and wp.isfinite(qhi[k])
    if valid:
        lower[i] = lo
        upper[i] = hi
        query_lower[i] = qlo
        query_upper[i] = qhi
    else:
        # Do not feed NaNs/infinities into a BVH refit. The latched error
        # disables all candidate queries and subsequent updates.
        lower[i] = wp.vec3(0.0)
        upper[i] = wp.vec3(0.0)
        query_lower[i] = wp.vec3(0.0)
        query_upper[i] = wp.vec3(0.0)
        wp.atomic_max(error, 0, 202)


BoxProof = wp.types.vector(length=5, dtype=wp.float64)  # verdict, witness axis, required gap


@wp.func_native(r"""
    wp::vec_t<5,double> proof;
    for(int k=0;k<5;++k)proof[k]=0.0;
    const auto certify=[&]()->int {
    using namespace ipc_gpu::ccd::detail;
    if(kind!=0&&kind!=1)return -11;
    double lo[4][3],hi[4][3],p[4][3],scale=0.0;
    for(int i=0;i<4;++i) {
        const int node=nodes[i];
        if(node<0||node>=centers.shape[0]||node>=previous.shape[0]
            ||node>=lower.shape[0]||node>=upper.shape[0])return -11;
        for(int k=0;k<3;++k) {
            lo[i][k]=double(lower.data[node][k]);
            hi[i][k]=double(upper.data[node][k]);
            p[i][k]=double(centers.data[node][k]);
            if(!finite(lo[i][k])||!finite(hi[i][k])||!finite(p[i][k])
                ||lo[i][k]>hi[i][k]||p[i][k]<lo[i][k]||p[i][k]>hi[i][k])return -202;
            if(check_history!=0&&!finite(double(previous.data[node][k])))return -101;
            scale=maximum(scale,maximum(fabs(lo[i][k]),fabs(hi[i][k])));
        }
    }
    if(!finite(d_hat)||d_hat<0.0)return -101;
    // A contact evaluator transports d_hat through F32 on its compact path.
    // Enclose that represented radius too, even when it rounded upwards.
    union FloatBits { float f; uint32_t u; } radius;
    radius.f=static_cast<float>(d_hat);
    if(double(radius.f)<d_hat)++radius.u;
    const double contact_radius=maximum(d_hat,double(radius.f));
    if(!finite(contact_radius))return 0;
    const double scaled_floor=multiply(point(128.0*DBL_EPSILON),point(scale)).hi;
    const double floor=maximum(1e-8,maximum(1e-10,scaled_floor));
    const double required=add(point(contact_radius),point(floor)).hi;
    if(!finite(required))return 0;
    proof[4]=required;
    const int split=kind==0?1:2;
    // If opposite vertex boxes intersect, their complete primitive regions
    // intersect too: no supporting-plane proof can possibly succeed.
    for(int a=0;a<split;++a)for(int b=split;b<4;++b) {
        bool overlap=true;
        for(int k=0;k<3;++k)overlap=overlap&&lo[a][k]<=hi[b][k]&&lo[b][k]<=hi[a][k];
        if(overlap)return 0;
    }
    // Coordinate axes are exact and need no normal or general projection.
    for(int k=0;k<3;++k) {
        double alo=INFINITY,ahi=-INFINITY,blo=INFINITY,bhi=-INFINITY;
        for(int i=0;i<4;++i) {
            if(i<split){alo=minimum(alo,lo[i][k]);ahi=maximum(ahi,hi[i][k]);}
            else {blo=minimum(blo,lo[i][k]);bhi=maximum(bhi,hi[i][k]);}
        }
        if(subtract(point(alo),point(bhi)).lo>required
            ||subtract(point(blo),point(ahi)).lo>required) {
            proof[1+k]=1.0;return 1;
        }
    }
    const auto normalized=[](Vec axis) {
        double magnitude=0.0;
        for(int k=0;k<3;++k) {
            if(!finite(axis.v[k]))return Vec{{0,0,0}};
            magnitude=maximum(magnitude,fabs(axis.v[k]));
        }
        if(!(magnitude>0.0))return Vec{{0,0,0}};
        for(int k=0;k<3;++k)axis.v[k]/=magnitude;
        return axis;
    };
    const auto separated=[&](Vec proposed) {
        const Vec axis=normalized(proposed);
        if(axis.v[0]==0.0&&axis.v[1]==0.0&&axis.v[2]==0.0)return false;
        Interval norm2=point(0.0);
        for(int k=0;k<3;++k)norm2=add(norm2,multiply(point(axis.v[k]),point(axis.v[k])));
        const double threshold=multiply(point(required),point(sqrt_upper(norm2.hi))).hi;
        if(!finite(threshold))return false;
        double alo=INFINITY,ahi=-INFINITY,blo=INFINITY,bhi=-INFINITY;
        for(int i=0;i<4;++i) {
            Interval projected=point(0.0);
            for(int k=0;k<3;++k) {
                const Interval coordinate=subtract({lo[i][k],hi[i][k]},point(p[3][k]));
                projected=add(projected,multiply(point(axis.v[k]),coordinate));
            }
            if(!finite(projected.lo)||!finite(projected.hi))return false;
            if(i<split){alo=minimum(alo,projected.lo);ahi=maximum(ahi,projected.hi);}
            else {blo=minimum(blo,projected.lo);bhi=maximum(bhi,projected.hi);}
        }
        const bool safe=subtract(point(alo),point(bhi)).lo>threshold
            ||subtract(point(blo),point(ahi)).lo>threshold;
        if(safe)for(int k=0;k<3;++k)proof[1+k]=axis.v[k];
        return safe;
    };
    if(kind==0) {
        const Vec a=normalized(difference(p[2],p[1]));
        const Vec b=normalized(difference(p[3],p[1]));
        const Vec normal=normalized(cross(a,b));
        if(separated(normal))return 1;
        for(int i=1;i<4;++i) {
            const Vec edge=normalized(difference(p[i==3?1:i+1],p[i]));
            if(separated(cross(normal,edge)))return 1;
        }
    } else {
        const Vec a=normalized(difference(p[1],p[0]));
        const Vec b=normalized(difference(p[3],p[2]));
        const Vec w=normalized(difference(p[2],p[0]));
        if(separated(cross(a,b))||separated(cross(cross(a,w),a))
            ||separated(cross(cross(b,w),b)))return 1;
    }
    for(int i=0;i<split;++i)for(int j=split;j<4;++j)
        if(separated(difference(p[i],p[j])))return 1;
    return 0;
    };
    proof[0]=static_cast<double>(certify());
    return proof;
""")
def _box_certificate(centers: wp.array[wp.vec3], previous: wp.array[wp.vec3],
                      lower: wp.array[wp.vec3], upper: wp.array[wp.vec3], nodes: wp.vec4i,
                      kind: int, d_hat: wp.float64, check_history: int) -> BoxProof: ...


@wp.func
def _box_pair(encoded: int, triangles: wp.array2d[int], edges: wp.array2d[int], info: TriMeshCollisionInfo):
    nodes = wp.vec4i(-1)
    kind = int(0)
    valid = True
    if encoded >= 0:
        if encoded >= info.vertex_colliding_triangles.shape[0] // 2:
            valid = False
        else:
            vertex = info.vertex_colliding_triangles[2 * encoded]
            triangle = info.vertex_colliding_triangles[2 * encoded + 1]
            if triangle < 0 or triangle >= triangles.shape[0]:
                valid = False
            else:
                nodes = wp.vec4i(vertex, triangles[triangle, 0], triangles[triangle, 1], triangles[triangle, 2])
    else:
        kind = 1
        slot = -(encoded + 1)
        if slot >= info.edge_colliding_edges.shape[0] // 2:
            valid = False
        else:
            first = info.edge_colliding_edges[2 * slot]
            second = info.edge_colliding_edges[2 * slot + 1]
            if first < 0 or second < 0 or first >= edges.shape[0] or second >= edges.shape[0]:
                valid = False
            else:
                nodes = wp.vec4i(edges[first, 2], edges[first, 3], edges[second, 2], edges[second, 3])
    return nodes, kind, valid


@wp.kernel
def _filter_box_pairs(centers: wp.array[wp.vec3], previous: wp.array[wp.vec3],
                      lower: wp.array[wp.vec3], upper: wp.array[wp.vec3],
                      triangles: wp.array2d[int], edges: wp.array2d[int], info: TriMeshCollisionInfo,
                      flags: wp.array[int], mass: wp.array[float], d_hat: wp.float64, validate_history: int,
                      source: wp.array[int], count: wp.array[int], workers: int,
                      epoch: wp.array[wp.uint64], current_epoch: wp.array[wp.uint64],
                      target: wp.array[int], retained: wp.array[int], error: wp.array[int]):
    if epoch[0] != current_epoch[0]:
        wp.atomic_max(error, 0, 11)
    if count[0] < 0 or count[0] > source.shape[0]:
        wp.atomic_max(error, 0, 12)
    if error[0] == 0:
        for at in range(wp.tid(), count[0], workers):
            encoded = source[at]
            nodes, kind, valid = _box_pair(encoded, triangles, edges, info)
            dynamic = False
            for role in range(4):
                node = nodes[role]
                if node < 0 or node >= centers.shape[0]:
                    valid = False
                else:
                    dynamic = dynamic or (flags[node] & 1 != 0 and mass[node] > 0.0)
            verdict = int(-11)
            if valid:
                proof = _box_certificate(centers, previous, lower, upper, nodes, kind, d_hat,
                                         int(dynamic and validate_history != 0))
                verdict = int(proof[0])
            if verdict < 0:
                wp.atomic_max(error, 0, -verdict)
            elif verdict == 0:
                slot = wp.atomic_add(retained, 0, 1)
                if slot < target.shape[0]:
                    target[slot] = encoded
                else:
                    wp.atomic_max(error, 0, 12)


@wp.kernel
def _publish_box_pairs(source: wp.array[int], retained: wp.array[int], target: wp.array[int],
                       count: wp.array[int], workers: int, epoch: wp.array[wp.uint64],
                       current_epoch: wp.array[wp.uint64], error: wp.array[int]):
    worker = wp.tid()
    if epoch[0] != current_epoch[0]:
        wp.atomic_max(error, 0, 11)
    if retained[0] < 0 or retained[0] > target.shape[0]:
        wp.atomic_max(error, 0, 12)
    if error[0] == 0:
        for at in range(worker, retained[0], workers):
            target[at] = source[at]
        if worker == 0:
            count[0] = retained[0]


class GsBoxEpochFilter:
    """Remove rigorously inactive GS interactions before dependency coloring.

    Call once after raw box detection/refresh and before rebuilding grouped
    queries. Closed blue boxes must enclose every subsequent represented GS
    iterate until the next detection checkpoint; bounded linear GS acceptance
    enforces this by convexity. Predictors and rigid trajectories are excluded.
    The filter's geometric zero is authoritative; floating-point packet/order
    bit identity with the unfiltered execution is not promised.
    """

    def __init__(self, contacts):
        if getattr(contacts.model, "body_count", 0):
            raise ValueError("GS box filtering supports bounded deformable trajectories only")
        if not getattr(contacts, "track_query_epoch", False):
            raise ValueError("GS box filtering requires tracked candidate epochs")
        self.contacts, self.device = contacts, contacts.device
        self.scratch = wp.empty(contacts.capacity, int, device=self.device)
        self.raw_count = wp.zeros(1, int, device=self.device)
        self.retained_count = wp.zeros(1, int, device=self.device)
        self.epoch = wp.zeros(1, wp.uint64, device=self.device)
        self.source_generation = None

    def apply(self, centers, previous, lower, upper, params):
        c, n = self.contacts, self.contacts.model.particle_count
        for name, values in (("centers", centers), ("previous", previous), ("lower", lower), ("upper", upper)):
            if values.dtype != wp.vec3 or values.shape != (n,) or values.device != self.device:
                raise TypeError(f"GS box filter {name} must match the device particle layout")
        # Import lazily to keep movement-bound construction independent of
        # contact.py, which imports this module during its own initialization.
        from .contact import parameters
        if isinstance(params, Mapping):
            p = parameters(d_hat=params["d_hat"], k_barrier=params["k_barrier"], dt=params["dt"],
                friction=params.get("friction", params.get("friction_coefficient", 0.0)),
                eps_v=params.get("eps_v", params.get("friction_velocity_epsilon", .001)),
                exact_geometry=params.get("exact_geometry", params.get("exact_computation_fallback", True)))
        else:
            if params.exact_geometry not in (0, 1):
                raise ValueError("Invalid GS box-filter contact geometry mode")
            p = parameters(d_hat=params.d_hat, k_barrier=params.k_barrier, dt=params.dt,
                friction=params.friction, eps_v=params.eps_v, exact_geometry=bool(params.exact_geometry))
            if not math.isfinite(params.eps) or params.eps <= 0:
                raise ValueError("Invalid GS box-filter contact epsilon")
        if self.source_generation == c.query_generation:
            raise ValueError("Refresh raw candidates before filtering another GS box epoch")
        self.source_generation = c.query_generation
        wp.copy(self.raw_count, c.candidate_count)
        wp.copy(self.epoch, c.query_epoch)
        self.retained_count.zero_()
        if c.capacity:
            wp.launch(_filter_box_pairs, c.fast_workers,
                [centers, previous, lower, upper, c.model.tri_indices, c.model.edge_indices, c.info,
                 c.model.particle_flags, c.model.particle_mass, p.d_hat, int(p.d_hat > 0 and p.k_barrier > 0),
                 c.candidate_ids, self.raw_count, c.fast_workers, self.epoch, c.query_epoch,
                 self.scratch, self.retained_count, c.error], device=self.device, block_dim=128)
            wp.launch(_publish_box_pairs, c.fast_workers,
                [self.scratch, self.retained_count, c.candidate_ids, c.candidate_count,
                 c.fast_workers, self.epoch, c.query_epoch, c.error], device=self.device)


configure_module(__name__, ("ccd.cuh",))
