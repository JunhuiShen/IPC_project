#include "DAT.h"

#include "broad_phase.h"
#include "segment_segment_distance.h"
#include "rigid_dat_trajectory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>

static void dat_atomic_min(std::atomic<double>& destination, double value)
{
    double previous = destination.load(std::memory_order_relaxed);
    while (value < previous && !destination.compare_exchange_weak(
        previous, value, std::memory_order_relaxed)) {}
}

template<class Function>
static void dat_for_pairs(std::size_t count, const Function& function)
{
    std::exception_ptr error;
    #pragma omp parallel for schedule(static) if(count >= 128)
    for (std::size_t pair = 0; pair < count; ++pair) {
        try {
            function(pair);
        } catch (...) {
            // Exceptions must not escape an OpenMP worker region.
            #pragma omp critical(dat_pair_error)
            {
                if (!error) error = std::current_exception();
            }
        }
    }
    if (error) std::rethrow_exception(error);
}

static Vec3 dat_capped_update(const Vec3& update, double limit)
{
    const double largest = update.cwiseAbs().maxCoeff();
    if (largest == 0) return update;
    // Avoid overflow even when the finite input's norm exceeds DBL_MAX.
    const Vec3 scaled = update / largest;
    const double length = scaled.norm();
    if (largest <= limit / length) return update;
    return scaled * (limit / length);
}

static double dat_norm_upper(const Vec3& component_bounds)
{
    const double scale=component_bounds.maxCoeff();
    if(scale==0)return 0;
    if(!std::isfinite(scale))return INFINITY;
    double squared_sum=0;
    for(int k=0;k<3;++k)if(component_bounds[k]>0) {
        const double ratio=std::nextafter(component_bounds[k]/scale,INFINITY);
        const double square=std::nextafter(ratio*ratio,INFINITY);
        squared_sum=std::nextafter(squared_sum+square,INFINITY);
    }
    const double root=std::nextafter(std::sqrt(squared_sum),INFINITY);
    return std::nextafter(scale*root,INFINITY);
}

static std::array<double,2> dat_plane_projection(const rigid_dat::Box& box,
    const Vec3& normal,const Vec3& point)
{
    double lo=0,hi=0;
    for(int k=0;k<3;++k)if(normal[k]!=0) {
        const double dl=std::nextafter(box.lo[k]-point[k],-INFINITY);
        const double dh=std::nextafter(box.hi[k]-point[k],INFINITY);
        const double pl=normal[k]>=0?normal[k]*dl:normal[k]*dh;
        const double ph=normal[k]>=0?normal[k]*dh:normal[k]*dl;
        lo=std::nextafter(lo+std::nextafter(pl,-INFINITY),-INFINITY);
        hi=std::nextafter(hi+std::nextafter(ph,INFINITY),INFINITY);
    }
    return {lo,hi};
}

// Work relative to the query vertex instead of subtracting rounded world-space
// closest points. For an exterior projection, search all finite triangle edges:
// barycentric sign regions alone need not pick the closest edge of an obtuse tri.
static Vec3 dat_vertex_triangle_separation(
    const Vec3& vertex, const Vec3& a, const Vec3& b, const Vec3& c)
{
    std::array<Vec3, 3> local{a - vertex, b - vertex, c - vertex};
    double scale = 0;
    for (const auto& p : local) if (!p.allFinite())
        throw std::domain_error("planar_DAT: relative geometry overflow");
    for (const auto& p : local) scale = std::max(scale, p.cwiseAbs().maxCoeff());
    if (scale == 0) return Vec3::Zero();
    // Uniform scaling and stableNorm avoid underflow/overflow of face normals.
    for (auto& p : local) p /= scale;
    const Vec3 cross = (local[1] - local[0]).cross(local[2] - local[0]);
    const double normal_length = cross.stableNorm();
    if (normal_length > 0) {
        const Vec3 normal = cross / normal_length;
        const Vec3 projection = local[0].dot(normal) * normal;
        bool interior = true;
        // Oriented edge tests avoid the squared Gram determinant, which can
        // cancel to zero for a thin, but nondegenerate, triangle.
        for (int edge = 0; edge < 3; ++edge) {
            const Vec3 direction = local[(edge + 1) % 3] - local[edge];
            if (direction.cross(projection - local[edge]).dot(normal) < 0)
                interior = false;
        }
        if (interior) return projection * scale;
    }
    Vec3 closest = Vec3::Zero();
    double minimum = std::numeric_limits<double>::infinity();
    for (int edge = 0; edge < 3; ++edge) {
        double parameter;
        const Vec3 candidate = segment_closest_point(
            Vec3::Zero(), local[edge], local[(edge + 1) % 3], parameter);
        const double distance = candidate.stableNorm();
        if (distance < minimum) {
            minimum = distance;
            closest = candidate;
        }
    }
    if (!std::isfinite(minimum))
        throw std::domain_error("planar_DAT: unresolved vertex/triangle distance");
    return closest * scale;
}

// The first side_size entries form primitive A; the remainder form B.
// separation points from A to B. Every corner of both primitives contributes,
// not just the closest points, so the whole deforming primitive stays on its side.
static void dat_constrain_pair(
    const std::array<int, 4>& nodes, int side_size, const Vec3& separation,
    double gamma, double query_radius, const std::vector<Vec3>& positions,
    const std::vector<Vec3>& updates, std::vector<std::atomic<double>>& factors)
{
    const double distance = separation.stableNorm();
    if (!separation.allFinite() || !std::isfinite(distance))
        throw std::domain_error("planar_DAT: unresolved pair distance");
    if (distance >= query_radius) return;
    if (!(distance > 0))
        throw std::domain_error("planar_DAT: zero-gap pair has no separating plane");
    const Vec3 normal = separation / distance;
    const Vec3& origin = positions[nodes[0]];
    std::array<double, 4> coordinate{}, velocity{};
    double left = -std::numeric_limits<double>::infinity();
    double right = std::numeric_limits<double>::infinity();
    double approach_a = 0, approach_b = 0;
    for (int i = 0; i < 4; ++i) {
        coordinate[i] = (positions[nodes[i]] - origin).dot(normal);
        velocity[i] = updates[nodes[i]].dot(normal);
        if (!std::isfinite(coordinate[i]) || !std::isfinite(velocity[i]))
            throw std::domain_error("planar_DAT: plane projection overflow");
        if (i < side_size) {
            left = std::max(left, coordinate[i]);
            approach_a = std::max(approach_a, velocity[i]);
        } else {
            right = std::min(right, coordinate[i]);
            approach_b = std::max(approach_b, -velocity[i]);
        }
    }
    const double gap = right - left;
    if (!(gap > 0) || !std::isfinite(gap))
        throw std::domain_error("planar_DAT: closest-point normal does not separate the primitives");

    // Use the paper's proportional-gap rule (Sec. 3.3), expressed without its
    // inconsistent printed lambda/sign conventions. For A -> B normal:
    // plane = q_A + a/(a+b) * (q_B-q_A). Side A gets a/(a+b) of the gap.
    // When one side is stationary/retreating its share may be zero; that side
    // is not truncated. This avoids freezing motion toward a stationary face.
    const double largest = std::max(approach_a, approach_b);
    const double a = largest > 0 ? approach_a / largest : 1;
    const double b = largest > 0 ? approach_b / largest : 1;
    const double budget_a = gap * (a / (a + b));
    const double budget_b = gap * (b / (a + b));
    for (int i = 0; i < 4; ++i) {
        const bool side_a = i < side_size;
        const double approach = side_a ? velocity[i] : -velocity[i];
        if (approach <= 0) continue; // parallel or moving away from the plane
        const double margin = side_a
            ? (left - coordinate[i]) + budget_a
            : (coordinate[i] - right) + budget_b;
        // Eq. 16: min(1, gamma*t_hit), applying relaxation exactly once.
        // Compare first to avoid overflow for an almost tangential update.
        const double allowed = gamma * margin;
        if (approach > allowed) {
            const double factor = allowed / approach;
            dat_atomic_min(factors[nodes[i]], std::clamp(factor, 0.0, 1.0));
        }
    }
}

std::vector<Vec3> planar_DAT(
    double gamma, double query_radius, const RefMesh& mesh,
    const std::vector<Vec3>& incremental_update,
    const std::vector<Vec3>& current_positions, double node_box_padding)
{
    const std::size_t count = current_positions.size();
    if (!std::isfinite(gamma) || gamma <= 0 || gamma >= 1
        || !std::isfinite(query_radius) || query_radius <= 0
        || !std::isfinite(node_box_padding) || node_box_padding < 1)
        throw std::invalid_argument("planar_DAT: require 0 < gamma < 1, r_q > 0, and padding >= 1");
    if (incremental_update.size() != count
        || count > static_cast<std::size_t>(std::numeric_limits<int>::max())
        || (mesh.num_positions != 0 && mesh.num_positions != count)
        || mesh.tris.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())
        || mesh.tris.size() % 3 != 0)
        throw std::invalid_argument("planar_DAT: inconsistent vertex or triangle counts");
    const auto valid_node = [count](int node) {
        return node >= 0 && static_cast<std::size_t>(node) < count;
    };
    for (std::size_t i = 0; i < mesh.tris.size(); i += 3) {
        const int a = mesh.tris[i], b = mesh.tris[i + 1], c = mesh.tris[i + 2];
        if (!valid_node(a) || !valid_node(b) || !valid_node(c)
            || a == b || b == c || a == c)
            throw std::invalid_argument("planar_DAT: invalid triangle connectivity");
    }
    for (int node : mesh.tet_nodes) if (!valid_node(node))
        throw std::invalid_argument("planar_DAT: invalid tet-node classification");
    for (int node : mesh.surface_nodes) if (!valid_node(node))
        throw std::invalid_argument("planar_DAT: invalid surface-node classification");
    if (!mesh.tet_nodes.empty()) {
        std::vector<unsigned char> interior(count, 0);
        for (int node : mesh.tet_nodes) interior[node] = 1;
        for (int node : mesh.surface_nodes) interior[node] = 0;
        for (int node : mesh.tris) if (interior[node])
            throw std::invalid_argument("planar_DAT: triangle vertex classified as solid interior");
    }
    const double limit = (0.5 * gamma) * query_radius;
    const double radius = query_radius * node_box_padding;
    if (!(limit > 0) || !std::isfinite(radius))
        throw std::invalid_argument("planar_DAT: displacement limit or box radius out of range");

    std::vector<Vec3> updates(count);
    std::vector<AABB> boxes(count);
    std::vector<std::atomic<double>> factors(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (!current_positions[i].allFinite() || !incremental_update[i].allFinite())
            throw std::invalid_argument("planar_DAT: nonfinite position or update");
        updates[i] = dat_capped_update(incremental_update[i], limit);
        boxes[i] = AABB(current_positions[i] - Vec3::Constant(radius),
                        current_positions[i] + Vec3::Constant(radius));
        if (!boxes[i].min.allFinite() || !boxes[i].max.allFinite())
            throw std::invalid_argument("planar_DAT: node box overflow");
        factors[i].store(1.0, std::memory_order_relaxed);
    }
    if (count == 0) return updates;

    // Fresh topology per call. Green triangles and red edges are unions of
    // these blue boxes. Zero additional d_hat padding: r_q already defines
    // the query neighborhood; there is no barrier stiffness/energy here.
    BroadPhase broad_phase;
    if (!mesh.tets.empty() || !mesh.tet_nodes.empty())
        broad_phase.initialize_surface_nodes(boxes, mesh, 0.0,
            BroadPhase::InitializationMode::DeformableSolver);
    else
        broad_phase.initialize(boxes, mesh, 0.0,
            BroadPhase::InitializationMode::DeformableSolver);
    const auto& cache = broad_phase.cache();
    dat_for_pairs(cache.nt_pairs.size(), [&](std::size_t i) {
        const auto& pair = cache.nt_pairs[i];
        const std::array<int, 4> nodes{
            pair.node, pair.tri_v[0], pair.tri_v[1], pair.tri_v[2]};
        const Vec3 separation = dat_vertex_triangle_separation(
            current_positions[nodes[0]], current_positions[nodes[1]],
            current_positions[nodes[2]], current_positions[nodes[3]]);
        dat_constrain_pair(nodes, 1, separation, gamma, query_radius,
            current_positions, updates, factors);
    });
    dat_for_pairs(cache.ss_pairs.size(), [&](std::size_t i) {
        const auto& pair = cache.ss_pairs[i];
        const std::array<int, 4> nodes{pair.v[0], pair.v[1], pair.v[2], pair.v[3]};
        const auto distance = segment_segment_distance(
            current_positions[nodes[0]], current_positions[nodes[1]],
            current_positions[nodes[2]], current_positions[nodes[3]]);
        dat_constrain_pair(nodes, 2, -distance.separation, gamma, query_radius,
            current_positions, updates, factors);
    });
    #pragma omp parallel for schedule(static) if(count >= 128)
    for (std::size_t i = 0; i < count; ++i)
        updates[i] *= factors[i].load(std::memory_order_relaxed);
    return updates;
}

GeneralDATResult general_planar_DAT(double gamma, double query_radius,
    const RefMesh& mesh, const std::vector<Vec3>& particle_updates,
    const std::vector<Vec3>& positions, const std::vector<Vec3>& centers,
    const std::vector<Vec3>& omega, const std::vector<Vec4>& previous_orientations,
    const std::vector<Vec3>& delta_centers, const std::vector<Vec3>& delta_omega,
    double dt, double node_box_padding)
{
    const std::size_t n = positions.size(), bodies = mesh.rb_nodes.size();
    if (bodies == 0) {
        if (!centers.empty() || !omega.empty() || !previous_orientations.empty()
            || !delta_centers.empty() || !delta_omega.empty())
            throw std::invalid_argument("general_DAT: unexpected rigid coordinates");
        GeneralDATResult result;
        const auto displacements=planar_DAT(gamma,query_radius,mesh,particle_updates,
            positions,node_box_padding);
        result.positions=positions;
        for(std::size_t v=0;v<n;++v) {
            result.positions[v]+=displacements[v];
            if(!result.positions[v].allFinite())throw std::domain_error("general_DAT: endpoint overflow");
        }
        return result;
    }
    if (!(gamma > 0 && gamma < 1) || !std::isfinite(gamma)
        || !(query_radius > 0) || !std::isfinite(query_radius)
        || !(node_box_padding >= 1) || !std::isfinite(node_box_padding)
        || !(dt > 0) || !std::isfinite(dt) || !(dt*dt>0) || !std::isfinite(dt*dt))
        throw std::invalid_argument("general_DAT: invalid filter parameters");
    if (particle_updates.size() != n || mesh.node_to_rb.size() != n
        || (mesh.num_positions && mesh.num_positions != n)
        || n > static_cast<std::size_t>(std::numeric_limits<int>::max()) || bodies>n
        || mesh.tris.size()>static_cast<std::size_t>(std::numeric_limits<int>::max())
        || centers.size() != bodies || omega.size() != bodies
        || previous_orientations.size() != bodies || delta_centers.size() != bodies
        || delta_omega.size() != bodies || mesh.ref_positions.size() != bodies
        || mesh.tris.size() % 3 || mesh.tets.size() % 4)
        throw std::invalid_argument("general_DAT: inconsistent mesh/coordinate sizes");
    const auto valid_node = [n](int v) { return v >= 0 && std::size_t(v) < n; };
    std::vector<int> local(n, -1);
    const auto array = [](const Vec3& x) { return rigid_dat::Vector{x[0],x[1],x[2]}; };
    std::vector<rigid_dat::Trajectory> paths(n);
    std::vector<rigid_dat::Box> initial_boxes(n);
    for (std::size_t b = 0; b < bodies; ++b) {
        if (mesh.rb_nodes[b].empty() || mesh.rb_nodes[b].size() != mesh.ref_positions[b].size()
            || !centers[b].allFinite() || !omega[b].allFinite()
            || !delta_centers[b].allFinite() || !delta_omega[b].allFinite()
            || omega[b].cwiseAbs().maxCoeff()>std::sqrt(std::numeric_limits<double>::max())/8
            || delta_omega[b].cwiseAbs().maxCoeff()>std::sqrt(std::numeric_limits<double>::max())/8
            || !previous_orientations[b].allFinite()
            || std::abs(previous_orientations[b].squaredNorm()-1) > 1e-10)
            throw std::invalid_argument("general_DAT: invalid rigid body");
        for (std::size_t j = 0; j < mesh.rb_nodes[b].size(); ++j) {
            const int v = mesh.rb_nodes[b][j];
            if (!valid_node(v) || local[v] >= 0 || mesh.node_to_rb[v] != int(b)
                || !mesh.ref_positions[b][j].allFinite())
                throw std::invalid_argument("general_DAT: inconsistent rigid ownership");
            local[v] = int(j);
            auto& path = paths[v];
            path.center = array(centers[b]); path.delta_center = array(delta_centers[b]);
            path.omega = array(omega[b]); path.delta_omega = array(delta_omega[b]);
            path.reference = array(mesh.ref_positions[b][j]); path.dt = dt;
            for (int k=0;k<4;++k) path.q0[k]=previous_orientations[b][k];
        }
    }
    for (std::size_t v=0;v<n;++v) {
        const int owner=mesh.node_to_rb[v];
        if (owner < -1 || owner >= int(bodies) || ((owner>=0) != (local[v]>=0))
            || !positions[v].allFinite() || !particle_updates[v].allFinite())
            throw std::invalid_argument("general_DAT: invalid node ownership/state");
        if (owner>=0) {
            const Vec3 expected=world_space_position(mesh.ref_positions[owner][local[v]],
                centers[owner],previous_orientations[owner],omega[owner],dt);
            const double tolerance=128*std::numeric_limits<double>::epsilon()
                *std::max({1.0,expected.cwiseAbs().maxCoeff(),positions[v].cwiseAbs().maxCoeff()});
            if (!expected.allFinite() || (expected-positions[v]).cwiseAbs().maxCoeff()>tolerance)
                throw std::invalid_argument("general_DAT: rigid proxies disagree with pose");
        }
    }
    for (std::size_t i=0;i<mesh.tris.size();i+=3) {
        const int a=mesh.tris[i],b=mesh.tris[i+1],c=mesh.tris[i+2];
        if (!valid_node(a)||!valid_node(b)||!valid_node(c)||a==b||b==c||a==c
            || mesh.node_to_rb[a]!=mesh.node_to_rb[b] || mesh.node_to_rb[a]!=mesh.node_to_rb[c])
            throw std::invalid_argument("general_DAT: invalid or mixed-owner triangle");
    }
    for(int v:mesh.tets) if(!valid_node(v)||mesh.node_to_rb[v]>=0)
        throw std::invalid_argument("general_DAT: invalid or rigid tetrahedron");
    std::vector<unsigned char> interior(n,0);
    for(int v:mesh.tet_nodes) {
        if(!valid_node(v)||mesh.node_to_rb[v]>=0) throw std::invalid_argument("general_DAT: invalid solid classification");
        interior[v]=1;
    }
    for(int v:mesh.surface_nodes) {
        if(!valid_node(v)) throw std::invalid_argument("general_DAT: invalid surface classification");
        interior[v]=0;
    }
    for(int v:mesh.tris) if(interior[v])
        throw std::invalid_argument("general_DAT: interior node on surface");

    const double limit=.5*gamma*query_radius, radius=query_radius*node_box_padding;
    if (!(limit>0)||!std::isfinite(radius))
        throw std::invalid_argument("general_DAT: invalid query extent");
    GeneralDATResult result;
    result.body_factors.assign(bodies,1);
    // The far-pair bound is a full curved-path sphere, never endpoint distance.
    for(std::size_t b=0;b<bodies;++b)
        for(int v:mesh.rb_nodes[b]) {
            initial_boxes[v]=rigid_dat::trajectory_bounds(paths[v],0,0);
            Vec3 initial_offset;
            for(int k=0;k<3;++k) initial_offset[k]=std::max(
                std::nextafter(std::abs(initial_boxes[v].lo[k]-positions[v][k]),INFINITY),
                std::nextafter(std::abs(initial_boxes[v].hi[k]-positions[v][k]),INFINITY));
            // Query neighborhoods are anchored at represented positions. The
            // mathematical phi(0) enclosure can differ by evaluation roundoff;
            // reserve that offset over the entire arc, not only at its endpoint.
            const double offset=dat_norm_upper(initial_offset);
            const double budget=offset<limit?std::nextafter(limit-offset,0.0):0;
            if(!(budget>0))throw std::domain_error("general_DAT: unresolved query-anchor roundoff");
            result.body_factors[b]=std::min(result.body_factors[b],rigid_dat::radius_step(paths[v],budget));
        }
    std::vector<Vec3> updates(n);
    std::vector<AABB> boxes(n);
    std::vector<std::atomic<double>> factors(n),body_factors(bodies);
    for(std::size_t b=0;b<bodies;++b) body_factors[b].store(result.body_factors[b]);
    for(std::size_t v=0;v<n;++v) {
        const int b=mesh.node_to_rb[v];
        if(b<0) updates[v]=dat_capped_update(particle_updates[v],limit);
        else if(delta_omega[b].cwiseAbs().maxCoeff()==0)
            updates[v]=result.body_factors[b]*delta_centers[b];
        else {
            // Plane allocation depends on local motion, not subtraction of two
            // rounded, potentially very large, world-space endpoints.
            auto local_path=paths[v]; local_path.center={0,0,0};
            const auto endpoint=rigid_dat::evaluate(local_path,result.body_factors[b]);
            const auto start=rigid_dat::evaluate(local_path,0);
            updates[v]=Vec3(endpoint[0]-start[0],endpoint[1]-start[1],endpoint[2]-start[2]);
        }
        boxes[v]=AABB(positions[v]-Vec3::Constant(radius),positions[v]+Vec3::Constant(radius));
        if(!updates[v].allFinite()||!boxes[v].min.allFinite()||!boxes[v].max.allFinite())
            throw std::invalid_argument("general_DAT: node extent overflow");
        factors[v].store(1);
    }
    BroadPhase phase;
    if(!mesh.tets.empty()||!mesh.tet_nodes.empty()) phase.initialize_surface_nodes(
        boxes,mesh,0,BroadPhase::InitializationMode::GeneralSolver);
    else phase.initialize(boxes,mesh,0,BroadPhase::InitializationMode::GeneralSolver);
    const auto& cache=phase.cache();
    struct EndpointConstraint {
        std::array<int,4> nodes{};
        Vec3 normal=Vec3::Zero();
        int split=0;
    };
    std::vector<EndpointConstraint> endpoint_constraints(cache.nt_pairs.size()+cache.ss_pairs.size());
    const auto constrain=[&](const std::array<int,4>& nodes,int split,const Vec3& separation,std::size_t index) {
        const double distance=separation.stableNorm();
        if(!separation.allFinite()||!std::isfinite(distance))
            throw std::domain_error("general_DAT: unresolved separation");
        if(distance>=query_radius)return;
        if(!(distance>0))throw std::domain_error("general_DAT: initially intersecting primitives");
        const Vec3 normal=separation/distance,origin=positions[nodes[0]];
        endpoint_constraints[index]={nodes,normal,split};
        double left=-INFINITY,right=INFINITY,a=0,b=0;
        bool rigid_pair=false;
        std::array<double,4> coordinate{},velocity{};
        for(int i=0;i<4;++i) {
            const int v=nodes[i],owner=mesh.node_to_rb[v];
            coordinate[i]=(positions[v]-origin).dot(normal); velocity[i]=updates[v].dot(normal);
            rigid_pair|=owner>=0;
            if(!std::isfinite(coordinate[i])||!std::isfinite(velocity[i]))
                throw std::domain_error("general_DAT: plane overflow");
            if(i<split) {left=std::max(left,coordinate[i]);a=std::max(a,velocity[i]);}
            else {right=std::min(right,coordinate[i]);b=std::max(b,-velocity[i]);}
        }
        const double gap=right-left;
        if(!(gap>0)||!std::isfinite(gap))throw std::domain_error("general_DAT: nonseparating normal");
        const double scale=std::max(a,b);
        a=scale>0?a/scale:1; b=scale>0?b/scale:1;
        // A curved vertex can return toward a plane despite retreating endpoint
        // motion. Give every rotating side an interior plane, then certify it.
        if(rigid_pair) {a=std::max(a,1e-3);b=std::max(b,1e-3);}
        const double budget_a=gap*(a/(a+b)),budget_b=gap*(b/(a+b));
        const Vec3 plane_point=origin+normal*(left+budget_a);
        std::array<double,4> certified_margins{};
        std::array<unsigned char,4> initial_side_safe{};
        if(rigid_pair) {
            // Materializing a plane point can move the represented plane out
            // of a narrow gap. Certify its actual sides with outward-rounded
            // projections before allowing plane_step to infer an initial side.
            for(int i=0;i<4;++i) {
                const auto x=array(positions[nodes[i]]);
                const auto projection=dat_plane_projection({x,x},normal,plane_point);
                const double lo=projection[0],hi=projection[1];
                if(!std::isfinite(lo)||!std::isfinite(hi)||(i<split?!(hi<0):!(lo>0)))
                    throw std::domain_error("general_DAT: rounded division plane is not separating");
                certified_margins[i]=i<split?-hi:lo;
                initial_side_safe[i]=1;
                if(mesh.node_to_rb[nodes[i]]>=0) {
                    const auto initial=dat_plane_projection(initial_boxes[nodes[i]],normal,plane_point);
                    initial_side_safe[i]=i<split?initial[1]<0:initial[0]>0;
                    certified_margins[i]=std::min(certified_margins[i],i<split?-initial[1]:initial[0]);
                }
            }
        }
        for(int i=0;i<4;++i) {
            const int v=nodes[i],owner=mesh.node_to_rb[v];
            if(owner>=0&&!initial_side_safe[i]) {
                throw std::domain_error("general_DAT: initial rigid trajectory is not on its intended plane side");
            }
            if(owner>=0 && delta_omega[owner].cwiseAbs().maxCoeff()>0) {
                const double alpha=rigid_dat::plane_step(paths[v],array(normal),array(plane_point),gamma);
                dat_atomic_min(body_factors[owner],alpha);
            } else {
                double approach=i<split?velocity[i]:-velocity[i];
                if(rigid_pair) {
                    // Both sides must use the same represented division plane.
                    // Bound the normal speed upwards, including a rigid body's
                    // radius cap, rather than trusting a rounded dot product.
                    double upper=0;
                    for(int k=0;k<3;++k) {
                        const double speed=owner<0?updates[v][k]:delta_centers[owner][k];
                        const double coefficient=i<split?normal[k]:-normal[k];
                        const double product=std::nextafter(coefficient*speed,INFINITY);
                        upper=std::nextafter(upper+product,INFINITY);
                    }
                    approach=owner<0?upper:std::nextafter(upper*result.body_factors[owner],INFINITY);
                }
                if(approach<=0)continue;
                const double margin=rigid_pair?certified_margins[i]
                    :(i<split?left-coordinate[i]+budget_a:coordinate[i]-right+budget_b);
                const double allowed=rigid_pair
                    ?std::nextafter(gamma*margin,0.0):gamma*margin;
                if(approach>allowed) {
                    double factor=std::clamp(allowed/approach,0.0,1.0);
                    if(rigid_pair)factor=std::nextafter(factor,0.0);
                    if(owner<0)dat_atomic_min(factors[v],factor);
                    else dat_atomic_min(body_factors[owner],
                        std::nextafter(result.body_factors[owner]*factor,0.0));
                }
            }
        }
    };
    dat_for_pairs(cache.nt_pairs.size(),[&](std::size_t i){
        const auto& p=cache.nt_pairs[i]; const std::array<int,4> nodes{p.node,p.tri_v[0],p.tri_v[1],p.tri_v[2]};
        constrain(nodes,1,dat_vertex_triangle_separation(positions[nodes[0]],positions[nodes[1]],positions[nodes[2]],positions[nodes[3]]),i);
    });
    dat_for_pairs(cache.ss_pairs.size(),[&](std::size_t i){
        const auto& p=cache.ss_pairs[i]; const std::array<int,4> nodes{p.v[0],p.v[1],p.v[2],p.v[3]};
        constrain(nodes,2,-segment_segment_distance(positions[nodes[0]],positions[nodes[1]],positions[nodes[2]],positions[nodes[3]]).separation,cache.nt_pairs.size()+i);
    });
    std::vector<double> node_factors(n);
    for(std::size_t v=0;v<n;++v)node_factors[v]=factors[v].load();
    for(std::size_t b=0;b<bodies;++b)result.body_factors[b]=body_factors[b].load();
    // Certify represented endpoints as well as the mathematical curved path.
    // COM addition and quaternion placement can otherwise round a positive gap
    // to zero. Backtracking only shortens already-certified trajectory prefixes.
    std::vector<Vec3> endpoints(n);
    result.centers.resize(bodies);result.omega.resize(bodies);result.orientations.resize(bodies);
    constexpr int endpoint_backtracks=std::numeric_limits<double>::digits+2;
    for(int attempt=0;;++attempt) {
        for(std::size_t v=0;v<n;++v)if(mesh.node_to_rb[v]<0) {
            const Vec3 displacement=updates[v]*node_factors[v];
            endpoints[v]=positions[v]+displacement;
        }
        for(std::size_t b=0;b<bodies;++b) {
            const double alpha=result.body_factors[b];
            result.centers[b]=centers[b]+alpha*delta_centers[b];
            result.omega[b]=omega[b]+alpha*delta_omega[b];
            result.orientations[b]=quaternion_from_angular_velocity(previous_orientations[b],result.omega[b],dt);
            for(std::size_t j=0;j<mesh.rb_nodes[b].size();++j)
                endpoints[mesh.rb_nodes[b][j]]=world_space_position(mesh.ref_positions[b][j],result.centers[b],result.orientations[b]);
        }
        std::vector<unsigned char> bad_node(n,0),bad_body(bodies,0);
        const auto mark=[&](int v){
            if(mesh.node_to_rb[v]<0)bad_node[v]=1;
            else bad_body[mesh.node_to_rb[v]]=1;
        };
        bool failed=false;
        for(std::size_t v=0;v<n;++v) {
            if(!endpoints[v].allFinite()) {failed=true;mark(int(v));continue;}
            Vec3 extent=Vec3::Zero();
            for(int k=0;k<3;++k) {
                if(endpoints[v][k]!=positions[v][k])
                    extent[k]=std::nextafter(std::abs(endpoints[v][k]-positions[v][k]),INFINITY);
            }
            if(dat_norm_upper(extent)>limit) {failed=true;mark(int(v));}
        }
        for(const auto& constraint:endpoint_constraints) {
            if(!constraint.split)continue;
            double left=-INFINITY,right=INFINITY;
            const Vec3& origin=endpoints[constraint.nodes[0]];
            for(int i=0;i<4;++i) {
                double lo=0,hi=0;
                for(int k=0;k<3;++k) {
                    const double d=endpoints[constraint.nodes[i]][k]-origin[k];
                    const double dl=std::nextafter(d,-INFINITY),dh=std::nextafter(d,INFINITY);
                    const double normal=constraint.normal[k];
                    lo=std::nextafter(lo+std::nextafter(normal*(normal>=0?dl:dh),-INFINITY),-INFINITY);
                    hi=std::nextafter(hi+std::nextafter(normal*(normal>=0?dh:dl),INFINITY),INFINITY);
                }
                if(i<constraint.split)left=std::max(left,hi);
                else right=std::min(right,lo);
            }
            if(!(right>left)) {
                failed=true;
                for(int v:constraint.nodes)mark(v);
            }
        }
        if(!failed) {
            result.positions=std::move(endpoints);
            return result;
        }
        if(attempt==endpoint_backtracks)throw std::domain_error("general_DAT: cannot certify represented endpoints");
        for(std::size_t b=0;b<bodies;++b)if(bad_body[b])
            result.body_factors[b]=attempt==endpoint_backtracks-1?0:.5*result.body_factors[b];
        for(std::size_t v=0;v<n;++v)if(bad_node[v])
            node_factors[v]=attempt==endpoint_backtracks-1?0:.5*node_factors[v];
    }
}
