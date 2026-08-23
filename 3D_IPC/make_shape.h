#pragma once
#include "physics.h"
#include <string>
#include <vector>

// Extruded regular n-gon rigid body. The polygon is built in the material
// x-y plane and extruded along material z by `thickness`; `orientation`
// rotates that prism into world space. Returns the rigid-body index.
int append_rigid_polygon(
    int number_of_nodes, DeformedState& state, RefMesh& ref_mesh,
    const Vec3& center, double radius, double density,
    double thickness = 0.001,
    const Vec3& v_com = Vec3::Zero(),
    const Vec4& orientation = Vec4(1.0, 0.0, 0.0, 0.0),
    const Vec3& omega = Vec3::Zero(),
    RigidBodyUpdateMode update_mode =
        RigidBodyUpdateMode::TranslationAndOrientation);

// Tetrahedral deformable regular n-gonal prism. The polygon is built in the
// material x-y plane and extruded along material z by `thickness`;
// `orientation` rotates that prism into world space. The tetrahedralization
// cones every outward boundary triangle to one interior center vertex. Returns
// the first global node index appended for this solid.
int append_deformable_polygon_prism(
    int number_of_nodes, DeformedState& state, RefMesh& ref_mesh,
    const Vec3& center, double radius, double density,
    double thickness,
    const Vec4& orientation = Vec4(1.0, 0.0, 0.0, 0.0));

// Load a linear TetGen mesh, recenter it at `center`, and uniformly scale it
// so its largest axis-aligned bounding-box extent is `target_max_extent`.
// `zero_based_index` describes the numbering in both input files. Any
// negatively oriented input tetrahedra are flipped before create_solid.
// Returns the first global node index appended for this solid.
int append_normalized_tetgen_solid(
    const std::string& node_filename,
    const std::string& element_filename,
    DeformedState& state, RefMesh& ref_mesh,
    const Vec3& center, double target_max_extent, double density,
    bool zero_based_index = false);

// Load a closed OBJ surface, remove unreferenced vertices, recenter its
// axis-aligned bounding box, and uniformly scale its largest extent to
// `target_max_extent`. `orientation` rotates the normalized surface before it
// is translated to `center`. The rigid body's total mass is the enclosed
// surface volume times `density`. Returns the appended rigid-body index.
int append_normalized_obj_rigid_body(
    const std::string& obj_filename,
    DeformedState& state, RefMesh& ref_mesh,
    const Vec3& center, double target_max_extent, double density,
    const Vec3& v_com = Vec3::Zero(),
    const Vec4& orientation = Vec4(1.0, 0.0, 0.0, 0.0),
    const Vec3& omega = Vec3::Zero(),
    RigidBodyUpdateMode update_mode =
        RigidBodyUpdateMode::TranslationAndOrientation);

// Grid counts: V = (nx + 1)(ny + 1), T = 2 nx ny.
int build_square_mesh(RefMesh& ref_mesh, DeformedState& state, std::vector<Vec2>& X, int nx, int ny, double width, double height, const Vec3& origin);

// Same grid and winding as build_square_mesh, with the cell diagonal
// alternating in a checkerboard pattern. For an even nx, the resulting
// topology is invariant under reflection across the grid's x midpoint.
int build_square_mesh_alternating_diagonals(
    RefMesh& ref_mesh, DeformedState& state, std::vector<Vec2>& X,
    int nx, int ny, double width, double height, const Vec3& origin);

// Cylinder: h = (2 pi radius/nu)sqrt(3)/2, n = max(1, round(length/h)); V = nu(n+1)+2, T = 2nu(n+1).
int build_cylinder_mesh(RefMesh& ref_mesh, DeformedState& state, std::vector<Vec2>& X, int nu, double radius, double length, const Vec3& center);

// Icosphere counts: V = 10(4^subdiv) + 2, T = 20(4^subdiv); X uses xz projection.
int build_sphere_mesh(RefMesh& ref_mesh, DeformedState& state, std::vector<Vec2>& X, int subdiv, double radius, const Vec3& center);

// OBJ transform: x' = origin + scale*x; each n-gon becomes n - 2 triangles.
int load_obj_mesh(const std::string& path, RefMesh& ref_mesh, DeformedState& state, double scale, const Vec3& origin);

// Flat-array overload of the same OBJ transform and triangulation.
void load_obj_mesh(const std::string& path, std::vector<Vec3>& verts, std::vector<int>& tris, double scale = 1.0, const Vec3& origin = Vec3::Zero());

// Sets Dm from an edge-length-preserving 2D flattening and A = |det(Dm)|/2.
void rebuild_triangle_rest_isometric(RefMesh& ref_mesh, const std::vector<Vec3>& x_rest, int t_begin, int t_end);

// Sets c_e = |edge|^2 / (A_left + A_right) within the vertex range.
void rebuild_hinge_c_e_3d(RefMesh& ref_mesh, const std::vector<Vec3>& x_rest, int v_begin, int v_end);
