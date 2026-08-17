#pragma once

#include "IPC_math.h"

#include <string>
#include <vector>

// Project-native ports of TGSL's TetGen readers in library/io/IO.h.
//
// zero_based_index describes the numbering used by the input file. The
// returned tetrahedron and triangle connectivity is always zero based.
// Successful calls replace the output vector; failures throw and leave it
// unchanged.

// Read a TetGen .node file. Only three-dimensional node files are supported;
// declared attributes and boundary markers are validated and discarded.
void read_tetgen_nodes(
    const std::string& filename,
    std::vector<Vec3>& positions,
    bool zero_based_index = false);

// Read a TetGen .ele file containing linear tetrahedra. Every four consecutive
// returned indices define one tetrahedron. Declared attributes are discarded.
void read_tetgen_tets(
    const std::string& filename,
    std::vector<int>& tets,
    bool zero_based_index = false);

// Read a TetGen .face file. Every three consecutive returned indices define
// one triangle. Declared boundary markers are discarded.
void read_tetgen_faces(
    const std::string& filename,
    std::vector<int>& faces,
    bool zero_based_index = false);

// Write a tetrahedral mesh as an OBJ visualization
void write_tet_mesh_obj(
    const std::string& filename,
    const std::vector<Vec3>& positions,
    const std::vector<int>& tets);
