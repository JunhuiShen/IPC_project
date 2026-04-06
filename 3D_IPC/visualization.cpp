#include "visualization.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace {

void write_aabb_wireframe(std::ofstream& out, const AABB& box, int& v_offset) {
    const Vec3& lo = box.min;
    const Vec3& hi = box.max;

    out << "v " << lo.x() << " " << lo.y() << " " << lo.z() << "\n";
    out << "v " << hi.x() << " " << lo.y() << " " << lo.z() << "\n";
    out << "v " << hi.x() << " " << hi.y() << " " << lo.z() << "\n";
    out << "v " << lo.x() << " " << hi.y() << " " << lo.z() << "\n";
    out << "v " << lo.x() << " " << lo.y() << " " << hi.z() << "\n";
    out << "v " << hi.x() << " " << lo.y() << " " << hi.z() << "\n";
    out << "v " << hi.x() << " " << hi.y() << " " << hi.z() << "\n";
    out << "v " << lo.x() << " " << hi.y() << " " << hi.z() << "\n";

    const int b = v_offset + 1; // OBJ indices are 1-based
    // bottom face
    out << "l " << b+0 << " " << b+1 << "\n";
    out << "l " << b+1 << " " << b+2 << "\n";
    out << "l " << b+2 << " " << b+3 << "\n";
    out << "l " << b+3 << " " << b+0 << "\n";
    // top face
    out << "l " << b+4 << " " << b+5 << "\n";
    out << "l " << b+5 << " " << b+6 << "\n";
    out << "l " << b+6 << " " << b+7 << "\n";
    out << "l " << b+7 << " " << b+4 << "\n";
    // verticals
    out << "l " << b+0 << " " << b+4 << "\n";
    out << "l " << b+1 << " " << b+5 << "\n";
    out << "l " << b+2 << " " << b+6 << "\n";
    out << "l " << b+3 << " " << b+7 << "\n";

    v_offset += 8;
}

} // namespace

void export_obj(const std::string& filename, const std::vector<Vec3>& x, const std::vector<int>& tris) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Error: cannot write " << filename << "\n";
        return;
    }

    for (const auto& p : x) out << "v " << p.x() << " " << p.y() << " " << p.z() << "\n";
    for (int t = 0; t < static_cast<int>(tris.size()); t += 3)
        out << "f " << (tris[t] + 1) << " " << (tris[t+1] + 1) << " " << (tris[t+2] + 1) << "\n";
}

void export_geo(const std::string& filename, const std::vector<Vec3>& x, const std::vector<int>& tris) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Error: cannot write " << filename << "\n";
        return;
    }

    int npoints   = static_cast<int>(x.size());
    int nprims    = static_cast<int>(tris.size()) / 3;
    int nvertices = nprims * 3;

    out << std::setprecision(10);

    out << "[\n";
    out << "    \"fileversion\", \"18.5.408\",\n";
    out << "    \"hasindex\", false,\n";
    out << "    \"pointcount\", " << npoints << ",\n";
    out << "    \"vertexcount\", " << nvertices << ",\n";
    out << "    \"primitivecount\", " << nprims << ",\n";
    out << "    \"info\", {},\n";

    // Topology — must be an array, not an object
    out << "    \"topology\",\n    [\n";
    out << "        \"pointref\",\n        [\n";
    out << "            \"indices\", [";
    for (int i = 0; i < static_cast<int>(tris.size()); ++i) {
        if (i > 0) out << ",";
        out << tris[i];
    }
    out << "]\n";
    out << "        ]\n";
    out << "    ],\n";

    // Attributes — must be an array, not an object
    out << "    \"attributes\",\n    [\n";
    out << "        \"pointattributes\",\n        [\n";
    out << "            [\n";
    // Descriptor array (alternating key/value, not a JSON object)
    out << "                [\"scope\",\"public\",\"type\",\"numeric\",\"name\",\"P\",\"options\",{}],\n";
    // Data array
    out << "                [\"size\",3,\"storage\",\"fpreal32\",\"values\",\n";
    out << "                    [\n";
    out << "                        \"size\", 3,\n";
    out << "                        \"storage\", \"fpreal32\",\n";
    out << "                        \"tuples\", [";
    for (int i = 0; i < npoints; ++i) {
        if (i > 0) out << ",";
        out << "[" << x[i].x() << "," << x[i].y() << "," << x[i].z() << "]";
    }
    out << "]\n";
    out << "                    ]\n";
    out << "                ]\n";
    out << "            ]\n";
    out << "        ]\n";
    out << "    ],\n";

    // Primitives
    out << "    \"primitives\",\n    [\n";
    out << "        [\n";
    out << "            [\"type\",\"Polygon_run\"],\n";
    out << "            [\n";
    out << "                \"startvertex\", 0,\n";
    out << "                \"nprimitives\", " << nprims << ",\n";
    out << "                \"nvertices_rle\", [3," << nprims << "]\n";
    out << "            ]\n";
    out << "        ]\n";
    out << "    ]\n";
    out << "]\n";
}

void export_usd(const std::string& filename, const std::vector<Vec3>& x, const std::vector<int>& tris) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Error: cannot write " << filename << "\n";
        return;
    }

    int npoints = static_cast<int>(x.size());
    int nprims  = static_cast<int>(tris.size()) / 3;

    out << std::setprecision(10);
    out << "#usda 1.0\n\n";
    out << "def Mesh \"mesh\"\n{\n";

    // Points
    out << "    point3f[] points = [";
    for (int i = 0; i < npoints; ++i) {
        if (i > 0) out << ", ";
        out << "(" << x[i].x() << ", " << x[i].y() << ", " << x[i].z() << ")";
    }
    out << "]\n";

    // Face vertex counts (all triangles = 3)
    out << "    int[] faceVertexCounts = [";
    for (int i = 0; i < nprims; ++i) {
        if (i > 0) out << ", ";
        out << "3";
    }
    out << "]\n";

    // Face vertex indices
    out << "    int[] faceVertexIndices = [";
    for (int i = 0; i < static_cast<int>(tris.size()); ++i) {
        if (i > 0) out << ", ";
        out << tris[i];
    }
    out << "]\n";

    out << "}\n";
}

void export_aabb_list(const std::string& filename, const std::vector<AABB>& boxes) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Error: cannot write " << filename << "\n";
        return;
    }
    out << std::setprecision(10);
    int v_offset = 0;
    for (const AABB& box : boxes)
        write_aabb_wireframe(out, box, v_offset);
}

int export_bvh_level(const std::string& filename, const std::vector<BVHNode>& nodes, int root, int depth) {
    if (root < 0 || nodes.empty()) return 0;

    struct Entry { int idx; int d; };
    std::vector<Entry> queue;
    queue.push_back({root, 0});

    std::vector<AABB> boxes;
    bool any_at_depth = false;
    for (int qi = 0; qi < static_cast<int>(queue.size()); ++qi) {
        const auto [ni, d] = queue[qi];
        const BVHNode& n = nodes[ni];
        if (d == depth) {
            any_at_depth = true;
            boxes.push_back(n.bbox);
        } else if (n.leafIndex >= 0) {
            boxes.push_back(n.bbox);
        } else {
            if (n.left  >= 0) queue.push_back({n.left,  d + 1});
            if (n.right >= 0) queue.push_back({n.right, d + 1});
        }
    }

    if (!any_at_depth) return 0;

    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Error: cannot write " << filename << "\n";
        return 0;
    }
    out << std::setprecision(10);
    int v_offset = 0;
    for (const AABB& box : boxes)
        write_aabb_wireframe(out, box, v_offset);

    return static_cast<int>(boxes.size());
}

void export_broad_phase_boxes(const std::string& filename, const BroadPhase& bp) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Error: cannot write " << filename << "\n";
        return;
    }

    out << std::setprecision(10);

    const BroadPhase::Cache& c = bp.cache();
    int v_offset = 0;

    out << "g node_boxes\n";
    for (const AABB& box : c.node_boxes)
        write_aabb_wireframe(out, box, v_offset);

    out << "g tri_boxes\n";
    for (const AABB& box : c.tri_boxes)
        write_aabb_wireframe(out, box, v_offset);

    out << "g edge_boxes\n";
    for (const AABB& box : c.edge_boxes)
        write_aabb_wireframe(out, box, v_offset);
}

void export_frame(const std::string& outdir, int frame, const std::vector<Vec3>& x, const std::vector<int>& tris,
                  ExportFormat fmt) {
    std::ostringstream ss;
    const char* ext = (fmt == ExportFormat::GEO) ? ".geo" : (fmt == ExportFormat::USD) ? ".usda" : ".obj";
    ss << outdir << "/frame_" << std::setw(4) << std::setfill('0') << frame << ext;
    if (fmt == ExportFormat::GEO)
        export_geo(ss.str(), x, tris);
    else if (fmt == ExportFormat::USD)
        export_usd(ss.str(), x, tris);
    else
        export_obj(ss.str(), x, tris);
}
