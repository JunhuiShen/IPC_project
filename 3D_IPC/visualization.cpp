#include "visualization.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

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
