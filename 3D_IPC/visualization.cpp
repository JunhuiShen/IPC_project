#include "visualization.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

void export_obj(const std::string& filename, const std::vector<Vec3>& x, const std::vector<Tri>& tris) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Error: cannot write " << filename << "\n";
        return;
    }

    for (const auto& p : x) out << "v " << p.x() << " " << p.y() << " " << p.z() << "\n";
    for (const auto& t : tris) out << "f " << (t.v[0] + 1) << " " << (t.v[1] + 1) << " " << (t.v[2] + 1) << "\n";
}

void export_frame(const std::string& outdir, int frame, const std::vector<Vec3>& x, const std::vector<Tri>& tris) {
    std::ostringstream ss;
    ss << outdir << "/frame_" << std::setw(4) << std::setfill('0') << frame << ".obj";
    export_obj(ss.str(), x, tris);
}
