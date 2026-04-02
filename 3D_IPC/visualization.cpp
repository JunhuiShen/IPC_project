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

void export_frame(const std::string& outdir, int frame, const std::vector<Vec3>& x, const std::vector<int>& tris) {
    std::ostringstream ss;
    ss << outdir << "/frame_" << std::setw(4) << std::setfill('0') << frame << ".obj";
    export_obj(ss.str(), x, tris);
}
