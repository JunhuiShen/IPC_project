#include "visualization.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>

using namespace math;

void export_obj(const std::string& filename,
                const Vec& x,
                const std::vector<std::pair<int,int>>& edges) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Error: cannot write " << filename << "\n";
        return;
    }

    int N = static_cast<int>(x.size() / 2);
    for (int i = 0; i < N; ++i) {
        Vec2 xi = get_xi(x, i);
        out << "v " << xi.x << " " << xi.y << " 0.0\n";
    }

    for (const auto& e : edges)
        out << "l " << (e.first + 1) << " " << (e.second + 1) << "\n";
}

void export_frame(const std::string& outdir,
                  int frame,
                  const Vec& x_combined,
                  const std::vector<std::pair<int,int>>& edges_combined) {
    std::ostringstream ss;
    ss << outdir << "/frame_" << std::setw(4) << std::setfill('0') << frame << ".obj";
    export_obj(ss.str(), x_combined, edges_combined);
}
