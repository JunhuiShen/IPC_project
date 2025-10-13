#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <system_error>
#include <cstdio>
#include <filesystem>
namespace fs = std::__fs::filesystem;

// simple 2D point
struct P2 { double x,y; };

// append a circle polygon centered at (cx,cy) with radius r, nSegments
// z=0 always
void appendCircle(std::ofstream &ofs, double cx, double cy, double r, int nSegments, int &vCount) {
    int startIdx = vCount + 1;
    for(int i=0;i<nSegments;++i){
        double theta = 2*M_PI*i/nSegments;
        double x = cx + r*std::cos(theta);
        double y = cy + r*std::sin(theta);
        ofs << "v " << x << " " << y << " 0\n";
        vCount++;
    }
    // faces: fan triangulation
    for(int i=1;i<nSegments-1;++i){
        ofs << "f " << startIdx << " " << startIdx+i << " " << startIdx+i+1 << "\n";
    }
}

int main(){
    // dynamics
    const double epsilon = -0.5;
    const double dt = 0.03;
    const int nFrames = 180;

    // visualization params
    const double visScaleX = 2.0;
    const double visScaleY = 2.0;
    const double pointRadiusVis = 0.1;
    const int circleSegments = 40;

    const std::string outDir = "obj_frames_2d";
    std::error_code ec;
    fs::create_directories(outDir, ec);

    // initial points
    P2 x1{0.0,  3.5};
    P2 x2{0.0, 0.5};
    P2 x3{2.0,  1.5};

    const double stepMul = std::exp(epsilon*dt); // x_dot = epsilon * x

    for(int f=0; f<nFrames; ++f){
        char name[256];
        std::snprintf(name,sizeof(name),"%s/frame_%04d.obj",outDir.c_str(),f);
        std::ofstream ofs(name);
        if(!ofs){ std::cerr<<"cannot write "<<name<<"\n"; return 1; }

        ofs << "# frame " << f << "\n";

        int vCount=0;

        // scaled visual positions
        double vx1=visScaleX*x1.x, vy1=visScaleY*x1.y;
        double vx2=visScaleX*x2.x, vy2=visScaleY*x2.y;
        double vx3=visScaleX*x3.x, vy3=visScaleY*x3.y;

        // circles for points
        appendCircle(ofs,vx1,vy1,pointRadiusVis,circleSegments,vCount);
        appendCircle(ofs,vx2,vy2,pointRadiusVis,circleSegments,vCount);
        appendCircle(ofs,vx3,vy3,pointRadiusVis,circleSegments,vCount);

        // line x1-x2
        ofs << "v " << vx1 << " " << vy1 << " 0\n";
        ofs << "v " << vx2 << " " << vy2 << " 0\n";
        int id1 = vCount+1;
        int id2 = vCount+2;
        ofs << "l " << id1 << " " << id2 << "\n";

        // advance x3
        x3.x *= stepMul;
    }

    std::cout<<"Wrote "<<nFrames<<" OBJ files in "<<outDir<<"\n";
    return 0;
}



