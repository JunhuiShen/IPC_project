// Host-only routing regression: no CUDA runtime or device execution.
#include "bridge.h"
#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    try {
        SimParams parameters=SimParams::zeros();
        parameters.fps=30;
        parameters.substeps=3;
        parameters.fixed_iters=true;
        parameters.use_ticcd=false;
        parameters.max_global_iters=3;
        parameters.node_box_min=.001;
        parameters.node_box_max=.01;
        parameters.theta_box_max=.1;
        parameters.node_box_update_count=3;
        const auto ordinary=ipc_gpu::solver::describe_step(parameters);
        if(ordinary.iterations!=3||ordinary.rebuild_interval!=3)
            throw std::runtime_error("ordinary GPU bridge mapping changed");
        parameters.use_dat=true;
        bool step_rejected=false,scene_rejected=false;
        try{(void)ipc_gpu::solver::describe_step(parameters);}
        catch(const std::invalid_argument& error){step_rejected=std::string(error.what()).find("CPU planar DAT")!=std::string::npos;}
        RefMesh mesh;DeformedState state;
        try{(void)ipc_gpu::solver::describe_scene(mesh,state,{},parameters);}
        catch(const std::invalid_argument& error){scene_rejected=std::string(error.what()).find("CPU planar DAT")!=std::string::npos;}
        if(!step_rejected||!scene_rejected)
            throw std::runtime_error("retired CUDA DAT request did not fail closed before scene/step setup");
        parameters.use_dat=false;parameters.use_ogc=true;
        step_rejected=scene_rejected=false;
        try{(void)ipc_gpu::solver::describe_step(parameters);}
        catch(const std::invalid_argument& error){step_rejected=std::string(error.what()).find("CPU isotropic OGC")!=std::string::npos;}
        try{(void)ipc_gpu::solver::describe_scene(mesh,state,{},parameters);}
        catch(const std::invalid_argument& error){scene_rejected=std::string(error.what()).find("CPU isotropic OGC")!=std::string::npos;}
        if(!step_rejected||!scene_rejected)
            throw std::runtime_error("CPU isotropic OGC request did not fail closed before scene/step setup");
        std::cout<<"GPU bridge host routing: ordinary mode retained, CPU-only DAT/OGC rejected PASS\n";
        return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
