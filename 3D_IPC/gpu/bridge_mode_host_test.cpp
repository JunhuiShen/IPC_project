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
        parameters.max_global_iters=3;
        parameters.node_box_min=.001;
        parameters.node_box_max=.01;
        parameters.theta_box_max=.1;
        parameters.node_box_update_count=3;
        const auto ordinary=ipc_gpu::solver::describe_step(parameters);
        if(ordinary.iterations!=3||ordinary.rebuild_interval!=3)
            throw std::runtime_error("ordinary GPU bridge mapping changed");
        // The modern cloth scheduler is shared by scalar and SIMD assembly.
        parameters.use_parallel=true;
        parameters.d_hat=.005;
        for(bool simd:{false,true}) {
            parameters.use_simd=simd;
            const auto step=ipc_gpu::solver::describe_step(parameters,2);
            if(step.cpu_energy_simd!=simd
                ||step.original_update_convention!=ipc_gpu::solver::OriginalUpdateConvention::CooperativeSquaredNorm)
                throw std::runtime_error("modern cloth backend/scheduler metadata changed");
            const auto serial_reference=ipc_gpu::solver::describe_step(parameters,1);
            if(serial_reference.cpu_energy_simd!=simd
                ||serial_reference.original_update_convention!=ipc_gpu::solver::OriginalUpdateConvention::StoredNoop)
                throw std::runtime_error("one-thread cloth metadata changed");
        }
        parameters.use_parallel=false;
        if(ipc_gpu::solver::describe_step(parameters,2).original_update_convention
            !=ipc_gpu::solver::OriginalUpdateConvention::StoredNoop)
            throw std::runtime_error("serial cloth dispatch became cooperative");
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
