        // Prescribed boundary inputs and file output stay on the host. All
        // evolving simulation state remains GPU between these launches.
        gpu_solver.begin_profile_frame(frame_index);
        for (int substep=0; substep<params.substeps; ++substep) {
            const double t_next=((frame_index-1)*params.substeps+substep+1)*params.dt();
            if (pin_updater) pin_updater(pins,t_next);
            auto gpu_step=ipc_gpu::solver::describe_step(params);
            // Match advance_one_frame_rb's clock. The CPU mixed/general
            // driver deliberately retains params.rigid_pin_time unchanged.
            if(has_rigid&&!uses_general_solver)gpu_step.rigid_pin_time=t_next;
            gpu_solver.advance_substep(gpu_step,
                ipc_gpu::solver::describe_pin_targets(state.deformed_positions.size(),pins),
                ipc_gpu::solver::describe_colliders(params));
            if (params.write_substeps) {
                ipc_gpu::solver::copy_snapshot(gpu_solver.snapshot(),state);
                if (substep_cb) substep_cb((frame_index-1)*params.substeps+substep,state.deformed_positions);
            }
            result.iterations+=params.max_global_iters;
        }
        if (!params.write_substeps)
            ipc_gpu::solver::copy_snapshot(gpu_solver.snapshot(),state);
        gpu_solver.end_profile_frame();
        result.converged=true; // completed the requested fixed sweeps
