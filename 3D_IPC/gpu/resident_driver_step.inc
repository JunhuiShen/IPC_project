        // Prescribed boundary inputs and file output stay on the host. All
        // evolving simulation state remains resident between these launches.
        resident_solver.begin_profile_frame(frame_index);
        for (int substep=0; substep<params.substeps; ++substep) {
            const double t_next=((frame_index-1)*params.substeps+substep+1)*params.dt();
            if (pin_updater) pin_updater(pins,t_next);
            resident_solver.advance_substep(ipc_gpu::resident::describe_step(params),
                ipc_gpu::resident::describe_pin_targets(state.deformed_positions.size(),pins),
                ipc_gpu::resident::describe_colliders(params));
            if (params.write_substeps) {
                ipc_gpu::resident::copy_snapshot(resident_solver.snapshot(),state);
                if (substep_cb) substep_cb((frame_index-1)*params.substeps+substep,state.deformed_positions);
            }
            result.iterations+=params.max_global_iters;
        }
        if (!params.write_substeps)
            ipc_gpu::resident::copy_snapshot(resident_solver.snapshot(),state);
        resident_solver.end_profile_frame();
        result.converged=true; // completed the requested fixed sweeps
