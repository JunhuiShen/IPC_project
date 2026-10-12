        if (params.use_parallel) {
            const int sweeps = params.fixed_iters
                ? std::min(params.max_global_iters - iteration + 1,
                    params.node_box_update_count - (iteration - 1) % params.node_box_update_count) : 1;
            // Reuse the existing contact helper/team scheduler and rollback
            // policy one complete color at a time. This intentionally retains
            // an OpenMP team launch per color in this correctness-first stage.
            solver_detail::GeneralSimdBatches gpu_color_batches;
            gpu_color_batches.batches = simd_batches.batches;
            gpu_color_batches.colors.resize(1);
            for (int sweep = 0; sweep < sweeps; ++sweep) {
                for (std::size_t color = 0; color < simd_batches.colors.size(); ++color) {
                    gpu_general.prepare_color(mixed_adjacency_workspace.color_groups[color],
                        cloth_nodes, solid_nodes, solid_begin, rigid_begin, ref_mesh,
                        simd_materials, pins, pin_map, params, xnew, xhat);
                    gpu_color_batches.colors[0] = simd_batches.colors[color];
                    solver_detail::run_general_simd_batches(gpu_color_batches, 1,
                        [&](int item, bool cooperative) { process_simd_batch(simd_batches.batches[item], cooperative); },
                        [&](int item) { snapshot_simd_batch(item, false); },
                        [&](int item) { snapshot_simd_batch(item, true); });
                }
            }
            iteration += sweeps - 1;
        } else {
            // Serial GS preserves [cloth][solid][rigid] order. Each deformable
            // block is its own GPU transaction, so dependent nodes are never
            // gathered before earlier updates have committed.
            for (int block = 0; block < rigid_begin + num_rbs; ++block) {
                solver_detail::GeneralSimdBatch batch;
                batch.blocks[0] = block; batch.size = 1;
                gpu_general.prepare_color({block}, cloth_nodes, solid_nodes,
                    solid_begin, rigid_begin, ref_mesh, simd_materials, pins,
                    pin_map, params, xnew, xhat);
                process_simd_batch(batch, false);
            }
        }
