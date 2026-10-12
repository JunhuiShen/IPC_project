        // GPU-only overlay: each color is an independent gather/compute/update
        // transaction. No next-color gather occurs before current updates join.
        if (use_v2_simd && !use_contact_sweep) {
            for (std::size_t color=0; color<color_groups.size(); ++color) {
                const auto& group=color_groups[color];
                gpu_prepare_color(color);
                for (int vi:group) color_rollback[vi]=xnew[vi];
                #pragma omp parallel for schedule(dynamic,1)
                for (std::size_t i=0; i<group.size(); ++i) {
                    if (color_derivative_failed.load(std::memory_order_relaxed)) continue;
                    try { process_vertex(group[i],false,true); }
                    catch (...) {
                        if (!color_derivative_failed.exchange(true,std::memory_order_relaxed))
                            color_derivative_error=std::current_exception();
                    }
                }
                if (color_derivative_failed.load(std::memory_order_relaxed)) {
                    for (int vi:group) xnew[vi]=color_rollback[vi];
                    std::rethrow_exception(color_derivative_error);
                }
            }
        } else if (use_contact_sweep) {
