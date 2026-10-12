        if (use_v2_simd) {
            // All members of the existing cooperative CPU contact team enter
            // here. One submits the color; its implicit barrier publishes all
            // views before helpers resume their unchanged contact/CCD work.
            #pragma omp single
            {
                if (!color_derivative_failed.load(std::memory_order_relaxed)) {
                    try { gpu_prepare_color(color); }
                    catch (...) {
                        if (!color_derivative_failed.exchange(true,std::memory_order_relaxed))
                            color_derivative_error=std::current_exception();
                    }
                }
            }
            return;
        }
