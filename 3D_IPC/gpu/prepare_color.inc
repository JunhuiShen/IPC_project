    static ipc_gpu::ColorBridge gpu_bridge;
    const auto gpu_prepare_color = [&](std::size_t color) {
        const auto& group=color_groups[color];
        gpu_bridge.prepare(group, ref_mesh, simd_materials, xnew, xhat,
            pins, pm, params, vertex_point_simd, vertex_elasticity_simd, vertex_bending);
        // SDF/friction preparation remains on the CPU in this stage.
        if (use_sdf_simd) {
            for (std::size_t offset=0; offset<group.size(); offset+=ipc_simd::tile_width) {
                const auto count=std::min(ipc_simd::tile_width,group.size()-offset);
                std::array<Vec3,ipc_simd::tile_width> positions, previous;
                std::array<physics_detail::SdfDerivatives,ipc_simd::tile_width> outputs;
                for (std::size_t e=0; e<count; ++e) {
                    const int vi=group[offset+e]; positions[e]=xnew[vi];
                    if (params.friction_coefficient>0) previous[e]=(*previous_positions)[vi];
                }
                physics_detail::compute_sdf_derivatives_tile(params,positions.data(),
                    params.friction_coefficient>0 ? previous.data() : nullptr,count,outputs.data());
                for (std::size_t e=0; e<count; ++e) vertex_sdf[group[offset+e]]=outputs[e];
            }
        }
    };
