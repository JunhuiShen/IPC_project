    // Explicitly reject unsupported GPU modes rather than silently run CPU.
    if (params.use_cloth_grid || params.use_ogc || params.use_ogc_solver
        || !params.use_parallel || !params.use_basic_experimental_v2 || !params.use_simd) {
        std::cerr << "GPU prototype requires conflict coloring, --use_parallel true "
                     "--use_basic_experimental true --use_simd true; grid/OGC are not ported.\n";
        return 2;
    }
    std::cout << "CUDA cloth/solid/rigid derivatives: " << ipc_gpu::device_name() << " (double precision)\n"
              << "GPU: membrane, bending, tetrahedral elasticity, point terms, rigid inertia and contact pullback; "
                 "CPU: gather, ordered accumulation, contact geometry, deformable contact/friction, "
                 "SDF, CCD, coloring, small solves and integration\n";
