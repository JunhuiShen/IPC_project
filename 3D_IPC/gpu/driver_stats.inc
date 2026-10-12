    const auto gpu_stats=ipc_gpu::statistics();
    std::cout << "CUDA calls=" << gpu_stats.calls << " launches=" << gpu_stats.launches
              << " point_records=" << gpu_stats.points << " triangle_records=" << gpu_stats.triangles
              << " hinge_records=" << gpu_stats.hinges
              << " solid_records=" << gpu_stats.solids
              << " rigid_contact_records=" << gpu_stats.rigid_contacts
              << " rigid_inertia_records=" << gpu_stats.rigid_inertia << "\n"
              << "CUDA event time: upload=" << gpu_stats.host_to_device_ms
              << " ms compute=" << gpu_stats.kernel_ms << " ms download=" << gpu_stats.device_to_host_ms << " ms\n";
