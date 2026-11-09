#pragma once

#include <cstddef>
#include <limits>

namespace ipc_gpu::solver::node_controller {

constexpr unsigned width=128;
enum class Ineligible {
    None, Empty, Unsupported, NoResidency, Overflow, WorkerCount, NodesNotCovered
};
struct Plan {
    unsigned blocks=0;
    Ineligible reason=Ineligible::Empty;
    constexpr bool eligible() const {return blocks!=0&&reason==Ineligible::None;}
};

// Host metadata only. No numerical state, row counts or errors are read back.
// Every old per-node stage still uses its original one-thread-per-node index.
constexpr Plan make_plan(std::size_t nodes,int workers,bool cooperative,
    int multiprocessors,int blocks_per_multiprocessor) {
    if(!nodes)return {0,Ineligible::Empty};
    if(!cooperative)return {0,Ineligible::Unsupported};
    if(multiprocessors<=0||blocks_per_multiprocessor<=0)return {0,Ineligible::NoResidency};
    const auto sms=static_cast<std::size_t>(multiprocessors);
    const auto per_sm=static_cast<std::size_t>(blocks_per_multiprocessor);
    if(sms>std::numeric_limits<std::size_t>::max()/per_sm)return {0,Ineligible::Overflow};
    const auto resident=sms*per_sm;
    if(resident>std::numeric_limits<unsigned>::max()
        ||resident>std::numeric_limits<std::size_t>::max()/width)return {0,Ineligible::Overflow};
    // One lane0 of block=worker owns the same exact arena as the old worker.
    if(workers<=0||static_cast<std::size_t>(workers)>resident)return {0,Ineligible::WorkerCount};
    if(nodes>resident*width)return {0,Ineligible::NodesNotCovered};
    return {static_cast<unsigned>(resident),Ineligible::None};
}

#if defined(__CUDACC__)
#define IPC_NODE_CONTROLLER_DEVICE __device__
#else
#define IPC_NODE_CONTROLLER_DEVICE
#endif

// The exact same controller is exercised by the host stage-order oracle.
// In CUDA it is device-only: Adapter owns a concrete grid_group and device
// stage methods. Early returns are confined to those methods, never this
// controller/kernel, so all grid threads reach every join.
template<class Adapter,class Control>
IPC_NODE_CONTROLLER_DEVICE inline void execute(Adapter& stage,Control* control) {
    stage.begin();stage.join();
    stage.prepare();stage.join();
    stage.fast();stage.join();
    stage.exact();stage.join();
    stage.gate();stage.join();
    stage.consume(-1);stage.join();
    for(int attempt=0;attempt<=64;++attempt) {
        // Following the preceding join, these flags are identical/stable for
        // every thread until the next join: prepare_trial changes neither.
        if(control->mode!=1||control->done)break;
        stage.prepare_trial(attempt);stage.join();
        stage.fast();stage.join();
        stage.exact();stage.join();
        stage.gate();stage.join();
        stage.consume(attempt);stage.join();
    }
}

#undef IPC_NODE_CONTROLLER_DEVICE

} // namespace ipc_gpu::solver::node_controller
