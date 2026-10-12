#pragma once

// Opt-in host/CUDA-stream interval tracing. No geometry, arithmetic, launch
// order, or scheduling policy is changed. Events are collected asynchronously
// and resolved only at substep/frame boundaries, never between kernels.
// CUDA intervals include idle gaps while the host feeds the stream; they are
// NOT kernel-active time. Exclusive intervals subtract direct child spans.
// Host times and frame wall time include profiling overhead; do not add host
// and CUDA time. Set IPC_GPU_PROFILE_CSV to a fresh output path to enable.
#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace ipc_gpu::resident {

enum class ResidentProfileCounter : std::size_t {
    Substeps, Iterations, ColorVisits,
    GuessRebuilds, GuessCandidateSum, GuessCandidateMax, GuessColorSum,
    GuessColorMax, GuessGraphEdgeMax,
    SolverRebuilds, SolverCandidateSum, SolverCandidateMax, SolverColorSum,
    SolverColorMax, SolverGraphEdgeMax,
    GlobalQueryBatches, GlobalQueryJobs, GlobalRetryJobs,
    GlobalQueueCapacityMax, GlobalWorkersMax, Count
};

class ResidentProfiler {
    using Clock=std::chrono::steady_clock;
    struct EventPair {cudaEvent_t begin=nullptr,end=nullptr;};
    struct Interval {
        std::string path;
        std::size_t parent;
        Clock::time_point begin;
        double host_ms=0,child_host_ms=0,child_cuda_ms=0;
        cudaStream_t stream=nullptr;
    };
    struct Total {std::size_t calls=0;double host=0,host_exclusive=0,cuda=0,cuda_exclusive=0;};
    static constexpr std::size_t no_parent=static_cast<std::size_t>(-1);
    std::vector<EventPair> events_;
    std::vector<Interval> intervals_;
    std::vector<std::size_t> stack_;
    std::map<std::string,Total> totals_;
    std::array<unsigned long long,static_cast<std::size_t>(ResidentProfileCounter::Count)> counters_{};
    std::ofstream output_;
    std::string path_,error_;
    Clock::time_point frame_start_;
    double overhead_ms_=0,root_host_ms_=0,root_cuda_ms_=0;
    int frame_=0;
    bool active_=false;
    static double elapsed(Clock::time_point a,Clock::time_point b) {
        return std::chrono::duration<double,std::milli>(b-a).count();
    }
    void status(cudaError_t e) noexcept {
        if(e!=cudaSuccess && error_.empty())error_=cudaGetErrorString(e);
    }
public:
    ResidentProfiler() {
        const char* value=std::getenv("IPC_GPU_PROFILE_CSV");
        if(value&&*value)path_=value;
    }
    ~ResidentProfiler() {
        for(auto& events:events_) {
            if(events.begin)cudaEventDestroy(events.begin);
            if(events.end)cudaEventDestroy(events.end);
        }
    }
    bool enabled()const{return !path_.empty();}
    bool active()const{return active_;}
    void begin_frame(int frame) {
        if(!enabled())return;
        if(active_)throw std::logic_error("GPU profile frame already active");
        if(!output_.is_open()) {
            std::ifstream existing(path_);
            if(existing.good())throw std::runtime_error("GPU profile output already exists: "+path_);
            output_.open(path_);
            if(!output_)throw std::runtime_error("cannot create GPU profile CSV: "+path_);
            output_<<"frame,stage,calls,host_inclusive_ms,host_exclusive_ms,cuda_inclusive_ms,cuda_exclusive_ms,frame_wall_ms,profile_cpu_overhead_ms"
                <<",substeps,iterations,color_visits,guess_broadphase_rebuilds,guess_candidate_sum,guess_candidate_max,guess_color_sum,guess_color_max,guess_graph_edge_max"
                <<",solver_broadphase_rebuilds,solver_candidate_sum,solver_candidate_max,solver_color_sum,solver_color_max,solver_graph_edge_max"
                <<",global_query_batches,global_query_jobs,global_retry_jobs,global_queue_capacity_max,global_workers_max\n";
            output_<<std::setprecision(12);
        }
        frame_=frame;totals_.clear();intervals_.clear();stack_.clear();counters_.fill(0);
        overhead_ms_=root_host_ms_=root_cuda_ms_=0;error_.clear();active_=true;
        frame_start_=Clock::now();
    }
    std::size_t begin(const char* name,cudaStream_t stream) {
        const auto overhead_start=Clock::now();
        const auto index=intervals_.size(),parent=stack_.empty()?no_parent:stack_.back();
        if(index==events_.size()) {
            EventPair pair;
            status(cudaEventCreate(&pair.begin));status(cudaEventCreate(&pair.end));
            events_.push_back(pair);
        }
        Interval interval;
        interval.path=parent==no_parent?name:intervals_[parent].path+"/"+name;
        interval.parent=parent;interval.stream=stream;
        status(cudaEventRecord(events_[index].begin,stream));
        intervals_.push_back(std::move(interval));stack_.push_back(index);
        const auto now=Clock::now();intervals_[index].begin=now;
        overhead_ms_+=elapsed(overhead_start,now);
        return index;
    }
    void end(std::size_t index) noexcept {
        const auto overhead_start=Clock::now();
        if(stack_.empty()||stack_.back()!=index) {
            error_="non-LIFO GPU profile scope";return;
        }
        auto& interval=intervals_[index];
        interval.host_ms=elapsed(interval.begin,overhead_start);
        status(cudaEventRecord(events_[index].end,interval.stream));
        stack_.pop_back();
        overhead_ms_+=elapsed(overhead_start,Clock::now());
    }
    void counter(ResidentProfileCounter counter,unsigned long long value,bool maximum=false) {
        auto& result=counters_[static_cast<std::size_t>(counter)];
        if(maximum)result=std::max(result,value);else result+=value;
    }
    bool in_global_guess()const {
        return !stack_.empty()&&intervals_[stack_.back()].path.find("global_guess")!=std::string::npos;
    }
    // The caller reaches this after the existing substep synchronization. The
    // final event wait accounts only for trailing event commands. Pool reuse
    // bounds storage by one substep rather than all 240 simulation frames.
    void flush() {
        if(!active_||intervals_.empty())return;
        const auto overhead_start=Clock::now();
        if(!stack_.empty())throw std::logic_error("GPU profile flush inside an open scope");
        for(std::size_t i=0;i<intervals_.size();++i)
            if(intervals_[i].parent==no_parent)status(cudaEventSynchronize(events_[i].end));
        for(std::size_t i=intervals_.size();i-- >0;) {
            auto& interval=intervals_[i];
            float milliseconds=0;
            status(cudaEventElapsedTime(&milliseconds,events_[i].begin,events_[i].end));
            auto& total=totals_[interval.path];++total.calls;
            total.host+=interval.host_ms;total.cuda+=milliseconds;
            total.host_exclusive+=interval.host_ms-interval.child_host_ms;
            total.cuda_exclusive+=double(milliseconds)-interval.child_cuda_ms;
            if(interval.parent!=no_parent) {
                auto& parent=intervals_[interval.parent];
                parent.child_host_ms+=interval.host_ms;parent.child_cuda_ms+=milliseconds;
            } else {root_host_ms_+=interval.host_ms;root_cuda_ms_+=milliseconds;}
        }
        intervals_.clear();
        overhead_ms_+=elapsed(overhead_start,Clock::now());
        if(!error_.empty())throw std::runtime_error("GPU profiling failed: "+error_);
    }
    void end_frame() {
        if(!active_)return;
        flush();
        const double wall=elapsed(frame_start_,Clock::now());
        // The frame row's exclusive host time captures driver work and timer
        // bookkeeping outside measured scopes. Its CUDA exclusive time is 0.
        totals_["frame"]={1,wall,wall-root_host_ms_,root_cuda_ms_,0};
        for(const auto& item:totals_) {
            const auto& total=item.second;
            output_<<frame_<<','<<item.first<<','<<total.calls<<','<<total.host<<','<<total.host_exclusive
                <<','<<total.cuda<<','<<total.cuda_exclusive<<','<<wall<<','<<overhead_ms_;
            for(const auto value:counters_)output_<<','<<value;
            output_<<'\n';
        }
        output_.flush();active_=false;
        if(!output_)throw std::runtime_error("failed writing GPU profile CSV: "+path_);
    }
    void abandon_frame()noexcept{active_=false;stack_.clear();intervals_.clear();}
};

inline ResidentProfiler& resident_profiler(){static ResidentProfiler profiler;return profiler;}
inline thread_local ResidentProfiler* active_resident_profiler=nullptr;

class ResidentProfileScope {
    ResidentProfiler* profiler_;
    std::size_t index_=0;
public:
    explicit ResidentProfileScope(const char* name,cudaStream_t stream=nullptr)
        :profiler_(active_resident_profiler) {
        if(profiler_)index_=profiler_->begin(name,stream);
    }
    ~ResidentProfileScope(){if(profiler_)profiler_->end(index_);}
    ResidentProfileScope(const ResidentProfileScope&)=delete;
    ResidentProfileScope& operator=(const ResidentProfileScope&)=delete;
};
inline void resident_profile_counter(ResidentProfileCounter counter,unsigned long long value=1,bool maximum=false) {
    if(active_resident_profiler)active_resident_profiler->counter(counter,value,maximum);
}
inline void resident_profile_broadphase(std::size_t candidates,int colors,std::size_t edges) {
    auto* profiler=active_resident_profiler;if(!profiler)return;
    const bool guess=profiler->in_global_guess();
    profiler->counter(guess?ResidentProfileCounter::GuessRebuilds:ResidentProfileCounter::SolverRebuilds,1);
    profiler->counter(guess?ResidentProfileCounter::GuessCandidateSum:ResidentProfileCounter::SolverCandidateSum,candidates);
    profiler->counter(guess?ResidentProfileCounter::GuessCandidateMax:ResidentProfileCounter::SolverCandidateMax,candidates,true);
    profiler->counter(guess?ResidentProfileCounter::GuessColorSum:ResidentProfileCounter::SolverColorSum,colors);
    profiler->counter(guess?ResidentProfileCounter::GuessColorMax:ResidentProfileCounter::SolverColorMax,colors,true);
    profiler->counter(guess?ResidentProfileCounter::GuessGraphEdgeMax:ResidentProfileCounter::SolverGraphEdgeMax,edges,true);
}
} // namespace ipc_gpu::resident
