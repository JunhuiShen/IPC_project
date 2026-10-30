#include "elastic.h"
#include "test_support.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ipc_gpu;
using namespace ipc_gpu::solver;
using Vec=std::array<double,3>;
static_assert(sizeof(Vec)==3*sizeof(double));
static void require(bool value,const std::string& label) {if(!value)throw std::runtime_error(label);}
static double (*triples(Vec* p))[3] {return reinterpret_cast<double(*)[3]>(p);}
template<class T>static void upload(GpuTestBuffer<T>& buffer,const std::vector<T>& values,cudaStream_t stream) {
    if(!values.empty())gpu_cuda_check(cudaMemcpyAsync(buffer.data(),values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice,stream));
    // The temporary host source may be destroyed after return. This is test
    // setup only, not part of the production worklist path.
    gpu_cuda_check(cudaStreamSynchronize(stream));
}
template<class T>static void equal(const GpuTestBuffer<T>& a,const GpuTestBuffer<T>& b,const std::string& label) {
    const auto x=a.download(),y=b.download();
    require(x.size()==y.size(),label+" size");
    if(!x.empty()&&std::memcmp(x.data(),y.data(),x.size()*sizeof(T)))throw std::runtime_error(label+" is not byte-identical");
}
static void equal_hinges(const GpuTestBuffer<Hinge>& a,const GpuTestBuffer<Hinge>& b) {
    const auto x=a.download(),y=b.download();
    require(x.size()==y.size(),"hinge gather size");
    // Compare every defined field, not the ABI tail padding after active_node.
    for(std::size_t i=0;i<x.size();++i) {
        require(std::memcmp(x[i].positions,y[i].positions,sizeof(x[i].positions))==0,"hinge gathered positions");
        require(std::memcmp(&x[i].coefficient,&y[i].coefficient,sizeof(double))==0,"hinge coefficient");
        require(std::memcmp(&x[i].rest_angle,&y[i].rest_angle,sizeof(double))==0,"hinge rest angle");
        require(x[i].active_node==y[i].active_node,"hinge active node");
    }
}
struct Stream {
    cudaStream_t value=nullptr;
    Stream(){gpu_cuda_check(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking));}
    ~Stream(){cudaStreamDestroy(value);}
    void wait()const{gpu_cuda_check(cudaStreamSynchronize(value));}
};
struct Fixture {
    // Four cloth nodes, four solid nodes, a rigid sentinel, a free point and a
    // negative-color fixed point. Cloth and solid share colors but no element.
    std::vector<Vec> x{{0,0,0},{1.01,0,0},{0,.99,.02},{1,1,.01},
        {2,0,0},{3.01,0,0},{2,1.02,0},{2,0,.99},{7,8,9},{-2,3,.1},{9,4,3}};
    std::vector<int> blocks{0,1,2,3,4,5,6,7,8,9,10};
    std::vector<int> colors{0,1,2,3,0,1,2,3,4,1,-1};
    std::vector<unsigned char> deformable{1,1,1,1,1,1,1,1,0,1,1};
    std::vector<unsigned char> solid{0,0,0,0,1,1,1,1,0,0,0};
    std::vector<unsigned char> pinned{1,0,0,0,0,1,0,0,0,0,0};
    std::vector<TriangleIncident> triangles;
    std::vector<SolidIncident> solids;
    std::vector<HingeIncident> hinges;
    std::vector<std::size_t> to,so,ho;
    void topology(int repeat) {
        triangles.clear();solids.clear();hinges.clear();
        to.assign(x.size()+1,0);so=to;ho=to;
        const int faces[2][3]={{0,1,2},{1,3,2}};
        for(std::size_t n=0;n<x.size();++n) {
            to[n]=triangles.size();so[n]=solids.size();ho[n]=hinges.size();
            for(int copy=0;copy<repeat;++copy) {
                for(int f=0;f<2;++f)for(int role=0;role<3;++role)if(faces[f][role]==int(n)) {
                    TriangleIncident t{};t.owner=static_cast<int>(n);
                    std::copy(faces[f],faces[f]+3,t.nodes);
                    t.material.dm_inverse[0]=1;t.material.dm_inverse[3]=1;t.material.area=.5;
                    t.material.shape[0]=role==0?-1:(role==1?1:0);
                    t.material.shape[1]=role==0?-1:(role==2?1:0);
                    triangles.push_back(t);
                }
                if(n<4) {
                    HingeIncident h{};h.owner=static_cast<int>(n);
                    const int corners[4]={1,2,0,3};
                    for(int a=0;a<4;++a){h.nodes[a]=corners[a];if(corners[a]==int(n))h.material.active_node=a;}
                    h.material.coefficient=1;hinges.push_back(h);
                }
                if(n>=4&&n<8) {
                    SolidIncident s{};s.owner=static_cast<int>(n);
                    for(int a=0;a<4;++a)s.nodes[a]=4+a;
                    for(int a=0;a<3;++a){s.material.dm_inverse[3*a+a]=1;s.material.shape[a]=n==4?-1:(n==std::size_t(5+a)?1:0);}
                    s.material.measure=1.0/6.0;solids.push_back(s);
                }
            }
        }
        to.back()=triangles.size();so.back()=solids.size();ho.back()=hinges.size();
    }
};
struct DeviceScene {
    GpuTestBuffer<int> blocks,colors;
    GpuTestBuffer<unsigned char> deformable,solid,pinned;
    GpuTestBuffer<double> masses,fractions;
    GpuTestBuffer<Vec> targets;
    GpuTestBuffer<TriangleIncident> triangles;
    GpuTestBuffer<SolidIncident> solids;
    GpuTestBuffer<HingeIncident> hinges;
    GpuTestBuffer<std::size_t> to,so,ho;
    DeformableScene view{};
    explicit DeviceScene(const Fixture& f):blocks(f.blocks),colors(f.colors),deformable(f.deformable),solid(f.solid),pinned(f.pinned),
        masses(std::vector<double>(f.x.size(),1)),fractions(std::vector<double>(f.x.size(),.9)),targets(f.x),
        triangles(f.triangles),solids(f.solids),hinges(f.hinges),to(f.to),so(f.so),ho(f.ho) {
        view={f.x.size(),f.triangles.size(),f.solids.size(),f.hinges.size(),blocks.data(),deformable.data(),solid.data(),pinned.data(),masses.data(),triples(targets.data()),triangles.data(),solids.data(),hinges.data(),to.data(),so.data(),ho.data()};
    }
    void update(const Fixture& f,cudaStream_t stream) {
        upload(blocks,f.blocks,stream);upload(colors,f.colors,stream);
        upload(triangles,f.triangles,stream);upload(solids,f.solids,stream);upload(hinges,f.hinges,stream);
        upload(to,f.to,stream);upload(so,f.so,stream);upload(ho,f.ho,stream);
        view.triangle_count=f.triangles.size();view.solid_count=f.solids.size();view.hinge_count=f.hinges.size();
    }
};
struct Pipeline {
    GpuTestBuffer<Vec> x,v,previous,predicted,proposed;
    GpuTestBuffer<Point> points;
    GpuTestBuffer<Triangle> triangles;
    GpuTestBuffer<Solid> solids;
    GpuTestBuffer<Hinge> hinges;
    GpuTestBuffer<Derivatives> point_out,triangle_out,solid_out,hinge_out,systems;
    GpuTestBuffer<int> error;
    DeformableState state{};ElasticWorkspace work{};
    explicit Pipeline(const Fixture& f):x(f.x),v(std::vector<Vec>(f.x.size(),Vec{.01,-.03,.02})),
        previous(std::vector<Vec>(f.x.size())),predicted(std::vector<Vec>(f.x.size())),proposed(std::vector<Vec>(f.x.size())),
        points(std::vector<Point>(f.x.size())),triangles(std::vector<Triangle>(f.triangles.size())),
        solids(std::vector<Solid>(f.solids.size())),hinges(std::vector<Hinge>(f.hinges.size())),
        point_out(std::vector<Derivatives>(f.x.size())),triangle_out(std::vector<Derivatives>(f.triangles.size())),
        solid_out(std::vector<Derivatives>(f.solids.size())),hinge_out(std::vector<Derivatives>(f.hinges.size())),
        systems(std::vector<Derivatives>(f.x.size())),error(std::vector<int>{0}) {
        state={triples(x.data()),triples(v.data()),triples(previous.data()),triples(predicted.data())};
        work={points.data(),triangles.data(),solids.data(),hinges.data(),point_out.data(),triangle_out.data(),solid_out.data(),hinge_out.data(),systems.data(),triples(proposed.data()),error.data()};
    }
    void compare(const Pipeline& other,const std::string& phase)const {
        equal(x,other.x,phase+" positions");equal(v,other.v,phase+" velocities");
        equal(previous,other.previous,phase+" previous");equal(predicted,other.predicted,phase+" predicted");
        equal(proposed,other.proposed,phase+" proposed");equal(points,other.points,phase+" points");
        equal(triangles,other.triangles,phase+" triangles");equal(solids,other.solids,phase+" solids");equal_hinges(hinges,other.hinges);
        equal(point_out,other.point_out,phase+" point derivatives");equal(triangle_out,other.triangle_out,phase+" triangle derivatives");
        equal(solid_out,other.solid_out,phase+" solid derivatives");equal(hinge_out,other.hinge_out,phase+" hinge derivatives");
        equal(systems,other.systems,phase+" reduced systems");equal(error,other.error,phase+" error status");
    }
};
static std::vector<std::size_t> read_indices(ElasticIndexRange range) {
    std::vector<std::size_t> result(range.count);
    if(range.count)gpu_cuda_check(cudaMemcpy(result.data(),range.indices,range.count*sizeof(std::size_t),cudaMemcpyDeviceToHost));
    return result;
}
static void check_lists(const Fixture& f,const GpuElasticWorklists& lists,int color_count) {
    auto selected=[&](std::size_t node,int color){return f.deformable.at(node)&&f.colors.at(f.blocks.at(node))==color;};
    for(int color=0;color<color_count;++color) {
        const auto view=lists.color(color);require(view.enabled,"missing enabled list");
        std::array<std::vector<std::size_t>,4> expected;
        for(std::size_t n=0;n<f.x.size();++n)if(selected(n,color))expected[0].push_back(n);
        for(std::size_t i=0;i<f.triangles.size();++i)if(selected(f.triangles[i].owner,color))expected[1].push_back(i);
        for(std::size_t i=0;i<f.solids.size();++i)if(selected(f.solids[i].owner,color))expected[2].push_back(i);
        for(std::size_t i=0;i<f.hinges.size();++i)if(selected(f.hinges[i].owner,color))expected[3].push_back(i);
        const ElasticIndexRange ranges[]={view.nodes,view.triangles,view.solids,view.hinges};
        for(int kind=0;kind<4;++kind)require(read_indices(ranges[kind])==expected[kind],"unstable or incomplete worklist, color "+std::to_string(color)+" kind "+std::to_string(kind));
    }
}
template<class F>static void must_throw(F f,const char* label) {
    bool threw=false;try{f();}catch(const std::exception&){threw=true;}require(threw,label);
}
static void check_configuration() {
    unsetenv("IPC_GPU_ELASTIC_COLOR_WORKLISTS");unsetenv("IPC_GPU_ELASTIC_NODE_THREADS");unsetenv("IPC_GPU_ELASTIC_ELEMENT_THREADS");
    {GpuElasticWorklists disabled;require(!disabled.enabled(),"worklists must default off");disabled.rebuild({},nullptr,0,0);require(!disabled.color(0).enabled,"disabled view");}
    for(const char* value:{"","yes","2","-1"}){setenv("IPC_GPU_ELASTIC_COLOR_WORKLISTS",value,1);must_throw([]{GpuElasticWorklists invalid;},"invalid enable flag accepted");}
    setenv("IPC_GPU_ELASTIC_COLOR_WORKLISTS","1",1);
    for(const char* name:{"IPC_GPU_ELASTIC_NODE_THREADS","IPC_GPU_ELASTIC_ELEMENT_THREADS"}) {
        for(const char* value:{"","0","3","256","-1"," 8","8x"}){setenv(name,value,1);must_throw([]{GpuElasticWorklists invalid;},"invalid thread width accepted");}
        unsetenv(name);
    }
}
static void lifecycle(unsigned nt,unsigned et) {
    setenv("IPC_GPU_ELASTIC_NODE_THREADS",std::to_string(nt).c_str(),1);
    setenv("IPC_GPU_ELASTIC_ELEMENT_THREADS",std::to_string(et).c_str(),1);
    Fixture f;f.topology(17);DeviceScene d(f);Pipeline scan(f),compact(f);Stream stream;
    GpuElasticWorklists lists;must_throw([&]{lists.color(0);},"unbuilt lists accepted");
    Parameters p{};p.gravity[1]=-9.81;p.kpin=100;p.dt2=.0001;p.mu=2;p.lambda=3;p.bending_stiffness=.001;
    SolidParameters sp{2,3};
    const int repeats[]={1,9,0,2,17,0,3};
    for(int phase=0;phase<7;++phase) {
        f.topology(repeats[phase]);
        // Preserve valid coloring: each four-node component has distinct
        // colors, independent components share them. Recolor in-place and
        // remap owner blocks, exercising invalidated metadata at the same GPU
        // addresses. Color4 is rigid-only, color5 is truly empty.
        std::vector<int> per_node{0,1,2,3,0,1,2,3,4,1,-1};
        for(int& color:per_node)if(color>=0&&color<4)color=(color+phase)%4;
        for(std::size_t n=0;n<f.x.size();++n)f.blocks[n]=static_cast<int>((n+phase)%f.x.size());
        for(std::size_t n=0;n<f.x.size();++n)f.colors[f.blocks[n]]=per_node[n];
        d.update(f,stream.value);lists.rebuild(d.view,d.colors.data(),static_cast<int>(f.x.size()),6,stream.value);
        check_lists(f,lists,6);
        require(lists.color(0).node_threads==nt&&lists.color(0).element_threads==et,"launch width configuration");
        require(lists.color(4).nodes.count==0&&lists.color(5).nodes.count==0,"rigid/empty color must have no deformable work");
        begin_deformable_substep(d.view,scan.state,.01,stream.value);begin_deformable_substep(d.view,compact.state,.01,stream.value);
        for(int sweep=0;sweep<2;++sweep)for(int color=0;color<6;++color) {
            const auto view=lists.color(color);p.bending_stiffness=(phase%2)?.0:.001;
            assemble_elastic_color(d.view,scan.state,scan.work,d.colors.data(),color,p,sp,stream.value);
            assemble_elastic_color(d.view,compact.state,compact.work,d.colors.data(),color,p,sp,stream.value,view);
            stream.wait();scan.compare(compact,"assembly");
            propose_deformable_color(d.view,scan.state,scan.work,d.colors.data(),color,.7,stream.value);
            propose_deformable_color(d.view,compact.state,compact.work,d.colors.data(),color,.7,stream.value,view);
            stream.wait();scan.compare(compact,"proposal");
            if(sweep==0) {
                commit_deformable_color(d.view,scan.state,scan.work,d.colors.data(),color,d.fractions.data(),stream.value);
                commit_deformable_color(d.view,compact.state,compact.work,d.colors.data(),color,d.fractions.data(),stream.value,view);
            } else {
                commit_deformable_endpoints(d.view,scan.state,scan.work,d.colors.data(),color,triples(scan.proposed.data()),stream.value);
                commit_deformable_endpoints(d.view,compact.state,compact.work,d.colors.data(),color,triples(compact.proposed.data()),stream.value,view);
            }
            stream.wait();scan.compare(compact,"commit");require(scan.error.download()[0]==0,"unexpected numerical error");
        }
        finish_deformable_substep(d.view,scan.state,.01,stream.value);finish_deformable_substep(d.view,compact.state,.01,stream.value);
        stream.wait();scan.compare(compact,"integration");
        const auto result=compact.x.download();require(result[8]==f.x[8],"rigid sentinel changed");require(result[10]==f.x[10],"negative-color fixed point changed");
    }
    // Existing failure semantics remain fail-closed: no commit after an
    // invalid fraction, a nonfinite endpoint, or a previous solver error.
    const int color=0;const auto view=lists.color(color);const auto ids=read_indices(view.nodes);
    require(!ids.empty(),"failure fixture needs active node");const auto before=compact.x.download();
    std::vector<double> fractions(f.x.size(),.9);fractions[ids.front()]=-1;
    upload(d.fractions,fractions,stream.value);
    commit_deformable_color(d.view,scan.state,scan.work,d.colors.data(),color,d.fractions.data(),stream.value);
    commit_deformable_color(d.view,compact.state,compact.work,d.colors.data(),color,d.fractions.data(),stream.value,view);
    stream.wait();scan.compare(compact,"invalid fraction");require(compact.x.download()==before&&compact.error.download()[0]==2,"invalid fraction committed");
    upload(scan.error,std::vector<int>{0},stream.value);upload(compact.error,std::vector<int>{0},stream.value);
    auto endpoints=before;endpoints[ids.front()][0]=std::numeric_limits<double>::quiet_NaN();GpuTestBuffer<Vec> accepted(endpoints);
    commit_deformable_endpoints(d.view,scan.state,scan.work,d.colors.data(),color,triples(accepted.data()),stream.value);
    commit_deformable_endpoints(d.view,compact.state,compact.work,d.colors.data(),color,triples(accepted.data()),stream.value,view);
    stream.wait();scan.compare(compact,"invalid endpoint");require(compact.x.download()==before&&compact.error.download()[0]==2,"invalid endpoint committed");
    propose_deformable_color(d.view,scan.state,scan.work,d.colors.data(),color,.7,stream.value);
    propose_deformable_color(d.view,compact.state,compact.work,d.colors.data(),color,.7,stream.value,view);
    stream.wait();scan.compare(compact,"prior error");
    must_throw([&]{propose_deformable_color(d.view,compact.state,compact.work,d.colors.data(),1,.7,stream.value,view);},"wrong-color view accepted");
    must_throw([&]{lists.color(-1);},"negative color accepted");must_throw([&]{lists.color(6);},"out-of-range color accepted");
    // Invalid owner, block and color each fail rebuild, and invalidate old
    // metadata instead of exposing a partial list. Subsequent valid rebuilds
    // must recover with the same capacity allocations.
    auto original=f.triangles;f.triangles[0].owner=-1;d.update(f,stream.value);
    must_throw([&]{lists.rebuild(d.view,d.colors.data(),11,6,stream.value);},"invalid owner accepted");
    must_throw([&]{lists.color(0);},"failed rebuild left metadata usable");f.triangles=original;
    const auto original_blocks=f.blocks;f.blocks[0]=11;d.update(f,stream.value);
    must_throw([&]{lists.rebuild(d.view,d.colors.data(),11,6,stream.value);},"invalid block accepted");f.blocks=original_blocks;
    const auto original_colors=f.colors;f.colors[f.blocks[0]]=6;d.update(f,stream.value);
    must_throw([&]{lists.rebuild(d.view,d.colors.data(),11,6,stream.value);},"invalid color accepted");f.colors=original_colors;
    d.update(f,stream.value);lists.rebuild(d.view,d.colors.data(),11,6,stream.value);check_lists(f,lists,6);
    DeformableScene empty{};lists.rebuild(empty,nullptr,0,0,stream.value);
    must_throw([&]{lists.color(0);},"zero-color scene exposed color");
    lists.rebuild(d.view,d.colors.data(),11,6,stream.value);check_lists(f,lists,6);
}
int main()try {
    check_configuration();lifecycle(1,8);lifecycle(32,64);lifecycle(128,1);
    unsetenv("IPC_GPU_ELASTIC_COLOR_WORKLISTS");unsetenv("IPC_GPU_ELASTIC_NODE_THREADS");unsetenv("IPC_GPU_ELASTIC_ELEMENT_THREADS");
    std::cout<<"PASS elastic scan/worklist byte identity: mixed cloth/solid/rigid, nondefault stream, stable IDs, 7 recolor/shrink/zero/grow rebuilds, empty/rigid/negative colors, bending on/off, both commits, failure checks, launch widths 1:8/32:64/128:1.\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
