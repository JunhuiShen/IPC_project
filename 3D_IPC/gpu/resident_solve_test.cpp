#include "IPC_math.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
struct SolveRecord {double h[9],g[3];};
struct SolveResult {double x[3];bool valid;};
void resident_test_solves(const SolveRecord*,SolveResult*,std::size_t);
int main()try{
    std::vector<SolveRecord> input;
    std::vector<Vec3> references;
    for(int sample=0;sample<100;++sample)for(int rank=0;rank<=3;++rank){
        Mat33 a=Mat33::Zero();
        for(int r=0;r<3;++r)for(int c=0;c<rank;++c)a(r,c)=std::sin((sample+1)*.37+(r+1)*(c+2)*.71);
        Mat33 h=a*a.transpose();
        if(rank==3)h+=.001*Mat33::Identity();
        Vec3 g(.13+.01*sample,-.21,.04);
        SolveRecord record{};for(int r=0;r<3;++r){record.g[r]=g[r];for(int c=0;c<3;++c)record.h[3*r+c]=h(r,c);}
        input.push_back(record);references.push_back(matrix3d_inverse(h)*g);
    }
    // Rank-preserving scale extremes and a full-rank fallback matrix.
    for(double scale:{1e-200,1e-100,1.,1e100,1e200}){
        Mat33 h=Mat33::Zero();h(0,0)=scale;h(1,1)=scale*1e-9;h(2,2)=scale*1e-10;
        Vec3 g=scale*Vec3(.1,.2,-.3);SolveRecord r{};
        for(int a=0;a<3;++a){r.g[a]=g[a];for(int b=0;b<3;++b)r.h[3*a+b]=h(a,b);}
        input.push_back(r);references.push_back(matrix3d_inverse(h)*g);
    }
    std::vector<SolveResult> output(input.size());resident_test_solves(input.data(),output.data(),input.size());
    double worst=0;
    for(std::size_t i=0;i<input.size();++i){
        if(!output[i].valid)throw std::runtime_error("GPU solve failed");
        Vec3 got(output[i].x[0],output[i].x[1],output[i].x[2]);
        if(!got.allFinite()||!references[i].allFinite())throw std::runtime_error("Nonfinite solve reference/result");
        const double error=(got-references[i]).norm()/(1+references[i].norm());worst=std::max(worst,error);
        if(error>1e-8)throw std::runtime_error("GPU minimum-norm solve mismatch at "+std::to_string(i)+" error="+std::to_string(error));
    }
    std::cout<<"PASS GPU node solves: "<<input.size()<<" regular/rank-deficient/scaled systems, worst relative error="<<worst<<'\n';
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
