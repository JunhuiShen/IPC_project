#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <filesystem>

// ======================================================
// Prelim stuff
// ======================================================

typedef std::vector<double> Vec;
namespace fs = std::__fs::filesystem;

struct Vec2 {
    double x, y;
};

struct Mat2 {
    double a11, a12, a21, a22;
};

Vec2 getXi(const Vec &x, int i) {
    return {x[2*i], x[2*i+1]};
}

void setXi(Vec &x, int i, const Vec2 &v) {
    x[2*i] = v.x; x[2*i+1] = v.y;
}

double norm2(const Vec &a, int i, int j) {
    double dx = a[2*i]   - a[2*j];
    double dy = a[2*i+1] - a[2*j+1];
    return dx*dx + dy*dy;
}

double norm(const Vec &a, int i, int j){
    return std::sqrt(norm2(a,i,j));
}

// Local spring energy
double springEnergy(const Vec &x, int i, int j, double k, double L) {
    double ell = norm(x,i,j);
    return 0.5 * k / L * (ell - L) * (ell - L);
}

// Total spring energy
double totalSpringEnergy(const Vec &x, double k, const std::vector<double> &L) {
    return springEnergy(x,0,1,k,L[0]) +
           springEnergy(x,1,2,k,L[1]) +
           springEnergy(x,2,3,k,L[2]);
}

// ======================================================
// Local Gradient
// ======================================================

// Analytic local spring gradient at node i
Vec2 localSpringGrad(int i, const Vec &x, double k, const std::vector<double> &L) {
    Vec2 g_i{0.0, 0.0};

    std::function<void(int,int,double)> contrib = [&](int a, int b, double Lref) {
        Vec2 xa = getXi(x,a), xb = getXi(x,b);
        double dx = xb.x - xa.x, dy = xb.y - xa.y;
        double ell = std::sqrt(dx*dx + dy*dy);
        if (ell < 1e-12) return;

        double coeff = k/Lref * (ell - Lref)/ell;
        double sgn = (i == b) ? +1.0 : -1.0;

        g_i.x += sgn * coeff * dx;
        g_i.y += sgn * coeff * dy;
    };

    int N = (int)L.size() + 1;
    if (i-1 >= 0)   contrib(i-1,i,L[i-1]);
    if (i+1 <= N-1) contrib(i,i+1,L[i]);

    return g_i;
}

// ======================================================
// Local Hessian
// ======================================================

// Analytic local spring Hessian (2x2 block at node i)
Mat2 localSpringHess(int i, const Vec &x, double k, const std::vector<double> &L) {
    Mat2 H_ii{0.0, 0.0, 0.0, 0.0};

    std::function<void(int,int,double)> contrib = [&](int a, int b, double Lref) {
        Vec2 xa = getXi(x,a), xb = getXi(x,b);
        double dx = xb.x - xa.x, dy = xb.y - xa.y;
        double ell = std::sqrt(dx*dx + dy*dy);
        if (ell < 1e-12) return;

        // Derivatives from d/dx of k/L * (ell - L) * (dx/ell, dy/ell)
        // Arrange as: coeff1 * I + coeff2 * [dx;dy][dx dy]
        double coeff1 = k/Lref * (ell - Lref)/ell;          // from (ell-L)/ell term
        double coeff2 = k/Lref * (Lref) / (ell*ell*ell);    // from derivative of ell

        // K_j = [Kxx, Kxy; Kxy; Kyy]
        double Kxx = coeff1 + coeff2*dx*dx;
        double Kyy = coeff1 + coeff2*dy*dy;
        double Kxy = coeff2*dx*dy;

        H_ii.a11 += Kxx; H_ii.a12 += Kxy;
        H_ii.a21 += Kxy; H_ii.a22 += Kyy;
    };

    int N = (int)L.size() + 1;
    if (i-1 >= 0)   contrib(i-1,i,L[i-1]);
    if (i+1 <= N-1) contrib(i,i+1,L[i]);

    return H_ii;
}

// ======================================================
// Barrier term (node-segment)
// ======================================================
struct BarrierPair {
    int node;   // i
    int seg0;   // j
    int seg1;   // j+1
};

std::vector<BarrierPair> build_barrier_pairs(int N) {
    std::vector<BarrierPair> pairs;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N - 1; ++j) {
            if (i == j || i == j + 1) continue; // skip self/adjacent
            pairs.push_back({i, j, j + 1});
        }
    }
    return pairs;
}

// Barrier energy and derivatives
double barrierEnergy(double d, double dhat) {
    if (d >= dhat) return 0.0;
//    d = std::max(d, 1e-12); // numerical safeguard
    return - (d - dhat) * (d - dhat) * std::log(d / dhat);
}

double barrierGrad(double d, double dhat) {
    if (d >= dhat) return 0.0;
//    d = std::max(d, 1e-12);
    return -2 * (d - dhat) * std::log(d / dhat) - (d - dhat) * (d - dhat) / d;
}

double barrierHess(double d, double dhat) {
    if (d >= dhat) return 0.0;
//    d = std::max(d, 1e-12);
    return -2 * std::log(d / dhat) - 4 * (d - dhat) / d + (d - dhat) * (d - dhat) / (d * d);
}

// ======================================================
// Compute point–segment distance
// ======================================================
double nodeSegmentDistance(const Vec2 &xi,
                           const Vec2 &xj,
                           const Vec2 &xjp1,
                           double &t, Vec2 &p, Vec2 &r)
{
    // Segment direction
    Vec2 seg = { xjp1.x - xj.x, xjp1.y - xj.y };
    double seg_len2 = seg.x * seg.x + seg.y * seg.y;

    // Handle degenerate segment
    if (seg_len2 < 1e-14) {
        t = 0.0;
        p = xj;
        r = { xi.x - p.x, xi.y - p.y };
        return std::sqrt(r.x * r.x + r.y * r.y);
    }

    // Project point onto segment
    Vec2 q = { xi.x - xj.x, xi.y - xj.y };
    double dot = q.x * seg.x + q.y * seg.y;
    t = dot / seg_len2;

    // Clamp to segment
    t = (t < 0.0) ? 0.0 : (t > 1.0 ? 1.0 : t);

    // Closest point
    p = { xj.x + t * seg.x, xj.y + t * seg.y };

    // Vector and distance
    r = { xi.x - p.x, xi.y - p.y };
    return std::sqrt(r.x * r.x + r.y * r.y);
}

static inline Vec2 mul(const Mat2& A, const Vec2& v){
    return { A.a11*v.x + A.a12*v.y, A.a21*v.x + A.a22*v.y };
}
static inline Mat2 mul(const Mat2& A, const Mat2& B){
    return {
            A.a11*B.a11 + A.a12*B.a21,  A.a11*B.a12 + A.a12*B.a22,
            A.a21*B.a11 + A.a22*B.a21,  A.a21*B.a12 + A.a22*B.a22
    };
}

// ======================================================
// LocalBarrierGrad
// ======================================================
// Build projectors T, P from segment direction s = x_{j+1} - x_j
// T = uu^T and P = I - uu^T
static inline void buildProjectors(const Vec2& xj, const Vec2& xk, Mat2& T, Mat2& P){
    Vec2 s{ xk.x - xj.x, xk.y - xj.y };
    double len2 = s.x*s.x + s.y*s.y;
    // Assume non-degenerate
    double inv = 1.0 / len2;
    T = { s.x*s.x*inv, s.x*s.y*inv, s.x*s.y*inv, s.y*s.y*inv };
    P = { 1.0 - T.a11, -T.a12, -T.a21, 1.0 - T.a22 };
}

Vec2 localBarrierGrad(int who, const Vec &x, int node, int seg0, int seg1, double dhat) {
    Vec2 xi   = getXi(x, node);
    Vec2 xj   = getXi(x, seg0);
    Vec2 xk   = getXi(x, seg1);

    double t; Vec2 p{}, r{};
    double d = nodeSegmentDistance(xi, xj, xk, t, p, r);
    if (d >= dhat) return {0,0};
    d = std::max(d, 1e-12);

    Vec2 n{ r.x/d, r.y/d };
    double bp = barrierGrad(d, dhat);

    Mat2 T{}, P{}; buildProjectors(xj, xk, T, P);
    Vec2 g_raw{ bp*n.x, bp*n.y };

    if (who == node) {
        // Gradient of {x_i} = P * (bp * n)
        return mul(P, g_raw);
    } else if (who == seg0 or who == seg1) {
        return {0,0};
    }
    return {0,0};
}


// ==============================
// LocalBarrierHess (node block only)
// ==============================
Mat2 localBarrierHess(int who, const Vec &x, int node, int seg0, int seg1, double dhat) {
    Vec2 xi   = getXi(x, node);
    Vec2 xj   = getXi(x, seg0);
    Vec2 xk   = getXi(x, seg1);

    double t; Vec2 p{}, r{};
    double d = nodeSegmentDistance(xi, xj, xk, t, p, r);
    if (d >= dhat) return {0,0,0,0};
    d = std::max(d, 1e-12);

    Vec2 n{ r.x/d, r.y/d };
    double bp  = barrierGrad(d, dhat);
    double bpp = barrierHess(d, dhat);

    // K = b'' nn^T + (b'/d)(I - nn^T)
    double nx=n.x, ny=n.y;
    Mat2 K{
            bpp*nx*nx + (bp/d)*(1 - nx*nx),
            (bpp - bp/d)*nx*ny,
            (bpp - bp/d)*nx*ny,
            bpp*ny*ny + (bp/d)*(1 - ny*ny)
    };

    Mat2 T{}, P{}; buildProjectors(xj, xk, T, P);

    if (who == node) {
        // H_{ii} = P K P
        Mat2 PK = mul(P, K);
        Mat2 H  = mul(PK, P);
        return H;
    } else {
        return {0,0,0,0};
    }
}

// ======================================================
// Finite-difference Hessian from energy
// ======================================================
double barrierPairEnergy(const Vec& x,int node,int seg0,int seg1,double dhat){
    double t; Vec2 p,r;
    double d=nodeSegmentDistance(getXi(x,node),getXi(x,seg0),getXi(x,seg1),t,p,r);
    return barrierEnergy(d,dhat);
}

Mat2 finiteDifferenceBarrierHessianEnergy(const Vec& x0,int node,int seg0,int seg1,
                                          double dhat,double h){
    Mat2 H{0,0,0,0};
    auto E=[&](const Vec& x){ return barrierPairEnergy(x,node,seg0,seg1,dhat); };
    auto shift=[&](Vec x,int sx,int sy){ x[2*node]+=sx*h; x[2*node+1]+=sy*h; return x; };

    Vec xp=shift(x0,+1,0), xm=shift(x0,-1,0);
    H.a11=(E(xp)-2*E(x0)+E(xm))/(h*h);

    Vec yp=shift(x0,0,+1), ym=shift(x0,0,-1);
    H.a22=(E(yp)-2*E(x0)+E(ym))/(h*h);

    Vec xpp=shift(x0,+1,+1), xpm=shift(x0,+1,-1),
            xmp=shift(x0,-1,+1), xmm=shift(x0,-1,-1);
    double Hxy=(E(xpp)-E(xpm)-E(xmp)+E(xmm))/(4*h*h);
    H.a12=Hxy; H.a21=Hxy;
    return H;
}

// ======================================================
// Tests
// ======================================================
void finiteDifferenceSpringTest(){
    std::cout<<std::scientific<<std::setprecision(3);
    double k=10.0; std::vector<double> L={1,1,1};
    Vec x={0,0,1,0.2,2,-0.1,3,0.3};

    std::cout<<"\n===== Finite Difference Test: SPRING Energy =====\n";
    std::cout<<"dt\t|grad err|\tgrad ratio\t|Hess err|\tHess ratio\n";

    for(int i=0;i<4;++i){
        Vec2 gA=localSpringGrad(i,x,k,L);
        Mat2 HA=localSpringHess(i,x,k,L);
        double prev_grad_err=0, prev_hess_err=0;
        std::cout<<"\nNode "<<i<<":\n";
        for(int n=2;n<=4;++n){
            double dt=std::pow(10.0,-n);
            Vec2 gN{0,0};
            for(int d=0;d<2;++d){
                Vec xp=x,xm=x;
                if(d==0){xp[2*i]+=dt; xm[2*i]-=dt;}
                else {xp[2*i+1]+=dt; xm[2*i+1]-=dt;}
                double Ep=totalSpringEnergy(xp,k,L);
                double Em=totalSpringEnergy(xm,k,L);
                double fd=(Ep-Em)/(2*dt);
                if(d==0) gN.x=fd; else gN.y=fd;
            }
            Mat2 HN{0,0,0,0};
            for(int d=0;d<2;++d){
                Vec xp=x,xm=x;
                if(d==0){xp[2*i]+=dt; xm[2*i]-=dt;}
                else {xp[2*i+1]+=dt; xm[2*i+1]-=dt;}
                Vec2 gp=localSpringGrad(i,xp,k,L);
                Vec2 gm=localSpringGrad(i,xm,k,L);
                Vec2 diff={(gp.x-gm.x)/(2*dt),(gp.y-gm.y)/(2*dt)};
                if(d==0){HN.a11=diff.x; HN.a21=diff.y;}
                else {HN.a12=diff.x; HN.a22=diff.y;}
            }
            double grad_err=std::hypot(gA.x-gN.x,gA.y-gN.y);
            double hess_err=std::sqrt((HA.a11-HN.a11)*(HA.a11-HN.a11)+
                                      (HA.a12-HN.a12)*(HA.a12-HN.a12)+
                                      (HA.a21-HN.a21)*(HA.a21-HN.a21)+
                                      (HA.a22-HN.a22)*(HA.a22-HN.a22));
            double grad_ratio=(prev_grad_err>0)?prev_grad_err/grad_err:0;
            double hess_ratio=(prev_hess_err>0)?prev_hess_err/hess_err:0;
            std::cout<<"1e-"<<n<<"\t"<<grad_err<<"\t"<<grad_ratio
                     <<"\t"<<hess_err<<"\t"<<hess_ratio<<"\n";
            prev_grad_err=grad_err; prev_hess_err=hess_err;
        }
    }
}

void makeWellConditionedBarrierCase(Vec& x,int& node,int& seg0,int& seg1,double& dhat){
    x={0,0,0,0,0,0};
    setXi(x,1,{0,0}); setXi(x,2,{1.3,0.2});
    dhat=0.8;
    Vec2 s0=getXi(x,1), s1=getXi(x,2);
    Vec2 seg={s1.x-s0.x,s1.y-s0.y};
    double len=std::sqrt(seg.x*seg.x+seg.y*seg.y);
    Vec2 n={-seg.y/len,seg.x/len};
    double t=0.4,d=0.3*dhat;
    Vec2 pt={s0.x+t*seg.x,s0.y+t*seg.y};
    setXi(x,0,{pt.x+d*n.x,pt.y+d*n.y});
    node=0; seg0=1; seg1=2;
}

void finiteDifferenceBarrierTest(){
    std::cout<<std::scientific<<std::setprecision(3);
    std::cout<<"\n===== Finite Difference Test: BARRIER Energy (energy-based FD) =====\n";
    std::cout<<"dt\t|grad err|\tgrad ratio\t|Hess err|\tHess ratio\n";
    Vec x; int node,seg0,seg1; double dhat;
    makeWellConditionedBarrierCase(x,node,seg0,seg1,dhat);

    Vec2 gA=localBarrierGrad(node,x,node,seg0,seg1,dhat);
    Mat2 HA=localBarrierHess(node,x,node,seg0,seg1,dhat);

    double prev_grad_err=0,prev_hess_err=0;
    for(int n=2;n<=4;++n){
        double h=std::pow(10.0,-n);
        // Gradient by central diff on energy
        Vec2 gN{0,0};
        {
            Vec xp=x,xm=x; xp[2*node]+=h; xm[2*node]-=h;
            double Ep=barrierPairEnergy(xp,node,seg0,seg1,dhat);
            double Em=barrierPairEnergy(xm,node,seg0,seg1,dhat);
            gN.x=(Ep-Em)/(2*h);
        }
        {
            Vec yp=x,ym=x; yp[2*node+1]+=h; ym[2*node+1]-=h;
            double Ep=barrierPairEnergy(yp,node,seg0,seg1,dhat);
            double Em=barrierPairEnergy(ym,node,seg0,seg1,dhat);
            gN.y=(Ep-Em)/(2*h);
        }
        Mat2 HN=finiteDifferenceBarrierHessianEnergy(x,node,seg0,seg1,dhat,h);
        double grad_err=std::hypot(gA.x-gN.x,gA.y-gN.y);
        double hess_err=std::sqrt((HA.a11-HN.a11)*(HA.a11-HN.a11)+
                                  (HA.a12-HN.a12)*(HA.a12-HN.a12)+
                                  (HA.a21-HN.a21)*(HA.a21-HN.a21)+
                                  (HA.a22-HN.a22)*(HA.a22-HN.a22));
        double grad_ratio=(prev_grad_err>0)?prev_grad_err/grad_err:0;
        double hess_ratio=(prev_hess_err>0)?prev_hess_err/hess_err:0;
        std::cout<<"1e-"<<n<<"\t"<<grad_err<<"\t"<<grad_ratio
                 <<"\t"<<hess_err<<"\t"<<hess_ratio<<"\n";
        prev_grad_err=grad_err; prev_hess_err=hess_err;
    }
}

// ---------------------------------------------------------
//  Finite-difference consistency test
// ---------------------------------------------------------
void finiteDifferenceBarrierScalarTest() {
    double dhat = 0.5;
    double d = 0.2;  // inside region d < dhat

    std::cout << "\n===== Finite Difference Test: SCALAR BARRIER =====\n";
    std::cout << "dt\t|grad err|\tgrad ratio\t|hess err|\thess ratio\n";

    double prev_grad_err = 0.0, prev_hess_err = 0.0;

    for (int n = 2; n <= 4; ++n) {
        double h = std::pow(10.0, -n);

        // finite-diff gradient
        double Ep = barrierEnergy(d + h, dhat);
        double Em = barrierEnergy(d - h, dhat);
        double fd_grad = (Ep - Em) / (2.0 * h);

        // finite-diff Hessian
        double bp_p = barrierGrad(d + h, dhat);
        double bp_m = barrierGrad(d - h, dhat);
        double fd_hess = (bp_p - bp_m) / (2.0 * h);

        double bp = barrierGrad(d, dhat);
        double bpp = barrierHess(d, dhat);

        double grad_err = std::abs(fd_grad - bp);
        double hess_err = std::abs(fd_hess - bpp);

        double grad_ratio = (prev_grad_err > 0) ? prev_grad_err / grad_err : 0.0;
        double hess_ratio = (prev_hess_err > 0) ? prev_hess_err / hess_err : 0.0;

        std::cout << "1e-" << n << "\t" << grad_err << "\t" << grad_ratio
                  << "\t" << hess_err << "\t" << hess_ratio << "\n";

        prev_grad_err = grad_err;
        prev_hess_err = hess_err;
    }
}

// ======================================================
// Generic finite-difference utilities
// ======================================================

double centralDiff(double Ep, double Em, double h) {
    return (Ep - Em) / (2.0 * h);
}

double secondCentralDiff(double Ep, double E0, double Em, double h) {
    return (Ep - 2.0 * E0 + Em) / (h * h);
}

// ======================================================
// Unified Consistency Test Utilities
// ======================================================

// Compare analytic gradient vs FD gradient from energy
Vec2 finiteDiffGradientEnergy(
        const Vec& x, int node,
        std::function<double(const Vec&)> energyFunc,
        double h)
{
    Vec2 gFD{0, 0};
    {
        Vec xp = x, xm = x;
        xp[2 * node] += h; xm[2 * node] -= h;
        gFD.x = centralDiff(energyFunc(xp), energyFunc(xm), h);
    }
    {
        Vec yp = x, ym = x;
        yp[2 * node + 1] += h; ym[2 * node + 1] -= h;
        gFD.y = centralDiff(energyFunc(yp), energyFunc(ym), h);
    }
    return gFD;
}

// Compare analytic Hessian vs FD Hessian from energy
Mat2 finiteDiffHessianEnergy(
        const Vec& x, int node,
        std::function<double(const Vec&)> energyFunc,
        double h)
{
    Mat2 HFD{0, 0, 0, 0};

    // xx
    Vec xp = x, xm = x;
    xp[2 * node] += h; xm[2 * node] -= h;
    double Ep = energyFunc(xp);
    double Em = energyFunc(xm);
    double E0 = energyFunc(x);
    HFD.a11 = secondCentralDiff(Ep, E0, Em, h);

    // yy
    Vec yp = x, ym = x;
    yp[2 * node + 1] += h; ym[2 * node + 1] -= h;
    Ep = energyFunc(yp);
    Em = energyFunc(ym);
    HFD.a22 = secondCentralDiff(Ep, E0, Em, h);

    // xy
    Vec xpp = x, xpm = x, xmp = x, xmm = x;
    xpp[2 * node] += h;  xpp[2 * node + 1] += h;
    xpm[2 * node] += h;  xpm[2 * node + 1] -= h;
    xmp[2 * node] -= h;  xmp[2 * node + 1] += h;
    xmm[2 * node] -= h;  xmm[2 * node + 1] -= h;

    double Epp = energyFunc(xpp);
    double Epm = energyFunc(xpm);
    double Emp = energyFunc(xmp);
    double Emm = energyFunc(xmm);
    double Hxy = (Epp - Epm - Emp + Emm) / (4 * h * h);
    HFD.a12 = Hxy; HFD.a21 = Hxy;

    return HFD;
}

// Compare analytic Hessian vs FD Hessian from gradient
Mat2 finiteDiffHessianGradient(
        const Vec& x, int node,
        std::function<Vec2(const Vec&)> gradFunc,
        double h)
{
    Mat2 HFD{0, 0, 0, 0};

    // derivative of grad.x wrt x and y
    {
        Vec xp = x, xm = x;
        xp[2 * node] += h; xm[2 * node] -= h;
        Vec2 gp = gradFunc(xp), gm = gradFunc(xm);
        HFD.a11 = (gp.x - gm.x) / (2 * h);
        HFD.a21 = (gp.y - gm.y) / (2 * h);
    }
    {
        Vec yp = x, ym = x;
        yp[2 * node + 1] += h; ym[2 * node + 1] -= h;
        Vec2 gp = gradFunc(yp), gm = gradFunc(ym);
        HFD.a12 = (gp.x - gm.x) / (2 * h);
        HFD.a22 = (gp.y - gm.y) / (2 * h);
    }
    return HFD;
}

// ======================================================
//  Unified energy-gradient-Hessian test driver
// ======================================================
void runFiniteDiffConsistencyTest(
        const std::string& name,
        const Vec& x0,
        int node,
        std::function<double(const Vec&)> energyFunc,
        std::function<Vec2(const Vec&)> gradFunc,
        std::function<Mat2(const Vec&)> hessFunc)
{
    std::cout << std::scientific << std::setprecision(3);
    std::cout << "\n===== Finite Difference Consistency Test: "
              << name << " =====\n";
    std::cout << "dt\t|grad err|\tgrad ratio\t|Hess err|\tHess ratio\n";

    double prev_grad_err = 0.0, prev_hess_err = 0.0;
    for (int n = 2; n <= 4; ++n) {
        double h = std::pow(10.0, -n);
        Vec2 gA = gradFunc(x0);
        Mat2 HA = hessFunc(x0);

        Vec2 gFD = finiteDiffGradientEnergy(x0, node, energyFunc, h);
        Mat2 HFD_E = finiteDiffHessianEnergy(x0, node, energyFunc, h);
        Mat2 HFD_G = finiteDiffHessianGradient(x0, node, gradFunc, h);

        double grad_err = std::hypot(gA.x - gFD.x, gA.y - gFD.y);
        double hess_err = std::sqrt(
                (HA.a11 - HFD_E.a11) * (HA.a11 - HFD_E.a11) +
                (HA.a12 - HFD_E.a12) * (HA.a12 - HFD_E.a12) +
                (HA.a21 - HFD_E.a21) * (HA.a21 - HFD_E.a21) +
                (HA.a22 - HFD_E.a22) * (HA.a22 - HFD_E.a22)
        );

        double grad_ratio = (prev_grad_err > 0) ? prev_grad_err / grad_err : 0;
        double hess_ratio = (prev_hess_err > 0) ? prev_hess_err / hess_err : 0;

        std::cout << "1e-" << n << "\t" << grad_err << "\t" << grad_ratio
                  << "\t" << hess_err << "\t" << hess_ratio << "\n";

        prev_grad_err = grad_err;
        prev_hess_err = hess_err;
    }
}

// ======================================================
//  Test runners for specific systems
// ======================================================
void testSpringSystem() {
    double k = 10.0;
    std::vector<double> L = {1, 1, 1};
    Vec x = {0, 0, 1, 0.2, 2, -0.1, 3, 0.3};

    for (int i = 0; i < 4; ++i) {
        std::stringstream ss;
        ss << "SPRING (node " << i << ")";
        runFiniteDiffConsistencyTest(
                ss.str(), x, i,
                [&](const Vec& X){ return totalSpringEnergy(X, k, L); },
                [&](const Vec& X){ return localSpringGrad(i, X, k, L); },
                [&](const Vec& X){ return localSpringHess(i, X, k, L); }
        );
    }
}

void testBarrierSystem() {
    Vec x; int node, seg0, seg1; double dhat;
    makeWellConditionedBarrierCase(x, node, seg0, seg1, dhat);

    runFiniteDiffConsistencyTest(
            "BARRIER (node-segment)",
            x, node,
            [&](const Vec& X){ return barrierPairEnergy(X, node, seg0, seg1, dhat); },
            [&](const Vec& X){ return localBarrierGrad(node, X, node, seg0, seg1, dhat); },
            [&](const Vec& X){ return localBarrierHess(node, X, node, seg0, seg1, dhat); }
    );
}

//==============================================================
// Energy components for Psi
//==============================================================

// Gravity energy
double gravityEnergy(const Vec& x,
                     const std::vector<double>& mass,
                     const Vec2& g_accel)
{
    double Eg = 0.0;
    int N = mass.size();
    for (int i = 0; i < N; ++i) {
        Vec2 xi = getXi(x, i);
        Eg += -mass[i] * (g_accel.x * xi.x + g_accel.y * xi.y);
    }
    return Eg;
}

// Total barrier energy
double totalBarrierEnergy(const Vec& x,
                          const std::vector<BarrierPair>& barriers,
                          double dhat)
{
    double Eb = 0.0;
    for (const auto& c : barriers) {
        double t; Vec2 p, r;
        double d = nodeSegmentDistance(getXi(x, c.node),
                                       getXi(x, c.seg0),
                                       getXi(x, c.seg1),
                                       t, p, r);
        Eb += barrierEnergy(d, dhat);
    }
    return Eb;
}

// Full Psi energy
double PsiEnergy(const Vec& x, const Vec& xhat,
                 const std::vector<double>& mass,
                 const std::vector<double>& L,
                 double dt, double k, const Vec2& g_accel,
                 const std::vector<BarrierPair>& barriers, double dhat)
{
    double Ein = 0.0;
    int N = mass.size();
    for (int i = 0; i < N; ++i) {
        Vec2 xi = getXi(x, i), xhi = getXi(xhat, i);
        double dx = xi.x - xhi.x, dy = xi.y - xhi.y;
        Ein += 0.5 * mass[i] * (dx * dx + dy * dy);
    }
    double Es = totalSpringEnergy(x, k, L);
    double Eb = totalBarrierEnergy(x, barriers, dhat);
    double Eg = gravityEnergy(x, mass, g_accel);

    return Ein + dt * dt * (Es + Eb + Eg);
}

//==============================================================
// Local Gradient and Local Hessian of Psi(x)
//==============================================================

// Local Gradient ∇_{x_i} Psi
Vec2 PsiLocalGrad(int i,
                  const Vec& x, const Vec& xhat,
                  const std::vector<double>& mass,
                  const std::vector<double>& L,
                  double dt, double k, const Vec2& g_accel,
                  const std::vector<bool>& is_fixed,
                  const std::vector<BarrierPair>& barriers,
                  double dhat)
{
    if (is_fixed[i]) return {0.0, 0.0};

    Vec2 xi = getXi(x, i), xhi = getXi(xhat, i);
    Vec2 gi{0.0, 0.0};

    // Mass term
    gi.x += mass[i] * (xi.x - xhi.x);
    gi.y += mass[i] * (xi.y - xhi.y);

    // Spring term
    Vec2 gs = localSpringGrad(i, x, k, L);
    gi.x += dt * dt * gs.x;
    gi.y += dt * dt * gs.y;

    // Gravity
    gi.x -= dt * dt * mass[i] * g_accel.x;
    gi.y -= dt * dt * mass[i] * g_accel.y;

    // Barrier forces
    for (const auto& c : barriers) {
        for (int who : {c.node, c.seg0, c.seg1}) {
            if (who != i) continue;
            Vec2 gb = localBarrierGrad(who, x, c.node, c.seg0, c.seg1, dhat);
            gi.x += dt * dt * gb.x;
            gi.y += dt * dt * gb.y;
        }
    }

    return gi;
}

// Local Hessian
Mat2 PsiLocalHess(int i,
                  const Vec& x,
                  const std::vector<double>& mass,
                  const std::vector<double>& L,
                  double dt, double k,
                  const std::vector<bool>& is_fixed,
                  const std::vector<BarrierPair>& barriers,
                  double dhat)
{
    if (is_fixed[i]) return {0, 0, 0, 0};

    Mat2 H{mass[i], 0, 0, mass[i]};

    // Spring term
    Mat2 Hs = localSpringHess(i, x, k, L);
    H.a11 += dt * dt * Hs.a11;
    H.a12 += dt * dt * Hs.a12;
    H.a21 += dt * dt * Hs.a21;
    H.a22 += dt * dt * Hs.a22;

    // Barrier term
    for (const auto& c : barriers) {
        for (int who : {c.node, c.seg0, c.seg1}) {
            if (who != i) continue;
            Mat2 Hb = localBarrierHess(who, x, c.node, c.seg0, c.seg1, dhat);
            H.a11 += dt * dt * Hb.a11;
            H.a12 += dt * dt * Hb.a12;
            H.a21 += dt * dt * Hb.a21;
            H.a22 += dt * dt * Hb.a22;
        }
    }

    return H;
}

//==============================================================
// Finite-Difference validation for one node block
//==============================================================
void finiteDifferencePsiLocalTest()
{
    std::cout << std::scientific << std::setprecision(3);
    std::cout << "\n===== Finite Difference Local Test: Psi(x) =====\n";
    std::cout << "h\t|grad err|\tgrad ratio\t|HessE err|\t|HessG err|\n";

    // simple test setup (similar to spring_test)
    double k = 10.0;
    std::vector<double> L = {1, 1, 1};
    Vec x = {0, 0, 1, 0.2, 2, -0.1, 3, 0.3};
    Vec xhat = x;
    std::vector<double> mass = {1.0, 2.0, 1.5, 1.0};
    Vec2 g_accel{0.0, -9.8};
    double dt = 0.02;
    double dhat = 0.4;
    std::vector<bool> is_fixed(mass.size(), false);
    auto barriers = build_barrier_pairs((int)mass.size());

    int i = 1; // test node 1
    auto E = [&](const Vec& X) {
        return PsiEnergy(X, xhat, mass, L, dt, k, g_accel, barriers, dhat);
    };
    auto G = [&](const Vec& X) {
        return PsiLocalGrad(i, X, xhat, mass, L, dt, k, g_accel,
                            is_fixed, barriers, dhat);
    };

    double prev_ge = 0, prev_he = 0, prev_hg = 0;
    for (int n = 2; n <= 4; ++n) {
        double h = std::pow(10.0, -n);

        // analytic
        Vec2 gA = PsiLocalGrad(i, x, xhat, mass, L, dt, k, g_accel,
                               is_fixed, barriers, dhat);
        Mat2 HA = PsiLocalHess(i, x, mass, L, dt, k, is_fixed, barriers, dhat);

        // FD gradient
        Vec2 gFD;
        Vec xp = x, xm = x;
        xp[2 * i] += h; xm[2 * i] -= h;
        gFD.x = (E(xp) - E(xm)) / (2 * h);

        Vec yp = x, ym = x;
        yp[2 * i + 1] += h; ym[2 * i + 1] -= h;
        gFD.y = (E(yp) - E(ym)) / (2 * h);

        // FD Hessian from energy
        Mat2 HE{0, 0, 0, 0};
        {
            Vec xp = x, xm = x;
            xp[2 * i] += h; xm[2 * i] -= h;
            HE.a11 = (E(xp) - 2 * E(x) + E(xm)) / (h * h);
        }
        {
            Vec yp = x, ym = x;
            yp[2 * i + 1] += h; ym[2 * i + 1] -= h;
            HE.a22 = (E(yp) - 2 * E(x) + E(ym)) / (h * h);
        }
        {
            Vec xpp = x, xpm = x, xmp = x, xmm = x;
            xpp[2 * i] += h; xpp[2 * i + 1] += h;
            xpm[2 * i] += h; xpm[2 * i + 1] -= h;
            xmp[2 * i] -= h; xmp[2 * i + 1] += h;
            xmm[2 * i] -= h; xmm[2 * i + 1] -= h;
            double Epp = E(xpp), Epm = E(xpm), Emp = E(xmp), Emm = E(xmm);
            double Hxy = (Epp - Epm - Emp + Emm) / (4 * h * h);
            HE.a12 = HE.a21 = Hxy;
        }

        // FD Hessian from gradient
        Mat2 HG{0, 0, 0, 0};
        {
            Vec xp = x, xm = x; xp[2 * i] += h; xm[2 * i] -= h;
            Vec2 gp = G(xp), gm = G(xm);
            HG.a11 = (gp.x - gm.x) / (2 * h);
            HG.a21 = (gp.y - gm.y) / (2 * h);
        }
        {
            Vec yp = x, ym = x; yp[2 * i + 1] += h; ym[2 * i + 1] -= h;
            Vec2 gp = G(yp), gm = G(ym);
            HG.a12 = (gp.x - gm.x) / (2 * h);
            HG.a22 = (gp.y - gm.y) / (2 * h);
        }

        auto hErr = [](const Mat2& A, const Mat2& B) {
            return std::sqrt((A.a11 - B.a11) * (A.a11 - B.a11) +
                             (A.a12 - B.a12) * (A.a12 - B.a12) +
                             (A.a21 - B.a21) * (A.a21 - B.a21) +
                             (A.a22 - B.a22) * (A.a22 - B.a22));
        };

        double ge = std::hypot(gA.x - gFD.x, gA.y - gFD.y);
        double he = hErr(HA, HE);
        double hg = hErr(HA, HG);

        double ge_ratio = (prev_ge > 0) ? prev_ge / ge : 0.0;
        double he_ratio = (prev_he > 0) ? prev_he / he : 0.0;
        double hg_ratio = (prev_hg > 0) ? prev_hg / hg : 0.0;

        std::cout << "1e-" << n << "\t" << ge << "\t" << ge_ratio
                  << "\t" << he << "\t" << he_ratio << "\n";

        prev_ge = ge; prev_he = he; prev_hg = hg;
    }
}

// ======================================================
// MAIN
// ======================================================
int main(){
    finiteDifferenceSpringTest();
    finiteDifferenceBarrierScalarTest();
    finiteDifferenceBarrierTest();
    testSpringSystem();
    testBarrierSystem();
    finiteDifferencePsiLocalTest();
    return 0;
}







