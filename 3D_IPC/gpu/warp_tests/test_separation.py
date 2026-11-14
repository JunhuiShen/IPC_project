"""Independent exact convex-hull oracle for the one-sided F32 certificate."""

from fractions import Fraction
from itertools import combinations
import unittest

import numpy as np
import warp as wp

from gpu import separation


@wp.kernel
def _filter(start: wp.array[wp.vec3],end: wp.array[wp.vec3],kind: wp.array[int],
            gap: wp.array[wp.float64],result: wp.array[int]):
    i=wp.tid();first=4*i
    result[i]=int(separation.safe(start,end,wp.vec4i(first,first+1,first+2,first+3),kind[i],gap[i]))


@wp.func_native(r"""
    double a[4][3],b[4][3];
    for(int i=0;i<4;++i)for(int k=0;k<3;++k){a[i][k]=start.data[4*index+i][k];b[i][k]=end.data[4*index+i][k];}
    return ipc_gpu::separation::safe(a,b,kind==0,gap);
""")
def _double_domain(start: wp.array[wp.vec3d],end: wp.array[wp.vec3d],index: int,kind: int,gap: wp.float64)->bool: ...


@wp.kernel
def _guarded(start: wp.array[wp.vec3d],end: wp.array[wp.vec3d],result: wp.array[int]):
    i=wp.tid()
    result[i]=int(_double_domain(start,end,i,0,wp.float64(0.0)))


separation.configure_module(__name__)


def _dot(a,b):
    return sum((x*y for x,y in zip(a,b)),Fraction(0))


def _solve(matrix,rhs):
    """Exact Gaussian elimination; singular subsets are covered by smaller ones."""
    n=len(rhs);rows=[list(matrix[i])+[rhs[i]] for i in range(n)]
    for column in range(n):
        pivot=next((i for i in range(column,n) if rows[i][column]),None)
        if pivot is None:return None
        rows[column],rows[pivot]=rows[pivot],rows[column]
        scale=rows[column][column]
        rows[column]=[x/scale for x in rows[column]]
        for i in range(n):
            if i!=column:
                factor=rows[i][column]
                rows[i]=[x-factor*y for x,y in zip(rows[i],rows[column])]
    return [row[-1] for row in rows]


def convex_hull_distance_squared(start,end,kind):
    """Exact minimum distance between endpoint hulls, without SAT axes.

    The Minkowski hull is a finite point set in R³. Its closest point has
    an affinely independent support of at most four vertices (Carathéodory).
    Enumerating those supports and their exact constrained quadratic minima
    gives its distance to the origin. This uses arbitrary-precision rational
    arithmetic and no production separation/CCD implementation.
    """
    split=1 if kind==0 else 2
    rows=[[tuple(Fraction.from_float(float(v)) for v in point) for point in values]
          for values in (start,end)]
    first={point for frame in rows for point in frame[:split]}
    second={point for frame in rows for point in frame[split:]}
    points=sorted({tuple(a[k]-b[k] for k in range(3)) for a in first for b in second})
    best=min(_dot(p,p) for p in points)
    if best==0:return best
    for count in range(2,min(4,len(points))+1):
        for subset in combinations(points,count):
            origin=subset[0]
            directions=[tuple(p[k]-origin[k] for k in range(3)) for p in subset[1:]]
            weights=_solve([[_dot(a,b) for b in directions] for a in directions],
                           [-_dot(a,origin) for a in directions])
            if weights is None or any(w<0 for w in weights) or sum(weights)>1:continue
            value=tuple(origin[k]+sum((w*d[k] for w,d in zip(weights,directions)),Fraction(0)) for k in range(3))
            best=min(best,_dot(value,value))
            if best==0:return best
    return best


def _ftz_kernel():
    @wp.kernel(module="unique",enable_backward=False)
    def kernel(start: wp.array[wp.vec3],end: wp.array[wp.vec3],kind: wp.array[int],
               gap: wp.array[wp.float64],result: wp.array[int]):
        i=wp.tid();first=4*i
        result[i]=int(separation.safe(start,end,wp.vec4i(first,first+1,first+2,first+3),kind[i],gap[i]))
    separation.configure_module(kernel)
    wp.set_module_options({"fast_math":True,"fuse_fp":True},module=kernel.module)
    return kernel


class SeparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init();cls.devices=[wp.get_device("cpu"),*wp.get_cuda_devices()]

    def evaluate(self,cases,device,kernel=_filter):
        start=np.concatenate([c[0] for c in cases]).astype(np.float32)
        end=np.concatenate([c[1] for c in cases]).astype(np.float32)
        result=wp.empty(len(cases),int,device=device)
        wp.launch(kernel,len(cases),[wp.array(start,wp.vec3,device=device),wp.array(end,wp.vec3,device=device),
            wp.array([c[2] for c in cases],int,device=device),
            wp.array([c[3] for c in cases],wp.float64,device=device),result],device=device)
        return result.numpy()

    def verify_accepted(self,cases,result):
        accepted=0
        for case,decision in zip(cases,result):
            if decision:
                start,end,kind,gap=case
                distance2=convex_hull_distance_squared(start.astype(np.float32),end.astype(np.float32),kind)
                self.assertGreater(distance2,Fraction.from_float(float(gap))**2)
                accepted+=1
        return accepted

    def test_coplanar_and_swept_vf_ee_against_exact_hull_oracle(self):
        vf=np.array([[.7,.7,0],[0,0,0],[1,0,0],[0,1,0]],np.float32)
        ee=np.array([[0,0,0],[1,1,0],[0,.5,0],[.5,1,0]],np.float32)
        crossing=np.array([[.25,.25,.5],[0,0,0],[1,0,0],[0,1,0]],np.float32)
        crossed=crossing.copy();crossed[0,2]=-.5
        touching=ee.copy();touching[2:]=[[0,1,0],[1,0,0]]
        cases=[(vf,vf,0,1e-8),(ee,ee,1,1e-8),
               (crossing,crossed,0,1e-8),(touching,touching,1,0.0)]
        for device in self.devices:
            result=self.evaluate(cases,device)
            np.testing.assert_array_equal(result,[1,1,0,0])
            self.assertEqual(self.verify_accepted(cases,result),2)

    def test_near_gap_floor_and_subnormal_geometry(self):
        floor=1e-8
        above=np.float32(floor)
        for _ in range(16):above=np.nextafter(above,np.float32(np.inf))
        below=np.nextafter(np.float32(floor),np.float32(0))
        tiny=np.nextafter(np.float32(0),np.float32(1))
        cases=[]
        for value,gap in ((above,floor),(below,floor),(tiny,0.0),(2*tiny,float(tiny))):
            start=np.array([[.25,.25,value],[0,0,0],[1,0,0],[0,1,0]],np.float32)
            cases.append((start,start,0,gap))
        start=cases[2][0].copy();end=start.copy();end[0,2]=-tiny
        cases.append((start,end,0,0.0))
        for device in self.devices:
            result=self.evaluate(cases,device)
            np.testing.assert_array_equal(result,[1,0,1,1,0])
            self.assertEqual(self.verify_accepted(cases,result),3)

    def test_huge_overflow_nonfinite_and_degenerate_inputs(self):
        large=np.array([[1e20,0,0],[0,0,0],[0,1,0],[0,0,1]],np.float32)
        maximum=np.finfo(np.float32).max
        overflow=np.array([[maximum,maximum,maximum],[-maximum,-maximum,-maximum],
                           [-maximum,-maximum,-maximum],[-maximum,-maximum,-maximum]],np.float32)
        zero=np.zeros((4,3),np.float32)
        cases=[(large,large,0,1e10),(overflow,overflow,0,1e-8),(zero,zero,0,0.0)]
        for invalid in (np.nan,np.inf,-np.inf):
            value=large.copy();value[3,1]=invalid;cases.append((value,value,0,1e-8))
        cases.extend([(large,large,0,np.nan),(large,large,0,-1.0)])
        for device in self.devices:
            result=self.evaluate(cases,device)
            self.assertEqual(result[0],1)
            np.testing.assert_array_equal(result[1:],0)
            self.verify_accepted(cases,result)

    def test_nonrepresentable_double_inputs_are_never_rounded_into_a_proof(self):
        base=np.array([[.25,.25,1],[0,0,0],[1,0,0],[0,1,0]],np.float64)
        changed=base.copy();changed[0,0]+=2.0**-30
        changed_end=base.copy();changed_end[0,2]+=2.0**-30
        starts=np.concatenate((base,changed,base))
        ends=np.concatenate((base,base,changed_end))
        for device in self.devices:
            result=wp.empty(3,int,device=device)
            wp.launch(_guarded,3,[wp.array(starts,wp.vec3d,device=device),wp.array(ends,wp.vec3d,device=device),result],device=device)
            np.testing.assert_array_equal(result.numpy(),[1,0,0])

    def test_random_oblique_sweeps_accepted_only_with_exact_hull_gap(self):
        rng=np.random.default_rng(4821);cases=[]
        fixtures=[np.array([[.7,.7,0],[0,0,0],[1,0,0],[0,1,0]]),
                  np.array([[0,0,0],[1,1,0],[0,.5,0],[.5,1,0]])]
        for index in range(16):
            rotation,_=np.linalg.qr(rng.normal(size=(3,3)))
            scale=10.0**rng.uniform(-3,3)
            start=(fixtures[index%2]@rotation.T*scale+rng.uniform(-2,2,size=3)*scale).astype(np.float32)
            end=(start.astype(np.float64)+rng.normal(size=(4,3))*.001*scale).astype(np.float32)
            cases.append((start,end,index%2,scale*1e-5))
        for device in self.devices:
            result=self.evaluate(cases,device)
            self.assertGreaterEqual(self.verify_accepted(cases,result),12)

    def test_subnormal_certificate_survives_surrounding_cuda_ftz(self):
        devices=wp.get_cuda_devices()
        if not devices:self.skipTest("No CUDA device is available")
        tiny=np.nextafter(np.float32(0),np.float32(1))
        start=np.array([[.25,.25,tiny],[0,0,0],[1,0,0],[0,1,0]],np.float32)
        crossed=start.copy();crossed[0,2]=-tiny
        cases=[(start,start,0,0.0),(start,crossed,0,0.0)]
        kernel=_ftz_kernel()
        for device in devices:
            result=self.evaluate(cases,device,kernel)
            np.testing.assert_array_equal(result,[1,0])
            self.verify_accepted(cases,result)


if __name__=="__main__":
    unittest.main()
