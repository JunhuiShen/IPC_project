"""Our energy terms on an unmodified stock Newton model.

Setup owns only additional coefficient/gradient/pin arrays. It does not modify
positions, masses, flags, topology, rest matrices, materials or coloring.
Stock triangle/bending viscous damping is absent from our constitutive laws;
the stock local solve and DAT controller remain the caller's responsibility.
"""
from dataclasses import dataclass
import numpy as np
import warp as wp
from . import energy


@wp.func
def evaluate_own_membrane_force_hessian(face:int,role:int,pos:wp.array[wp.vec3],pos_anchor:wp.array[wp.vec3],
    triangles:wp.array2d[int],inverse:wp.mat22,area:float,mu:float,lam:float,viscous_damping:float,dt:float,
    own_materials:wp.array2d[float]):
    value=energy.triangle(pos[triangles[face,0]],pos[triangles[face,1]],pos[triangles[face,2]],inverse,area,
        own_materials[face,0],own_materials[face,1],role)
    return energy.unpack(value)


@wp.func
def evaluate_own_bending_force_hessian(edge:int,role:int,pos:wp.array[wp.vec3],pos_anchor:wp.array[wp.vec3],
    edges:wp.array2d[int],rest_angles:wp.array[float],rest_lengths:wp.array[float],stiffness:float,
    viscous_damping:float,dt:float,coefficients:wp.array[float],own_stiffness:wp.array[float]):
    if edges[edge,0]<0 or edges[edge,1]<0 or own_stiffness[edge]<=0.0 or coefficients[edge]<=0.0:
        return wp.vec3(0.0),wp.mat33(0.0)
    native_role=role+2
    if role>=2:native_role=role-2
    value=energy.hinge(pos[edges[edge,2]],pos[edges[edge,3]],pos[edges[edge,0]],pos[edges[edge,1]],
        coefficients[edge],rest_angles[edge],own_stiffness[edge],native_role)
    return energy.unpack(value)


@wp.func
def evaluate_own_solid_force_hessian(tet:int,role:int,pos_anchor:wp.array[wp.vec3],pos:wp.array[wp.vec3],
    tets:wp.array2d[int],inverse:wp.mat33,mu:float,lam:float,viscous_damping:float,dt:float,
    gradients:wp.array2d[wp.vec3],measures:wp.array[float]):
    value=energy.solid(pos[tets[tet,0]],pos[tets[tet,1]],pos[tets[tet,2]],pos[tets[tet,3]],
        inverse,gradients[tet,role],measures[tet],mu,lam)
    return energy.unpack(value)


@wp.func
def evaluate_own_pin_force_hessian(particle:int,pos:wp.array[wp.vec3],targets:wp.array[wp.vec3],stiffness:float):
    return -stiffness*(pos[particle]-targets[particle]),stiffness*wp.identity(n=3,dtype=float)


@dataclass(frozen=True)
class MaterialData:
    triangle_materials: object
    edge_coefficients: object
    bending_stiffness: object
    pin_mask: object
    pin_targets: object
    pin_stiffness: float
    tet_gradients: object
    tet_measures: object

    def arguments(self):
        return [self.triangle_materials,self.edge_coefficients,self.bending_stiffness,self.pin_mask,self.pin_targets,float(self.pin_stiffness),
            self.tet_gradients,self.tet_measures]


def prepare(model,*,triangle_materials=None,edge_coefficients=None,bending_stiffness=None,pin_mask=None,pin_targets=None,pin_stiffness=None,
            tet_gradients=None,tet_measures=None):
    """Read model/rest metadata and own only the appended energy inputs.

    Existing IPC energy metadata is reused when present. A stock model has no
    finite pins by default; its inactive driven vertices stay hard constraints.
    Motion targets may be GPU-authored, so an existing target array is aliased.
    """
    device=model.device;meta=getattr(model,'ipc',None);n=model.particle_count
    def fallback(value,name):return getattr(meta,name,None) if value is None else value
    triangle_materials=fallback(triangle_materials,'dat_triangle_materials')
    edge_coefficients=fallback(edge_coefficients,'edge_coefficient')
    bending_stiffness=fallback(bending_stiffness,'dat_bending_stiffness')
    pin_mask=fallback(pin_mask,'pinned');pin_targets=fallback(pin_targets,'pin_targets')
    tet_gradients=fallback(tet_gradients,'tet_grad_N');tet_measures=fallback(tet_measures,'tet_measure')
    if pin_stiffness is None:pin_stiffness=float(getattr(meta,'params',{}).get('kpin',0.0))
    if not np.isfinite(pin_stiffness) or pin_stiffness<0:raise ValueError('Invalid finite-pin stiffness')
    def array(value,dtype,shape,name):
        if isinstance(value,wp.array):
            if value.dtype!=dtype or value.shape!=shape or value.device!=device:raise ValueError('Invalid '+name+' layout')
            return value
        values=np.asarray(value)
        if values.shape!=shape or not np.isfinite(values).all():raise ValueError('Invalid '+name)
        return wp.array(values,dtype=dtype,device=device)
    if edge_coefficients is None:
        x=model.particle_q.numpy().astype(np.float64)
        edges=model.edge_indices.numpy() if model.edge_count else np.empty((0,4),np.int32)
        values=np.zeros(model.edge_count,np.float32)
        for i,(a,b,e0,e1) in enumerate(edges):
            if a<0 or b<0:continue
            edge=x[e1]-x[e0]
            area=.5*(np.linalg.norm(np.cross(edge,x[a]-x[e0]))+np.linalg.norm(np.cross(edge,x[b]-x[e0])))
            if area>0:values[i]=np.dot(edge,edge)/area
        edge_coefficients=values
    if pin_mask is None:pin_mask=np.zeros(n,np.int32)
    if triangle_materials is None:
        triangle_materials=model.tri_materials.numpy()[:,:2].copy() if model.tri_count else np.empty((0,2),np.float32)
    if bending_stiffness is None:
        bending_stiffness=model.edge_bending_properties.numpy()[:,0].copy() if model.edge_count else np.empty(0,np.float32)
    if pin_targets is None:pin_targets=model.particle_q
    if tet_gradients is None or tet_measures is None:
        inverse=model.tet_poses.numpy().astype(np.float64) if model.tet_count else np.empty((0,3,3))
        gradients=np.empty((model.tet_count,4,3),np.float32);measure=np.empty(model.tet_count,np.float32)
        for i,dm in enumerate(inverse):
            gradients[i,1:]=dm;gradients[i,0]=-dm.sum(axis=0)
            determinant=np.linalg.det(dm)
            if not np.isfinite(determinant) or determinant==0:raise ValueError('Invalid stock tetrahedron rest matrix')
            measure[i]=1.0/(6.0*abs(determinant))
        if tet_gradients is None:tet_gradients=wp.array(gradients,dtype=wp.vec3,ndim=2,device=device)
        if tet_measures is None:tet_measures=measure
    # wp.vec3 array shapes omit the value's vector dimension.
    if not isinstance(pin_targets,wp.array):
        values=np.asarray(pin_targets)
        if values.shape!=(n,3) or not np.isfinite(values).all():raise ValueError('Invalid pin targets')
        pin_targets=wp.array(values,dtype=wp.vec3,device=device)
    if not isinstance(tet_gradients,wp.array):
        values=np.asarray(tet_gradients)
        if values.shape!=(model.tet_count,4,3) or not np.isfinite(values).all():raise ValueError('Invalid tetrahedron gradients')
        tet_gradients=wp.array(values,dtype=wp.vec3,ndim=2,device=device)
    coefficients_host=edge_coefficients.numpy() if isinstance(edge_coefficients,wp.array) else np.asarray(edge_coefficients)
    mask_host=pin_mask.numpy() if isinstance(pin_mask,wp.array) else np.asarray(pin_mask)
    stiffness_host=bending_stiffness.numpy() if isinstance(bending_stiffness,wp.array) else np.asarray(bending_stiffness)
    materials_host=triangle_materials.numpy() if isinstance(triangle_materials,wp.array) else np.asarray(triangle_materials)
    if not np.isfinite(coefficients_host).all() or np.any(coefficients_host<0):raise ValueError('Invalid bending coefficients')
    if not np.isin(mask_host,[0,1]).all():raise ValueError('Invalid finite-pin mask')
    if not np.isfinite(stiffness_host).all() or np.any(stiffness_host<0):raise ValueError('Invalid bending stiffness')
    if not np.isfinite(materials_host).all():raise ValueError('Invalid triangle materials')
    return MaterialData(array(triangle_materials,wp.float32,(model.tri_count,2),'triangle materials'),
        array(edge_coefficients,wp.float32,(model.edge_count,),'edge coefficients'),
        array(bending_stiffness,wp.float32,(model.edge_count,),'bending stiffness'),
        array(pin_mask,wp.int32,(n,),'pin mask'),array(pin_targets,wp.vec3,(n,),'pin targets'),float(pin_stiffness),
        array(tet_gradients,wp.vec3,(model.tet_count,4),'tet gradients'),
        array(tet_measures,wp.float32,(model.tet_count,),'tet measures'))


energy.configure_module(__name__)
