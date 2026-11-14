"""Fixed float32 query corpus; no imports from or writes to the solver."""
import numpy as np


def synthetic_cases():
    result=[]
    for kind in [0,1]:
        if kind==0:
            start=np.array([[.2,.2,1],[0,0,0],[1,0,0],[0,1,0]],np.float32)
        else:
            start=np.array([[-1,0,1],[1,0,1],[0,-1,0],[0,1,0]],np.float32)
        variants=[]
        variants.append(('stationary',start.copy(),start.copy()))
        end=start.copy();end[:,2]+=2
        variants.append(('common_translation',start.copy(),end))
        end=start.copy();end[:1 if kind==0 else 2,2]-=1;end[1 if kind==0 else 2:,2]+=1
        variants.append(('multiple_crossing',start.copy(),end))
        end=start.copy();end[0,2]=-1
        variants.append(('single_crossing',start.copy(),end))
        value=start.copy();value[0,2]=0
        if kind==1:value[1,2]=0
        end=value.copy();end[:,2]+=2
        variants.append(('initial_contact',value,end))
        value=np.zeros((4,3),np.float32);value[0,0]=-1
        if kind==1:value[1,0]=-1
        end=value.copy();end[:,0]+=2
        variants.append(('degenerate_common_translation',value,end))
        for label,at in [('nan_start',0),('infinite_end',1)]:
            value=start.copy();end=start.copy()
            (value if at==0 else end)[0,0]=np.nan if at==0 else np.inf
            variants.append((label,value,end))
        for name,value,end in variants:
            for exact in [0,1]:
                for preserve in [0,1]:
                    result.append({'name':f'{"vf" if kind==0 else "ee"}_{name}_exact{exact}_gap{preserve}',
                        'start':value.copy(),'end':end.copy(),'kind':kind,'exact_geometry':exact,'preserve_gap':preserve})
    return result
