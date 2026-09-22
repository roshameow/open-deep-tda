"""Analytic repair demonstrations; not a real-data quality benchmark.

python examples/repair_structural_layout.py
The source is known planar, so these are feasible regression cases. No source
coordinates are supplied to the solver: it sees distances, required cycles and
the failed initial layout. All returned results must pass independent checkers.
"""
import json
import numpy as np

from open_deep_tda.topology import distance_matrix
from open_deep_tda.structural_dual_layout import solve_structural_layout


SQUARE=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
CYCLE=[(0,1),(1,2),(2,3),(3,0)]


def run_cases():
    broken=np.array([[0.,0.],[3.,.1],[1.4,.2],[1.4+np.sqrt(1-.35**2),-.15]])
    cases=[
        ('intact',SQUARE,SQUARE.copy(),[CYCLE]),
        ('missing_ring',SQUARE,broken,[CYCLE]),
        ('wrong_correspondence',SQUARE,SQUARE[[0,2,1,3]],[CYCLE]),
        ('external_filling',np.vstack([SQUARE,[10.,10.]]),np.vstack([SQUARE,[.5,.5]]),[CYCLE]),
        ('merged_classes',np.vstack([SQUARE,SQUARE+[4.,0.]]),np.vstack([SQUARE,SQUARE]),
         [CYCLE,[(i+4,j+4) for i,j in CYCLE]]),
        ('wrong_global_bridge',np.column_stack(([0.,1.,10.,11.],np.zeros(4))),
         np.column_stack(([0.,1.,100.,101.],np.zeros(4))),[]),
    ]
    reports={}
    for name,source,initial,cycles in cases:
        radii=dict(birth_radius=1.,survival_radius=1.2) if cycles else {}
        result=solve_structural_layout(distance_matrix(source),initial,cycles,h0_tolerance=.05,**radii)
        assert result['status']=='certified', (name,result['status'])
        reports[name]=dict(status=result['status'],rounds=len(result['history']),
                          h0_max_merge_error=result['certificate']['h0']['max_merge_error'],
                          h1_rank=result['certificate']['h1']['surviving_rank'] if cycles else None,
                          embedding=result['embedding'].tolist())
    return reports


if __name__=='__main__':
    print(json.dumps(run_cases(),indent=2,allow_nan=False))
