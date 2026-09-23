"""Small analytic API check: one declared source cycle, PH trained and checked.

Synthetic correctness check, NOT a real-data quality or out-of-sample claim.
No data, model, coordinate array or plot is written to disk.
"""
import json
import numpy as np
from open_deep_tda import DeepTDA
from open_deep_tda._ph_guided_training import GuidedLimits
from open_deep_tda._ph_guided_source import TeacherLimits


def main():
    theta=2*np.pi*np.arange(16)/16
    X=np.column_stack((np.cos(theta),np.sin(theta)))
    source_cycle=[[(i,(i+1)%16) for i in range(16)]]
    model=DeepTDA(steps=6,warmup_steps=0,topology_interval=1,
                  h0_size=16,h1_size=16,subset_bank_size=1,
                  standardize=False,missing_indicators=False)
    model.fit_with_topology_guidance(X,cycles=source_cycle,
             birth_radius=.5,survival_radius=.9,strategy='single',
             limits=GuidedLimits(max_vertices=16,max_feature_workspace_bytes=100000,
                                 teacher_steps=0,contact_steps=0,cut_steps=0,max_seconds=60),
             teacher_limits=TeacherLimits(max_vertices=16,max_pairs=120,max_iterations=10,max_seconds=30))
    print(json.dumps({'scope':'16 supplied TRAIN rows only; synthetic analytic circle',
                      'certificate':model.report_['guided_training']['certificate'],
                      'sampled_native_h1_updates':model.training_coverage_['h1_updates'],
                      'query_topology_guarantee':False},indent=2))

if __name__=='__main__':main()
