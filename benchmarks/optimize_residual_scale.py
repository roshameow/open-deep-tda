"""Second bounded TRAIN-only ablation: condition ONLY the residual MLP input.

Same official HAR TRAIN subject split and metrics as optimize_geometry.py.
No class labels or official TEST data. Run --preregister before --run.
The scalar sqrt(561) counters reference normalization's small coordinate RMS;
it does not change the reference metric, PCA skip, PH target, or output units.
"""
import math

import optimize_geometry as experiment

SCALE = math.sqrt(561)
experiment.CANDIDATES = [
    dict(name='graph_r1_unscaled', steps=2400, geometry_objective='fuzzy_graph',
         fuzzy_repulsion=1., lambda_h0=0., lambda_h1=0., residual_input_scale=1.),
    dict(name='graph_r1_conditioned', steps=2400, geometry_objective='fuzzy_graph',
         fuzzy_repulsion=1., lambda_h0=0., lambda_h1=0., residual_input_scale=SCALE),
    dict(name='graph_r5_conditioned', steps=2400, geometry_objective='fuzzy_graph',
         fuzzy_repulsion=5., lambda_h0=0., lambda_h1=0., residual_input_scale=SCALE),
    dict(name='graph_weak_conditioned', steps=2400, geometry_objective='fuzzy_graph',
         fuzzy_repulsion=1., lambda_h0=.1, lambda_h1=.01, residual_input_scale=SCALE),
    dict(name='stress_conditioned', steps=2400, residual_input_scale=SCALE),
]
experiment.PLAN = dict(experiment.PLAN, protocol='residual-conditioning-ablation-v1',
                       candidates=experiment.CANDIDATES,
                       hypothesis='reference normalization makes per-coordinate RMS small; condition neural branch without changing PH/PCA metric',
                       restrictions='No class labels, official TEST rows, best-seed selection or early stopping. Separate registration from first pilot.')

if __name__ == '__main__':
    experiment.main()
