"""Allowlisted aggregate export. Never copies coordinates, IDs, labels or logs."""
import json
import math
import re
from pathlib import Path

DATASETS=('fashion','har','coil')
METHODS=('pca','strong','directA','umap')
METRIC_KEYS={f'{split}_{metric}_{k}' for split in ('train','test') for metric in ('overlap','trustworthiness','continuity') for k in (5,15,50)}
SCALAR_KEYS=METRIC_KEYS|{'fit_seconds','transform_seconds','accuracy','balanced_accuracy','ari','nmi','pose_adjacency','reference_pose_adjacency','optimizer_nonconvergences'}
A_KEYS={'all_queries','normalizer_population','max_residual_difference','gradient_median','gradient_max','gradient_gtol_count','optimizer_success','objective_increases'}
AUDIT_KEYS={'queries','population','k','strict_recall','tie_aware_recall'}
FORBIDDEN_KEYS={'query_ids','sample_ids','nonconverged_ids','train_ids','test_ids','exact_query_ids','per_query','embeddings','coordinates','predictions','raw_logs','model_path','traceback','center','positive_ids','fit_statuses'}


def validate_public_aggregate(value):
    """Strict public result schema plus recursive privacy tripwires."""
    def walk(x):
        if isinstance(x,dict):
            for key,v in x.items():
                if key in FORBIDDEN_KEYS:raise ValueError('Disallowed raw field: '+key)
                walk(v)
        elif isinstance(x,list):
            for v in x:walk(v)
        elif isinstance(x,str):
            if any(token in x for token in ('/Users/','/home/','/tmp/','docs/','file://','\\Users\\')):raise ValueError('Local/research path in public artifact')
            if x.startswith(('/', './', '../')):raise ValueError('Filesystem path in public result')
        elif isinstance(x,float) and not math.isfinite(x):raise ValueError('Nonfinite public metric')
    walk(value)
    allowed_root={'schema','provenance_manifest','protocol','dataset_metadata','records','summary','graph_audits','model_diagnostics','runtime','required_negative_facts','integrity','figures','environment'}
    if set(value)-allowed_root:raise ValueError('Unknown public root field')
    if value.get('schema')!='graph-core-confirmation-summary-v1':raise ValueError('Unknown schema')
    records=value['records']
    expected={(d,m,s) for d in DATASETS for m in METHODS for s in ([0] if m=='pca' else [0,1,2])}
    observed=[(r['dataset'],r['method'],r['seed']) for r in records]
    if len(observed)!=30 or set(observed)!=expected:raise ValueError('Exactly all30 dataset/method/seed records are required')
    for record in records:
        allowed={'dataset','method','seed','status','A_audit','initializer'}|SCALAR_KEYS
        if set(record)-allowed:raise ValueError('Unknown record field')
        if record['status']=='completed' and not METRIC_KEYS<=record.keys():raise ValueError('Missing a declared scale/domain metric')
    summaries=value['summary']
    if len(summaries)!=12 or {(r['dataset'],r['method']) for r in summaries}!={(d,m) for d in DATASETS for m in METHODS}:raise ValueError('Expected all12 dataset/method summaries')
    import statistics
    for row in summaries:
        if set(row)-{'dataset','method','status','completed','expected','statistics'}:raise ValueError('Unknown summary field')
        matching=[r for r in records if r['dataset']==row['dataset'] and r['method']==row['method'] and r['status']=='completed']
        expected=1 if row['method']=='pca' else 3
        if row['expected']!=expected or row['completed']!=len(matching):raise ValueError('Misleading seed completion count')
        if row['status']=='complete' and len(matching)!=expected:raise ValueError('Incomplete mean mislabeled complete')
        for key,entry in row['statistics'].items():
            if key not in SCALAR_KEYS or set(entry)!={'mean','sample_std','values'}:raise ValueError('Unknown statistic')
            vals=[r[key] for r in matching if key in r]
            if entry['values']!=vals or not vals:raise ValueError('Statistic values differ from retained seeds')
            if abs(entry['mean']-statistics.mean(vals))>1e-10:raise ValueError('Mean differs from retained seeds')
            if len(vals)==1:
                if entry['sample_std'] is not None:raise ValueError('One-run control must not have variance')
            elif abs(entry['sample_std']-statistics.stdev(vals))>1e-10:raise ValueError('Sample std differs from retained seeds')
    return True


def export_aggregate(directory):
    directory=Path(directory)
    result=json.loads((directory/'results.json').read_text())
    reg=json.loads((directory/'preregistration.json').read_text())
    runner=json.loads((directory/'runner_status.json').read_text())
    environment=reg.get('environment',{})
    env_path=directory/'environment.json'
    if env_path.exists():environment=json.loads(env_path.read_text())
    environment={k:environment[k] for k in ('platform','packages') if k in environment}
    records=[]
    for row in result['records']:
        out={k:row[k] for k in ('dataset','method','seed','status')}
        out.update({k:row[k] for k in SCALAR_KEYS if k in row})
        if 'A_audit' in row:
            out['A_audit']={k:row['A_audit'][k] for k in A_KEYS if k in row['A_audit']}
            out['A_audit']['optimizer_nonconvergence']=len(row['A_audit'].get('nonconverged_ids',[]))
        if 'initializer' in row:out['initializer']={k:row['initializer'][k] for k in ('components','center_rule')}
        records.append(out)
    summary=[]
    for row in result['summary']:
        summary.append({**{k:row[k] for k in ('dataset','method','status','completed','expected')},'statistics':{k:v for k,v in row['statistics'].items() if k in SCALAR_KEYS}})
    graphs=[]
    diagnostics=[]
    for dataset in DATASETS:
        for seed in (0,1,2):
            path=directory/dataset/f'graph{seed}'/'result.json'
            if path.exists():
                graph=json.loads(path.read_text())
                graphs.append(dict(dataset=dataset,seed=seed,status=graph['status'],seconds=graph.get('seconds'),audits={k:{n:v for n,v in a.items() if n in AUDIT_KEYS} for k,a in graph.get('audits',{}).items()}))
            path=directory/dataset/f'directA-seed{seed}'/'result.json'
            if path.exists():
                r=json.loads(path.read_text());keys=('fit_seconds','optimization_seconds','transform_seconds','query_count','optimizer_successes','optimizer_nonconvergences','nonfinite_or_exceptions','max_gradient_linf','positive_updates','negative_updates','directed_edges','unvisited_edges','unit')
                diagnostics.append(dict(dataset=dataset,seed=seed,**{k:r[k] for k in keys if k in r}))
    stages={k:{n:v for n,v in row.items() if n in ('status','exit_code','wall_seconds','max_combined_active_rss_bytes','reason')} for k,row in runner['stages'].items()}
    output=dict(schema='graph-core-confirmation-summary-v1',environment=environment,provenance_manifest='graph_core_provenance.json',
        protocol=dict(seeds=[0,1,2],methods=list(METHODS),PCA='one full-SVD control per dataset',source_neighbors=15,author_neighbors_argument=16,layout_epochs=300,A_optimizer=reg['A_options'],strong_configs=reg['configs'],
            preprocessing='Fashion/COIL full official-TRAIN-fitted unwhitened PCA64; HAR TRAIN-only standardization on official subject split; source float32 retained',
            geometry='512 TRAIN queries and1024 TEST queries (COILall480) against FULLTRAIN; TRAIN self excluded; exact ranks k5/15/50',
            labels='Read only after all30 fitting jobs terminal and outputs hash-frozen. AllTEST uniform15NN; TRAIN KMeans n_init10, count=unique TRAIN labels, TEST ARI/NMI.',
            adaptations=['Shared audited ANN15 fitting graph; author self column makes16','Author force_approximation_algorithm=True retains precomputed COIL graph','Process-local graph provider for unchanged strong training','Component-aware V2 initialization and exact all-anchor A query mapper'],
            limitations=['Historically seen TEST, not virgin external validation','Single fixed protocol, three stochastic seeds, not dataset-level confidence','Unequal compute/objectives; no topology or official-equivalence claim','COIL strong means documented stress600 control, not best possible']),
        dataset_metadata={d:{k:reg['plan']['datasets'][d][k] for k in ('n','m','d')} for d in DATASETS},
        records=records,summary=summary,graph_audits=graphs,model_diagnostics=diagnostics,
        runtime=dict(status=runner['status'],elapsed_seconds=runner['elapsed_seconds'],parallel_children=2,native_threads_per_child=1,sampled_combined_rss_gib_limit=8,wall_cap_seconds=2700,stages=stages,
            note='Method times omit shared graph/preprocessing; sampled RSS is not a hard memory quota; JIT/loading/serialization scope differs'),
        required_negative_facts=['Fashion direct+A ARI/NMI below author UMAP','HAR direct+A ARI/NMI below old strong graph control; accuracy below UMAP','Fashion direct+A transformation substantially slower than author UMAP and strong MLP','COIL direct+A ARI/NMI slightly below UMAP despite higher overlap/accuracy','Extreme finite A query positions exist; full-range panels retain them, TRAIN-range panels are display only'],
        integrity=dict(as_run_registered_source_count=len(reg['sources']),as_run_source_hashes_verified=result['source_hashes_unchanged'],published_model_data=False))
    validate_public_aggregate(output)
    return output


def write_aggregate(directory,destination):
    data=export_aggregate(directory)
    Path(destination).write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n')
    return data
