"""Allowlisted FIXED9 compact followup export; historical FIXED30 is untouched."""
import json,hashlib,math,statistics
from pathlib import Path
import numpy as np
if __package__:
    from .common import ROOT,DATASETS,member
    from .publish import FORBIDDEN_KEYS
else:
    from common import ROOT,DATASETS,member
    from publish import FORBIDDEN_KEYS

DATASET_NAMES=('fashion','har','coil')

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def metric_record(r):
    result={k:r[k] for k in ('train_metrics','test_metrics')}
    result['classification']={k:r['classification'][k] for k in ('accuracy','balanced_accuracy','all_test_rows')}
    result['kmeans']={k:r['kmeans'][k] for k in ('ari','nmi','n_clusters')}
    if 'pose' in r:
        result['pose']={side:{k:v for k,v in r['pose'][side].items() if k in ('adjacent_view_recall','mean_neighbor_angle_degrees','queries')} for side in ('reference','embedding')}
    return result

def validate_compact_public(value):
    def walk(x):
        if isinstance(x,dict):
            for k,v in x.items():
                if k in FORBIDDEN_KEYS or k in {'outside_ids','original_nonconverged_ids','refinement_nonconverged_ids','query_index','path','raw_outputs'}:raise ValueError('Raw/local field in compact publication')
                walk(v)
        elif isinstance(x,list):
            for v in x:walk(v)
        elif isinstance(x,str):
            if x.startswith(('/','../','./')) or any(t in x for t in ('/Users/','/home/','/tmp/','docs/','file://')):raise ValueError('Local path in compact publication')
        elif isinstance(x,float) and not math.isfinite(x):raise ValueError('Nonfinite metric')
    walk(value)
    allowed={'schema','protocol','records','summary','totals','integrity','provenance','regressions','figures','historical_artifacts','timing_scope'}
    if set(value)-allowed:raise ValueError('Unknown compact publication field')
    if value.get('schema')!='graph-core-compact-confirmation-v2':raise ValueError('Wrong schema')
    records=value['records'];expected={(d,s) for d in DATASET_NAMES for s in (0,1,2)}
    if len(records)!=9 or {(r['dataset'],r['seed']) for r in records}!=expected:raise ValueError('All nine fixed arms required')
    allowed_record={'dataset','seed','status','mapping_domain','train_metrics','test_metrics','classification','kmeans','pose','query_count','outside_count','changed_rows','fit_file_unchanged','inside_count','inside_bytes_unchanged','all_final_finite','all_final_feasible','old_max_norm','final_max_norm','max_final_violation','feasibility_tolerance','phase_I_failures','phase_II_failures','phase_II_successes','original_unbounded_transform_seconds','refinement_seconds','validation_seconds','scoring_seconds','max_cached_objective_difference','max_cached_gradient_difference','TRAIN_centers_reproduce_all_old_clusters','readonly_baselines','changes_vs_unbounded','as_run_public_API_parity'}
    for r in records:
        if set(r)-allowed_record:raise ValueError('Unknown compact record field')
        if r.get('mapping_domain')!='train_hull':raise ValueError('Cannot relabel historical unbounded arms')
        if r['status']=='completed':
            for key in ('train_metrics','test_metrics'):
                if set(r[key])!={'5','15','50'}:raise ValueError('All scales required')
            expected_queries={'fashion':10000,'har':2947,'coil':480}[r['dataset']]
            if r['query_count']!=expected_queries:raise ValueError('All original TEST outputs required')
            if not r['inside_bytes_unchanged']:raise ValueError('Inside rows changed')
            if r['outside_count']+r['inside_count']!=r['query_count']:raise ValueError('Query exclusion/count mismatch')
    if len(value['summary'])!=3:raise ValueError('All datasets required')
    for summary in value['summary']:
        rows=[r for r in records if r['dataset']==summary['dataset'] and r['status']=='completed']
        if summary['completed']!=len(rows) or summary['expected']!=3:raise ValueError('Bad completion count')
        for key,entry in summary['statistics'].items():
            def extract(r):
                if key.startswith(('train_','test_')):
                    domain,metric,k=key.split('_');return r[domain+'_metrics'][k][metric]
                if key in ('accuracy','balanced_accuracy'):return r['classification'][key]
                if key in ('ari','nmi'):return r['kmeans'][key]
                return r[key]
            vals=[extract(r) for r in rows]
            if entry['values']!=vals or abs(entry['mean']-statistics.mean(vals))>1e-12:raise ValueError('Mean differs from all retained seeds')
            expected_std=statistics.stdev(vals) if len(vals)>1 else None
            if entry['sample_std']!=expected_std and (entry['sample_std'] is None or expected_std is None or abs(entry['sample_std']-expected_std)>1e-12):raise ValueError('Sample std mismatch')
    completed=[r for r in records if r['status']=='completed']
    if value['totals']['queries']!=sum(r['query_count'] for r in completed) or value['totals']['outside_refined']!=sum(r['outside_count'] for r in completed):raise ValueError('Global query counts mismatch')
    return True

def export_compact(directory,prior_directory):
    directory=Path(directory);prior=Path(prior_directory);generated=read(directory/'generation_results.json');scored=read(directory/'scoring_results.json');bygen={(r['dataset'],r['seed']):r for r in generated};byscore={(r['dataset'],r['seed']):r for r in scored}
    records=[]
    for d in DATASET_NAMES:
        for s in (0,1,2):
            g=bygen.get((d,s));r=byscore.get((d,s))
            if not g or not r or g['status']!='completed' or r['status']!='completed':records.append(dict(dataset=d,seed=s,mapping_domain='train_hull',status='failed_or_missing'));continue
            a=directory/f'{d}-seed{s}';old=prior/d/f'directA-seed{s}';inside=g['query_count']-g['outside_count']
            oldY=np.load(old/'test.npy');newY=np.load(a/'test.npy');assert oldY.shape==newY.shape and oldY.dtype==newY.dtype
            changed=int(np.any(oldY.view(np.uint8).reshape(len(oldY),-1)!=newY.view(np.uint8).reshape(len(newY),-1),axis=1).sum())
            assert changed==g['outside_count'] and digest(old/'fit.npy')==digest(a/'fit.npy')
            baselines={('unbounded_A' if m=='directA' else m):metric_record(v['record']) for m,v in r['readonly_baselines'].items()}
            q=read(a/'query_metrics.json');oq=read(old/'query_metrics.json');assert q['test_ids']==oq['test_ids'] and q['train']==oq['train']
            changes={}
            for k in ('5','15','50'):
                delta=np.asarray(q['test'][k])-np.asarray(oq['test'][k]);changes[k]={metric:dict(mean=float(delta[:,i].mean()),improved=int((delta[:,i]>0).sum()),worsened=int((delta[:,i]<0).sum()),unchanged=int((delta[:,i]==0).sum())) for i,metric in enumerate(('overlap','trustworthiness','continuity'))}
            # Already post-barrier completed artifacts; no model execution.
            y,_=member(ROOT/DATASETS[d]['annotations'],DATASETS[d]['test_labels'])
            with np.load(old/'evaluation_predictions.npz') as f:op=f['classification']
            with np.load(a/'evaluation_predictions.npz') as f:npred=f['classification']
            correct_old=op==y;correct_new=npred==y
            classification_changes=dict(predictions_changed=int((op!=npred).sum()),correct_gained=int((~correct_old&correct_new).sum()),correct_lost=int((correct_old&~correct_new).sum()),accuracy_delta=r['classification']['accuracy']-baselines['unbounded_A']['classification']['accuracy'])
            parity=g.get('public_parity',{})
            row=dict(dataset=d,seed=s,status='completed',mapping_domain='train_hull',**metric_record(r),query_count=g['query_count'],outside_count=g['outside_count'],changed_rows=changed,fit_file_unchanged=True,inside_count=inside,inside_bytes_unchanged=g['inside_bytes_unchanged'],all_final_finite=g['all_final_finite'],all_final_feasible=g['all_final_feasible'],old_max_norm=g['old_max_norm'],final_max_norm=g['final_max_norm'],max_final_violation=g['max_final_violation'],feasibility_tolerance=g['tolerance'],phase_I_failures=len(g['original_nonconverged_ids']),phase_II_failures=len(g['refinement_nonconverged_ids']),phase_II_successes=g['outside_count']-len(g['refinement_nonconverged_ids']),original_unbounded_transform_seconds=g['original_unbounded_transform_seconds'],refinement_seconds=g['refinement_seconds'],validation_seconds=g['validation']['seconds'],scoring_seconds=r.get('seconds',0.),max_cached_objective_difference=g['validation']['max_objective_difference'],max_cached_gradient_difference=g['validation']['max_gradient_linf_difference'],TRAIN_centers_reproduce_all_old_clusters=r['kmeans']['original_prediction_match'],readonly_baselines=baselines,changes_vs_unbounded=dict(test_geometry=changes,classification=classification_changes,ari_delta=r['kmeans']['ari']-baselines['unbounded_A']['kmeans']['ari'],nmi_delta=r['kmeans']['nmi']-baselines['unbounded_A']['kmeans']['nmi']))
            if parity:row['as_run_public_API_parity']={k:parity[k] for k in ('outside_queries','inside_queries','seconds','max_physical_linf_difference','public_center_exact','public_unit_exact','public_positive_ids_exact','public_probabilities_exact','public_probability_max_difference','all_routed_as_expected')}
            records.append(row)
    summaries=[]
    fields=['accuracy','balanced_accuracy','ari','nmi','outside_count','old_max_norm','final_max_norm','original_unbounded_transform_seconds','refinement_seconds','validation_seconds','scoring_seconds']+[f'{domain}_{metric}_{k}' for domain in ('train','test') for metric in ('overlap','trustworthiness','continuity') for k in (5,15,50)]
    for d in DATASET_NAMES:
        rows=[r for r in records if r['dataset']==d and r['status']=='completed'];stats={}
        for key in fields:
            def extract(r):
                if key.startswith(('train_','test_')):domain,metric,k=key.split('_');return r[domain+'_metrics'][k][metric]
                if key in ('accuracy','balanced_accuracy'):return r['classification'][key]
                if key in ('ari','nmi'):return r['kmeans'][key]
                return r[key]
            vals=[extract(r) for r in rows]
            if vals:stats[key]=dict(values=vals,mean=statistics.mean(vals),sample_std=statistics.stdev(vals) if len(vals)>1 else None)
        summaries.append(dict(dataset=d,completed=len(rows),expected=3,status='completed' if len(rows)==3 else 'incomplete',statistics=stats))
    complete=[r for r in records if r['status']=='completed'];integrity=read(directory/'integrity.json');freeze=directory/('source_freeze.json' if (directory/'source_freeze.json').exists() else 'preregistration.json');runner=read(directory/'runner_status.json')
    frozen=read(freeze);authorized={}
    for path,h in frozen.get('authorized',{}).items():
        authorized['production/'+Path(path).name]=h
    if not authorized:authorized={k:v for k,v in frozen.get('sources',{}).items() if k.startswith('production/')}
    value=dict(schema='graph-core-compact-confirmation-v2',protocol=dict(name='FIXED9-official-compact-v1',public_reproducer='graph-core-compact-followup-public-v2',mapping_domain='train_hull',historical_protocol='unbounded-v1; original30 records unchanged',selection='Fixed authorized TRAIN-hull followup motivated by TRAIN-internal validation; no official TEST tuning or seed selection',TEST_previously_seen=True,layout_refits=0,source_graph_rebuilds=0,query_exclusions=0,clipping=False,metric_population='Same512 TRAIN/1024 TEST queries vsFULLTRAIN; COILall480TEST; allTEST classification',norm_units='Euclidean norm in TRAIN-centered layout divided by TRAIN edge-median unit',solver='Exact production CompactMap, outside-only SLSQP over TRAIN convex hull; inside rows bitwise unchanged, feasible visited incumbent retained; no global optimum guarantee',labels='New all9 output/hash barrier before labels or saved cluster predictions'),records=records,summary=summaries,
        totals=dict(queries=sum(r['query_count'] for r in complete),outside_refined=sum(r['outside_count'] for r in complete),inside_bitwise_unchanged=sum(r['inside_count'] for r in complete),phase_II_successes=sum(r['phase_II_successes'] for r in complete),phase_II_failures=sum(r['phase_II_failures'] for r in complete),original_phase_I_failures_retained=sum(r['phase_I_failures'] for r in complete),original_unbounded_transform_seconds=sum(r['original_unbounded_transform_seconds'] for r in complete),refinement_seconds=sum(r['refinement_seconds'] for r in complete),validation_seconds=sum(r['validation_seconds'] for r in complete),scoring_seconds=sum(r['scoring_seconds'] for r in complete),followup_wall_seconds=runner.get('seconds')),
        timing_scope='Historical original unbounded transformation PLUS separately measured outside-only refinement; validation/API parity/scoring separate. NOT a fresh full train_hull benchmark timing.',integrity=dict(as_run_original_inputs_and_sources_unchanged_at_closeout=integrity['unchanged'],inside_rows_unchanged=all(r['inside_bytes_unchanged'] for r in complete),frozen_generation_before_labels=True,old_TRAIN_geometry_reused=True,old_TRAIN_KMeans_assignments_verified=all(r['TRAIN_centers_reproduce_all_old_clusters'] for r in complete)),
        provenance=dict(as_run_authorized_source_hashes=authorized,as_run_source_freeze_sha256=digest(freeze),as_run_global_output_barrier_sha256=digest(directory/'embeddings_frozen.json'),historical_aggregate='graph_core_confirmation.json',historical_aggregate_sha256=digest(ROOT/'benchmarks/results/graph_core_confirmation.json'),historical_source_manifest='graph_core_provenance.json',historical_source_manifest_sha256=digest(ROOT/'benchmarks/results/graph_core_provenance.json'),note='Separate followup; old numbers/source manifest remain historical and are not relabeled current default'),
        regressions=['Fashion accuracy decreases for seeds0/1; boundedness is not universal metric improvement','Fashion compact ARI/NMI remain below author UMAP','HAR compact ARI/NMI remain below old strong; accuracy remains below UMAP on average','COIL compact ARI/NMI remain below UMAP','Exact A full transformation still includes expensive historical unbounded phase; refinement-only time is not total inference cost'])
    validate_compact_public(value);return value


def write_compact(directory,prior_directory,destination):
    value=export_compact(directory,prior_directory)
    Path(destination).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');return value
