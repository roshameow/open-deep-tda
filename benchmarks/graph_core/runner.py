#!/usr/bin/env python3
"""External two-child scheduler, bounded45min; no agent required to stay alive."""
import os,sys,json,time,signal,subprocess,traceback
from pathlib import Path
if __package__:
    from .common import OUT,CODE,PLAN,DATASETS,registration,dump,sha,now,armdir,graphdir,base
else:
    from common import OUT,CODE,PLAN,DATASETS,registration,dump,sha,now,armdir,graphdir,base

def rss_tree(pids):
    rows=[tuple(map(int,l.split())) for l in subprocess.check_output(['ps','-axo','pid=,ppid=,rss='],text=True).splitlines() if len(l.split())==3];owned=set(pids);old=-1
    while old!=len(owned):old=len(owned);owned.update(p for p,pp,r in rows if pp in owned)
    return sum(r for p,pp,r in rows if p in owned)*1024

def main():
    reg=registration();start=time.monotonic();env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'}
    marker=dict(status='running',pid=os.getpid(),started_utc=now(),registration_sha256=sha(OUT/'preregistration.json'),task_id=os.environ.get('PI_SUBAGENT_TASK_ID'),parallel_mode='maximum2 isolated children, each1 native thread',global_cap_seconds=2700,stages={})
    with (OUT/'run_started.json').open('x') as f:json.dump(marker,f,indent=2)
    state=marker;active=[]
    def persist():
        state['elapsed_seconds']=time.monotonic()-start;state['active']=[dict(job=x['name'],pid=x['proc'].pid,seconds=time.monotonic()-x['start']) for x in active];dump(OUT/'checkpoint.json',state)
    def phase(jobs):
        queue=list(jobs)
        while queue or active:
            elapsed=time.monotonic()-start
            if elapsed>=2700:
                for item in active:
                    os.killpg(item['proc'].pid,signal.SIGKILL);item['proc'].wait();item['file'].close();state['stages'][item['name']]=dict(status='timeout',reason='global2700seconds',wall_seconds=time.monotonic()-item['start'])
                active.clear()
                for j in queue:state['stages'][j['name']]=dict(status='skipped',reason='global budget')
                queue.clear();persist();break
            while queue and len(active)<2:
                j=queue.pop(0)
                if any(state['stages'].get(dep,{}).get('status')!='completed' for dep in j.get('deps',[])):
                    state['stages'][j['name']]=dict(status='skipped',reason='failed dependency');continue
                path=OUT/'logs'/(j['name']+'.log');path.parent.mkdir(exist_ok=True);f=path.open('x');args=[sys.executable,str(CODE/j.get('script','worker.py'))]+j['args'];proc=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                active.append(dict(**j,proc=proc,file=f,start=time.monotonic(),peak=0))
            try:rss=rss_tree([x['proc'].pid for x in active]) if active else 0
            except Exception:rss=0
            if rss>8*2**30:
                for item in active:
                    if item['proc'].poll() is None:os.killpg(item['proc'].pid,signal.SIGKILL)
                    item['reason']='combined_sampled_RSS_limit'
            for item in active[:]:
                item['peak']=max(item['peak'],rss);t=time.monotonic()-item['start']
                if item['proc'].poll() is None and t>item['timeout']:os.killpg(item['proc'].pid,signal.SIGKILL);item['reason']='stage_timeout'
                code=item['proc'].poll()
                if code is not None:
                    item['proc'].wait();item['file'].close();status='completed' if code==0 else 'failed';row=dict(status=status,exit_code=code,wall_seconds=t,max_combined_active_rss_bytes=item['peak'])
                    if 'reason' in item:row.update(status='failed',reason=item['reason'])
                    state['stages'][item['name']]=row;active.remove(item)
            persist()
            if queue or active:time.sleep(.5) # bounded external scheduler, never foreground agent polling
    try:
        phase([dict(name='prepare-'+d,args=['prepare',d],timeout=90) for d in DATASETS])
        phase([dict(name=f'graph-{d}-{s}',args=['graph',d,str(s)],timeout=240,deps=['prepare-'+d]) for d in DATASETS for s in (0,1,2)])
        fits=[]
        for d in DATASETS:
            fits.append(dict(name=f'fit-{d}-pca-0',args=['fit',d,'0','pca'],timeout=90,deps=['prepare-'+d]))
            for s in (0,1,2):
                for m in ('strong','directA','umap'):fits.append(dict(name=f'fit-{d}-{m}-{s}',args=['fit',d,str(s),m],timeout=PLAN['resources']['fit_timeouts'][m],deps=[f'graph-{d}-{s}']))
        phase(fits)
        # Global label barrier includes terminal failures/skips, no remaining fits.
        assert not active and all(j['name'] in state['stages'] for j in fits)
        hashes={}
        for d in DATASETS:
            for path in base(d).glob('*-seed*/*.npy'):hashes[str(path)]=sha(path)
        dump(OUT/'embeddings_frozen.json',dict(frozen_utc=now(),terminal_fit_jobs=len(fits),fit_statuses={j['name']:state['stages'][j['name']] for j in fits},embeddings=hashes,labels_not_read_by_runner=True))
        phase([dict(name=j['name'].replace('fit-','eval-',1),args=['evaluate']+j['args'][1:],timeout=180,deps=[j['name']]) for j in fits])
        phase([dict(name='report-and-independent-closeout',script='report.py',args=[],timeout=180)])
        state['status']='completed' if all(x['status']=='completed' for x in state['stages'].values()) else 'incomplete'
    except BaseException:
        state.update(status='runner_failed',traceback=traceback.format_exc())
        for item in active:
            if item['proc'].poll() is None:os.killpg(item['proc'].pid,signal.SIGKILL);item['proc'].wait()
            item['file'].close()
        active.clear()
    state.update(finished_utc=now(),elapsed_seconds=time.monotonic()-start);persist();dump(OUT/'runner_status.json',state)
    # Even if the global cap leaves no report time, state lists every failure.
    inventory=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name not in ('runner_artifacts.json','runner.log'):
            inventory.append(dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)))
    dump(OUT/'runner_artifacts.json',dict(files=inventory,scope='Local run artifacts: do not publish raw data/models; self/live runner log excluded'))
if __name__=='__main__':main()
