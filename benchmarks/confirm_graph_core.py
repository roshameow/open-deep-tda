#!/usr/bin/env python3
"""Register or run the fixed official-split graph benchmark. Never downloads data."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--preregister',action='store_true',help='synthetic compatibility preflight, then TRAIN-only registration')
    action.add_argument('--run',action='store_true',help='run the existing unchanged registration with a bounded supervisor')
    action.add_argument('--preflight-only',action='store_true',help='synthetic compatibility tests only; reads no dataset cache')
    parser.add_argument('--output',type=Path,required=True,help='new local artifact directory, normally under outputs/')
    parser.add_argument('--protocol',choices=('unbounded-v1','compact-v2'),default='unbounded-v1',help='legacy fixed30 unbounded benchmark, or separate fixed9 TRAIN-hull followup; production API defaults to train_hull')
    parser.add_argument('--prior-output',type=Path,help='completed unbounded-v1 output, required when registering compact-v2')
    args=parser.parse_args(argv)
    if __package__:
        from .graph_core.common import validate_output
    else:
        from graph_core.common import validate_output
    output=validate_output(args.output)
    code=Path(__file__).resolve().parent/'graph_core'
    env={**os.environ,'GRAPH_CORE_OUTPUT':str(output),'PYTHONDONTWRITEBYTECODE':'1'}
    if args.prior_output is not None and args.protocol!='compact-v2':parser.error('--prior-output belongs only to compact-v2')
    if args.protocol=='compact-v2' and args.preregister and args.prior_output is None:parser.error('compact-v2 registration requires --prior-output')
    if args.run:
        if not (output/'preregistration.json').exists():parser.error('Missing preregistration; do not run without it')
        if (output/'run_started.json').exists():parser.error('Run already started; no retry/overwrite allowed')
        registered=json.loads((output/'preregistration.json').read_text())['plan']['protocol']
        expected='graph-core-compact-followup-public-v2' if args.protocol=='compact-v2' else 'graph-core-official-confirmation-public-port-v1'
        if registered!=expected:parser.error('Protocol differs from registration; historical and compact runs cannot be relabeled')
        command=[sys.executable,str(code/'compact.py'),'supervise'] if args.protocol=='compact-v2' else [sys.executable,str(code/'runner.py')]
        return subprocess.call(command,env=env)
    if output.exists() and any(output.iterdir()):parser.error('Preflight/registration requires a new empty output directory')
    output.mkdir(parents=True,exist_ok=True)
    if args.protocol=='compact-v2':
        with (output/'compact_preflight.log').open('x') as log:
            subprocess.run([sys.executable,str(code/'compact.py'),'preflight'],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=60,check=True)
        if args.preregister:
            with (output/'freeze.log').open('x') as log:
                subprocess.run([sys.executable,str(code/'compact.py'),'freeze',str(args.prior_output.resolve())],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=90,check=True)
        return 0
    for phase,timeout in [('preflight',120),('preflight_strong',60)]:
        with (output/(phase+'.log')).open('x') as log:
            subprocess.run([sys.executable,str(code/'worker.py'),phase],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout,check=True)
    if args.preregister:
        with (output/'freeze.log').open('x') as log:
            subprocess.run([sys.executable,str(code/'worker.py'),'freeze'],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=60,check=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
