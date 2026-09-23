"""Public analytic example emits a bounded certificate, no coordinates/model."""
import json
import os
from pathlib import Path
import subprocess
import sys


def test_guided_analytic_example(tmp_path):
    root=Path(__file__).resolve().parents[1]
    env=os.environ.copy()
    env['PYTHONPATH']=str(root/'python') + os.pathsep + env.get('PYTHONPATH','')
    run=subprocess.run([sys.executable,str(root/'examples/run_ph_guided.py')],cwd=tmp_path,
                       env=env,capture_output=True,text=True,timeout=45,check=True)
    result=json.loads(run.stdout)
    assert result['certificate']['accepted']
    assert result['certificate']['selected_rank']==1
    assert result['sampled_native_h1_updates']>0
    assert result['query_topology_guarantee'] is False
    assert not list(tmp_path.iterdir())
