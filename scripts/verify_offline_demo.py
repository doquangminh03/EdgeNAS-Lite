"""Verify archived pilot behavior without Gemini, dataset or checkpoint."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

from src.proposal.rule_based import evaluate_retrieved_candidates, retrieve_candidates_for_requirement


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    output=Path(args.output_dir).resolve();output.mkdir(parents=True,exist_ok=False)
    base=json.loads(Path('results/proposals/low_latency_balanced_demo.json').read_text())['requirement']
    scenarios=[('balanced','balanced',.2,12,'local_mac_cpu_01',False,0,'yolo26n_kitti_pilot_imgsz416_cpu'),
               ('accuracy','accuracy',.24,12,'local_mac_cpu_01',False,0,'yolo26n_kitti_pilot_imgsz512_cpu'),
               ('no_feasible','balanced',.9,1,'local_mac_cpu_01',False,1,None),
               ('hardware_mismatch','balanced',.2,12,'other_cpu',False,1,None),
               ('strict_archived','balanced',.2,12,'local_mac_cpu_01',True,1,None)]
    results=[]
    for name,goal,accuracy,latency,hardware,strict,expected_exit,selected in scenarios:
        request=deepcopy(base);request['request_id']='verify_'+name
        request['constraints']['minimum_map50_95']=accuracy
        request['constraints']['maximum_median_latency_ms']=latency
        request['preferences']['optimization_goal']=goal
        path=output/(name+'_request.json');path.write_text(json.dumps(request,indent=2)+'\n')
        report=output/(name+'_proposal.json')
        command=[sys.executable,'-m','src.proposal.cli',str(path),'--hardware-id',hardware,'--output',str(report)]
        if strict:command.append('--strict-comparison')
        completed=subprocess.run(command,capture_output=True,text=True)
        (output/(name+'.log')).write_text(completed.stdout+completed.stderr)
        assert completed.returncode==expected_exit, completed.stderr
        proposal=json.loads(report.read_text());selection=proposal['selection']
        assert (selection['selected_candidate_id'] if selection else None)==selected
        if name=='balanced':
            assert selection['evaluated_candidate_count']==7 and selection['feasible_candidate_count']==4
        results.append({'scenario':name,'exit_code':completed.returncode,'selected':selected,'status':proposal['proposal_status']})
    retrieval=retrieve_candidates_for_requirement('configs/requests/low_latency_balanced_demo.yaml')
    for item in retrieval['candidates']:
        del item['record']['benchmark']['protocol_version']
    missing=evaluate_retrieved_candidates(retrieval,expected_hardware_id='local_mac_cpu_01')
    assert not missing['evaluations'] and len(missing['excluded_candidate_ids'])==7
    results.append({'scenario':'missing_metadata','excluded':7,'entry_point':'evaluate_retrieved_candidates'})
    (output/'verification.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
