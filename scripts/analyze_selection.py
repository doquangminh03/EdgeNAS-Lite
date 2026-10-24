"""Archived evidence sensitivity, Pareto frontier and equal-budget search replay."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import yaml

from src.proposal.rule_based import propose_from_knowledge
from src.requirement_parser.parser import load_requirement
from src.candidate_selector.selector import score_components, _ranking_key


WEIGHTS = {'equal': (1, 1, 1), 'accuracy_heavy': (8, 1, 1),
           'latency_heavy': (1, 8, 1), 'size_heavy': (1, 1, 8)}
SEEDS = (11, 29, 47, 83, 101)


def trajectory(order, initial, rows, budget):
    seen = set(initial)
    best = []
    first_feasible = first_best = None
    for step in range(budget + 1):
        ranked = [row for row in rows if row['candidate_id'] in seen]
        winner = ranked[0] if ranked else None
        if winner and first_feasible is None:
            first_feasible = step
        if winner and winner['candidate_id'] == rows[0]['candidate_id'] and first_best is None:
            first_best = step
        best.append({'step': step, 'best_score': winner['score'] if winner else None})
        if step < budget and step < len(order):
            seen.add(order[step])
    return {'trajectory': best, 'experiments_to_feasible': first_feasible,
            'experiments_to_best': first_best, 'failure_to_find_feasible': first_feasible is None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--orders', help='Optional archived LLM orders keyed by request ID; each value is a list of candidate IDs.')
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--model')
    parser.add_argument('--max-calls', type=int, default=0)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error('Output already exists.')
    provider = None
    audit_path = output.with_name(output.stem + '_live_calls.json')
    if args.live:
        if not args.model or args.max_calls < 60 or audit_path.exists():
            parser.error('Live requires explicit --model --max-calls 60 and new audit output.')
        from src.llm_agent.gemini_provider import GeminiProvider
        provider = GeminiProvider(args.model)
    live_audit = {'prompt_source_sha256': hashlib.sha256(Path('src/llm_agent/candidate_proposer.py').read_bytes()).hexdigest(), 'model_id': args.model, 'trials': {}, 'maximum_calls': 60,
                  'limitations': ['Seeds control random baseline only; LLM trials are not independent statistical replicates.',
                                 'Provider exposes no token/cost metadata through this adapter.']}
    public_pool = {'fixed_configuration': yaml.safe_load(Path('configs/search_space_llm_pool.yaml').read_text())['fixed_configuration']}
    requirement_path = Path('configs/requests/low_latency_balanced_demo.yaml')
    req = load_requirement(requirement_path)
    proposal = propose_from_knowledge(requirement_path, expected_hardware_id='local_mac_cpu_01')
    ranking = proposal['selection']['ranking']
    policies = {}
    for name, weights in WEIGHTS.items():
        scored = deepcopy(ranking)
        for row in scored:
            components = score_components(row['metrics'], req)
            row['score'] = round(sum(weight * components[key] for weight, key in zip(weights, ('accuracy_headroom','latency_headroom','model_size_headroom'))) / sum(weights), 6)
        policies[name] = sorted(scored, key=lambda row: _ranking_key(row, 'balanced'))
    all_rows = [{'candidate_id': item['record']['candidate_id'], 'metrics': {
        'map50_95': item['record']['accuracy']['map50_95'],
        'median_latency_ms': item['record']['benchmark']['median_latency_ms'],
        'model_size_mb': item['record']['model']['model_size_mb']}} for item in proposal['candidates']]
    def dominates(a, b):
        x, y = a['metrics'], b['metrics']
        return (x['map50_95'] >= y['map50_95'] and x['median_latency_ms'] <= y['median_latency_ms'] and x['model_size_mb'] <= y['model_size_mb'] and x != y)
    pareto = [row for row in all_rows if not any(dominates(other,row) for other in all_rows)]
    orders = json.loads(Path(args.orders).read_text()) if args.orders else {}
    search = []
    initial = ['yolo26n_kitti_pilot']
    pool = sorted([row['candidate_id'] for row in all_rows if row['candidate_id'] not in initial], key=lambda cid: next(item['record']['accuracy']['image_size'] for item in proposal['candidates'] if item['record']['candidate_id']==cid))
    # Requirements, seeds, initial evidence and budget fixed before strategy evaluation.
    for name, accuracy, latency in [('balanced',.2,12),('accuracy',.24,12),('relaxed',.2,20),('impossible',.9,1)]:
        request = req.to_dict(); request['request_id'] = 'replay_' + name
        request['constraints']['minimum_map50_95'] = accuracy
        request['constraints']['maximum_median_latency_ms'] = latency
        from src.proposal.rule_based import evaluate_retrieved_candidates
        from src.requirement_parser.parser import parse_requirement_data
        from src.candidate_selector.selector import select_candidate
        retrieval = {**proposal, 'requirement':request}
        evaluated = evaluate_retrieved_candidates(retrieval, expected_hardware_id='local_mac_cpu_01')
        rows = select_candidate(parse_requirement_data(request), evaluated['evaluations'])['ranking']
        for seed in SEEDS:
            random_order = list(pool); random.Random(seed).shuffle(random_order)
            methods={'ascending':pool,'random_without_replacement':random_order}
            if request['request_id'] in orders:
                order=orders[request['request_id']]
                if len(order)!=len(set(order)) or any(cid not in pool for cid in order):
                    raise ValueError('Invalid saved LLM order.')
                methods['llm_saved_order']=order
            live_trial = None
            if provider:
                from scripts.evaluate_hidden_search import run_trial
                from src.llm_agent.candidate_proposer import request_proposal
                oracle = {item['record']['accuracy']['image_size']: {
                    'candidate_id': item['record']['candidate_id'],
                    'image_size': item['record']['accuracy']['image_size'],
                    'map50_95': item['record']['accuracy']['map50_95'],
                    'median_latency_ms': item['record']['benchmark']['median_latency_ms'],
                    'model_size_mb': item['record']['model']['model_size_mb'],
                } for item in proposal['candidates']}
                def save_trial(trial):
                    live_audit['trials'][request['request_id'] + '_' + str(seed)] = trial
                    audit_path.parent.mkdir(parents=True, exist_ok=True)
                    audit_path.write_text(json.dumps(live_audit, indent=2, allow_nan=False) + '\n')
                live_trial = run_trial(provider, request_proposal, public_pool, request,
                                       oracle, [640], 3, rows, save_trial)
                methods['llm_live_archived_reveal'] = [oracle[size]['candidate_id'] for size in live_trial['order']]
            search.append({'request':request, 'seed':seed, 'initial_evidence':initial, 'budget':3,
                           'llm_status':'LIVE_ARCHIVED_REVEAL' if live_trial else ('SAVED_ORDER' if 'llm_saved_order' in methods else 'NOT_RUN'),
                           'methods':{method:{'order':order[:3],**trajectory(order,initial,rows,3)} for method,order in methods.items()},
                           'proposal_latency_seconds':sum(call['elapsed_seconds'] for call in live_trial['calls']) if live_trial else None, 'token_usage':None,'cost':None})
    source_hashes = {item['record_path']:hashlib.sha256(Path(item['record_path']).read_bytes()).hexdigest() for item in proposal['candidates']}
    result={'version':'policy_sensitivity_replay_v1','evidence_kind':'archived_measurement_replay',
            'weights':WEIGHTS,'sensitivity':policies,'pareto_frontier':pareto,'search_replay':search,'source_sha256':source_hashes,
            'limitations':['Equal weights do not equalize effect: headroom scales differ.',
                           'File size is constant and does not measure runtime RAM.',
                           'Search replay has hypothetical experiment slots, not measured execution cost.',
                           'Archived software environments differ; no controlled resolution-only comparison.',
                           'Saved orders are exploratory unless generated with the exact initial evidence and hidden metrics withheld.',
                           'No superiority claim; inspect llm_status to distinguish planned and executed trials.']}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')


if __name__=='__main__':
    main()
