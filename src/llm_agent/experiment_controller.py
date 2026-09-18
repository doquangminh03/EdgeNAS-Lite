"""One budgeted Gemini experiment, with measured evidence and a durable run report.

Run from the project root. No automatic retry/resume and no overwriting measurements.
Hardware identity is operator supplied, not automatically detected.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace

import yaml


def load_backend():
    from src.llm_agent import candidate_proposer as proposer
    from src.candidate_runner import runner
    from src.knowledge_database.loader import load_knowledge_base
    from src.knowledge_database.query import query_candidates
    from src.proposal.compatibility import check_candidate_compatibility
    from src.proposal.rule_based import propose_from_knowledge
    from src.llm_agent.gemini_provider import GeminiProvider
    return SimpleNamespace(proposer=proposer, runner=runner, load=load_knowledge_base,
                           query=query_candidates, check=check_candidate_compatibility,
                           rank=propose_from_knowledge, provider=GeminiProvider)


def atomic_write(path, text):
    """Replace only controller-owned reports or a checked index, on the same filesystem."""
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_report(path, report):
    atomic_write(path, json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def register_candidate(index_path, root, candidate_id, record_path, expected_text, loader):
    """Validate a prospective index before replacing it; refuse edits observed during the run."""
    if index_path.read_text(encoding='utf-8') != expected_text:
        raise ValueError('Knowledge index changed during this run; inspect before registering.')
    index = yaml.safe_load(expected_text)
    if not isinstance(index, dict) or not isinstance(index.get('candidate_records'), list):
        raise ValueError('Invalid knowledge index.')
    relative = record_path.relative_to(root).as_posix()
    for entry in index['candidate_records']:
        if entry.get('candidate_id') == candidate_id or entry.get('record_path') == relative:
            raise ValueError('Candidate ID or record path is already indexed.')
    index['candidate_records'].append({'candidate_id': candidate_id, 'record_path': relative})
    text = yaml.safe_dump(index, sort_keys=False, allow_unicode=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.candidate_index_', suffix='.yaml', dir=index_path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            stream.write(text)
        loader(Path(temporary), root)
        if index_path.read_text(encoding='utf-8') != expected_text:
            raise ValueError('Knowledge index changed before registration.')
        os.replace(temporary, index_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run_experiment(*, root, requirement_file, hardware_id, template_candidate_id,
                   run_id, pool_file='configs/search_space_llm_pool.yaml',
                   base_file='configs/search_space.yaml', index_file='knowledge/index.yaml',
                   budget=1, dry_run=False, model=None, backend=None):
    if type(budget) is not int or budget != 1:
        raise ValueError('This version supports exactly one new experiment per run.')
    if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('run-id must contain only letters, digits, underscores and hyphens.')
    if not isinstance(hardware_id, str) or not hardware_id or any(c.isspace() for c in hardware_id):
        raise ValueError('A non-empty hardware-id without whitespace is required.')
    root = Path(root).resolve()
    if Path.cwd().resolve() != root:
        raise ValueError('Run this command from the project root (dataset paths are relative).')
    api = backend or load_backend()
    def resolve(value):
        path = (root / value).resolve()
        path.relative_to(root)
        return path
    index_path = resolve(index_file)
    index_text = index_path.read_text(encoding='utf-8')
    pool = api.proposer.load_search_space(resolve(pool_file), check_paths=False)
    base = api.proposer.load_search_space(resolve(base_file), check_paths=False)
    for key in ('fixed_configuration', 'evaluation', 'candidate_naming'):
        if pool[key] != base[key]:
            raise ValueError('Pool differs from baseline: ' + key)
    fixed = pool['fixed_configuration']
    if (fixed['model']['family'], fixed['model']['scale']) != ('YOLO26', 'n'):
        raise ValueError('Only the YOLO26n pilot is supported.')
    rank_kwargs = dict(project_root=root, index_file=index_path,
                       model_family=fixed['model']['family'], expected_hardware_id=hardware_id)
    baseline = api.rank(resolve(requirement_file), **rank_kwargs)
    requirement = baseline['requirement']
    if requirement['target'] != dict(task=fixed['task'], device=fixed['deployment']['device'], dataset=fixed['dataset']['name']):
        raise ValueError('Requirement and pool targets differ.')
    entries = api.query(api.load(index_path, root), dataset=fixed['dataset']['name'],
                        model_family=fixed['model']['family'], device=fixed['deployment']['device'])
    context = api.proposer.collect_context(pool, entries, hardware_id, root, api.check)
    if not context['measured_evidence']:
        raise ValueError('No compatible measured evidence for this model and hardware.')
    report = dict(schema_version='1.0', run_id=run_id, status='planned', dry_run=dry_run,
                  experiment_budget=1, experiments_attempted=0, hardware_id=hardware_id,
                  requirement=requirement, context=context, baseline=baseline,
                  created_at_utc=datetime.now(timezone.utc).isoformat(),
                  limitations=['Hardware ID is operator supplied.',
                               'Pilot compatibility does not verify full software/environment equivalence.',
                               'No automatic retry or resume; inspect partial results after failure.'])
    if not context['available_image_sizes']:
        report['status'] = 'no_available_candidate'
        return report
    # Check template and benchmark identity before spending an API request.
    api.runner.load_template_candidate(pool, root, template_candidate_id)
    benchmark_config = api.runner.load_benchmark_config(resolve(pool['evaluation']['benchmark_config']))
    api.runner.resolve_hardware_id(benchmark_config, hardware_id)
    config_path = root / 'configs' / ('llm_experiment_' + run_id + '.yaml')
    proposal_path = root / 'results/proposals' / ('llm_experiment_' + run_id + '.json')
    report_path = root / 'results/llm_experiments' / (run_id + '.json')
    for path in (config_path, proposal_path, report_path):
        if os.path.lexists(path):
            raise FileExistsError('Run output already exists: ' + str(path))
    if dry_run:
        return report
    lock = root / 'results/.llm_experiment.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    # One controller writer per checkout. Manual index edits must also be avoided during a run.
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps({'run_id': run_id, 'pid': os.getpid()}))
    try:
        if index_path.read_text(encoding='utf-8') != index_text:
            raise ValueError('Index changed during preflight; retry with fresh context.')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2)
        try:
            provider = api.provider(**({'model': model} if model else {}))
            report['provider'] = {'name': 'gemini', 'model': getattr(provider, 'model', model)}
            report['status'] = 'requesting_proposal'
            save_report(report_path, report)
            proposal = api.proposer.request_proposal(provider, requirement, pool, context)
            plan = api.proposer.build_single_search_space(pool, proposal, context['available_image_sizes'], run_id)
            candidate_id = api.proposer.candidate_id_for(plan, proposal['image_size'])
            record_path = root / 'results/candidates' / (candidate_id + '.json')
            for path in [record_path] + api.runner.benchmark_result_paths(root, candidate_id):
                if os.path.lexists(path):
                    raise FileExistsError('Candidate or partial benchmark already exists: ' + str(path))
            report.update(candidate_id=candidate_id, proposal=proposal, execution_config=str(config_path.relative_to(root)),
                          status='proposed_unmeasured')
            api.proposer.write_new_pair(config_path, plan, proposal_path,
                                       dict(report, pool_snapshot=pool, prompt_version=api.proposer.PROMPT_VERSION,
                                            system_prompt=api.proposer.SYSTEM_PROMPT))
            save_report(report_path, report)
            api.runner.run_candidates(config_path, candidate_id, True, False,
                                      template_candidate_id=template_candidate_id, hardware_id=hardware_id)
            report.update(status='evaluating', experiments_attempted=1)
            save_report(report_path, report)
            report['runner_result'] = api.runner.run_candidates(
                config_path, candidate_id, False, False,
                template_candidate_id=template_candidate_id, hardware_id=hardware_id)
            record = api.runner.load_json(record_path)
            if record.get('candidate_id') != candidate_id:
                raise ValueError('Measured candidate ID differs from proposal.')
            for section in ('accuracy', 'benchmark'):
                if record.get(section, {}).get('image_size') != proposal['image_size']:
                    raise ValueError('Measured image size differs from proposal.')
            compatibility = api.check(record, expected_hardware_id=hardware_id)
            report['compatibility'] = compatibility
            if compatibility['status'] != 'compatible':
                report['status'] = 'rejected_incompatible'
                save_report(report_path, report)
                return report
            # Also verify model/checkpoint against the pool, as the compatibility policy alone does not.
            evidence = api.proposer.collect_context(pool, [{'record': record, 'record_path': record_path}], hardware_id, root, api.check)
            if not evidence['measured_evidence']:
                raise ValueError('Measured model/checkpoint does not match the pool.')
            report['status'] = 'registering'
            save_report(report_path, report)
            register_candidate(index_path, root, candidate_id, record_path, index_text, api.load)
            report['index_updated'] = True
            save_report(report_path, report)
            report['final_selection'] = api.rank(resolve(requirement_file), **rank_kwargs)
            report['status'] = 'completed'
            report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
            save_report(report_path, report)
            return report
        except Exception as error:
            report.update(status='failed', error_type=type(error).__name__, error=str(error))
            save_report(report_path, report)
            raise
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requirement', required=True)
    parser.add_argument('--hardware-id', required=True)
    parser.add_argument('--template-candidate-id', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--pool', default='configs/search_space_llm_pool.yaml')
    parser.add_argument('--base', default='configs/search_space.yaml')
    parser.add_argument('--index', default='knowledge/index.yaml')
    parser.add_argument('--budget', type=int, choices=[1], default=1)
    parser.add_argument('--model')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        report = run_experiment(root=Path.cwd(), requirement_file=args.requirement,
                                hardware_id=args.hardware_id, template_candidate_id=args.template_candidate_id,
                                run_id=args.run_id, pool_file=args.pool, base_file=args.base, index_file=args.index,
                                budget=args.budget, dry_run=args.dry_run, model=args.model)
    except Exception as error:
        parser.exit(2, f'{type(error).__name__}: {error}\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report['status'] in ('planned', 'completed', 'no_available_candidate') else 1


if __name__ == '__main__':
    raise SystemExit(main())
