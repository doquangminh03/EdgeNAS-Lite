"""Sequential budget controller; never repeats an attempted experiment."""
import argparse
import json
from pathlib import Path
import re

from src.llm_agent.experiment_controller import run_experiment, save_report


def run_budget(*, root, batch_id, budget, requirement_file, hardware_id,
               template_candidate_id, resume=False, dry_run=False, model=None):
    root = Path(root).resolve()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', batch_id):
        raise ValueError('Invalid batch ID.')
    if type(budget) is not int or not 1 <= budget <= 7:
        raise ValueError('Budget must be an integer from 1 to 7.')

    settings = dict(
        budget=budget,
        requirement_file=requirement_file,
        hardware_id=hardware_id,
        template_candidate_id=template_candidate_id,
        model=model,
    )
    folder = root / 'results/llm_batches'
    path = folder / (batch_id + '.json')
    lock = folder / (batch_id + '.lock')

    if resume:
        state = json.loads(path.read_text(encoding='utf-8'))
        if state['settings'] != settings:
            raise ValueError('Resume settings must match the original batch.')
    else:
        if path.exists():
            raise FileExistsError('Batch exists; use --resume.')
        state = dict(
            schema_version='1.0',
            batch_id=batch_id,
            settings=settings,
            status='planned',
            slots=[],
        )

    def execute(run_id, preview=False):
        return run_experiment(
            root=root,
            requirement_file=requirement_file,
            hardware_id=hardware_id,
            template_candidate_id=template_candidate_id,
            run_id=run_id,
            budget=1,
            dry_run=preview,
            model=model,
        )

    if dry_run:
        slot = len(state['slots']) + 1
        if slot > budget:
            return dict(state, dry_run=True)
        preview = execute(f'{batch_id}_{slot:03d}', True)
        return dict(state, dry_run=True, next_slot_preview=preview)

    folder.mkdir(parents=True, exist_ok=True)
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(batch_id)

    try:
        if resume:
            state = json.loads(path.read_text(encoding='utf-8'))
            if state['settings'] != settings:
                raise ValueError('Batch settings changed.')
        else:
            with path.open('x', encoding='utf-8') as stream:
                json.dump(state, stream, indent=2)

        # Reconcile interrupted slots from their saved child reports.
        for slot in state['slots']:
            if slot['status'] != 'running':
                continue
            child = (
                root / 'results/llm_experiments'
                / (slot['run_id'] + '.json')
            )
            if child.exists():
                result = json.loads(child.read_text(encoding='utf-8'))
                slot['status'] = result['status']
            else:
                slot['status'] = 'interrupted_unknown'

        accepted = (
            'completed',
            'rejected_incompatible',
            'no_available_candidate',
        )
        blocked = [
            slot for slot in state['slots']
            if slot['status'] not in accepted
        ]
        if blocked:
            state['status'] = 'needs_review'
            save_report(path, state)
            raise ValueError(
                'An unfinished or failed slot needs review; '
                'it will not be rerun automatically.'
            )

        if any(
            slot['status'] == 'no_available_candidate'
            for slot in state['slots']
        ):
            state['status'] = 'exhausted'
            save_report(path, state)
            return state

        while len(state['slots']) < budget:
            run_id = f"{batch_id}_{len(state['slots']) + 1:03d}"

            # Reserve before API/model execution.
            # Failed attempts remain part of the budget.
            slot = dict(run_id=run_id, status='running')
            state['slots'].append(slot)
            state['status'] = 'running'
            save_report(path, state)
            print('Experiment:', run_id, flush=True)

            try:
                result = execute(run_id)
            except Exception as error:
                slot.update(status='failed', error=str(error))
                state['status'] = 'needs_review'
                save_report(path, state)
                raise

            slot['status'] = result['status']
            slot['candidate_id'] = result.get('candidate_id')

            if result['status'] == 'no_available_candidate':
                state['status'] = 'exhausted'
                save_report(path, state)
                return state

            if result['status'] not in (
                'completed', 'rejected_incompatible'
            ):
                state['status'] = 'needs_review'
                save_report(path, state)
                raise ValueError(
                    'Unexpected child status; inspect the run report.'
                )

            save_report(path, state)

        state['status'] = 'budget_finished'
        save_report(path, state)
        return state
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-id', required=True)
    parser.add_argument('--budget', required=True, type=int)
    parser.add_argument('--requirement', required=True)
    parser.add_argument('--hardware-id', required=True)
    parser.add_argument('--template-candidate-id', required=True)
    parser.add_argument('--model')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    try:
        result = run_budget(
            root=Path.cwd(),
            batch_id=args.batch_id,
            budget=args.budget,
            requirement_file=args.requirement,
            hardware_id=args.hardware_id,
            template_candidate_id=args.template_candidate_id,
            model=args.model,
            resume=args.resume,
            dry_run=args.dry_run,
        )
    except Exception as error:
        parser.exit(2, f'{type(error).__name__}: {error}\n')

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
