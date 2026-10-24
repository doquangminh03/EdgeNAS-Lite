"""Evaluate pre-labelled language cases from saved responses; live is opt-in."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from src.llm_agent.interpreter import interpret_requirement, StaticResponseProvider


def evaluate_case(case, response_text):
    try:
        response = interpret_requirement(case['text'], request_id=case['id'],
                                         provider=StaticResponseProvider(response_text))
    except ValueError as error:
        return {'schema_correct': False, 'status_correct': False,
                'field_correctness': {}, 'error': str(error), 'manual_review': 'NOT_RUN'}
    fields = {}
    if case['expected']['status'] == response['status'] == 'ready':
        for section in ('target', 'constraints', 'preferences'):
            for key, value in case['expected']['requirement'][section].items():
                actual = response['requirement'][section].get(key)
                fields[section + '.' + key] = actual == value
    return {'schema_correct': True, 'status_correct': response['status'] == case['expected']['status'],
            'field_correctness': fields,
            'manual_review': 'PENDING' if response['status'] == 'needs_clarification' else 'NOT_APPLICABLE',
            'interpretation': response}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', default='configs/evaluation/language_heldout_v1.json')
    parser.add_argument('--responses', help='Directory of CASE_ID.json containing response_text and provenance.')
    parser.add_argument('--output', required=True)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--model')
    parser.add_argument('--max-calls', type=int, default=0)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error('Output already exists.')
    cases_path = Path(args.cases)
    cases = json.loads(cases_path.read_text())
    if args.live and (not args.model or args.max_calls < len(cases['cases'])):
        parser.error('Live requires --model and an explicit --max-calls covering every case.')
    provider = None
    if args.live:
        from src.llm_agent.gemini_provider import GeminiProvider
        provider = GeminiProvider(args.model)
    rows = []
    for case in cases['cases']:
        saved = Path(args.responses) / (case['id'] + '.json') if args.responses else None
        envelope = None
        if provider:
            from src.llm_agent.prompts import build_interpretation_messages
            from src.llm_agent import prompts
            envelope = {'response_text': provider.complete(build_interpretation_messages(case['text'], request_id=case['id'])),
                        'model_id': args.model, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                        'prompt_sha256': hashlib.sha256(Path(prompts.__file__).read_bytes()).hexdigest(),
                        'token_usage': None, 'cost': None}
        elif saved and saved.is_file():
            envelope = json.loads(saved.read_text())
        rows.append({'case_id': case['id'], 'status': 'RUN' if envelope else 'NOT_RUN',
                     'response': envelope,
                     'evaluation': evaluate_case(case, envelope['response_text']) if envelope else None})
    result = {'evaluation_version': 'language_heldout_v1', 'cases_sha256': hashlib.sha256(cases_path.read_bytes()).hexdigest(),
              'mode': 'live' if args.live else 'saved_responses', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'rubric': cases['rubric'], 'cases': rows,
              'limitations': ['Schema tests are not live model quality evidence.',
                              'Clarification relevance requires named reviewer and rubric; PENDING is not a pass.']}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
