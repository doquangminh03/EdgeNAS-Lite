"""Create explicit review lists; never stage or mutate Git metadata."""
import json
from pathlib import Path
import subprocess

BASE = Path('results/remediation/thread_fix_20261024')
prior = json.loads(Path('results/remediation/audit_20261024/commit_candidates.json').read_text())
include = {row['path']: row['category'] for row in prior['include']}
include['tests/test_cpu_benchmark.py'] = 'thread_control_regression'
include['docs/THREAD_CONTROL_FIX.md'] = 'thread_control_documentation'
for path in Path('results/benchmarks/campaigns/threads1_fixed_20261024').rglob('*'):
    if path.is_file() and not path.name.startswith('._'):
        include[str(path)] = 'verified_1_thread_campaign_v2'
for path in BASE.rglob('*'):
    if path.is_file() and not path.name.startswith('._') and path.name != 'tests.log':
        include[str(path)] = 'thread_fix_evidence'
for name in ('commit_include.txt', 'commit_exclude.txt', 'commit_candidates.json', 'source_sha256.json', 'final_review.json'):
    include[str(BASE / name)] = 'thread_fix_evidence'
raw = subprocess.run(['git', 'ls-files', '--others', '--exclude-standard', '-z'], capture_output=True, check=True).stdout
untracked = {path.decode() for path in raw.split(b'\0') if path}
old_reasons = {row['path']: row['reason'] for row in prior['exclude']}
exclude = []
for path in sorted(untracked - set(include)):
    reason = old_reasons.get(path, 'intermediate_fixture_failure_log_keep_locally')
    exclude.append({'path': path, 'reason': reason})
result = {'policy': 'Proposal only; nothing staged. Keep old invalid campaign with thread mismatch annotation; new v2 campaign remains outside the selection index.',
          'include': [{'path': path, 'category': category} for path, category in sorted(include.items())],
          'exclude': exclude, 'always_exclude_patterns': prior['always_exclude_patterns']}
(BASE / 'commit_candidates.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
(BASE / 'commit_include.txt').write_text(''.join(path + '\n' for path in sorted(include)))
(BASE / 'commit_exclude.txt').write_text(''.join(row['path'] + '\n' for row in exclude))
print('Proposed paths:', len(include), '; excluded existing paths:', len(exclude))
