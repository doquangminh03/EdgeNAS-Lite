"""Independent check of the fixed 1-thread campaign; no model execution."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

BASE = Path('results/remediation/thread_fix_20261024')
CAMPAIGN = Path('results/benchmarks/campaigns/threads1_fixed_20261024')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def percentile(values):
    ordered = sorted(values)
    position = (len(ordered) - 1) * .95
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def main():
    campaign = json.loads((CAMPAIGN / 'campaign.json').read_text())
    assert campaign['status'] == 'completed'
    assert campaign['protocol']['version'] == 'cpu_campaign_v2'
    assert campaign['protocol']['cpu_threads'] == 1
    assert campaign['protocol']['backend_setup_predictions'] == 1
    assert len(campaign['results']) == 35
    assert len(campaign['source_sha256']) == 2
    for name, expected in campaign['source_sha256'].items():
        assert sha256(Path(name)) == expected, name
    assert sha256(Path(campaign['checkpoint']['path'])) == campaign['checkpoint']['sha256']
    manifest = campaign['dataset']['images']
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()
    assert hashlib.sha256(manifest_bytes).hexdigest() == campaign['dataset']['manifest_sha256']
    assert json.loads((CAMPAIGN / 'image_manifest.json').read_text()) == campaign['dataset']
    assert len(manifest) == 100
    for image in manifest:
        assert sha256(Path(image['path'])) == image['sha256']
    image_names = [str(Path(image['path']).relative_to('datasets/kitti/images/val')) for image in manifest]
    assert [(r['session'], r['resolution']) for r in campaign['results']] == [
        (session['session'], size) for session in campaign['schedule'] for size in session['resolution_order']
    ]
    sizes = (416, 448, 480, 512, 576, 608, 640)
    signatures = set()
    setup_counts = Counter()
    verified = []
    summaries = {}
    for size in sizes:
        rows = [row for row in campaign['results'] if row['resolution'] == size]
        assert len(rows) == 5 and {row['session'] for row in rows} == {1, 2, 3, 4, 5}
        pool, medians = [], []
        for row in rows:
            path = Path(row['result_file'])
            raw = json.loads(path.read_text())
            assert raw['raw_samples'] == row['raw_samples']
            assert raw['protocol']['version'] == 'cpu_threads_v2'
            assert raw['protocol']['requested_cpu_threads'] == row['requested_cpu_threads'] == 1
            assert raw['protocol']['image_size'] == size
            assert raw['protocol']['warmup_runs'] == 10
            assert raw['protocol']['repetitions_per_image'] == 3
            assert raw['protocol']['backend_setup_predictions'] == 1
            assert raw['hardware_id'] == campaign['hardware_id']
            environment = raw['environment']
            assert environment['campaign_id'] == campaign['campaign_id']
            assert environment['session_id'] == row['session']
            assert environment['requested_cpu_threads'] == environment['torch_num_threads'] == row['observed_cpu_threads'] == 1
            observations = environment['thread_observations']
            for stage in ('before_warmup', 'after_warmup', 'before_timing', 'after_timing'):
                assert type(observations[stage]) is int and observations[stage] == 1
            setup_counts[observations['after_backend_setup']] += 1
            signatures.add(tuple(environment[k] for k in (
                'system', 'architecture', 'python_version', 'torch_version',
                'ultralytics_version', 'numpy_version', 'torch_num_threads',
            )))
            samples = raw['raw_samples']
            assert len(samples) == 300
            assert raw['selected_image_files'] == image_names
            assert Counter((s['image'], s['repetition']) for s in samples) == Counter(
                (image, repetition) for image in image_names for repetition in (1, 2, 3)
            )
            for sample in samples:
                assert type(sample['torch_num_threads_before']) is int and sample['torch_num_threads_before'] == 1
                assert type(sample['torch_num_threads_after']) is int and sample['torch_num_threads_after'] == 1
                assert math.isfinite(sample['latency_ms']) and sample['latency_ms'] > 0
            values = [sample['latency_ms'] for sample in samples]
            median = round(statistics.median(values), 3)
            p95 = round(percentile(values), 3)
            assert median == raw['summary']['median_latency_ms'] == row['median_latency_ms']
            assert p95 == raw['summary']['p95_latency_ms'] == row['p95_latency_ms']
            if row['session'] >= 3:
                pool.extend(values)
                medians.append(median)
            verified.append({'path': str(path), 'sha256': sha256(path), 'session': row['session'], 'resolution': size, 'samples': len(samples)})
        summary = {'pooled_sample_count': len(pool), 'pooled_median_latency_ms': statistics.median(pool),
                   'pooled_p95_latency_ms': percentile(pool), 'session_median_population_sd_ms': statistics.pstdev(medians)}
        for key, value in summary.items():
            assert math.isclose(value, campaign['summary'][str(size)][key], rel_tol=1e-12, abs_tol=1e-12)
        assert campaign['summary'][str(size)]['included_sessions'] == [3, 4, 5]
        summaries[size] = summary
    assert len(signatures) == 1
    prior = json.loads((BASE / 'prior_evidence_sha256.json').read_text())
    prior_changes = [name for name, digest in prior.items() if sha256(Path(name)) != digest]
    assert not prior_changes, prior_changes
    historical = []
    tree = subprocess.run(['git', 'ls-tree', '-rz', '--full-tree', 'HEAD'], capture_output=True, check=True).stdout
    for entry in tree.split(b'\0'):
        if not entry:
            continue
        info, name = entry.split(b'\t', 1)
        mode, kind, oid = info.split()
        path = name.decode()
        if kind != b'blob' or not path.startswith(('results/', 'knowledge/', 'artifacts/', 'configs/')):
            continue
        data = Path(path).read_bytes()
        assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == oid.decode(), path
        historical.append(path)
    report = {'status': 'PASS', 'campaign': str(CAMPAIGN / 'campaign.json'),
              'campaign_sha256': sha256(CAMPAIGN / 'campaign.json'),
              'protocol_version': 'cpu_campaign_v2', 'completed_runs': 35, 'raw_samples': 10500,
              'retained_samples': 6300, 'requested_threads': 1, 'observed_threads': 1,
              'all_sample_thread_boundaries_match': True,
              'after_backend_setup_observations': dict(setup_counts),
              'environment_signature_count': len(signatures), 'source_hashes_match_campaign_start': True,
              'prior_evidence_files_unchanged': len(prior), 'tracked_historical_files_unchanged': len(historical),
              'summary': summaries, 'raw_files': verified,
              'limitations': ['Thread counts are checked at prediction boundaries, not continuously.',
                              'No new accuracy validation or index promotion.',
                              'The host was used for documentation while the campaign ran; not a dedicated isolated host.',
                              'Old campaign requested 1 but recorded 7; no automatic cross-protocol performance claim.']}
    with (BASE / 'campaign_verification.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('raw_files', 'summary')}, indent=2))


if __name__ == '__main__':
    main()
