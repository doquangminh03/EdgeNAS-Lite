import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml
from scripts.benchmark_campaign import make_plan, execute_campaign
from scripts.evaluate_language import evaluate_case
from scripts.dataset_manifest import build_manifest
from src.proposal.compatibility import check_strict_comparison, STRICT_PROVENANCE_FIELDS
from src.proposal.rule_based import propose_from_knowledge


class TestRemediationTools(unittest.TestCase):
    def test_strict_mode_excludes_archived_records(self):
        result = propose_from_knowledge('configs/requests/low_latency_balanced_demo.yaml',
                                       expected_hardware_id='local_mac_cpu_01', strict_comparison=True)
        self.assertEqual(result['proposal_status'], 'no_compatible_candidates')
        self.assertEqual(result['evaluations'], [])

    def test_strict_comparison_rejects_changed_environment(self):
        record = {'candidate_id':'a','comparison_provenance':dict.fromkeys(STRICT_PROVENANCE_FIELDS,'fixture')}
        second = copy.deepcopy(record)
        self.assertEqual(check_strict_comparison([record,second])['status'],'compatible')
        second['comparison_provenance']['torch_version']='different'
        self.assertEqual(check_strict_comparison([record,second])['status'],'incompatible')
        del second['comparison_provenance']['checkpoint_sha256']
        self.assertTrue(any(p['reason']=='missing' for p in check_strict_comparison([record,second])['problems']))

    def test_language_fields_and_manual_review_are_separate(self):
        cases=json.loads(Path('configs/evaluation/language_heldout_v1.json').read_text())['cases']
        case=cases[0]
        response={'status':'ready','requirement':copy.deepcopy(case['expected']['requirement']),'questions':[],'reasons':[]}
        result=evaluate_case(case,json.dumps(response))
        self.assertTrue(result['schema_correct'])
        self.assertTrue(all(result['field_correctness'].values()))
        response['requirement']['constraints']['maximum_median_latency_ms']=99
        self.assertFalse(evaluate_case(case,json.dumps(response))['field_correctness']['constraints.maximum_median_latency_ms'])
        question={'status':'needs_clarification','requirement':None,'questions':['What latency?'],'reasons':[]}
        self.assertEqual(evaluate_case(cases[12],json.dumps(question))['manual_review'],'PENDING')

    def test_archived_language_responses_still_validate(self):
        count = 0
        for pattern in ('semantic_eval_v1_*.json', 'clarification_eval_v1_*.json'):
            for path in Path('results/llm_evaluations').glob(pattern):
                archived = json.loads(path.read_text())
                for row in archived['results']:
                    requirement = row['expected_requirement']
                    case = {'id': requirement['request_id'] if requirement else row['case_id'],
                            'text': row['user_text'],
                            'expected': {'status': row['expected_status'], 'requirement': requirement}}
                    with self.subTest(case=row['case_id']):
                        result = evaluate_case(case, json.dumps(row['actual']))
                        self.assertTrue(result['schema_correct'])
                        self.assertTrue(result['status_correct'])
                        self.assertTrue(all(result['field_correctness'].values()))
                    count += 1
        self.assertEqual(count, 9)

    def test_file_inputs_reject_nonfinite_values_in_every_constraint(self):
        from src.requirement_parser.parser import load_requirement, RequirementValidationError
        base=json.loads(Path('configs/evaluation/language_heldout_v1.json').read_text())['cases'][0]['expected']['requirement']
        with tempfile.TemporaryDirectory() as directory:
            for field in base['constraints']:
                for value in (float('nan'),float('inf'),float('-inf')):
                    request=copy.deepcopy(base);request['constraints'][field]=value
                    for suffix,text in [('yaml',yaml.safe_dump(request)),('json',json.dumps(request))]:
                        path=Path(directory)/('request.'+suffix);path.write_text(text)
                        with self.subTest(field=field,value=value,suffix=suffix):
                            with self.assertRaisesRegex(RequirementValidationError,field):
                                load_requirement(path)
                request=copy.deepcopy(base);request['constraints'][field]=1.23456789
                text=json.dumps(request).replace('1.23456789','1e309')
                path=Path(directory)/'overflow.json';path.write_text(text)
                with self.assertRaisesRegex(RequirementValidationError,field):load_requirement(path)

    def test_campaign_executes_multiple_resolutions_and_retains_failed_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve()
            (root/'images').mkdir()
            (root/'images/one.png').write_bytes(b'fixture')
            (root/'model.pt').write_bytes(b'fixture')
            config=yaml.safe_load(Path('configs/benchmark_cpu.yaml').read_text())
            config['model']['checkpoint']='model.pt'
            config['dataset']['images_dir']='images'
            config['selection']['sample_size']=1
            config['protocol']['repetitions_per_image']=1
            path=root/'config.yaml';path.write_text(yaml.safe_dump(config))
            output=root/'campaign'
            plan=make_plan(path,root=root,run_id='fixture',hardware_id='fixture_cpu',output_dir=output,
                           resolutions=(416,640),session_count=2,stabilization_sessions=1)
            def run(command,**kwargs):
                cfg=yaml.safe_load(Path(command[3]).read_text())
                result={'selected_image_files':['one.png'],'raw_samples':[{'image':'one.png','latency_ms':10,'torch_num_threads_before':1,'torch_num_threads_after':1}],
                        'summary':{'median_latency_ms':10,'p95_latency_ms':10},
                        'protocol':{'version':'cpu_threads_v2','requested_cpu_threads':1},
                        'environment':{'requested_cpu_threads':1,'torch_num_threads':1,
                                       'thread_observations':dict.fromkeys(('before_warmup','after_warmup','before_timing','after_timing'),1)}}
                Path(cfg['output']['result_file']).write_text(json.dumps(result))
            with patch('scripts.benchmark_campaign.subprocess.run',side_effect=run) as runner:
                result=execute_campaign(plan,base_config=path,root=root,output_dir=output)
            self.assertEqual(runner.call_count,4)
            self.assertEqual(result['status'],'completed')
            self.assertEqual(len(list(output.rglob('benchmark_imgsz_*.yaml'))),4)
            self.assertEqual(set(result['summary']),{'416','640'})
            failed=root/'failed'
            plan=make_plan(path,root=root,run_id='failed',hardware_id='fixture_cpu',output_dir=failed,
                           session_count=2,stabilization_sessions=1)
            with patch('scripts.benchmark_campaign.subprocess.run',side_effect=RuntimeError('fixture failure')):
                with self.assertRaises(RuntimeError):
                    execute_campaign(plan,base_config=path,root=root,output_dir=failed)
            self.assertEqual(json.loads((failed/'campaign.json').read_text())['status'],'failed')
            mismatch = root/'mismatch'
            plan=make_plan(path,root=root,run_id='mismatch',hardware_id='fixture_cpu',output_dir=mismatch,
                           session_count=2,stabilization_sessions=1)
            def wrong_threads(command, **kwargs):
                run(command, **kwargs)
                cfg=yaml.safe_load(Path(command[3]).read_text())
                result_path=Path(cfg['output']['result_file'])
                result=json.loads(result_path.read_text())
                result['environment']['torch_num_threads']=7
                result_path.write_text(json.dumps(result))
            from scripts.benchmark_campaign import CampaignError
            with patch('scripts.benchmark_campaign.subprocess.run',side_effect=wrong_threads):
                with self.assertRaisesRegex(CampaignError, 'Thread mismatch'):
                    execute_campaign(plan,base_config=path,root=root,output_dir=mismatch)
            report=json.loads((mismatch/'campaign.json').read_text())
            self.assertEqual(report['status'],'failed')
            self.assertEqual(report['results'],[])
            self.assertIsNone(report['summary'])


    def test_dataset_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for split in ('train','val'):
                (root/'images'/split).mkdir(parents=True)
                (root/'labels'/split).mkdir(parents=True)
                (root/'images'/split/'one.png').write_bytes(b'same image')
                (root/'labels'/split/'one.txt').write_text('0 0.5 0.5 0.2 0.2\n')
            with self.assertRaisesRegex(ValueError,'overlap'):
                build_manifest(root)
