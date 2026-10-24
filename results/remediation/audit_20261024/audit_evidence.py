from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import statistics
import subprocess
import sys

root = Path.cwd()
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=False)
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()
def save(name, value):
    (out/name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
def percentile(values, p):
    v=sorted(values); position=(len(v)-1)*p/100; lower=int(position); upper=min(lower+1,len(v)-1)
    return v[lower]+(v[upper]-v[lower])*(position-lower)

snapshot=Path(sys.argv[2]) if len(sys.argv)>2 else Path('results/remediation/continuation_20261024/final_source_sha256.json')
expected=json.loads(snapshot.read_text()); matches=[]; differences=[]
for name, old in expected.items():
    actual=digest(Path(name)) if Path(name).is_file() else None
    (matches if actual==old else differences).append({'path':name,'expected':old,'actual':actual})
save('original_source_snapshot_check.json', {'manifest':str(snapshot), 'manifest_sha256':digest(snapshot), 'checked':len(expected), 'matched':len(matches), 'differences':differences})
assert not differences, differences

# Compare bytes with HEAD blob identity, independently of index/stat/mode cache.
result=subprocess.run(['git','ls-tree','-rz','--full-tree','HEAD'],capture_output=True,check=True)
tracked=[]
for entry in result.stdout.split(b'\0'):
    if not entry: continue
    info, name=entry.split(b'\t',1); mode,kind,oid=info.split()
    if kind!=b'blob': continue
    path=Path(name.decode()); data=path.read_bytes() if path.is_file() else None
    actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest() if data is not None else None
    tracked.append({'path':str(path),'head_blob':oid.decode(),'working_blob':actual,'content_equal':actual==oid.decode(),'head_mode':mode.decode()})
historical=[r for r in tracked if r['path'].startswith(('results/','knowledge/','artifacts/','configs/'))]
save('tracked_content_vs_HEAD.json',{'head':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip(),'tracked_blob_count':len(tracked),'historical_blob_count':len(historical),'historical_changed':[r for r in historical if not r['content_equal']],'changed':[r for r in tracked if not r['content_equal']],'files':tracked})
assert all(row['content_equal'] for row in historical)

base=Path('results/benchmarks/campaigns/continuation_measured_v2_20261024')
campaign=json.loads((base/'campaign.json').read_text()); old_check=json.loads(Path('results/remediation/continuation_20261024/campaign_verification.json').read_text())
assert digest(base/'campaign.json')==old_check['campaign_sha256']
assert campaign['status']=='completed' and len(campaign['results'])==35
manifest=campaign['dataset']['images']
assert len(manifest)==100 and len({r['path'] for r in manifest})==100
manifest_hash=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert manifest_hash==campaign['dataset']['manifest_sha256']
assert json.loads((base/'image_manifest.json').read_text())==campaign['dataset']
assert digest(Path(campaign['checkpoint']['path']))==campaign['checkpoint']['sha256']
assert all(digest(Path(row['path']))==row['sha256'] for row in manifest)
images=[str(Path(row['path']).relative_to('datasets/kitti/images/val')) for row in manifest]
assert len(campaign['schedule'])==5
sizes={416,448,480,512,576,608,640}
for schedule in campaign['schedule']:
    assert set(schedule['resolution_order'])==sizes and len(schedule['resolution_order'])==7
actual_order=[(r['session'],r['resolution']) for r in campaign['results']]
expected_order=[(s['session'],size) for s in campaign['schedule'] for size in s['resolution_order']]
assert actual_order==expected_order
signatures=set(); per_run=[]; raw_total=0; pooled_total=0; summaries={}; observed_threads=Counter()
for size in sorted(sizes):
    rows=[r for r in campaign['results'] if r['resolution']==size]
    assert len(rows)==5 and {r['session'] for r in rows}=={1,2,3,4,5}
    medians=[]; pool=[]
    for row in rows:
        path=Path(row['result_file']);raw=json.loads(path.read_text());samples=raw['raw_samples']
        assert row['raw_samples']==samples
        assert raw['selected_image_files']==images
        assert len(samples)==300
        assert Counter((s['image'],s['repetition']) for s in samples)==Counter((image,rep) for image in images for rep in (1,2,3))
        values=[s['latency_ms'] for s in samples]
        assert all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in values)
        median=round(statistics.median(values),3);p95=round(percentile(values,95),3)
        assert raw['summary']['median_latency_ms']==median==row['median_latency_ms']
        assert raw['summary']['p95_latency_ms']==p95==row['p95_latency_ms']
        assert raw['summary']['sample_count']==300
        assert raw['protocol']['image_size']==size
        assert raw['environment']['campaign_id']==campaign['campaign_id']
        assert raw['environment']['session_id']==row['session']
        observed_threads[raw['environment']['torch_num_threads']]+=1
        assert raw['hardware_id']==campaign['hardware_id']
        env=raw['environment'];signatures.add(tuple(env[k] for k in ('system','architecture','python_version','torch_version','ultralytics_version','numpy_version','torch_num_threads')))
        raw_total+=len(samples)
        if row['session']>=3:pool.extend(values);medians.append(median)
        per_run.append({'path':str(path),'sha256':digest(path),'session':row['session'],'resolution':size,'samples':len(samples),'median_ms':median,'p95_ms':p95})
    computed={'pooled_sample_count':len(pool),'pooled_median_latency_ms':statistics.median(pool),'pooled_p95_latency_ms':percentile(pool,95),'session_median_population_sd_ms':statistics.pstdev(medians)}
    for k,v in computed.items(): assert math.isclose(campaign['summary'][str(size)][k],v,rel_tol=1e-12,abs_tol=1e-12)
    assert campaign['summary'][str(size)]['included_sessions']==[3,4,5]
    assert campaign['summary'][str(size)]['excluded_stabilization_sessions']==[1,2]
    for k,v in computed.items(): assert math.isclose(old_check['summary'][str(size)][k],v,rel_tol=1e-12,abs_tol=1e-12)
    pooled_total+=len(pool);summaries[size]=computed
assert len(signatures)==1 and raw_total==10500 and pooled_total==6300
save('campaign_raw_audit.json',{'campaign_sha256':digest(base/'campaign.json'),'campaign_hash_matches_previous_verification':True,'completed_runs':len(per_run),'raw_samples':raw_total,'retained_samples':pooled_total,'unique_images':len(images),'current_checkpoint_hash_matches':True,'all_current_image_hashes_match':True,'environment_signature_count':len(signatures),'requested_cpu_threads':campaign['protocol']['cpu_threads'],'observed_torch_num_threads':dict(observed_threads),'numerical_integrity':'PASS','declared_thread_protocol':'MISMATCH','thread_recording_scope':'environment measured when building result, not per sample','summary':summaries,'raw_files':per_run})
print(json.dumps({'source_snapshot_checked':len(expected),'source_snapshot_mismatches':len(differences),'tracked_files':len(tracked),'historical_files':len(historical),'historical_content_changes':sum(not r['content_equal'] for r in historical),'campaign_runs':len(per_run),'raw_samples':raw_total,'retained_samples':pooled_total},indent=2))
