#!/usr/bin/env python3
"""Execute the twelve MANIFEST-ARCH-SLICE checks and emit JSON."""
from __future__ import annotations
import argparse, hashlib, json, re, subprocess, sys, tempfile
from pathlib import Path

CHECK_IDS=[
'manifest.schema_version','manifest.roadmap_scenarios_covered','manifest.roadmap_requirements_covered',
'manifest.requirement_has_exactly_one_owner','manifest.owners_resolve','manifest.statuses_valid',
'manifest.commands_valid','manifest.counts_positive','manifest.inventory_rows_owned',
'manifest.inventory_digest_reproducible','manifest.decisions_represented','manifest.scenario_counts_match_roadmap']
def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def roadmap_requirements(text:str)->set[str]:
 return {x for x in re.findall(r'\*\*([A-Z]{2,4}-\d{3})(?:\s+—[^*]*)?\.?\*\*',text) if not x.startswith('DL-')}
def roadmap_scenarios(text:str)->dict[str,tuple[int,int]]:
 out={}
 for line in text.splitlines():
  m=re.match(r'\| `([A-Z][A-Z0-9-]+(?:-V1)?|MANIFEST-ARCH-SLICE)` \| ([^|]+) \|',line)
  if not m: continue
  nums=[int(x) for x in re.findall(r'\d+',m.group(2))]
  if nums: out[m.group(1)]=(nums[0],nums[-1])
 return out
def execute(root:Path,manifest_path:Path,mutant:str|None):
 manifest=json.loads(manifest_path.read_text());
 if mutant=='MUT-MANIFEST-ORPHAN':
  manifest['requirements'][0]['owned_by_scenario']=None
 roadmap=(root/manifest['sources']['roadmap']).read_text(); decisions_text=(root/manifest['sources']['decisions']).read_text()
 rreq=roadmap_requirements(roadmap); rsc=roadmap_scenarios(roadmap); scenarios=manifest.get('scenarios',{}); allowed=set(manifest.get('allowed_statuses',[]))
 results={}
 def check(cid,fn):
  try:
   val=fn();
   if val is False: raise AssertionError('condition was false')
   results[cid]=None
  except Exception as e: results[cid]=f'{type(e).__name__}: {e}'
 check(CHECK_IDS[0],lambda: manifest['schema_version']==1 or (_ for _ in ()).throw(AssertionError('schema_version must equal 1')))
 check(CHECK_IDS[1],lambda: set(scenarios)==set(rsc) or (_ for _ in ()).throw(AssertionError(f'manifest={sorted(scenarios)} roadmap={sorted(rsc)}')))
 check(CHECK_IDS[2],lambda: {x['id'] for x in manifest['requirements']}==rreq or (_ for _ in ()).throw(AssertionError('requirement coverage differs from roadmap')))
 def owners():
  bad=[x['id'] for x in manifest['requirements'] if not isinstance(x.get('owned_by_scenario'),str) or not x['owned_by_scenario']]
  assert not bad,f'missing owner: {bad}'
 check(CHECK_IDS[3],owners)
 check(CHECK_IDS[4],lambda: not (bad:=[x['id'] for x in manifest['requirements'] if x.get('owned_by_scenario') and x['owned_by_scenario'] not in scenarios]) or (_ for _ in ()).throw(AssertionError(f'unknown owner: {bad}')))
 def statuses():
  vals=[x.get('status') for x in manifest['requirements']]+[x.get('status') for x in scenarios.values()]
  assert allowed and all(x in allowed for x in vals),f'invalid status in {vals}'
 check(CHECK_IDS[5],statuses)
 def commands():
  bad=[]
  for sid,s in scenarios.items():
   c=s.get('command');
   if not isinstance(c,list) or not c or not all(isinstance(x,str) and x for x in c) or sid not in c: bad.append(sid)
  assert not bad,f'invalid commands: {bad}'
 check(CHECK_IDS[6],commands)
 check(CHECK_IDS[7],lambda: all(isinstance(s.get(k),int) and s[k]>0 for s in scenarios.values() for k in ('logical_count','test_count')) or (_ for _ in ()).throw(AssertionError('counts must be positive integers')))
 inv=manifest['inventory']; meta=json.loads((root/inv['metadata']).read_text()); rows=[json.loads(x) for x in (root/inv['symbols']).read_text().splitlines() if x]
 def inv_owned():
  assert meta['discovered_rows']>0 and len(rows)==meta['discovered_rows'],'inventory count mismatch/zero'
  bad=[x['row_id'] for x in rows if not x.get('owner') or not x.get('status')]; assert not bad,f'unowned inventory rows: {bad[:5]}'
 check(CHECK_IDS[8],inv_owned)
 def inv_digest():
  assert sha(root/inv['search_output'])==meta['search_output_sha256'],'search digest mismatch'
  assert sha(root/inv['symbols'])==meta['symbols_sha256'],'symbols digest mismatch'
  with tempfile.TemporaryDirectory() as d:
   subprocess.run([sys.executable,str(root/'scripts/conformance/inventory.py'),'--root',str(root),'--output-dir',d],check=True,capture_output=True,text=True)
   assert (Path(d)/'search-output.txt').read_bytes()==(root/inv['search_output']).read_bytes(),'scanner search output drift'
   assert (Path(d)/'symbols.jsonl').read_bytes()==(root/inv['symbols']).read_bytes(),'scanner symbol output drift'
 check(CHECK_IDS[9],inv_digest)
 def decisions():
  expected={'DL-003','DL-009','DL-023','DL-024','DL-025','DL-022'}; actual={x['id'] for x in manifest.get('decisions',[])}
  assert actual==expected,f'decision set {actual}'
  assert all(x in decisions_text for x in expected),'decision source missing entry'
 check(CHECK_IDS[10],decisions)
 def counts():
  bad={sid:(s['logical_count'],s['test_count'],rsc.get(sid)) for sid,s in scenarios.items() if rsc.get(sid)!=(s['logical_count'],s['test_count'])}
  assert not bad,f'count mismatches: {bad}'
 check(CHECK_IDS[11],counts)
 failed=[{'id':x,'message':results[x]} for x in CHECK_IDS if results[x] is not None]
 return {'schema_version':1,'scenario':'MANIFEST-ARCH-SLICE','collected':len(CHECK_IDS),'executed':len(CHECK_IDS),'logical':len(CHECK_IDS),'concrete':len(CHECK_IDS),'passed':[x for x in CHECK_IDS if results[x] is None],'failed':failed,'errors':[],'skipped':[],'ignored':[],'filtered':[],'underlying_exit':1 if failed else 0}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--result',type=Path,required=True); ap.add_argument('--mutant'); a=ap.parse_args()
 result=execute(a.root.resolve(),a.manifest.resolve(),a.mutant); a.result.write_text(json.dumps(result,sort_keys=True)+'\n'); print(json.dumps(result,sort_keys=True)); raise SystemExit(result['underlying_exit'])
if __name__=='__main__': main()
