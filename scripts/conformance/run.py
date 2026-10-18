#!/usr/bin/env python3
"""Evidence-capturing conformance scenario runner (stdlib only)."""
from __future__ import annotations
import argparse, datetime, hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path
EXCLUDES={'.git','target','__pycache__','.pytest_cache'}
def sha_bytes(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def tree_digest(root:Path)->str:
 h=hashlib.sha256()
 for p in sorted(root.rglob('*'),key=lambda x:x.relative_to(root).as_posix()):
  rel=p.relative_to(root); 
  if any(x in EXCLUDES for x in rel.parts) or rel.parts[:3]==('tests','conformance','evidence') or p.suffix=='.pyc': continue
  kind=b'L' if p.is_symlink() else (b'F' if p.is_file() else b'D')
  h.update(rel.as_posix().encode()+b'\0'+kind+b'\0')
  if p.is_symlink(): h.update(os.readlink(p).encode())
  elif p.is_file(): h.update(p.read_bytes())
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--scenario',required=True); ap.add_argument('--expect-logical',type=int,required=True); ap.add_argument('--expect-tests',type=int,required=True); ap.add_argument('--phase',choices=['red','green'],required=True); ap.add_argument('--mutant'); ap.add_argument('--evidence',type=Path); a=ap.parse_args()
 root=Path(__file__).resolve().parents[2]; mp=root/'scripts/conformance/manifest.json'; manifest=json.loads(mp.read_text())
 if a.scenario not in manifest['scenarios']: ap.error(f'unknown scenario {a.scenario}')
 declared=manifest['scenarios'][a.scenario]
 if min(a.expect_logical,a.expect_tests)<=0 or (a.expect_logical,a.expect_tests)!=(declared['logical_count'],declared['test_count']): ap.error('expected counts must be positive and equal manifest declarations')
 if a.scenario!='MANIFEST-ARCH-SLICE': ap.error(f'{a.scenario} is planned but has no AS-000 executable adapter')
 pre=tree_digest(root)
 with tempfile.TemporaryDirectory() as d:
  result_path=Path(d)/'result.json'; cmd=[sys.executable,str(root/'tests/conformance/manifest_arch_slice.py'),'--root',str(root),'--manifest',str(mp),'--result',str(result_path)]
  if a.mutant: cmd += ['--mutant',a.mutant]
  env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'); cp=subprocess.run(cmd,text=True,capture_output=True,env=env)
  if not result_path.exists(): print(cp.stdout+cp.stderr,file=sys.stderr); raise SystemExit(2)
  result=json.loads(result_path.read_text())
 post=tree_digest(root)
 violations=[]
 for k,expected in [('logical',a.expect_logical),('concrete',a.expect_tests),('collected',a.expect_tests),('executed',a.expect_tests)]:
  if result.get(k)!=expected: violations.append(f'{k}={result.get(k)} expected={expected}')
 for k in ('errors','skipped','ignored','filtered'):
  if result.get(k): violations.append(f'{k} must be empty')
 if pre!=post: violations.append('repository tree changed during scenario')
 failed={x['id'] for x in result.get('failed',[])}
 if a.phase=='green':
  if cp.returncode!=0 or result.get('underlying_exit')!=0 or failed: violations.append('green requires zero underlying exit and no failures')
 else:
  profiles=declared.get('red_profiles',{}); key=a.mutant or 'baseline'; expected=set(profiles.get(key,profiles.get('baseline',[])))
  if cp.returncode==0 or result.get('underlying_exit')==0: violations.append('red requires nonzero underlying exit')
  if not expected or failed!=expected: violations.append(f'red failures={sorted(failed)} expected={sorted(expected)}')
 evidence={'schema_version':1,'scenario':a.scenario,'phase':a.phase,'mutant':a.mutant,'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'argv':cmd,'display_command':' '.join(cmd),'runner_exit':0 if not violations else 2,'underlying_exit':result.get('underlying_exit'),'counts':{k:result.get(k) for k in ('logical','concrete','collected','executed')},'passed':result.get('passed',[]),'failed':result.get('failed',[]),'errors':result.get('errors',[]),'skipped':result.get('skipped',[]),'ignored':result.get('ignored',[]),'filtered':result.get('filtered',[]),'manifest_sha256':sha_bytes(mp.read_bytes()),'inventory_sha256':sha_bytes((root/manifest['inventory']['symbols']).read_bytes()),'tree_sha256_before':pre,'tree_sha256_after':post,'violations':violations}
 out=a.evidence or root/'target/conformance'/f"{a.scenario.lower()}-{a.phase}{'-'+a.mutant.lower() if a.mutant else ''}.json"; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(evidence,indent=2,sort_keys=True)+'\n')
 print(f"{a.scenario}: logical={result.get('logical')} concrete={result.get('concrete')} passed={len(result.get('passed',[]))} failed={len(failed)} phase={a.phase}")
 print(f'evidence: {out}')
 if violations:
  for x in violations: print(f'ERROR: {x}',file=sys.stderr)
  raise SystemExit(2)
if __name__=='__main__': main()
