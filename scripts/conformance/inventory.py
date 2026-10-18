#!/usr/bin/env python3
"""Create the deterministic AS-000 SQL/Database caller inventory."""
from __future__ import annotations
import argparse, hashlib, json, re, subprocess
from pathlib import Path

PATTERN = re.compile(r"rusqlite|\bConnection\b|\bDatabase\b|\bSELECT\b|\bINSERT\b|\bUPDATE\b|\bDELETE\b|CREATE\s+TABLE|ALTER\s+TABLE|DROP\s+TABLE|\bPRAGMA\b|query_row|execute_batch|\.prepare\s*\(|\.transaction\s*\(|\.conn\s*\(", re.I)
SYMBOL = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:fn|struct|enum|trait)\s+([A-Za-z_][A-Za-z0-9_]*)|^\s*impl(?:<[^>]*>)?\s+([^\{]+)")
TEXT_SUFFIXES={'.rs','.toml','.py','.nu'}

def sha(data: bytes)->str: return hashlib.sha256(data).hexdigest()
def canonical(obj)->bytes: return (json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False)+'\n').encode()
def git(root:Path,*args:str)->str:
 return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
def surface(path:str)->str:
 if path.startswith('graphrag-cli/'): return 'cli'
 if path.startswith('graphrag-mcp/'): return 'mcp'
 if path.startswith('graphrag-py/'): return 'python'
 if path.startswith('graphrag-nushell/'): return 'nushell'
 if '/tests/' in path: return 'test'
 if '/examples/' in path: return 'example'
 return 'core'
def classify(text:str,path:str)->tuple[str,str]:
 if path.endswith('Cargo.toml') and 'rusqlite' in text: return ('sql_dependency','dependency')
 if re.search(r'\buse\s+rusqlite|rusqlite::',text): return ('sql_import','import')
 if re.search(r'\b(pub\s+)?struct\s+Database|pub\s+use\s+db::Database',text): return ('public_database','definition')
 if re.search(r'\bDatabase::open\b',text): return ('database_constructor','construct')
 if re.search(r'[:<&]\s*(?:Mutex\s*<\s*)?Database\b|impl\s+[^\n]*Database',text): return ('database_parameter','type_reference')
 if re.search(r'\bConnection\b|\.conn\s*\(',text): return ('connection_access','connection')
 if re.search(r'\b(SELECT|INSERT|UPDATE|DELETE|CREATE\s+TABLE|ALTER\s+TABLE|DROP\s+TABLE|PRAGMA)\b|query_row|execute_batch|\.prepare\s*\(|\.transaction\s*\(',text,re.I): return ('sql_literal','sql')
 return ('database_call','reference')
def owner_for(path:str,kind:str)->str:
 if kind in {'sql_dependency','sql_import','sql_literal','connection_access','public_database'} and path.startswith('graphrag-core/'):
  return 'SQLITE-SEMANTICS-V1'
 return 'CONSUMER-CUTOVER-V1'
def scan(root:Path):
 tracked=git(root,'ls-files').splitlines(); rows=[]; raw=[]
 for path in sorted(tracked):
  if path.startswith(('scripts/conformance/','tests/conformance/')): continue
  p=root/path
  if p.suffix not in TEXT_SUFFIXES or not p.is_file(): continue
  lines=p.read_text(encoding='utf-8',errors='strict').splitlines(); current='<module>'
  for number,text in enumerate(lines,1):
   sm=SYMBOL.match(text)
   if sm: current=(sm.group(1) or sm.group(2) or '<module>').strip()
   if not PATTERN.search(text): continue
   normalized=' '.join(text.strip().split()); kind,call=classify(normalized,path)
   target='Database' if re.search(r'\bDatabase\b',text) else ('rusqlite' if 'rusqlite' in text else 'SQL')
   raw.append(f'{path}:{number}:{normalized}')
   ident='\0'.join([path,current,kind,target,normalized]).encode()
   rows.append({'schema_version':1,'row_id':sha(ident),'surface':surface(path),'kind':kind,'path':path,'line':number,'symbol':current,'call_kind':call,'target':target,'normalized_match':normalized,'owner':owner_for(path,kind),'status':'legacy','native_test_commands':['cargo test --workspace --all-targets'],'notes':''})
 rows.sort(key=lambda x:(x['path'],x['line'],x['kind'],x['symbol'],x['target'],x['row_id']))
 raw.sort()
 return rows, ('\n'.join(raw)+'\n').encode()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]); ap.add_argument('--output-dir',type=Path,required=True); args=ap.parse_args()
 root=args.root.resolve(); out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
 rows,raw=scan(root); rows_bytes=b''.join(canonical(x) for x in rows)
 (out/'search-output.txt').write_bytes(raw); (out/'symbols.jsonl').write_bytes(rows_bytes)
 meta={'schema_version':1,'git_commit':git(root,'rev-parse','HEAD'),'scanner_argv':['python3','scripts/conformance/inventory.py','--output-dir','tests/conformance/inventory'],'discovered_rows':len(rows),'search_output_sha256':sha(raw),'symbols_sha256':sha(rows_bytes),'untracked_relevant':sorted(p for p in git(root,'ls-files','--others','--exclude-standard').splitlines() if Path(p).suffix in TEXT_SUFFIXES and not p.startswith(('scripts/conformance/','tests/conformance/')))}
 (out/'metadata.json').write_bytes(canonical(meta)); print(json.dumps(meta,sort_keys=True))
if __name__=='__main__': main()
