from __future__ import annotations
import subprocess, sys, tempfile, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN=[sys.executable,str(ROOT/'scripts/conformance/run.py')]
class RunnerContract(unittest.TestCase):
 def invoke(self,*args):
  with tempfile.TemporaryDirectory() as d:
   return subprocess.run(RUN+list(args)+['--evidence',str(Path(d)/'e.json')],cwd=ROOT,text=True,capture_output=True)
 def test_rejects_zero_expected_count(self):
  r=self.invoke('--scenario','MANIFEST-ARCH-SLICE','--expect-logical','0','--expect-tests','12','--phase','green'); self.assertNotEqual(r.returncode,0)
 def test_rejects_manifest_count_mismatch(self):
  r=self.invoke('--scenario','MANIFEST-ARCH-SLICE','--expect-logical','12','--expect-tests','11','--phase','green'); self.assertNotEqual(r.returncode,0)
 def test_rejects_scenario_without_adapter(self):
  r=self.invoke('--scenario','ARCH-STATIC-V1','--expect-logical','10','--expect-tests','10','--phase','green'); self.assertNotEqual(r.returncode,0)
 def test_green_contract(self):
  r=self.invoke('--scenario','MANIFEST-ARCH-SLICE','--expect-logical','12','--expect-tests','12','--phase','green'); self.assertEqual(r.returncode,0,r.stderr)
 def test_named_mutant_red_contract(self):
  r=self.invoke('--scenario','MANIFEST-ARCH-SLICE','--expect-logical','12','--expect-tests','12','--phase','red','--mutant','MUT-MANIFEST-ORPHAN'); self.assertEqual(r.returncode,0,r.stderr)
if __name__=='__main__': unittest.main()
