from pp_op.wannier_centroid_op import CalWC
import unittest, shutil, os, numpy as np
from pathlib import Path
from dflow.python import (
    OPIO
)

class TestWC(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.confs = Path("tests/data.qe/test.002/data").absolute()
        self.wfc = {
            "ori": Path("tests/data.qe/test.002/back/wfc.raw").absolute()
        }
        self.work_dir = Path("tests/tmp")
        self.work_dir.mkdir()
        self.last_cwd = os.getcwd()
        os.chdir(self.work_dir)
    
    def tearDown(self) -> None:
        os.chdir(self.last_cwd)
        if self.work_dir.is_dir():
            shutil.rmtree(self.work_dir)
        return super().tearDown()

    def test_001(self):
        wc_op = CalWC()
        op_in = {
            "confs": self.confs,
            "wannier_function_centers": self.wfc
        }
        op_out = wc_op.execute(op_in)
        wc = np.loadtxt(op_out["wannier_centroid"]["ori"], dtype = float)
        self.assertTupleEqual(wc.shape, (10, 64 * 3))
