from mocked_ops import MockedCollectWFC
from spectra_flow.utils import read_conf
import unittest, shutil, os
from pathlib import Path
import numpy as np

class TestCollect(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.confs = Path("tests/data.qe/test.003/data").absolute()
        self.conf_fmt = {
            "fmt": "deepmd/raw",
            "type_map": ["O", "H"]
        }
        self.work_dir = Path("tests/tmp")
        self.work_dir.mkdir()
        self.last_cwd = os.getcwd()
        os.chdir(self.work_dir)
        self.backs = [Path(f"task.{i:06d}") for i in range(2)]
        for p in self.backs:
            p.mkdir()
    
    def tearDown(self) -> None:
        os.chdir(self.last_cwd)
        if self.work_dir.is_dir():
            shutil.rmtree(self.work_dir)
        return super().tearDown()


    def test_collect_all_success(self):
        collect_op = MockedCollectWFC()
        ori_confs = read_conf(self.confs, self.conf_fmt)
        op_in = {
            "mlwf_setting": {},
            "confs": self.confs,
            "conf_fmt": self.conf_fmt,
            "backward": self.backs
        }
        op_out = collect_op.execute(op_in)
        wfc = op_out["wannier_function_centers"]
        final_confs = op_out["final_confs"]
        final_conf_fmt = op_out["final_conf_fmt"]
        final_confs = read_conf(final_confs, final_conf_fmt)
        self.assertEqual(final_confs.get_nframes(), 2)
        self.assert_(np.allclose(ori_confs["coords"], final_confs["coords"]))
        self.assert_(op_out["failed_confs"].exists())
        wfc1 = np.loadtxt(wfc["test1"], dtype = float).reshape(-1, 10, 3)
        wfc2 = np.loadtxt(wfc["test2"], dtype = float).reshape(-1, 9, 3)
        self.assert_(np.allclose(wfc1, np.zeros((2, 10, 3))))
        self.assert_(np.allclose(wfc2, np.ones((2, 9, 3))))

    def test_collect_one_fail(self):
        collect_op = MockedCollectWFC(success_list = [True, False])
        ori_confs = read_conf(self.confs, self.conf_fmt)
        op_in = {
            "mlwf_setting": {},
            "confs": self.confs,
            "conf_fmt": {
                "fmt": "deepmd/raw",
                "type_map": ["O", "H"]
            },
            "backward": self.backs
        }
        op_out = collect_op.execute(op_in)
        wfc = op_out["wannier_function_centers"]

        final_confs = op_out["final_confs"]
        final_conf_fmt = op_out["final_conf_fmt"]
        final_confs = read_conf(final_confs, final_conf_fmt)
        self.assertEqual(final_confs.get_nframes(), 1)
        self.assert_(np.allclose(ori_confs[0]["coords"], final_confs["coords"]))

        failed_confs = op_out["failed_confs"]
        self.assert_(failed_confs.exists())
        failed_confs = read_conf(failed_confs, {"fmt": "deepmd/npy"})
        self.assertEqual(failed_confs.get_nframes(), 1)
        self.assert_(np.allclose(ori_confs[1]["coords"], failed_confs["coords"]))

        wfc1 = np.loadtxt(wfc["test1"], dtype = float).reshape(-1, 10, 3)
        wfc2 = np.loadtxt(wfc["test2"], dtype = float).reshape(-1, 9, 3)
        self.assert_(np.allclose(wfc1, np.zeros((1, 10, 3))))
        self.assert_(np.allclose(wfc2, np.ones((1, 9, 3))))
        
    def test_collect_all_fail(self):
        collect_op = MockedCollectWFC(success_list = [False, False])
        op_in = {
            "mlwf_setting": {},
            "confs": self.confs,
            "conf_fmt": {
                "fmt": "deepmd/raw",
                "type_map": ["O", "H"]
            },
            "backward": self.backs
        }
        self.assertRaises(AssertionError, collect_op.execute, op_in)