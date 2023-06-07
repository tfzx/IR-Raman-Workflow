from pathlib import Path
import numpy as np
from spectra_flow.SuperOP.mlwf_reader import MLWFReaderQeW90
from spectra_flow.mlwf.qe_wannier90 import PrepareQeWann, RunQeWann, CollectWann
import unittest, shutil, os, imp
from spectra_flow.utils import read_conf

class TestReader(unittest.TestCase):
    def test_default(self):
        mlwf_setting = {}
        mlwf = MLWFReaderQeW90(mlwf_setting)
        self.assertDictEqual(mlwf_setting, {})
        self.assert_(isinstance(mlwf.name, str))
        self.assert_(mlwf.run_nscf)
        self.assert_(len(mlwf.qe_iter) == 1)
        self.assert_(mlwf.dft_params is not None)
        self.assert_(mlwf.scf_params is not None)
        self.assert_(mlwf.nscf_params is not None)
        self.assert_(mlwf.pw2wan_params is not None)
        self.assert_(mlwf.w90_params is not None)
        self.assert_(mlwf.multi_w90_params is None)
        self.assert_(mlwf.atomic_species is not None)
        self.assert_(not mlwf.with_efield)
        self.assert_(mlwf.efields is None)
        self.assert_(mlwf.ef_type == "enthalpy")
        self.assertRaises(AssertionError, mlwf.get_kgrid)
        self.assertDictEqual(
            mlwf.get_w90_params_dict(),
            {"": mlwf.w90_params}
        )
        self.assertDictEqual(
            mlwf.get_qe_params_dict(),
            {"ori": mlwf.scf_params}
        )
        
    def test_copy(self):
        mlwf_setting = {}
        mlwf = MLWFReaderQeW90(mlwf_setting, if_copy = False)
        self.assert_(mlwf_setting is mlwf.mlwf_setting)
        self.assert_(isinstance(mlwf.name, str))
        self.assert_(mlwf.run_nscf)
        self.assert_(len(mlwf.qe_iter) == 1)
        self.assert_(mlwf.dft_params is not None)
        self.assert_(mlwf.scf_params is not None)
        self.assert_(mlwf.nscf_params is not None)
        self.assert_(mlwf.pw2wan_params is not None)
        self.assert_(mlwf.w90_params is not None)
        self.assert_(mlwf.multi_w90_params is None)
        self.assert_(mlwf.atomic_species is not None)
        self.assert_(not mlwf.with_efield)
        self.assert_(mlwf.efields is None)
        self.assert_(mlwf.ef_type == "enthalpy")
        self.assertRaises(AssertionError, mlwf.get_kgrid)
        self.assertDictEqual(
            mlwf.get_w90_params_dict(),
            {"": mlwf.w90_params}
        )
        self.assertDictEqual(
            mlwf.get_qe_params_dict(),
            {"ori": mlwf.scf_params}
        )


class TestPrepareQeW90(unittest.TestCase):
    def setUp(self) -> None:
        self.confs = Path("tests/data.qe/test.004/uploads/system/train_confs").absolute()
        self.conf_fmt = {
            "fmt": "deepmd/npy",
            "type_map": ["O", "H"]
        }
        self.wc_python_path = Path("tests/data.qe/test.004/cal_dipole.py").absolute()
        self.wc_python = imp.load_source("wc_python", str(self.wc_python_path))
        self.work_dir = Path("tests/tmp")
        self.work_dir.mkdir()
        self.last_cwd = os.getcwd()
        os.chdir(self.work_dir)
        return super().setUp()
    
    def tearDown(self) -> None:
        os.chdir(self.last_cwd)
        if self.work_dir.is_dir():
            shutil.rmtree(self.work_dir)
        return super().tearDown()
    
    def test_rewriter(self):
        prepare = PrepareQeWann()
        rewrite_atoms, rewrite_proj = prepare.get_w90_rewriter(self.wc_python)
        conf_sys = read_conf(self.confs, self.conf_fmt)
        atoms2 = self.wc_python.rewrite_atoms(conf_sys)
        proj2 = self.wc_python.rewrite_proj(conf_sys)
        atoms = rewrite_atoms(conf_sys) # type: ignore
        proj = rewrite_proj(conf_sys)   # type: ignore
        self.assert_(isinstance(atoms, np.ndarray)) 
        self.assert_((atoms == "C1").all())
        self.assert_((atoms == atoms2).all())
        self.assertDictEqual(proj, {"C1": "sp2"})
        self.assertDictEqual(proj, proj2)
