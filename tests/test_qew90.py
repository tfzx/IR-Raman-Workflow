from spectra_flow.SuperOP.mlwf_reader import MLWFReaderQeW90
import unittest, shutil, os

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
        self.assertTupleEqual(mlwf.get_kgrid(), (None, None))
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
        self.assertTupleEqual(mlwf.get_kgrid(), (None, None))
        self.assertDictEqual(
            mlwf.get_w90_params_dict(),
            {"": mlwf.w90_params}
        )
        self.assertDictEqual(
            mlwf.get_qe_params_dict(),
            {"ori": mlwf.scf_params}
        )