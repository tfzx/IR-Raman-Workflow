from mlwf_op.inputs import complete_qe
import unittest, shutil, os, dpdata
from pathlib import Path
from dflow.python import (
    OPIO
)
import numpy as np

class TestInputs(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.confs = dpdata.System("tests/data.qe/test.001/data", fmt = "deepmd/raw", type_map = ["O", "H"])
    
    def tearDown(self) -> None:
        return super().tearDown()

    def test_base(self):
        inputs_params, kpoints = complete_qe(
            input_params = {
                "control": {
                    "title": "water"
                },
                "system": {
                    "ecutwfc": 50,
                    "nosym": True,
                    "input_dft": "PBE",
                    "ibrav": 0
                },
                "electrons": {
                    "conv_thr": 1e-05
                },
                "ions": None,
                "cell": None
            }
        )
        self.assertDictEqual(
            inputs_params, 
            {
                "control": {
                    "title": "water"
                },
                "system": {
                    "ecutwfc": 50,
                    "nosym": True,
                    "input_dft": "PBE",
                    "ibrav": 0
                },
                "electrons": {
                    "conv_thr": 1e-05
                },
                "ions": None,
                "cell": None
            }
        )
        self.assertIsNone(kpoints)

    def test_scf(self):
        inputs_params, kpoints = complete_qe(
            input_params = {
                "control": {
                    "title": "water"
                },
                "system": {
                    "ecutwfc": 50,
                    "nosym": True,
                    "input_dft": "PBE",
                    "ibrav": 0
                },
                "electrons": {
                    "conv_thr": 1e-05
                },
                "ions": None,
                "cell": None
            },
            calculation = "scf"
        )
        self.assertDictEqual(
            inputs_params, 
            {
                "control": {
                    "title": "water",
                    "calculation": "scf"
                },
                "system": {
                    "ecutwfc": 50,
                    "nosym": True,
                    "input_dft": "PBE",
                    "ibrav": 0
                },
                "electrons": {
                    "conv_thr": 1e-05
                },
                "ions": None,
                "cell": None
            }
        )
        self.assertIsNone(kpoints)

    def test_scf_kpoints(self):
        inputs_params, kpoints = complete_qe(
            input_params = {
                "control": {
                    "title": "water",
                    "calculation": "scf"
                },
                "system": {
                    "ecutwfc": 50,
                    "nosym": True,
                    "input_dft": "PBE",
                    "ibrav": 0
                },
                "electrons": {
                    "conv_thr": 1e-05
                },
                "ions": None,
                "cell": None
            },
            k_grid = (1, 1, 1)
        )
        self.assertDictEqual(
            inputs_params, 
            {
                "control": {
                    "title": "water",
                    "calculation": "scf"
                },
                "system": {
                    "ecutwfc": 50,
                    "nosym": True,
                    "input_dft": "PBE",
                    "ibrav": 0
                },
                "electrons": {
                    "conv_thr": 1e-05
                },
                "ions": None,
                "cell": None
            }
        )
        self.assertDictEqual(
            kpoints,
            {
                "type": "automatic",
                "k_grid": (1, 1, 1)
            }
        )

    def test_nscf_kpoints(self):
        inputs_params, kpoints = complete_qe(
            input_params = {
                "control": {
                    "title": "water",
                    "calculation": "nscf"
                },
                "system": {
                    "ecutwfc": 50,
                    "nosym": True,
                    "input_dft": "PBE",
                    "ibrav": 0
                },
                "electrons": {
                    "conv_thr": 1e-05
                },
                "ions": None,
                "cell": None
            },
            k_grid = (1, 1, 1)
        )
        self.assertDictEqual(
            inputs_params, 
            {
                "control": {
                    "title": "water",
                    "calculation": "nscf"
                },
                "system": {
                    "ecutwfc": 50,
                    "nosym": True,
                    "input_dft": "PBE",
                    "ibrav": 0
                },
                "electrons": {
                    "conv_thr": 1e-05
                },
                "ions": None,
                "cell": None
            }
        )
        self.assertTrue((np.abs(kpoints["k_grid"] - np.array([0.0, 0.0, 0.0, 1.0])) < 1e-5).all())
        kpoints["k_grid"] = True
        self.assertDictEqual(
            kpoints,
            {
                "type": "crystal",
                "k_grid": True
            }
        )

    def test_cal_nscf_kpoints(self):
        inputs_params, kpoints = complete_qe(
            input_params = {
                "control": {
                    "title": "water"
                },
                "system": {
                    "ecutwfc": 50,
                    "nosym": True,
                    "input_dft": "PBE",
                    "ibrav": 0
                },
                "electrons": {
                    "conv_thr": 1e-05
                },
                "ions": None,
                "cell": None
            },
            calculation = "nscf",
            k_grid = (1, 1, 1)
        )
        self.assertDictEqual(
            inputs_params, 
            {
                "control": {
                    "title": "water",
                    "calculation": "nscf"
                },
                "system": {
                    "ecutwfc": 50,
                    "nosym": True,
                    "input_dft": "PBE",
                    "ibrav": 0
                },
                "electrons": {
                    "conv_thr": 1e-05
                },
                "ions": None,
                "cell": None
            }
        )
        self.assertTrue((np.abs(kpoints["k_grid"] - np.array([0.0, 0.0, 0.0, 1.0])) < 1e-5).all())
        kpoints["k_grid"] = True
        self.assertDictEqual(
            kpoints,
            {
                "type": "crystal",
                "k_grid": True
            }
        )

    def test_confs(self):
        inputs_params, kpoints = complete_qe(
            input_params = {
                "control": {
                    "title": "water",
                    "calculation": "scf"
                },
                "system": {
                    "ecutwfc": 50,
                    "nosym": True,
                    "input_dft": "PBE",
                    "ibrav": 0
                },
                "electrons": {
                    "conv_thr": 1e-05
                },
                "ions": None,
                "cell": None
            },
            confs = self.confs
        )
        self.assertDictEqual(
            inputs_params, 
            {
                "control": {
                    "title": "water",
                    "calculation": "scf"
                },
                "system": {
                    "ecutwfc": 50,
                    "nosym": True,
                    "input_dft": "PBE",
                    "ibrav": 0,
                    "ntyp": 2,
                    "nat": 3
                },
                "electrons": {
                    "conv_thr": 1e-05
                },
                "ions": None,
                "cell": None
            }
        )
        self.assertIsNone(kpoints)

    def test_all(self):
        inputs_params, kpoints = complete_qe(
            input_params = {
                "control": {
                    "title": "water"
                },
                "system": {
                    "ecutwfc": 50,
                    "nosym": True,
                    "input_dft": "PBE",
                    "ibrav": 0
                },
                "electrons": {
                    "conv_thr": 1e-05
                },
                "ions": None,
                "cell": None
            },
            calculation = "scf",
            k_grid = (1, 1, 1),
            confs = self.confs
        )
        self.assertDictEqual(
            inputs_params, 
            {
                "control": {
                    "title": "water",
                    "calculation": "scf"
                },
                "system": {
                    "ecutwfc": 50,
                    "nosym": True,
                    "input_dft": "PBE",
                    "ibrav": 0,
                    "ntyp": 2,
                    "nat": 3
                },
                "electrons": {
                    "conv_thr": 1e-05
                },
                "ions": None,
                "cell": None
            }
        )
        self.assertDictEqual(
            kpoints,
            {
                "type": "automatic",
                "k_grid": (1, 1, 1)
            }
        )