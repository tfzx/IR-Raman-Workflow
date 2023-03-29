from mlwf_op.inputs import complete_qe
import unittest, shutil, os
from pathlib import Path
from dflow.python import (
    OPIO
)

class TestInputs(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()
    
    def tearDown(self) -> None:
        return super().tearDown()

    def test_001(self):
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
            confs = None
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
        return