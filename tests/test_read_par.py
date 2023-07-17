import json
import unittest, shutil, os, dpdata
from pathlib import Path
from spectra_flow.read_par import read_par
from spectra_flow.utils import load_json

class TestReadPar(unittest.TestCase):
    def setUp(self) -> None:
        self.work_dir = Path("tests/tmp")
        self.work_dir.mkdir()
        self.last_cwd = os.getcwd()
        os.chdir(self.work_dir)
        self.par_from_file = Path("parameters.json")
        self.par_from_file.write_text(json.dumps({
            "config": {
                "global": {
                    "type_map": ["C"]
                },
                "dipole": {
                    "dft_type": "qe",
                    "mlwf_setting": "mlwf.json",
                    "task_setting": "task.json",
                },
                "polar": "polar.json",
                "deep_wannier": {
                    "train_inputs": "dwann.json"
                },
                "deep_polar": {
                    "train_inputs": "dpolar.json"
                }
            },
            "uploads": {}
        }))
        self.par = Path("parameters2.json")
        self.par.write_text(json.dumps({
            "config": {
                "global": {
                    "type_map": ["C"]
                },
                "dipole": {
                    "dft_type": "qe",
                    "mlwf_setting": {
                        "mlwf": "test"
                    },
                    "task_setting": {
                        "task": "test"
                    },
                },
                "polar": {
                    "polar": "test"
                },
                "deep_wannier": {
                    "train_inputs": {
                        "dwann": "test"
                    }
                },
                "deep_polar": {
                    "train_inputs": {
                        "dpolar": "test"
                    }
                }
            },
            "uploads": {}
        }))
        self.task_f = Path("task.json")
        self.task_f.write_text(json.dumps({
            "task": "test"
        }))
        self.mlwf_f = Path("mlwf.json")
        self.mlwf_f.write_text(json.dumps({
            "mlwf": "test"
        }))
        self.polar_f = Path("polar.json")
        self.polar_f.write_text(json.dumps({
            "polar": "test"
        }))
        self.dwann_f = Path("dwann.json")
        self.dwann_f.write_text(json.dumps({
            "dwann": "test"
        }))
        self.dpolar_f = Path("dpolar.json")
        self.dpolar_f.write_text(json.dumps({
            "dpolar": "test"
        }))
        return super().setUp()

    def tearDown(self) -> None:
        os.chdir(self.last_cwd)
        if self.work_dir.is_dir():
            shutil.rmtree(self.work_dir)
        return super().tearDown()

    def test_read(self):
        inputs = read_par(load_json(self.par))
        self.assertDictEqual(inputs, {
            "global_config": {
                "name": "system",
                "calculation": "ir",
                "dt": 0.0003,
                "nstep": 10000,
                "window": 1000,
                "temperature": 300,
                "width": 240,
                "type_map": ["C"]
            },
            "polar_setting": {
                "polar": "test"
            },
            "mlwf_setting": {
                "mlwf": "test"
            },
            "task_setting": {
                "task": "test"
            },
            "dwann_setting": {
                "train_inputs": {
                    "dwann": "test"
                }
            },
            "dpolar_setting": {
                "train_inputs": {
                    "dpolar": "test"
                }
            }
        })

    def test_read_from_file(self):
        inputs = read_par(load_json(self.par_from_file))
        self.assertDictEqual(inputs, {
            "global_config": {
                "name": "system",
                "calculation": "ir",
                "dt": 0.0003,
                "nstep": 10000,
                "window": 1000,
                "temperature": 300,
                "width": 240,
                "type_map": ["C"]
            },
            "polar_setting": {
                "polar": "test"
            },
            "mlwf_setting": {
                "mlwf": "test"
            },
            "task_setting": {
                "task": "test"
            },
            "dwann_setting": {
                "train_inputs": {
                    "dwann": "test"
                }
            },
            "dpolar_setting": {
                "train_inputs": {
                    "dpolar": "test"
                }
            }
        })