import numpy as np
from mocked_ops import MockedPrepare
import unittest, shutil, os
from pathlib import Path
from dflow.python import (
    OPIO
)

class TestPrepare(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.confs = Path("tests/data.qe/test.001/data").absolute()
        self.conf_fmt = {
            "fmt": "deepmd/raw",
            "type_map": ["O", "H"]
        }
        self.pseudo = Path("tests/data.qe/test.001/pseudo").absolute()
        self.wc_python_path = Path("tests/data.qe/test.001/cal_dipole.py").absolute()
        self.work_dir = Path("tests/tmp")
        self.work_dir.mkdir()
        self.last_cwd = os.getcwd()
        os.chdir(self.work_dir)
    
    def tearDown(self) -> None:
        os.chdir(self.last_cwd)
        if self.work_dir.is_dir():
            shutil.rmtree(self.work_dir)
        return super().tearDown()

    def run_t(self, group_size: int, if_cal_python: bool = False):
        prepare = MockedPrepare()
        op_in = OPIO({
            "mlwf_setting": {},
            "task_setting": {"group_size": group_size},
            "confs": self.confs,
            "conf_fmt": self.conf_fmt,
            "pseudo": self.pseudo
        })
        if if_cal_python:
            op_in["cal_dipole_python"] = self.wc_python_path
        op_out = prepare.execute(op_in)
        task_path = op_out["task_path"]
        frames_list = op_out["frames_list"]
        return prepare, task_path, frames_list

    def test_001(self):
        prepare, task_path, frames_list = self.run_t(1)
        self.assertListEqual(prepare.conf_path, [f"conf.{i:06d}" for i in range(4)])
        self.assertListEqual(prepare.task_path, [f"task.{i:06d}" for i in [0, 1, 2, 3]])
        self.assertEqual(len(task_path), 4)
        self.assertListEqual(frames_list, [(0, 1), (1, 2), (2, 3), (3, 4)])

    def test_002(self):
        prepare, task_path, frames_list = self.run_t(2)
        self.assertListEqual(prepare.conf_path, [f"conf.{i:06d}" for i in range(4)])
        self.assertListEqual(prepare.task_path, [f"task.{i:06d}" for i in [0, 0, 1, 1]])
        self.assertEqual(len(task_path), 2)
        self.assertListEqual(frames_list, [(0, 2), (2, 4)])

    def test_003(self):
        prepare, task_path, frames_list = self.run_t(3)
        self.assertListEqual(prepare.conf_path, [f"conf.{i:06d}" for i in range(4)])
        self.assertListEqual(prepare.task_path, [f"task.{i:06d}" for i in [0, 0, 0, 1]])
        self.assertEqual(len(task_path), 2)
        self.assertListEqual(frames_list, [(0, 3), (3, 4)])
        
    def test_004(self):
        prepare, task_path, frames_list = self.run_t(4)
        self.assertListEqual(prepare.conf_path, [f"conf.{i:06d}" for i in range(4)])
        self.assertListEqual(prepare.task_path, [f"task.{i:06d}" for i in [0, 0, 0, 0]])
        self.assertEqual(len(task_path), 1)
        self.assertListEqual(frames_list, [(0, 4)])
    
    def test_wc_python(self):
        prepare, task_path, frames_list = self.run_t(1, if_cal_python = True)
        self.assertListEqual(prepare.conf_path, [f"conf.{i:06d}" for i in range(4)])
        self.assertListEqual(prepare.task_path, [f"task.{i:06d}" for i in [0, 1, 2, 3]])
        self.assertEqual(len(task_path), 4)
        self.assertListEqual(frames_list, [(0, 1), (1, 2), (2, 3), (3, 4)])
        self.assert_(prepare.wc_python.test())