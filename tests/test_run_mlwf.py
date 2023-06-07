from typing import List
from mocked_ops import MockedRunMLWF
import unittest, shutil, os
from pathlib import Path
from dflow.python import (
    OPIO
)

class TestRunMLWF(unittest.TestCase):
    def setUp(self) -> None:
        self.work_dir = Path("tests/tmp")
        self.work_dir.mkdir()
        self.last_cwd = os.getcwd()
        os.chdir(self.work_dir)
        self.task_path = Path(f"task.{0:06d}")
        self.task_path.mkdir()
        self.frames = (0, 2)
        for i in range(2):
            conf_p = self.task_path / f"conf.{i:06d}"
            conf_p.mkdir()
        return super().setUp()
    
    def tearDown(self) -> None:
        os.chdir(self.last_cwd)
        if self.work_dir.is_dir():
            shutil.rmtree(self.work_dir)
        return super().tearDown()
    
    def test_all_success(self):
        run = MockedRunMLWF(
            success_list_test1 = [True, True],
            success_list_test2 = [True, True],
            success_list_test3 = [True, True],
        )
        op_in = OPIO({
            "task_path": self.task_path,
            "mlwf_setting": {},
            "task_setting": {},
            "frames": self.frames
        })
        op_out = run.execute(op_in)
        backs: List[Path] = op_out["backward"]
        for p in backs:
            self.assert_((p / "test1").exists())
            self.assert_((p / "test2.out").exists())
            self.assert_((p / "test3.xyz").exists())
            self.assert_((p / "run.log").exists())
        
    def test_one_fail_1(self):
        run = MockedRunMLWF(
            success_list_test1 = [True, False],
            success_list_test2 = [True, True],
            success_list_test3 = [True, True],
        )
        op_in = OPIO({
            "task_path": self.task_path,
            "mlwf_setting": {},
            "task_setting": {},
            "frames": self.frames
        })
        op_out = run.execute(op_in)
        backs: List[Path] = op_out["backward"]
        p = backs[0]
        self.assert_((p / "test1").exists())
        self.assert_((p / "test2.out").exists())
        self.assert_((p / "test3.xyz").exists())
        self.assert_((p / "run.log").exists())
        p = backs[1]
        self.assert_(not (p / "test1").exists())
        self.assert_(not (p / "test2.out").exists())
        self.assert_(not (p / "test3.xyz").exists())
        self.assert_((p / "run.log").exists())
        
    def test_one_fail_2(self):
        run = MockedRunMLWF(
            success_list_test1 = [True, True],
            success_list_test2 = [True, False],
            success_list_test3 = [True, True],
        )
        op_in = OPIO({
            "task_path": self.task_path,
            "mlwf_setting": {},
            "task_setting": {},
            "frames": self.frames
        })
        op_out = run.execute(op_in)
        backs: List[Path] = op_out["backward"]
        p = backs[0]
        self.assert_((p / "test1").exists())
        self.assert_((p / "test2.out").exists())
        self.assert_((p / "test3.xyz").exists())
        self.assert_((p / "run.log").exists())
        p = backs[1]
        self.assert_(not (p / "test1").exists())
        self.assert_((p / "test2.out").exists())
        self.assert_(not (p / "test3.xyz").exists())
        self.assert_((p / "run.log").exists())
        
    def test_one_fail_3(self):
        run = MockedRunMLWF(
            success_list_test1 = [True, True],
            success_list_test2 = [True, True],
            success_list_test3 = [True, False],
        )
        op_in = OPIO({
            "task_path": self.task_path,
            "mlwf_setting": {},
            "task_setting": {},
            "frames": self.frames
        })
        op_out = run.execute(op_in)
        backs: List[Path] = op_out["backward"]
        p = backs[0]
        self.assert_((p / "test1").exists())
        self.assert_((p / "test2.out").exists())
        self.assert_((p / "test3.xyz").exists())
        self.assert_((p / "run.log").exists())
        p = backs[1]
        self.assert_(not (p / "test1").exists())
        self.assert_((p / "test2.out").exists())
        self.assert_((p / "test3.xyz").exists())
        self.assert_((p / "run.log").exists())
        
    def test_all_fail_1(self):
        run = MockedRunMLWF(
            success_list_test1 = [False, False],
            success_list_test2 = [True, True],
            success_list_test3 = [True, True],
        )
        op_in = OPIO({
            "task_path": self.task_path,
            "mlwf_setting": {},
            "task_setting": {},
            "frames": self.frames
        })
        op_out = run.execute(op_in)
        backs: List[Path] = op_out["backward"]
        for p in backs:
            self.assert_(not (p / "test1").exists())
            self.assert_(not (p / "test2.out").exists())
            self.assert_(not (p / "test3.xyz").exists())
            self.assert_((p / "run.log").exists())
        
    def test_all_fail_2(self):
        run = MockedRunMLWF(
            success_list_test1 = [True, True],
            success_list_test2 = [False, False],
            success_list_test3 = [True, True],
        )
        op_in = OPIO({
            "task_path": self.task_path,
            "mlwf_setting": {},
            "task_setting": {},
            "frames": self.frames
        })
        op_out = run.execute(op_in)
        backs: List[Path] = op_out["backward"]
        for p in backs:
            self.assert_(not (p / "test1").exists())
            self.assert_((p / "test2.out").exists())
            self.assert_(not (p / "test3.xyz").exists())
            self.assert_((p / "run.log").exists())
        
    def test_all_fail_3(self):
        run = MockedRunMLWF(
            success_list_test1 = [True, True],
            success_list_test2 = [True, True],
            success_list_test3 = [False, False],
        )
        op_in = OPIO({
            "task_path": self.task_path,
            "mlwf_setting": {},
            "task_setting": {},
            "frames": self.frames
        })
        op_out = run.execute(op_in)
        backs: List[Path] = op_out["backward"]
        for p in backs:
            self.assert_(not (p / "test1").exists())
            self.assert_((p / "test2.out").exists())
            self.assert_((p / "test3.xyz").exists())
            self.assert_((p / "run.log").exists())
        
    def test_all_fail_4(self):
        run = MockedRunMLWF(
            success_list_test1 = [False, True],
            success_list_test2 = [True, False],
            success_list_test3 = [True, True],
        )
        op_in = OPIO({
            "task_path": self.task_path,
            "mlwf_setting": {},
            "task_setting": {},
            "frames": self.frames
        })
        op_out = run.execute(op_in)
        backs: List[Path] = op_out["backward"]
        p = backs[0]
        self.assert_(not (p / "test1").exists())
        self.assert_(not (p / "test2.out").exists())
        self.assert_(not (p / "test3.xyz").exists())
        self.assert_((p / "run.log").exists())
        p = backs[1]
        self.assert_(not (p / "test1").exists())
        self.assert_((p / "test2.out").exists())
        self.assert_(not (p / "test3.xyz").exists())
        self.assert_((p / "run.log").exists())
        
    def test_all_fail_5(self):
        run = MockedRunMLWF(
            success_list_test1 = [True, False],
            success_list_test2 = [False, True],
            success_list_test3 = [True, True],
        )
        op_in = OPIO({
            "task_path": self.task_path,
            "mlwf_setting": {},
            "task_setting": {},
            "frames": self.frames
        })
        op_out = run.execute(op_in)
        backs: List[Path] = op_out["backward"]
        p = backs[0]
        self.assert_(not (p / "test1").exists())
        self.assert_((p / "test2.out").exists())
        self.assert_(not (p / "test3.xyz").exists())
        self.assert_((p / "run.log").exists())
        p = backs[1]
        self.assert_(not (p / "test1").exists())
        self.assert_(not (p / "test2.out").exists())
        self.assert_(not (p / "test3.xyz").exists())
        self.assert_((p / "run.log").exists())
        
    def test_all_fail_6(self):
        run = MockedRunMLWF(
            success_list_test1 = [True, True],
            success_list_test2 = [False, True],
            success_list_test3 = [True, False],
        )
        op_in = OPIO({
            "task_path": self.task_path,
            "mlwf_setting": {},
            "task_setting": {},
            "frames": self.frames
        })
        op_out = run.execute(op_in)
        backs: List[Path] = op_out["backward"]
        p = backs[0]
        self.assert_(not (p / "test1").exists())
        self.assert_((p / "test2.out").exists())
        self.assert_(not (p / "test3.xyz").exists())
        self.assert_((p / "run.log").exists())
        p = backs[1]
        self.assert_(not (p / "test1").exists())
        self.assert_((p / "test2.out").exists())
        self.assert_((p / "test3.xyz").exists())
        self.assert_((p / "run.log").exists())
        
    def test_all_fail_7(self):
        run = MockedRunMLWF(
            success_list_test1 = [False, True],
            success_list_test2 = [True, True],
            success_list_test3 = [True, False],
        )
        op_in = OPIO({
            "task_path": self.task_path,
            "mlwf_setting": {},
            "task_setting": {},
            "frames": self.frames
        })
        op_out = run.execute(op_in)
        backs: List[Path] = op_out["backward"]
        p = backs[0]
        self.assert_(not (p / "test1").exists())
        self.assert_(not (p / "test2.out").exists())
        self.assert_(not (p / "test3.xyz").exists())
        self.assert_((p / "run.log").exists())
        p = backs[1]
        self.assert_(not (p / "test1").exists())
        self.assert_((p / "test2.out").exists())
        self.assert_((p / "test3.xyz").exists())
        self.assert_((p / "run.log").exists())
        