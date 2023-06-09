from pathlib import Path
from typing import Dict, Union
import numpy as np, dpdata
from spectra_flow.SuperOP.mlwf_reader import MLWFReaderQeW90
from spectra_flow.mlwf.qe_wannier90 import PrepareQeWann, RunQeWann
import unittest, shutil, os, imp
from unittest import mock
from spectra_flow.utils import read_conf

class TestReader(unittest.TestCase):
    def test_default(self):
        mlwf_setting = {}
        mlwf = MLWFReaderQeW90(mlwf_setting)
        self.assertDictEqual(mlwf_setting, {})
        self.assertIsInstance(mlwf.name, str)
        self.assertTrue(mlwf.run_nscf)
        self.assertListEqual(mlwf.qe_iter, ["ori"])
        self.assertIsNotNone(mlwf.dft_params)
        self.assertIsNotNone(mlwf.scf_params)
        self.assertIsNotNone(mlwf.nscf_params)
        self.assertIsNotNone(mlwf.pw2wan_params)
        self.assertIsNotNone(mlwf.w90_params)
        self.assertIsNone(mlwf.multi_w90_params)
        self.assertIsNotNone(mlwf.atomic_species)
        self.assertFalse(mlwf.with_efield)
        self.assertIsNone(mlwf.efields)
        self.assertTrue(mlwf.ef_type == "enthalpy")
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
        self.assertIs(mlwf_setting, mlwf.mlwf_setting)
        self.assertIsInstance(mlwf.name, str)
        self.assertTrue(mlwf.run_nscf)
        self.assertListEqual(mlwf.qe_iter, ["ori"])
        self.assertIsNotNone(mlwf.dft_params)
        self.assertIsNotNone(mlwf.scf_params)
        self.assertIsNotNone(mlwf.nscf_params)
        self.assertIsNotNone(mlwf.pw2wan_params)
        self.assertIsNotNone(mlwf.w90_params)
        self.assertIsNone(mlwf.multi_w90_params)
        self.assertIsNotNone(mlwf.atomic_species)
        self.assertFalse(mlwf.with_efield)
        self.assertIsNone(mlwf.efields)
        self.assertTrue(mlwf.ef_type == "enthalpy")
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
        self.conf_sys = read_conf(self.confs, self.conf_fmt)
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
        atoms2 = self.wc_python.rewrite_atoms(self.conf_sys)
        proj2 = self.wc_python.rewrite_proj(self.conf_sys)
        atoms = rewrite_atoms(self.conf_sys) # type: ignore
        proj = rewrite_proj(self.conf_sys)   # type: ignore
        self.assertIsInstance(atoms, np.ndarray)
        self.assertTrue((atoms == "C1").all())
        self.assertTrue((atoms == atoms2).all())
        self.assertDictEqual(proj, {"C1": "sp2"})
        self.assertDictEqual(proj, proj2)

    @mock.patch("spectra_flow.mlwf.qe_wannier90.get_pw_w90_writers")
    @mock.patch("spectra_flow.mlwf.qe_wannier90.get_qe_writers")
    def test_get_writers_1(self, mock_get_qe_writers: mock.Mock, mock_get_pw_w90_writers: mock.Mock):
        mock_get_qe_writers.return_value = ("scf_writer", "nscf_writer", "input_scf", "input_nscf")
        mock_get_pw_w90_writers.return_value = ("pw2wan_writer", "wannier90_writer")
        prepare = PrepareQeWann()
        mlwf_setting: Dict[str, Union[str, dict]] = {
            "kmesh": {
                "scf_grid": (3, 4, 5),
                "nscf_grid": (1, 2, 3),
            }
        }
        mlwf_setting = prepare.init_inputs(mlwf_setting, self.conf_sys, None)
        self.assert_("name" in mlwf_setting)
        self.assert_("dft_params" in mlwf_setting)
        self.assert_("wannier90_params" in mlwf_setting)

        mock_get_qe_writers.assert_called_once()
        call_args1 = mock_get_qe_writers.call_args
        self.assertIsInstance(call_args1[0][0], dpdata.System)
        self.assertTupleEqual(call_args1[0][1], (3, 4, 5))
        self.assertTupleEqual(call_args1[0][2], (1, 2, 3))
        self.assertIsInstance(call_args1[0][3], dict)
        self.assertTrue(call_args1[0][4] is None or isinstance(call_args1[0][4], dict))
        self.assertIsInstance(call_args1[0][5], dict)
        self.assertTrue(call_args1[0][6])

        mock_get_pw_w90_writers.assert_called_once()
        call_args2 = mock_get_pw_w90_writers.call_args
        self.assertIsInstance(call_args2[0][0], str)
        self.assertIsInstance(call_args2[0][1], dpdata.System)
        self.assertIsInstance(call_args2[0][2], dict)
        self.assertIsInstance(call_args2[0][3], dict)
        self.assertTupleEqual(call_args2[0][4], (1, 2, 3))
        self.assertEqual(call_args2[0][5], "input_scf")
        self.assertEqual(call_args2[0][6], "input_nscf")
        self.assertIs(call_args2[0][7], None)
        self.assertIs(call_args2[0][8], None)

        for scf_writer in prepare.scf_writers.values():
            self.assertEqual(scf_writer, "scf_writer")
        for nscf_writer in prepare.nscf_writers.values():
            self.assertEqual(nscf_writer, "nscf_writer")
        for pw2wan_writers in prepare.pw2wan_writers.values():
            for pw2wan_writer in pw2wan_writers.values():
                self.assertEqual(pw2wan_writer, "pw2wan_writer")
        for wannier90_writers in prepare.wannier90_writers.values():
            for wannier90_writer in wannier90_writers.values():
                self.assertEqual(wannier90_writer, "wannier90_writer")

    @mock.patch("spectra_flow.mlwf.qe_wannier90.get_pw_w90_writers")
    @mock.patch("spectra_flow.mlwf.qe_wannier90.get_qe_writers")
    def test_get_writers_2(self, mock_get_qe_writers: mock.Mock, mock_get_pw_w90_writers: mock.Mock):
        mock_get_qe_writers.return_value = ("scf_writer", "nscf_writer", "input_scf", "input_nscf")
        mock_get_pw_w90_writers.return_value = ("pw2wan_writer", "wannier90_writer")
        prepare = PrepareQeWann()
        mlwf_setting: Dict[str, Union[str, dict]] = {
            "dft_params": {
                "cal_type": "scf"
            },
            "kmesh": {
                "scf_grid": (3, 4, 5),
                "nscf_grid": (1, 2, 3),
            }
        }
        mlwf_setting = prepare.init_inputs(mlwf_setting, self.conf_sys, None)
        self.assert_("name" in mlwf_setting)
        self.assert_("dft_params" in mlwf_setting)
        self.assert_("wannier90_params" in mlwf_setting)

        mock_get_qe_writers.assert_called_once()
        call_args1 = mock_get_qe_writers.call_args
        self.assertIsInstance(call_args1[0][0], dpdata.System)
        self.assertTupleEqual(call_args1[0][1], (3, 4, 5))
        self.assertTupleEqual(call_args1[0][2], (1, 2, 3))
        self.assertIsInstance(call_args1[0][3], dict)
        self.assertTrue(call_args1[0][4] is None or isinstance(call_args1[0][4], dict))
        self.assertIsInstance(call_args1[0][5], dict)
        self.assertFalse(call_args1[0][6])

        mock_get_pw_w90_writers.assert_called_once()
        call_args2 = mock_get_pw_w90_writers.call_args
        self.assertIsInstance(call_args2[0][0], str)
        self.assertIsInstance(call_args2[0][1], dpdata.System)
        self.assertIsInstance(call_args2[0][2], dict)
        self.assertIsInstance(call_args2[0][3], dict)
        self.assertTupleEqual(call_args2[0][4], (3, 4, 5))
        self.assertEqual(call_args2[0][5], "input_scf")
        self.assertEqual(call_args2[0][6], "input_nscf")
        self.assertIs(call_args2[0][7], None)
        self.assertIs(call_args2[0][8], None)

        for scf_writer in prepare.scf_writers.values():
            self.assertEqual(scf_writer, "scf_writer")
        self.assertFalse(hasattr(prepare, "nscf_writers"))
        for pw2wan_writers in prepare.pw2wan_writers.values():
            for pw2wan_writer in pw2wan_writers.values():
                self.assertEqual(pw2wan_writer, "pw2wan_writer")
        for wannier90_writers in prepare.wannier90_writers.values():
            for wannier90_writer in wannier90_writers.values():
                self.assertEqual(wannier90_writer, "wannier90_writer")

    @mock.patch("spectra_flow.mlwf.qe_wannier90.get_pw_w90_writers")
    @mock.patch("spectra_flow.mlwf.qe_wannier90.get_qe_writers")
    def test_get_writers_3(self, mock_get_qe_writers: mock.Mock, mock_get_pw_w90_writers: mock.Mock):
        mock_get_qe_writers.return_value = ("scf_writer", "nscf_writer", "input_scf", "input_nscf")
        mock_get_pw_w90_writers.return_value = ("pw2wan_writer", "wannier90_writer")
        prepare = PrepareQeWann()
        mlwf_setting: Dict[str, Union[str, dict]] = {
            "name": "test",
            "dft_params": {
                "cal_type": "scf+nscf",
                "scf_params": {
                    "test": "scf"
                },
                "nscf_params": {
                    "test": "nscf"
                },
                "pw2wan_params": {
                    "test": "pw2wan"
                },
                "atomic_species": {
                    "test": "atomic_species"
                }
            },
            "wannier90_params": {
                "test": "wannier90"
            },
            "kmesh": {
                "scf_grid": (3, 4, 5),
                "nscf_grid": (1, 2, 3),
            }
        }
        mlwf_setting = prepare.init_inputs(mlwf_setting, self.conf_sys, None)

        mock_get_qe_writers.assert_called_once()
        call_args1 = mock_get_qe_writers.call_args
        self.assertIsInstance(call_args1[0][0], dpdata.System)
        self.assertTupleEqual(call_args1[0][1], (3, 4, 5))
        self.assertTupleEqual(call_args1[0][2], (1, 2, 3))
        self.assertEqual(call_args1[0][3]["test"], "scf")
        self.assertEqual(call_args1[0][4]["test"], "nscf")
        self.assertEqual(call_args1[0][5]["test"], "atomic_species")
        self.assertTrue(call_args1[0][6])

        mock_get_pw_w90_writers.assert_called_once()
        call_args2 = mock_get_pw_w90_writers.call_args
        self.assertIsInstance(call_args2[0][0], str)
        self.assertIsInstance(call_args2[0][1], dpdata.System)
        self.assertEqual(call_args2[0][2]["test"], "pw2wan")
        self.assertEqual(call_args2[0][3]["test"], "wannier90")
        self.assertTupleEqual(call_args2[0][4], (1, 2, 3))
        self.assertEqual(call_args2[0][5], "input_scf")
        self.assertEqual(call_args2[0][6], "input_nscf")
        self.assertIs(call_args2[0][7], None)
        self.assertIs(call_args2[0][8], None)

        for scf_writer in prepare.scf_writers.values():
            self.assertEqual(scf_writer, "scf_writer")
        for nscf_writer in prepare.nscf_writers.values():
            self.assertEqual(nscf_writer, "nscf_writer")
        for pw2wan_writers in prepare.pw2wan_writers.values():
            for pw2wan_writer in pw2wan_writers.values():
                self.assertEqual(pw2wan_writer, "pw2wan_writer")
        for wannier90_writers in prepare.wannier90_writers.values():
            for wannier90_writer in wannier90_writers.values():
                self.assertEqual(wannier90_writer, "wannier90_writer")

    def test_init_inputs(self):
        prepare = PrepareQeWann()
        mlwf_setting: Dict[str, Union[str, dict]] = {
            "kmesh": {
                "scf_grid": (1, 1, 1)
            }
        }
        mlwf_setting = prepare.init_inputs(mlwf_setting, self.conf_sys, None)
        self.assert_("name" in mlwf_setting)
        mlwf = MLWFReaderQeW90(mlwf_setting)
        self.assert_("dft_params" in mlwf_setting)
        self.assert_("wannier90_params" in mlwf_setting)
        prepare.prep_one_frame(0)
        ori_path = Path("ori")
        self.assertTrue(ori_path.exists())
        self.assertTrue((ori_path / mlwf.scf_name('ori')).exists())
        self.assertTrue((ori_path / mlwf.nscf_name('ori')).exists())
        self.assertTrue((ori_path / f"{mlwf.seed_name('ori', '')}.pw2wan").exists())
        self.assertTrue((ori_path / f"{mlwf.seed_name('ori', '')}.win").exists())


class TestRunQeW90(unittest.TestCase):
    def setUp(self) -> None:
        self.work_dir = Path("tests/tmp")
        self.work_dir.mkdir()
        self.last_cwd = os.getcwd()
        os.chdir(self.work_dir)
        self.back_dir = Path("back")
        self.back_dir.mkdir()
        self.ori_task = Path("ori")
        self.ori_task.mkdir()
        return super().setUp()
    
    def tearDown(self) -> None:
        os.chdir(self.last_cwd)
        if self.work_dir.is_dir():
            shutil.rmtree(self.work_dir)
        return super().tearDown()
    
    @mock.patch.object(RunQeWann, "run_one_subtask")
    def test_run_1(self, mock_run: mock.Mock):
        mlwf_setting: Dict[str, Union[str, dict]] = {
            "kmesh": {
                "scf_grid": (1, 1, 1)
            }
        }
        MLWFReaderQeW90(mlwf_setting, if_copy = False)
        run = RunQeWann()
        run.init_cmd(mlwf_setting, {})
        run.run_one_frame(self.back_dir)
        mock_run.assert_called_once()
        call_args, call_kwargs = mock_run.call_args
        self.assertEqual(call_args[0], "ori")
        self.assertIsInstance(call_args[1], Path)
        self.assertIsInstance(call_args[2], Path)
        if len(call_args) > 3:
            self.assertFalse(call_args[3])
        if "copy_out" in call_kwargs:
            self.assertFalse(call_kwargs["copy_out"])
    
class TestRunQeW90Subtask(unittest.TestCase):
    def setUp(self) -> None:
        self.last_cwd = os.getcwd()
        self.work_dir = Path("tests/tmp")
        self.work_dir.mkdir()
        self.back_dir = (self.work_dir / Path("back")).absolute()
        self.back_dir.mkdir()
        self.ori_task = self.work_dir / Path("ori")
        self.ori_task.mkdir()
        self.tar_out = (self.work_dir / Path("ori_out")).absolute()
        os.chdir(self.ori_task)
        self.out_dir = Path("test_out")
        self.out_dir.mkdir()
        self.w90_xyz = Path("test_ori__centres.xyz")
        self.w90_xyz.write_text("test!")
        return super().setUp()
    
    def tearDown(self) -> None:
        os.chdir(self.last_cwd)
        if self.work_dir.is_dir():
            shutil.rmtree(self.work_dir)
        return super().tearDown()
    
    @mock.patch.object(RunQeWann, "run")
    def test_subtask_1(self, mock_run: mock.Mock):
        mlwf_setting: Dict[str, Union[str, dict]] = {
            "name": "test",
            "kmesh": {
                "scf_grid": (1, 1, 1)
            }
        }
        MLWFReaderQeW90(mlwf_setting)
        Path("scf_ori.in").write_text("test scf!")
        Path("nscf_ori.in").write_text("test nscf!")
        Path("test_ori_.pw2wan").write_text("test pw2wan!")
        Path("test_ori_.win").write_text("test w90!")

        run = RunQeWann()
        run.init_cmd(mlwf_setting, {})
        run.run_one_subtask("ori", self.back_dir, self.out_dir, copy_out = False, tar_dir = self.tar_out)
        mock_run.assert_called()
        self.assertEqual(mock_run.call_count, 5)
        for call_args in mock_run.call_args_list:
            self.assertIsInstance(call_args[0][0], str)
        self.assertTrue((self.back_dir / self.w90_xyz).exists())
        self.assertFalse(self.tar_out.exists())

    @mock.patch.object(RunQeWann, "run")
    def test_subtask_2(self, mock_run: mock.Mock):
        mlwf_setting: Dict[str, Union[str, dict]] = {
            "name": "test",
            "dft_params": {
                "cal_type": "scf"
            },
            "kmesh": {
                "scf_grid": (1, 1, 1)
            }
        }
        MLWFReaderQeW90(mlwf_setting)
        Path("scf_ori.in").write_text("test scf!")
        Path("test_ori_.pw2wan").write_text("test pw2wan!")
        Path("test_ori_.win").write_text("test w90!")

        run = RunQeWann()
        run.init_cmd(mlwf_setting, {})
        run.run_one_subtask("ori", self.back_dir, self.out_dir, copy_out = False, tar_dir = self.tar_out)
        mock_run.assert_called()
        self.assertEqual(mock_run.call_count, 4)
        for call_args in mock_run.call_args_list:
            self.assertIsInstance(call_args[0][0], str)
        self.assertTrue((self.back_dir / self.w90_xyz).exists())
        self.assertFalse(self.tar_out.exists())

    @mock.patch.object(RunQeWann, "run")
    def test_subtask_3(self, mock_run: mock.Mock):
        mlwf_setting: Dict[str, Union[str, dict]] = {
            "name": "test",
            "kmesh": {
                "scf_grid": (1, 1, 1)
            }
        }
        MLWFReaderQeW90(mlwf_setting)
        Path("scf_ori.in").write_text("test scf!")
        Path("nscf_ori.in").write_text("test nscf!")
        Path("test_ori_.pw2wan").write_text("test pw2wan!")
        Path("test_ori_.win").write_text("test w90!")

        run = RunQeWann()
        run.init_cmd(mlwf_setting, {})
        run.run_one_subtask("ori", self.back_dir, self.out_dir, copy_out = True, tar_dir = self.tar_out)
        mock_run.assert_called()
        self.assertEqual(mock_run.call_count, 5)
        for call_args in mock_run.call_args_list:
            self.assertIsInstance(call_args[0][0], str)
        self.assertTrue((self.back_dir / self.w90_xyz).exists())
        self.assertTrue(self.tar_out.exists())
