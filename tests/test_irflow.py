import unittest, shutil, os
from pathlib import Path
from dflow import Workflow, Step, upload_artifact
from spectra_flow.ir_flow.ir import IRflow
from spectra_flow.utils import bohrium_login

class TestIRflow(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.work_dir = Path("tests/tmp")
        self.work_dir.mkdir()
        self.last_cwd = os.getcwd()
        os.chdir(self.work_dir)
        bohrium_login(debug = True)
        data_path = Path("test_data")
        data_path.touch()
        self.data = upload_artifact(data_path)
    
    def tearDown(self) -> None:
        os.chdir(self.last_cwd)
        if self.work_dir.is_dir():
            shutil.rmtree(self.work_dir)
        return super().tearDown()
    
    def test_steps(self):
        IRflow.check_steps()
        
    def test_steps_list(self):
        IRflow.check_steps_list()
    
    def test_io_dict(self):
        IRflow.check_io_dict()
    
    def test_run_list(self):
        self.assertSetEqual(
            set(IRflow.to_run_list({
                "start_step": "dipole", 
                "end_step": "cal_ir", 
                "dft_type": "qe",
                "run_md": True
            })),
            {"dipole", "train_wann", "predict_dipole", "cal_ir", "md"}
        )
        self.assertSetEqual(
            set(IRflow.to_run_list({
                "start_step": "dipole", 
                "end_step": "cal_ir", 
                "dft_type": "qe",
                "run_md": False
            })),
            {"dipole", "train_wann", "predict_dipole", "cal_ir"}
        )
        self.assertSetEqual(
            set(IRflow.to_run_list({
                "start_step": "cal_ir", 
                "end_step": "cal_ir", 
                "dft_type": "qe",
                "run_md": False
            })),
            {"cal_ir"}
        )
        self.assertSetEqual(
            set(IRflow.to_run_list({
                "start_step": "cal_ir", 
                "end_step": "cal_ir", 
                "dft_type": "qe",
                "run_md": True
            })),
            {"cal_ir"}
        )
        self.assertSetEqual(
            set(IRflow.to_run_list({
                "start_step": "dipole", 
                "end_step": "train_wann", 
                "dft_type": "qe",
                "run_md": True
            })),
            {"dipole", "train_wann"}
        )
        self.assertSetEqual(
            set(IRflow.to_run_list({
                "start_step": "predict_dipole", 
                "end_step": "predict_dipole", 
                "dft_type": "qe",
                "run_md": True
            })),
            {"predict_dipole", "md"}
        )
        self.assertSetEqual(
            set(IRflow.to_run_list({
                "start_step": "predict_dipole", 
                "end_step": "predict_dipole", 
                "dft_type": "qe",
                "run_md": False
            })),
            {"predict_dipole"}
        )
    
    def test_build_01(self):
        ir_template = IRflow(
            "ir", 
            {
                "start_step": "dipole", 
                "end_step": "cal_ir", 
                "dft_type": "qe",
                "run_md": True
            },
            executors = {
                "base": None,
                "run": None,
                "cal": None,
                "train": None,
                "predict": None,
                "deepmd_lammps": None,
            },
            debug = True
        )
        in_p = ir_template.input_parameters
        in_a = ir_template.input_artifacts
        self.assertSetEqual(
            set(in_p.keys()), {
                "mlwf_setting",
                "task_setting",
                "train_conf_fmt",
                "dwann_setting",
                "global_config",
                "init_conf_fmt",
            }
        )
        self.assertSetEqual(
            set(in_a.keys()), {
                "train_confs",
                "pseudo",
                "cal_dipole_python",
                "init_conf",
                "dp_model",
            }
        )
        self.assert_(in_a["cal_dipole_python"].optional)

    def test_build_02(self):
        ir_template = IRflow(
            "ir", 
            {
                "start_step": "dipole", 
                "end_step": "cal_ir", 
                "dft_type": "qe",
                "run_md": False
            },
            executors = {
                "base": None,
                "run": None,
                "cal": None,
                "train": None,
                "predict": None,
                "deepmd_lammps": None,
            },
            debug = True
        )
        in_p = ir_template.input_parameters
        in_a = ir_template.input_artifacts
        self.assertSetEqual(
            set(in_p.keys()), {
                "mlwf_setting",
                "task_setting",
                "train_conf_fmt",
                "dwann_setting",
                "global_config",
                "sys_fmt",
            }
        )
        self.assertSetEqual(
            set(in_a.keys()), {
                "train_confs",
                "pseudo",
                "cal_dipole_python",
                "sampled_system"
            }
        )
        self.assert_(in_a["cal_dipole_python"].optional)