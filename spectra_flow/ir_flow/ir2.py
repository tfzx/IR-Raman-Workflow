from spectra_flow.base_workflow import AdaptFlow, BasicSteps
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from dflow import (
    Step,
    Steps,
    Executor,
    Inputs,
    Outputs,
    InputParameter,
    InputArtifact,
    OutputArtifact,
    upload_artifact
)
from spectra_flow.mlwf.qe_wannier90 import (
    PrepareQeWann,
    RunQeWann,
    CollectWann
)
from spectra_flow.mlwf.qe_cp import (
    PrepareCP,
    RunCPWF,
    CollectCPWF
)
from spectra_flow.mlwf.mlwf_steps import MLWFSteps
from spectra_flow.post.wannier_centroid_op import CalWC
from spectra_flow.ir_flow.dipole_steps import DipoleSteps
from spectra_flow.ir_flow.train_steps import TrainDwannSteps
from spectra_flow.ir_flow.predict_steps import PredictSteps
from spectra_flow.ir_flow.ir_steps import IRsteps
from spectra_flow.sample.deepmd_lmp_op import DpLmpSample

class IRflow(AdaptFlow):
    steps_list = [DipoleSteps, TrainDwannSteps, PredictSteps, IRsteps]
    @classmethod
    def get_io_dict(cls) -> Dict[BasicSteps, Dict[str, List[Tuple[BasicSteps, str]]]]:
        return {
            DipoleSteps: {
                "input_setting": [(None, "input_setting")],
                "task_setting": [(None, "task_setting")],
                "conf_fmt": [(None, "train_conf_fmt")],
                "confs": [(None, "train_confs")],
                "pseudo": [(None, "pseudo")],
                "cal_dipole_python": [(None, "cal_dipole_python")],
            },
            TrainDwannSteps: {
                "conf_fmt": [(None, "train_conf_fmt")],
                "confs": [(None, "train_confs")],
                "dp_setting": [(None, "dp_setting")],
                "wannier_centroid": [(DipoleSteps, "wannier_centroid"), (None, "train_label")],
            },
            PredictSteps: {
                "dp_setting": [(None, "dp_setting")],
                "sampled_system": [(None, "sampled_system")],
                "sys_fmt": [(None, "sys_fmt")],
                "dwann_model": [(TrainDwannSteps, "dwann_model"), (None, "dwann_model")],
                "cal_dipole_python": [(None, "cal_dipole_python")],
            },
            IRsteps: {
                "global": [(None, "global")],
                "total_dipole": [(PredictSteps, "total_dipole"), (None, "total_dipole")],
            }
        }

    def __init__(self, 
            name: str,
            run_config: dict,
            executors: Dict[str, Executor],
            upload_python_packages: List[Union[str, Path]] = None,
            debug = False
        ):
        self.run_config = run_config
        self.executors = executors
        self.upload_python_packages = upload_python_packages
        super().__init__(name, self.to_run_list(run_config), debug = debug)
    
    def build_templates(self, run_list: List[BasicSteps]) -> Dict[BasicSteps, Steps]:
        build_dict = {
            DipoleSteps: self.build_dipole_temp,
            TrainDwannSteps: self.build_train_temp,
            PredictSteps: self.build_predict_temp,
            IRsteps: self.build_ir_temp,
        }
        return {step: build_dict[step]() for step in run_list}
        
    def to_run_list(self, run_config: dict):
        name_list = ["dipole", "train", "predict", "cal_ir"]
        name_dict = {name_list[i]: i for i in range(4)}
        return [self.steps_list[i] for i in range(name_dict[run_config["start_steps"]], name_dict[run_config["end_steps"]] + 1)]
    
    def build_dipole_temp(self):
        if self.run_config["dft_type"] == "qe":
            prepare_op = PrepareQeWann
            run_op = RunQeWann
            collect_op = CollectWann
        elif self.run_config["dft_type"] == "qe_cp":
            prepare_op = PrepareCP
            run_op = RunCPWF
            collect_op = CollectCPWF
        else:
            raise NotImplementedError()
        
        mlwf_template = MLWFSteps(
            "mlwf",
            prepare_op,
            run_op,
            collect_op,
            self.executors["base"],
            self.executors["run"],
            self.executors["cal"],
            self.upload_python_packages
        )
        return DipoleSteps(
            "Cal-Dipole",
            mlwf_template,
            CalWC,
            self.executors["cal"],
            self.upload_python_packages
        )

    def build_train_temp(self):
        return TrainDwannSteps(
            "train-dwann",
            self.executors["train"],
            self.upload_python_packages
        )
    
    def build_predict_temp(self):
        return PredictSteps(
            "predict",
            self.executors["predict"],
            self.executors["cal"],
            self.upload_python_packages
        )
    
    def build_ir_temp(self):
        return IRsteps(
            "Cal-IR",
            self.executors["cal"],
            self.upload_python_packages
        )
    
if __name__ == "__main__":
    from spectra_flow.utils import bohrium_login, load_json
    bohrium_login(load_json("../../examples/account.json"))
    
    from dflow.plugins.dispatcher import DispatcherExecutor
    ex = DispatcherExecutor(
        machine_dict={
            "batch_type": "Bohrium",
            "context_type": "Bohrium",
            "remote_profile": {
                "input_data": {
                    "job_type": "container",
                    "platform": "ali",
                    "scass_type" : "c16_m32_cpu",
                    "image_name": "registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6",
                },
            },
        },
    )
    flow = IRflow(
        "ir", 
        {
            "start_steps": "dipole", 
            "end_steps": "cal_ir", 
            "dft_type": "qe",
            "provide_sample": False
        },
        executors = {
            "base": ex,
            "run": ex,
            "cal": ex,
            "train": ex,
            "predict": ex,
            "deepmd_lammps": ex,
        }
        )