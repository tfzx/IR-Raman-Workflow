from spectra_flow.base_workflow import AdaptiveFlow, BasicSteps, StepKeyPair
from typing import Dict, Iterable, List, Optional, Tuple, Union
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
from spectra_flow.ir_flow.dipole_steps import DipoleSteps
from spectra_flow.ir_flow.train_steps import TrainDwannSteps
from spectra_flow.ir_flow.predict_steps import PredictSteps
from spectra_flow.ir_flow.ir_steps import IRsteps
from spectra_flow.MD.md_steps import MDSteps
from spectra_flow.utils import (
    get_executor
)
from spectra_flow.read_par import read_par

def prep_par(parameters: Dict[str, dict], run_config: dict, debug: bool = False):
    inputs = read_par(parameters)
    if debug:
        print(list(inputs))
    name_dict = {IRflow.main_steps[i]: i for i in range(4)}
    name_dict["md"] = -1
    run_tree = IRflow.run_from_inputs(inputs)
    assert len(run_tree) > 0, "Cannot run any step from the inputs!"
    if "dipole" in parameters["config"]:
        run_config["dft_type"] = parameters["config"]["dipole"]["dft_type"]
    if "start_steps" in run_config:
        start_i = name_dict[run_config["start_steps"]]
        if "end_steps" in run_config:
            end_i = name_dict[run_config["end_steps"]]
        else:
            end_i = 0
            for step_name in run_tree:
                end_i = max(end_i, name_dict[step_name])
        if end_i < start_i:
            raise AssertionError(f"Cannot start at {run_config['start_steps']}")
        run_list = [IRflow.main_steps[i] for i in range(start_i, end_i + 1)]
        if "predict" in run_list and (not run_config.get("provide_sample", False)):
            run_list += ["md"]
    else:
        if "end_steps" in run_config:
            end_steps = run_config["end_steps"]
        else:
            for step_name in reversed(IRflow.main_steps):
                if step_name in run_tree:
                    end_steps = step_name
                    break
            else:
                end_steps = "md"
        run_list = run_tree[end_steps]
        if run_config.get("provide_sample", False) and "md" in run_list:
            print("[WARNING] 'provide_sample' in run_config is set to be True, but MD is necessary!")
    return inputs, run_list
    


def build_ir(
        name: str,
        parameters: dict, 
        machine: dict, 
        run_config: dict = None, 
        upload_python_packages = None, 
        with_parallel = True,
        debug = False
    ):
    executors = {}
    for ex_name, exec_config in machine["executors"].items():
        executors[ex_name] = get_executor(exec_config)
    if run_config is None:
        run_config = {}
    inputs, run_list = prep_par(parameters, run_config, debug)
    ir_template = IRflow(
        "IR-Flow", 
        run_config, 
        executors, 
        upload_python_packages, 
        run_list,
        with_parallel,
        debug
    )
    in_p, in_a = ir_template.get_inputs_list()
    input_parameters = {key: inputs[key] for key in in_p if key in inputs}
    input_artifacts_path = {key: inputs[key] for key in in_a if key in inputs}
    input_artifacts = {}
    for in_key, path in input_artifacts_path.items():
        input_artifacts[in_key] = upload_artifact(path)
    ir_step = Step(
        name,
        ir_template,
        parameters = input_parameters,
        artifacts = input_artifacts
    )
    return ir_step

class IRflow(AdaptiveFlow):
    all_steps = {
        "dipole": DipoleSteps, 
        "train": TrainDwannSteps, 
        "md": MDSteps, 
        "predict": PredictSteps, 
        "cal_ir": IRsteps
    }
    steps_list = ["dipole", "train", "md", "predict", "cal_ir"]
    parallel_steps = [["dipole"], ["train", "md"], ["predict"], ["cal_ir"]]
    main_steps = ["dipole", "train", "predict", "cal_ir"]
    @classmethod
    def get_io_dict(cls) -> Dict[str, Dict[str, List[StepKeyPair]]]:
        return {
            "dipole": {
                "mlwf_setting": [(None, "mlwf_setting")],
                "task_setting": [(None, "task_setting")],
                "conf_fmt": [(None, "train_conf_fmt")],
                "confs": [(None, "train_confs")],
                "pseudo": [(None, "pseudo")],
                "cal_dipole_python": [(None, "cal_dipole_python")],
            },
            "train": {
                "conf_fmt": [(None, "train_conf_fmt")],
                "confs": [(None, "train_confs")],
                "dp_setting": [(None, "dp_setting")],
                "wannier_centroid": [(None, "train_label"), ("dipole", "wannier_centroid")],
            },
            "md": {
                "global": [(None, "global")],
                "init_conf_fmt": [(None, "init_conf_fmt")],
                "init_conf": [(None, "init_conf")],
                "dp_model": [(None, "dp_model")],
            },
            "predict": {
                "dp_setting": [(None, "dp_setting")],
                "sampled_system": [(None, "sampled_system"), ("md", "sampled_system")],
                "sys_fmt": [(None, "sys_fmt"), ("md", "sys_fmt")],
                "dwann_model": [(None, "dwann_model"), ("train", "dwann_model")],
                "cal_dipole_python": [(None, "cal_dipole_python")],
            },
            "cal_ir": {
                "global": [(None, "global")],
                "total_dipole": [("predict", "total_dipole"), (None, "total_dipole")],
            }
        }

    def __init__(self, 
            name: str,
            run_config: dict,
            executors: Dict[str, Executor],
            upload_python_packages: List[Union[str, Path]] = None,
            run_list: Iterable[str] = None,
            with_parallel: bool = True,
            debug: bool = False
        ):
        self.run_config = run_config
        self.executors = executors
        self.upload_python_packages = upload_python_packages
        if run_list is None:
            run_list = self.to_run_list(run_config)
        super().__init__(name, run_list, with_parallel, debug)
    
    def build_templates(self, run_list: List[str]) -> Dict[str, Steps]:
        build_dict = {
            DipoleSteps: self.build_dipole_temp,
            TrainDwannSteps: self.build_train_temp,
            MDSteps: self.build_md_temp,
            PredictSteps: self.build_predict_temp,
            IRsteps: self.build_ir_temp,
        }
        return {step_name: build_dict[self.all_steps[step_name]]() for step_name in run_list}
        
    def to_run_list(self, run_config: dict):
        name_dict = {self.main_steps[i]: i for i in range(4)}
        run_list = [self.main_steps[i] for i in range(name_dict[run_config["start_steps"]], name_dict[run_config["end_steps"]] + 1)]
        if not run_config.get("provide_sample", False):
            run_list += ["md"]
        return run_list
    
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
            self.executors["cal"],
            self.upload_python_packages
        )

    def build_train_temp(self):
        return TrainDwannSteps(
            "train-dwann",
            self.executors["train"],
            self.upload_python_packages
        )
    
    def build_md_temp(self):
        return MDSteps(
            "MD",
            self.executors["deepmd_lammps"],
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
    IRflow.check_steps_list()
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
        },
        debug = False
    )