from dflow.step import Step
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
from pathlib import Path
from dflow import (
    OPTemplate,
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
from dflow.python import PythonOPTemplate
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
import spectra_flow
from spectra_flow.SuperOP.mlwf_steps import MLWFSteps
from spectra_flow.SuperOP.dipole_steps import DipoleSteps
from spectra_flow.dp.dp_train import DWannTrain
from spectra_flow.dp.predict_steps import PredictSteps
from spectra_flow.ir_flow.ir_op import CalIR
from spectra_flow.MD.deepmd_lmp_op import DpLmpSample
from spectra_flow.utils import (
    get_executor
)
from spectra_flow.read_par import read_par
from spectra_flow.base_workflow import AdaptiveFlow, StepType, SuperOP, StepKeyPair, StepKey

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
    if "start_step" in run_config:
        start_i = name_dict[run_config["start_step"]]
        if "end_step" in run_config:
            end_i = name_dict[run_config["end_step"]]
        else:
            end_i = 0
            for step_name in run_tree:
                end_i = max(end_i, name_dict[step_name])
        if end_i < start_i:
            raise AssertionError(f"Cannot start at {run_config['start_step']}")
        end_step = IRflow.main_steps[end_i]
        run_md = run_config.get("run_md", True)
        if "md" in run_tree[end_step] and (not run_md):
            raise AssertionError("'run_md' in run_config is False, but MD is necessary!")
        run_list = [IRflow.main_steps[i] for i in range(start_i, end_i + 1)]
        if "predict_dipole" in run_list and run_md:
            run_list += ["md"]
    else:
        if "end_step" in run_config:
            end_step = run_config["end_step"]
        else:
            for step_name in reversed(IRflow.main_steps):
                if step_name in run_tree:
                    end_step = step_name
                    break
            else:
                end_step = "md"
        run_list = run_tree[end_step]
        if (not run_config.get("run_md", True)) and "md" in run_list:
            raise AssertionError("'run_md' in run_config is False, but MD is necessary!")
    return inputs, run_list
    


def build_ir(
        name: str,
        parameters: dict, 
        machine: dict, 
        run_config: Optional[dict] = None, 
        upload_python_packages: Optional[List[Union[str, Path]]] = None, 
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
        name = "IR-Flow", 
        run_config = run_config, 
        executors = executors, 
        upload_python_packages = upload_python_packages, 
        run_list = run_list,
        given_inputs = inputs.keys(),
        with_parallel = with_parallel,
        debug = debug
    )
    in_p = ir_template.input_parameters
    in_a = ir_template.input_artifacts
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
        "train_wann": DWannTrain, 
        "md": DpLmpSample, 
        "predict_dipole": PredictSteps, 
        "cal_ir": CalIR
    }
    steps_list = ["dipole", "train_wann", "md", "predict_dipole", "cal_ir"]
    parallel_steps = [["dipole"], ["train_wann", "md"], ["predict_dipole"], ["cal_ir"]]
    main_steps = ["dipole", "train_wann", "predict_dipole", "cal_ir"]
    @classmethod
    def get_io_dict(cls) -> Dict[str, Dict[str, List[StepKeyPair]]]:
        this = StepKey()
        dipole = StepKey("dipole")
        train_wann = StepKey("train_wann")
        md = StepKey("md")
        predict_dipole = StepKey("predict_dipole")
        return {
            "dipole": {
                "mlwf_setting": [this.mlwf_setting],
                "task_setting": [this.task_setting],
                "conf_fmt": [this.train_conf_fmt],
                "confs": [this.train_confs],
                "pseudo": [this.pseudo],
                "cal_dipole_python": [this.cal_dipole_python],
            },
            "train_wann": {
                "conf_fmt": [this.labeled_sys_fmt, dipole.final_conf_fmt],
                "labeled_sys": [this.labeled_sys, dipole.labeled_confs],
                "dp_setting": [this.dwann_setting],
            },
            "md": {
                "global": [this.global_config],
                "conf_fmt": [this.init_conf_fmt],
                "init_conf": [this.init_conf],
                "dp_model": [this.dp_model],
            },
            "predict_dipole": {
                "dp_setting": [this.dwann_setting],
                "sampled_system": [this.sampled_system, md.sampled_system],
                "sys_fmt": [this.sys_fmt, md.sys_fmt],
                "frozen_model": [this.dwann_model, train_wann.frozen_model],
                "cal_dipole_python": [this.cal_dipole_python],
            },
            "cal_ir": {
                "global": [this.global_config],
                "total_dipole": [predict_dipole.total_tensor, this.total_dipole],
            }
        }

    def __init__(self, 
            name: str,
            run_config: dict,
            executors: Dict[str, Optional[Executor]],
            upload_python_packages: Optional[List[Union[str, Path]]] = None,
            run_list: Optional[Iterable[str]] = None,
            given_inputs: Optional[Iterable[str]] = None, 
            pri_source: Optional[str] = None, 
            with_parallel: bool = True,
            debug: bool = False
        ):
        self.run_config = run_config
        self.executors = executors
        self.python_op_executor = {
            "train_wann": self.executors["train"],
            "md": self.executors["deepmd_lammps"],
            "cal_ir": self.executors["cal"],
        }
        if upload_python_packages is None:
            upload_python_packages = []
        up_py_set = set(upload_python_packages)
        up_py_set.update(spectra_flow.__path__)
        self.upload_python_packages = list(up_py_set)
        if run_list is None:
            run_list = self.to_run_list(run_config)
        super().__init__(
            name = name, 
            run_list = run_list,
            given_inputs = given_inputs,
            pri_source = pri_source,
            with_parallel = with_parallel, 
            debug = debug
        )
    
    def build_templates(self, run_list: List[str]) -> Dict[str, OPTemplate]:
        build_dict: Dict[StepType, Callable[[], OPTemplate]] = {
            DipoleSteps: self.build_dipole_temp,
            DWannTrain: self.build_train_temp,
            DpLmpSample: self.build_md_temp,
            PredictSteps: self.build_predict_temp,
            CalIR: self.build_ir_temp,
        }
        return {step_name: build_dict[self.all_steps[step_name]]() for step_name in run_list}

    @classmethod
    def to_run_list(cls, run_config: dict):
        name_dict = {cls.main_steps[i]: i for i in range(4)}
        run_list = [cls.main_steps[i] for i in range(name_dict[run_config["start_step"]], name_dict[run_config["end_step"]] + 1)]
        if run_config.get("run_md", True) and "predict_dipole" in run_list:
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
            prepare_op, # type: ignore
            run_op, # type: ignore
            collect_op, # type: ignore
            self.executors["base"],
            self.executors["run"],
            self.executors["cal"],
            self.upload_python_packages
        )
        return DipoleSteps(
            "cal-dipole",
            mlwf_template,
            self.executors["cal"],
            self.upload_python_packages
        )

    def build_train_temp(self):
        return PythonOPTemplate(
            DWannTrain,  # type: ignore
            image="registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6",
            python_packages = self.upload_python_packages # type: ignore
        )
    
    def build_md_temp(self):
        return PythonOPTemplate(
            DpLmpSample,  # type: ignore
            image="registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6",
            python_packages = self.upload_python_packages # type: ignore
        )
    
    def build_predict_temp(self):
        return PredictSteps(
            "predict-dipole",
            "dipole",
            self.executors["predict"],
            self.executors["cal"],
            self.upload_python_packages
        )
    
    def build_ir_temp(self):
        return PythonOPTemplate(
            CalIR,  # type: ignore
            image="registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6",
            python_packages = self.upload_python_packages # type: ignore
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
    ir_template = IRflow(
        "ir", 
        {
            "start_step": "dipole", 
            "end_step": "cal_ir", 
            "dft_type": "qe",
            "run_md": True
        },
        executors = {
            "base": ex,
            "run": ex,
            "cal": ex,
            "train": ex,
            "predict": ex,
            "deepmd_lammps": ex,
        },
        debug = True
    )