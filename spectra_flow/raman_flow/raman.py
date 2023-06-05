from dflow.step import Step
from typing import Dict, Iterable, List, Optional, Tuple, Union
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
from spectra_flow.mlwf.mlwf_steps import MLWFSteps
from spectra_flow.raman_flow.polar_steps import PolarSteps
from spectra_flow.dp.dp_train import DPolarTrain
from spectra_flow.dp.predict_steps import PredictSteps
from spectra_flow.raman_flow.raman_op import CalRaman
from spectra_flow.MD.deepmd_lmp_op import DpLmpSample
from spectra_flow.utils import (
    get_executor
)
from spectra_flow.read_par import read_par
from spectra_flow.base_workflow import AdaptiveFlow, SuperOP, StepKeyPair, StepKey

def prep_par(parameters: Dict[str, dict], run_config: dict, debug: bool = False):
    inputs = read_par(parameters)
    if debug:
        print(list(inputs))
    name_dict = {RamanFlow.main_steps[i]: i for i in range(4)}
    name_dict["md"] = -1
    run_tree = RamanFlow.run_from_inputs(inputs)
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
        run_list = [RamanFlow.main_steps[i] for i in range(start_i, end_i + 1)]
        if "predict" in run_list and (not run_config.get("provide_sample", False)):
            run_list += ["md"]
    else:
        if "end_steps" in run_config:
            end_steps = run_config["end_steps"]
        else:
            for step_name in reversed(RamanFlow.main_steps):
                if step_name in run_tree:
                    end_steps = step_name
                    break
            else:
                end_steps = "md"
        run_list = run_tree[end_steps]
        if run_config.get("provide_sample", False) and "md" in run_list:
            print("[WARNING] 'provide_sample' in run_config is set to be True, but MD is necessary!")
    return inputs, run_list
    


def build_raman(
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
    ir_template = RamanFlow(
        name = "IR-Flow", 
        run_config = run_config, 
        executors = executors, 
        upload_python_packages = upload_python_packages, 
        run_list = run_list,
        given_inputs = inputs.keys(),
        with_parallel = with_parallel,
        debug = debug
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

class RamanFlow(AdaptiveFlow):
    all_steps = {
        "polar": PolarSteps, 
        "train": DPolarTrain, 
        "md": DpLmpSample, 
        "predict": PredictSteps, 
        "cal_raman": CalRaman
    }
    steps_list = ["polar", "train", "md", "predict", "cal_raman"]
    parallel_steps = [["polar"], ["train", "md"], ["predict"], ["cal_raman"]]
    main_steps = ["polar", "train", "predict", "cal_raman"]
    @classmethod
    def get_io_dict(cls) -> Dict[str, Dict[str, List[StepKeyPair]]]:
        this = StepKey()
        polar = StepKey("polar")
        train = StepKey("train")
        md = StepKey("md")
        predict = StepKey("predict")
        return {
            "polar": {
                "polar_setting": [this.polar_setting],
                "mlwf_setting": [this.mlwf_setting],
                "task_setting": [this.task_setting],
                "conf_fmt": [this.train_conf_fmt],
                "confs": [this.train_confs],
                "pseudo": [this.pseudo],
                "cal_dipole_python": [this.cal_dipole_python],
            },
            "train": {
                "conf_fmt": [this.train_conf_fmt],
                "confs": [this.train_confs],
                "dp_setting": [this.dp_setting],
                "label": [this.train_label, polar.polarizability],
            },
            "md": {
                "global": [this.global_config],
                "conf_fmt": [this.init_conf_fmt],
                "init_conf": [this.init_conf],
                "dp_model": [this.dp_model],
            },
            "predict": {
                "dp_setting": [this.dp_setting],
                "sampled_system": [this.sampled_system, md.sampled_system],
                "sys_fmt": [this.sys_fmt, md.sys_fmt],
                "frozen_model": [this.dpolar_model, train.frozen_model],
                "cal_dipole_python": [this.cal_dipole_python],
            },
            "cal_raman": {
                "global": [this.global_config],
                "total_polar": [predict.total_tensor, this.total_polar],
            }
        }

    def __init__(self, 
            name: str,
            run_config: dict,
            executors: Dict[str, Executor],
            upload_python_packages: List[Union[str, Path]] = None,
            run_list: Iterable[str] = None,
            given_inputs: Optional[Iterable[str]] = None, 
            pri_source: str = None, 
            with_parallel: bool = True,
            debug: bool = False
        ):
        self.run_config = run_config
        self.executors = executors
        self.python_op_executor = {
            "train": self.executors["train"],
            "md": self.executors["deepmd_lammps"],
            "cal_raman": self.executors["cal"],
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
        build_dict = {
            PolarSteps: self.build_polar_temp,
            DPolarTrain: self.build_train_temp,
            DpLmpSample: self.build_md_temp,
            PredictSteps: self.build_predict_temp,
            CalRaman: self.build_raman_temp,
        }
        return {step_name: build_dict[self.all_steps[step_name]]() for step_name in run_list}

    def to_run_list(self, run_config: dict):
        name_dict = {self.main_steps[i]: i for i in range(4)}
        run_list = [self.main_steps[i] for i in range(name_dict[run_config["start_steps"]], name_dict[run_config["end_steps"]] + 1)]
        if not run_config.get("provide_sample", False):
            run_list += ["md"]
        return run_list
    
    def build_polar_temp(self):
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
        return PolarSteps(
            "Cal-Polar",
            mlwf_template,
            self.executors["base"],
            self.executors["cal"],
            self.upload_python_packages
        )

    def build_train_temp(self):
        return PythonOPTemplate(
            DPolarTrain, 
            image="registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6",
            python_packages = self.upload_python_packages
        )
    
    def build_md_temp(self):
        return PythonOPTemplate(
            DpLmpSample, 
            image="registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6",
            python_packages = self.upload_python_packages
        )
    
    def build_predict_temp(self):
        return PredictSteps(
            "predict",
            "polar",
            self.executors["predict"],
            self.executors["cal"],
            self.upload_python_packages
        )
    
    def build_raman_temp(self):
        return PythonOPTemplate(
            CalRaman, 
            image="registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6",
            python_packages = self.upload_python_packages
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
    RamanFlow.check_steps_list()
    flow = RamanFlow(
        "raman", 
        {
            "start_steps": "polar", 
            "end_steps": "cal_raman", 
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
        debug = True
    )