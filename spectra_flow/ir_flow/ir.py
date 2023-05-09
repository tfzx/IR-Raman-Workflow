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
from dflow.io import Inputs, Outputs
from dflow.python import (
    PythonOPTemplate,
)
from dflow.step import Step
import spectra_flow
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
from spectra_flow.dp.dp_train import DWannTrain
from spectra_flow.sample.deepmd_lmp_op import DpLmpSample
from spectra_flow.ir_flow.predict_steps import PredictSteps
from spectra_flow.ir_flow.ir_op import CalIR
from spectra_flow.utils import (
    complete_by_default,
    get_executor
)


def build_ir(global_config: dict, machine_config: dict, run_config: dict = None):
    executors = {}
    for name, exec_config in machine_config["executors"].items():
        executors[name] = get_executor(exec_config)
    if run_config is None:
        run_config = {}
    checker = IRchecker(global_config, machine_config)
    assert checker.check_all()
    run_config = complete_by_default(run_config, checker.get_run_config())
    ir_inputs = IRinputs(run_config)
    input_parameters, input_artifacts_path = checker.get_inputs(*ir_inputs.get_inputs_list())
    input_artifacts = {}
    for name, path in input_artifacts_path.items():
        input_artifacts[name] = upload_artifact(path)
    ir_template = IRflow(checker.global_global["name"], run_config, executors)
    ir_step = Step(
        checker.global_global["name"],
        ir_template,
        parameters = input_parameters,
        artifacts = input_artifacts
    )
    return ir_step


class IRinputs:
    step_list = ["dipole", "train", "predict", "cal_ir"]
    def __init__(self, run_config: dict) -> None:
        self.run_config = run_config
        self.run_input_parameters: Dict[str, List[str]] = {
            "dipole": [
                "input_setting",
                "task_setting",
                "train_conf_fmt"
            ],
            "train": ["dp_setting"],
            "predict": [],
            "cal_ir": ["global"]
        }
        self.run_input_artifacts: Dict[str, List[str]] = {
            "dipole": ["pseudo"],
            "train": [],
            "predict": [],
            "cal_ir": []
        }
        self.start_input_parameters: Dict[str, List[str]] = {
            "dipole": [],
            "train": ["train_conf_fmt"],
            "predict": [],
            "cal_ir": []
        }
        self.start_input_artifacts: Dict[str, List[str]] = {
            "dipole": ["train_confs"],
            "train": ["train_confs", "train_label"],
            "predict": ["dwann_model"],
            "cal_ir": ["total_dipole"]
        }
        if run_config["provide_sample"]:
            self.run_input_parameters["predict"] += ["sys_fmt"]
            self.run_input_artifacts["predict"] += ["sampled_system"]
        else:
            self.run_input_parameters["predict"] += ["init_conf_fmt", "global"]
            self.run_input_artifacts["predict"] += ["init_conf", "dp_model"]
        self.run_range = list(range(
            self.step_list.index(run_config["start_steps"]),
            self.step_list.index(run_config["end_steps"]) + 1
        ))

    def get_inputs_list(self):
        input_parameters = set()
        input_artifacts = set()
        for i in self.run_range:
            step = self.step_list[i]
            input_parameters.update(self.run_input_parameters[step])
            input_artifacts.update(self.run_input_artifacts[step])
        start_steps = self.run_config["start_steps"]
        input_parameters.update(self.start_input_parameters[start_steps])
        input_artifacts.update(self.start_input_artifacts[start_steps])
        return list(input_parameters), list(input_artifacts)


class IRchecker:
    step_list = ["dipole", "train", "predict", "cal_ir"]
    _default_config = {
        "config": {
            "global": {
                "name": "system",
                "calculation": "ir",
                "dt": 0.0003,
                "nstep": 10000,
                "window": 1000,
                "temperature": 300,
                "width": 240
            },
        },
        "uploads": {
            "frozen_model": {},
            "system": {},
            "other": {}
        }
    }
    
    def __init__(self, global_config: dict, machine_config: dict) -> None:
        self.global_config = global_config
        self.machine_config = machine_config
        self._default()
        self._init_reader()
        pass

    @property
    def config(self) -> dict:
        return self.global_config["config"]

    @property
    def uploads(self) -> dict:
        return self.global_config["uploads"]
    
    def check_all(self):
        if "type_map" not in self.global_global:
            print("[Error] Missing 'type_map' in global_config")
            return False
        if "mass_map" not in self.global_global:
            print("[Error] Missing 'mass_map' in global_config")
            return False
        self.run_config = {
            "provide_sample": self.check_provide_sample()
        }
        self.if_run_l, self.if_start_l = tuple(zip(
            self.check_dipole(), 
            self.check_train(), 
            self.check_predict(), 
            self.check_ir(), 
        ))
        if True not in self.if_start_l:
            print("[Error] Missing some inputs!")
            return False
        return True

    def get_run_config(self):
        if not hasattr(self, "if_run_l") or not hasattr(self, "if_start_l"):
            if not self.check_all():
                return
        for i in reversed(range(4)):
            if self.if_start_l[i]:
                id_start = i
                break
        else:
            raise RuntimeError("Check fail!")
        for i in range(id_start, 4):
            if self.if_run_l[i]:
                id_end = i
            else:
                break
        run_config = {
            "start_steps": self.step_list[id_start],
            "end_steps": self.step_list[id_end],
            "provide_sample": self.check_provide_sample()
        }
        if run_config["start_steps"] == "dipole":
            run_config["dipole_config"] = {
                "dft_type": self.config["dipole"]["dft_type"]
            }
        return run_config

    def get_inputs(self, input_parameters_l: List[str], input_artifacts_l: List[str]):
        self._input_parameters = {}
        self._input_artifacts = {}
        for key in input_parameters_l:
            self._read_parameters(key)
        for key in input_artifacts_l:
            self._read_artifacts(key)
        return self._input_parameters, self._input_artifacts

    def _default(self):
        complete_by_default(self.global_config, self._default_config)
        self.global_global = self.global_config["config"]["global"]
        if "dipole" in self.config:
            complete_by_default(self.config["dipole"]["input_setting"], {"name": self.global_global["name"]})

    def _init_reader(self):
        self._inputs_p_reader = {
            "global": lambda: self.config["global"],
            "input_setting": lambda: self.config["dipole"]["input_setting"],
            "task_setting": lambda: self.config["dipole"]["task_setting"],
            "dp_setting": lambda: self.config["deep_model"],
            "train_conf_fmt": lambda: None,
            "init_conf_fmt": lambda: None,
            "sys_fmt": lambda: None,
        }
        self._inputs_a_reader = {
            "dp_model": lambda: self.uploads["frozen_model"]["deep_potential"],
            "dwann_model": lambda: self.uploads["frozen_model"]["deep_wannier"],
            "train_confs": lambda: self.load_system("train_confs"),
            "sampled_system": lambda: self.load_system("sampled_system"),
            "init_conf": lambda: self.load_system("init_conf"),
            "pseudo": lambda: self.uploads["other"]["pseudo"],
            "train_label": lambda: self.uploads["other"]["train_label"],
            "total_dipole": lambda: self.uploads["other"]["total_dipole"],
        }
        self._sys_fmt_map = {
            "train_confs": "train_conf_fmt",
            "sampled_system": "sys_fmt",
            "init_conf": "init_conf_fmt",
        }

    def _read_parameters(self, key: str):
        fun = self._inputs_p_reader[key]
        out = fun()
        if out is not None:
            self._input_parameters[key] = out

    def _read_artifacts(self, key: str):
        fun = self._inputs_a_reader[key]
        out = fun()
        if isinstance(out, tuple):
            self._input_artifacts[key] = out[0]
            self._input_parameters[self._sys_fmt_map[key]] = out[1]
        else:
            self._input_artifacts[key] = out

    def check_provide_sample(self):
        return "sampled_system" in self.uploads["system"]

    def check_sample(self):
        if_run = "init_conf" in self.uploads["system"]
        if_run = if_run and "deep_potential" in self.uploads["frozen_model"]
        return if_run

    def check_dipole(self):
        if_run = "dipole" in self.config
        if_run = if_run and "train_confs" in self.uploads["system"]
        if_start = if_run
        return if_run, if_start
    
    def check_train(self):
        if_run = "deep_model" in self.config
        if_start = if_run and "train_confs" in self.uploads["system"]
        if_start = if_start and "train_label" in self.uploads["other"]
        return if_run, if_start
    
    def check_predict(self):
        if_run = self.check_provide_sample() or self.check_sample()
        if_start = if_run and "deep_wannier" in self.uploads["frozen_model"]
        if_start = if_start and "deep_model" in self.config
        return if_run, if_start
    
    def check_ir(self):
        if_run = True
        if_start = if_run and "total_dipole" in self.uploads["other"]
        return if_run, if_start

    def load_system(self, name: str):
        sys = self.uploads["system"][name]
        sys_path = sys["path"]
        sys_fmt = {
            "type_map": self.global_global["type_map"]
        }
        if "fmt" in sys:
            sys_fmt["fmt"] = sys["fmt"]
        return sys_path, sys_fmt
    
    # def load_dipole(self):
    #     dipole_config = self.config["dipole"]
    #     input_setting = dipole_config["input_setting"]
    #     complete_by_default(input_setting, {"name": self.global_global["name"]})
    #     task_setting = dipole_config["task_setting"]
    #     self._input_parameters.update({
    #         "input_setting": input_setting,
    #         "task_setting": task_setting
    #     })
    #     if self.run_config["start_steps"] == "dipole":
    #         train_confs, train_conf_fmt = self.load_system("train_confs")
    #         self._input_artifacts["train_confs"] = train_confs
    #         self._input_parameters["train_conf_fmt"] = train_conf_fmt
    #     if dipole_config["dft_type"] in ["qe", "qe_cp"]:
    #         self._input_artifacts["pseudo"] = self.uploads["other"]["pseudo"]

    # def load_train(self):
    #     self._input_parameters["dp_setting"] = self.config["deep_model"]
    #     if self.run_config["start_steps"] == "train":
    #         train_confs, train_conf_fmt = self.load_system("train_confs")
    #         train_label = self.uploads["other"]["train_label"]
    #         self._input_parameters.update({
    #             "train_conf_fmt": train_conf_fmt
    #         })
    #         self._input_artifacts.update({
    #             "train_confs": train_confs,
    #             "train_label": train_label
    #         })
    
    # def load_sample(self):
    #     init_conf, init_conf_fmt = self.load_system("init_conf")
    #     self._input_parameters.update({
    #         "global": self.global_global,
    #         "init_conf_fmt": init_conf_fmt
    #     })
    #     self._input_artifacts.update({
    #         "init_conf": init_conf,
    #         "dp_model": self.uploads["frozen_model"]["deep_potential"]
    #     })

    # def load_predict(self):
    #     if self.run_config["provide_sample"]:
    #         sampled_system, sys_fmt = self.load_system("sampled_system")
    #         self._input_parameters["sys_fmt"] = sys_fmt
    #         self._input_artifacts["sampled_system"] = sampled_system
    #     else:
    #         self.load_sample()
    #     if self.run_config["start_steps"] == "predict":
    #         self._input_parameters["dp_setting"] = self.config["deep_model"]
    #         self._input_artifacts["dwann_model"] = self.uploads["frozen_model"]["deep_wannier"]
    
    # def load_ir(self):
    #     self._input_parameters["global"] = self.global_global
    #     if self.run_config["start_steps"] == "cal_ir":
    #         self._input_artifacts["total_dipole"] = self.uploads["other"]["total_dipole"]


class IRflow(Steps):
    step_list = ["dipole", "train", "predict", "cal_ir"]
    def __init__(self, 
            name: str,
            run_config: dict,
            executors: Dict[str, Executor],
            upload_python_packages: List[Union[str, Path]] = None
        ):
        self.run_config = run_config
        self.executors = executors
        ir_inputs = IRinputs(run_config)
        input_parameters_l, input_artifacts_l = ir_inputs.get_inputs_list()
        print("IRflow: inputs_list: ")
        print("parameters:", input_parameters_l)
        print("artifacts:", input_artifacts_l)
        _input_parameters_temp = {
            "global": InputParameter(type = dict, value = {}),
            "input_setting": InputParameter(type = dict, value = {}),
            "task_setting": InputParameter(type = dict, value = {}),
            "dp_setting": InputParameter(type = dict, value = {}),
            "train_conf_fmt": InputParameter(type = dict, value = {}),
            "init_conf_fmt": InputParameter(type = dict, value = {}),
            "sys_fmt": InputParameter(type = dict, value = {}),
        }
        _input_artifacts_temp = {
            "dp_model": InputArtifact(),
            "dwann_model": InputArtifact(),
            "pseudo": InputArtifact(optional = True),
            "train_confs": InputArtifact(),
            "sampled_system": InputArtifact(),
            "init_conf": InputArtifact(),
            "train_label": InputArtifact(),
            "total_dipole": InputArtifact()
        }
        _output_artifacts_temp = {
            "dipole": ["wannier_centroid"],
            "train": ["dwann_model"],
            "predict": ["total_dipole"],
            "cal_ir": ["ir"]
        }
        self._input_parameters = {
            key: _input_parameters_temp[key] for key in input_parameters_l
        }
        self._input_artifacts = {
            key: _input_artifacts_temp[key] for key in input_artifacts_l
        }
        self._output_artifacts = {
            key: OutputArtifact() for key in _output_artifacts_temp[run_config["end_steps"]]
        }

        super().__init__(
            name = name,
            inputs = Inputs(
                parameters = self._input_parameters,
                artifacts = self._input_artifacts
            ),
            outputs = Outputs(
                artifacts = self._output_artifacts
            )
        )
        if not upload_python_packages:
            self.upload_python_packages = spectra_flow.__path__
        else:
            self.upload_python_packages = upload_python_packages + spectra_flow.__path__
        self.build_all()

    def build_all(self):
        build_list = {
            "dipole": self.build_dipole_steps,
            "train": self.build_train_step,
            "predict": self.build_predict_steps,
            "cal_ir": self.build_ir_step
        }
        build_list[self.run_config["end_steps"]]()
    
    def build_dipole_steps(self):
        config = self.run_config["dipole_config"]
        
        if config["dft_type"] == "qe":
            prepare_op = PrepareQeWann
            run_op = RunQeWann
            collect_op = CollectWann
        elif config["dft_type"] == "qe_cp":
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
        dipole_template = DipoleSteps(
            "Cal-Dipole",
            mlwf_template,
            CalWC,
            self.executors["cal"],
            self.upload_python_packages
        )

        if self.run_config["start_steps"] == "dipole":
            self.cal_dipole = Step(
                "Cal-Dipole",
                dipole_template,
                parameters = {
                    "input_setting": self.inputs.parameters["input_setting"],
                    "task_setting": self.inputs.parameters["task_setting"],
                    "conf_fmt": self.inputs.parameters["train_conf_fmt"]
                },
                artifacts = {
                    "confs": self.inputs.artifacts["train_confs"],
                    "pseudo": self.inputs.artifacts["pseudo"]
                }
            )
            out = {
                "train_confs": self.inputs.artifacts["train_confs"],
                "train_conf_fmt": self.inputs.parameters["train_conf_fmt"]
            }
        else:
            raise NotImplementedError()
        
        out["train_label"] = self.cal_dipole.outputs.artifacts["wannier_centroid"]
        if self.run_config["end_steps"] == "dipole":
            self.add(self.cal_dipole)
            self.outputs.artifacts["wannier_centroid"]._from = out["train_label"]
        print("build dipole")
        return out, self.cal_dipole
    
    def build_train_step(self):
        dp_setting = self.inputs.parameters["dp_setting"]
        if self.run_config["start_steps"] == "train":
            train_confs = self.inputs.artifacts["train_confs"]
            train_conf_fmt = self.inputs.parameters["train_conf_fmt"]
            train_label = self.inputs.artifacts["train_label"]
        else:
            out, step = self.build_dipole_steps()
            self.add(step)
            train_confs = out["train_confs"]
            train_conf_fmt = out["train_conf_fmt"]
            train_label = out["train_label"]
        
        self.train_dwann = Step(
            "train-dwann",
            PythonOPTemplate(
                DWannTrain, 
                image = "registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6",
                python_packages = self.upload_python_packages
            ),
            artifacts = {
                "confs": train_confs,
                "label": train_label
            },
            parameters = {
                "conf_fmt": train_conf_fmt,
                "dp_setting": dp_setting
            },
            executor = self.executors["train"]
        )

        out = {
            "dwann_model": self.train_dwann.outputs.artifacts["frozen_model"]
        }
        if self.run_config["end_steps"] == "train":
            self.add(self.train_dwann)
            self.outputs.artifacts["dwann_model"]._from = out["dwann_model"]
        print("build train")
        return out, self.train_dwann
    
    def build_lmp_step(self):
        self.lmp_step = Step(
            "Lammps",
            PythonOPTemplate(
                DpLmpSample, 
                image = "registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6",
                python_packages = self.upload_python_packages
            ),
            artifacts = {
                "init_conf": self.inputs.artifacts["init_conf"],
                "dp_model": self.inputs.artifacts["dp_model"]
            },
            parameters = {
                "global": self.inputs.parameters["global"],
                "conf_fmt": self.inputs.parameters["init_conf_fmt"]
            },
            executor = self.executors["deepmd_lammps"]
        )
        print("build deepmd_lammps")
        return {
            "sampled_system": self.lmp_step.outputs.artifacts["sampled_system"],
            "sys_fmt": self.lmp_step.outputs.parameters["sys_fmt"]
        }, self.lmp_step
    
    def build_predict_steps(self):
        step_l = []

        if self.run_config["start_steps"] == "predict":
            dwann_model = self.inputs.artifacts["dwann_model"]
        else:
            out, step = self.build_train_step()
            step_l.append(step)
            dwann_model = out["dwann_model"]

        if self.run_config["provide_sample"]:
            sampled_system = self.inputs.artifacts["sampled_system"]
            sys_fmt = self.inputs.parameters["sys_fmt"]
        else:
            out, step = self.build_lmp_step()
            step_l.append(step)
            sampled_system = out["sampled_system"]
            sys_fmt = out["sys_fmt"]
        
        if len(step_l) > 0:
            self.add(step_l)
        
        predict_template = PredictSteps(
            "predict",
            self.executors["predict"],
            self.executors["cal"],
            self.upload_python_packages
        )
        self.predict_dipole = Step(
            "Predict-dipole",
            predict_template,
            parameters = {
                "dp_setting": self.inputs.parameters["dp_setting"],
                "sys_fmt": sys_fmt
            },
            artifacts = {
                "sampled_system": sampled_system,
                "dwann_model": dwann_model
            }
        )
        
        out = {
            "total_dipole": self.predict_dipole.outputs.artifacts["total_dipole"]
        }
        if self.run_config["end_steps"] == "predict":
            self.add(self.predict_dipole)
            self.outputs.artifacts["total_dipole"]._from = out["total_dipole"]
        print("build predict")
        return out, self.predict_dipole
    
    def build_ir_step(self):
        if self.run_config["start_steps"] == "cal_ir":
            total_dipole = self.inputs.artifacts["total_dipole"]
        else:
            out, step = self.build_predict_steps()
            self.add(step)
            total_dipole = out["total_dipole"]
        self.cal_ir = Step(
            "Cal-IR",
            PythonOPTemplate(
                CalIR, 
                image = "registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6",
                python_packages = self.upload_python_packages
            ),
            parameters = {
                "global": self.inputs.parameters["global"]
            },
            artifacts = {
                "total_dipole": total_dipole
            },
            executor = self.executors["cal"]
        )
        
        out = {
            "ir": self.cal_ir.outputs.artifacts["ir"]
        }
        if self.run_config["end_steps"] == "cal_ir":
            self.add(self.cal_ir)
            self.outputs.artifacts["ir"]._from = out["ir"]
        print("build ir")
        return out, self.cal_ir
