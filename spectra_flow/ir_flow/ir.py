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
    OutputArtifact
)
from dflow.io import Inputs, Outputs
from dflow.python import (
    PythonOPTemplate,
)
from dflow.step import Step
from dflow.plugins.dispatcher import DispatcherExecutor
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


class IRflow(Steps):
    def __init__(self, 
            name: str,
            run_config: dict,
            executors: Dict[str, Executor],
            upload_python_packages: List[Union[str, Path]] = None
        ):
        self.run_config = run_config
        self.executors = executors

        self._input_parameters = {
            "global": InputParameter(type = dict, value = {}),
            "input_setting": InputParameter(type = dict, value = {}),
            "task_setting": InputParameter(type = dict, value = {}),
            "dp_setting": InputParameter(type = dict, value = {}),
            "conf_fmt": InputParameter(type = dict, value = {}),
            "sys_fmt": InputParameter(type = dict, value = {}),
            "input_conf_fmt": InputParameter(type = dict, value = {})
        }
        self._input_artifacts = {
            "dp_model": InputArtifact(),
            "confs": InputArtifact(),
            "pseudo": InputArtifact(),
            "wannier_centroid": InputArtifact(),
            "sampled_system": InputArtifact(),
            "dwann_model": InputArtifact(),
            "input_conf": InputArtifact(),
            "total_dipole": InputArtifact()
        }
        self._output_artifacts = {
            "wannier_centroid": OutputArtifact(),
            "dwann_model": OutputArtifact(),
            "total_dipole": OutputArtifact(),
            "ir": OutputArtifact()
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
            self.executors.get("collect", None),
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
                    "conf_fmt": self.inputs.parameters["conf_fmt"]
                },
                artifacts = {
                    "confs": self.inputs.artifacts["confs"],
                    "pseudo": self.inputs.artifacts["pseudo"]
                }
            )
            out = {
                "confs": self.inputs.artifacts["confs"],
                "conf_fmt": self.inputs.parameters["conf_fmt"]
            }
        else:
            raise NotImplementedError()
        
        out["wannier_centroid"] = self.cal_dipole.outputs.artifacts["wannier_centroid"]
        if self.run_config["end_steps"] == "dipole":
            self.add(self.cal_dipole)
            self.outputs.artifacts["wannier_centroid"]._from = out["wannier_centroid"]
        print("build dipole")
        return out, self.cal_dipole
    
    def build_train_step(self):
        dp_setting = self.inputs.parameters["dp_setting"]
        if self.run_config["start_steps"] == "train":
            confs = self.inputs.artifacts["confs"]
            conf_fmt = self.inputs.parameters["conf_fmt"]
            label = self.inputs.artifacts["wannier_centroid"]
        else:
            out, step = self.build_dipole_steps()
            self.add(step)
            confs = out["confs"]
            conf_fmt = out["conf_fmt"]
            label = out["wannier_centroid"]
        
        self.train_dwann = Step(
            "train-dwann",
            PythonOPTemplate(
                DWannTrain, 
                image = "registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6",
                python_packages = self.upload_python_packages
            ),
            artifacts = {
                "confs": confs,
                "label": label
            },
            parameters = {
                "conf_fmt": conf_fmt,
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
                "input_conf": self.inputs.artifacts["input_conf"],
                "dp_model": self.inputs.artifacts["dp_model"]
            },
            parameters = {
                "global": self.inputs.parameters["global"],
                "conf_fmt": self.inputs.parameters["input_conf_fmt"]
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
