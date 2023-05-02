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
    Slices,
    OP
)
from dflow.step import Step
import spectra_flow
from spectra_flow.ir_flow.dipole_steps import DipoleSteps
from spectra_flow.dp.dp_train import DWannTrain

class TrainDwannSteps(Steps):
    def __init__(self, 
            name: str,
            dipole_template: DipoleSteps,
            train_executor: Executor,
            upload_python_packages: List[Union[str, Path]] = None
        ):
        self._input_parameters = {
            "input_setting": InputParameter(type = dict, value = {}),
            "task_setting": InputParameter(type = dict, value = {}),
            "train_inputs": InputParameter(type = dict, value = {}),
        }
        self._input_artifacts = {
            "confs": InputArtifact(),
            "pseudo": InputArtifact()
        }
        self._output_artifacts = {
            "wannier_centroid": OutputArtifact(),
            "frozen_model": OutputArtifact(),
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
            upload_python_packages = spectra_flow.__path__
        self.build_steps(
            dipole_template, 
            train_executor, 
            upload_python_packages
        )

    def build_steps(
            self, 
            dipole_template: DipoleSteps,
            train_executor: Executor,
            upload_python_packages: List[Union[str, Path]]
        ):
        input_setting = self.inputs.parameters["input_setting"]
        task_setting = self.inputs.parameters["task_setting"]
        train_inputs = self.inputs.parameters["train_inputs"]
        confs_artifact = self.inputs.artifacts["confs"]
        pseudo_artifact = self.inputs.artifacts["pseudo"]
        cal_dipole_step = Step(
            name = "cal-dipole",
            template = dipole_template,
            parameters = {
                "input_setting": input_setting,
                "task_setting": task_setting
            },
            artifacts = {
                "confs": confs_artifact,
                "pseudo": pseudo_artifact
            }
        )
        self.add(cal_dipole_step)

        train_dw_step = Step(
            "train-dw",
            PythonOPTemplate(
                DWannTrain, 
                image="registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6",
                python_packages = upload_python_packages
            ),
            artifacts={
                "confs": confs_artifact,
                "dp_train_inputs": train_inputs,
                "tensor": cal_dipole_step.outputs.artifacts["wannier_centroid"]
            },
            parameters = {
                "dp_train_inputs": train_inputs,
            },
            executor = train_executor
        )
        self.add(train_dw_step)
        self.outputs.artifacts["wannier_centroid"]._from = cal_dipole_step.outputs.artifacts["wannier_centroid"]
        self.outputs.artifacts["frozen_model"]._from = train_dw_step.outputs.artifacts["frozen_model"]