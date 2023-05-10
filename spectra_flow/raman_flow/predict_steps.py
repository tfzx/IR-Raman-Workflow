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
import spectra_flow
from spectra_flow.post.total_polar_op import CalTotalPolar
from spectra_flow.dp.dp_predict import DPolarPredict

class PredictSteps(Steps):
    def __init__(self, 
            name: str,
            predict_executor: Executor,
            cal_executor: Executor,
            upload_python_packages: List[Union[str, Path]] = None
        ):
        self._input_parameters = {
            "dp_setting": InputParameter(type = dict, value = {}),
            "sys_fmt": InputParameter(type = dict, value = {}),
        }
        self._input_artifacts = {
            "sampled_system": InputArtifact(),
            "dpolar_model": InputArtifact(),
        }
        self._output_artifacts = {
            "predicted_polar": OutputArtifact(),
            "total_polar": OutputArtifact(),
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
            predict_executor, 
            cal_executor,
            upload_python_packages
        )

    def build_steps(
            self, 
            predict_executor: Executor,
            cal_executor: Executor,
            upload_python_packages: List[Union[str, Path]]
        ):
        dp_setting = self.inputs.parameters["dp_setting"]
        sys_fmt = self.inputs.parameters["sys_fmt"]
        sampled_system_artifact = self.inputs.artifacts["sampled_system"]
        dpolar_model_artifact = self.inputs.artifacts["dpolar_model"]
        
        predict_polar = Step(
            "predict-polar",
            PythonOPTemplate(
                DPolarPredict, 
                image="registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6",
                python_packages = upload_python_packages
            ),
            artifacts = {
                "sampled_system": sampled_system_artifact,
                "frozen_model": dpolar_model_artifact
            },
            parameters = {
                "dp_setting": dp_setting,
                "sys_fmt": sys_fmt
            },
            executor = predict_executor
        )
        self.add(predict_polar)

        total_polar = Step(
            "total-polar",
            PythonOPTemplate(
                CalTotalPolar, 
                image="registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6",
                python_packages = upload_python_packages
            ),
            artifacts={
                "polar": predict_polar.outputs.artifacts["predicted_tensor"]
            },
            executor = cal_executor
        )
        self.add(total_polar)

        self.outputs.artifacts["predicted_polar"]._from = predict_polar.outputs.artifacts["predicted_tensor"]
        self.outputs.artifacts["total_polar"]._from = total_polar.outputs.artifacts["total_polar"]