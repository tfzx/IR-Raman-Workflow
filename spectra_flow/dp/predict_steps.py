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
    OutputParameter,
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
from spectra_flow.base_workflow import SuperOP
from spectra_flow.post.total_dipole_op import CalTotalDipole
from spectra_flow.post.total_polar_op import CalTotalPolar
from spectra_flow.dp.dp_predict import DWannPredict, DPolarPredict

class PredictSteps(SuperOP):
    @classmethod
    def get_inputs(cls) -> Tuple[Dict[str, InputParameter], Dict[str, InputArtifact]]:
        return {
            "dp_setting": InputParameter(type = dict, value = {}),
            "sys_fmt": InputParameter(type = dict, value = {}),
        }, {
            "sampled_system": InputArtifact(),
            "frozen_model": InputArtifact(),
            "cal_dipole_python": InputArtifact(optional = True)
        }
    
    @classmethod
    def get_outputs(cls) -> Tuple[Dict[str, OutputParameter], Dict[str, OutputArtifact]]:
        return {}, {
            "predicted_tensor": OutputArtifact(),
            "total_tensor": OutputArtifact(),
        }

    def __init__(self, 
            name: str,
            tensor_type: str,
            predict_executor: Optional[Executor],
            cal_executor: Optional[Executor],
            upload_python_packages: Optional[List[Union[str, Path]]] = None
        ):
        super().__init__(name)
        if not upload_python_packages:
            upload_python_packages = list(spectra_flow.__path__)
        tensor_type = tensor_type.lower().strip()
        self.build_steps(
            tensor_type,
            predict_executor, 
            cal_executor,
            upload_python_packages
        )

    def build_steps(
            self, 
            tensor_type: str,
            predict_executor: Optional[Executor],
            cal_executor: Optional[Executor],
            upload_python_packages: List[Union[str, Path]]
        ):
        dp_setting = self.inputs.parameters["dp_setting"]
        sys_fmt = self.inputs.parameters["sys_fmt"]
        sampled_system = self.inputs.artifacts["sampled_system"]
        frozen_model = self.inputs.artifacts["frozen_model"]
        cal_dipole_python = self.inputs.artifacts["cal_dipole_python"]
        
        if tensor_type == "dipole":
            predict_op = DWannPredict
            total_tensor_op = CalTotalDipole
            tensor_name = "wannier_centroid"
        elif tensor_type == "polar":
            predict_op = DPolarPredict
            total_tensor_op = CalTotalPolar
            tensor_name = "polar"
        else:
            raise RuntimeError(f"Tensor_type {tensor_type} is not supported!")

        predict_step = Step(
            f"predict-{tensor_type}",
            PythonOPTemplate(
                predict_op,  # type: ignore
                image="registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6",
                python_packages = upload_python_packages # type: ignore
            ),
            artifacts = {
                "sampled_system": sampled_system,
                "frozen_model": frozen_model
            },
            parameters = {
                "dp_setting": dp_setting,
                "sys_fmt": sys_fmt
            },
            executor = predict_executor
        )
        self.add(predict_step)

        total_tensor = Step(
            f"total-{tensor_type}",
            PythonOPTemplate(
                total_tensor_op,  # type: ignore
                image="registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6",
                python_packages = upload_python_packages # type: ignore
            ),
            artifacts={
                "confs": sampled_system,
                "cal_dipole_python": cal_dipole_python,
                tensor_name: predict_step.outputs.artifacts["predicted_tensor"]
            },
            parameters = {
                "conf_fmt": sys_fmt,
            },
            executor = cal_executor
        )
        self.add(total_tensor)

        self.outputs.artifacts["predicted_tensor"]._from = predict_step.outputs.artifacts["predicted_tensor"]
        self.outputs.artifacts["total_tensor"]._from = total_tensor.outputs.artifacts[f"total_{tensor_type}"]