from typing import Dict, List, Tuple, Union
from pathlib import Path
from dflow import (
    Step,
    Executor,
    InputParameter,
    InputArtifact,
    OutputParameter,
    OutputArtifact
)
from dflow.python import (
    PythonOPTemplate,
)
from dflow.step import Step
import spectra_flow
from spectra_flow.base_workflow import BasicSteps
from spectra_flow.ir_flow.ir_op import CalIR

class IRsteps(BasicSteps):
    @classmethod
    def get_inputs(cls) -> Tuple[Dict[str, InputParameter], Dict[str, InputArtifact]]:
        return {
            "global": InputParameter(type = dict, value = {}),
        }, {
            "total_dipole": InputArtifact(),
        }
    
    @classmethod
    def get_outputs(cls) -> Tuple[Dict[str, OutputParameter], Dict[str, OutputArtifact]]:
        return {}, {
            "ir": OutputArtifact(),
        }

    def __init__(self, 
            name: str,
            cal_executor: Executor,
            upload_python_packages: List[Union[str, Path]] = None
        ):
        super().__init__(name)
        if not upload_python_packages:
            upload_python_packages = spectra_flow.__path__
        self.build_steps(
            cal_executor, 
            upload_python_packages
        )

    def build_steps(
            self, 
            cal_executor: Executor,
            upload_python_packages: List[Union[str, Path]]
        ):

        cal_ir_step = Step(
            "Cal-IR",
            PythonOPTemplate(
                CalIR, 
                image = "registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6",
                python_packages = upload_python_packages
            ),
            parameters = {
                "global": self.inputs.parameters["global"]
            },
            artifacts = {
                "total_dipole": self.inputs.artifacts["total_dipole"]
            },
            executor = cal_executor
        )
        self.add(cal_ir_step)
        self.outputs.artifacts["ir"]._from = cal_ir_step.outputs.artifacts["ir"]