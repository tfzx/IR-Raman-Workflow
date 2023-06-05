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
from spectra_flow.base_workflow import SuperOP
from spectra_flow.dp.dp_train import DWannTrain

class TrainDwannSteps(SuperOP):
    @classmethod
    def get_inputs(cls) -> Tuple[Dict[str, InputParameter], Dict[str, InputArtifact]]:
        return {
            "conf_fmt": InputParameter(type = dict, value = {}),
            "dp_setting": InputParameter(type = dict, value = {}),
        }, {
            "confs": InputArtifact(),
            "wannier_centroid": InputArtifact()
        }
    
    @classmethod
    def get_outputs(cls) -> Tuple[Dict[str, OutputParameter], Dict[str, OutputArtifact]]:
        return {}, {
            "dwann_model": OutputArtifact(),
        }

    def __init__(self, 
            name: str,
            train_executor: Executor,
            upload_python_packages: List[Union[str, Path]] = None
        ):
        super().__init__(name)
        if not upload_python_packages:
            upload_python_packages = spectra_flow.__path__
        self.build_steps(
            train_executor, 
            upload_python_packages
        )

    def build_steps(
            self, 
            train_executor: Executor,
            upload_python_packages: List[Union[str, Path]]
        ):
        conf_fmt = self.inputs.parameters["conf_fmt"]
        dp_setting = self.inputs.parameters["dp_setting"]
        confs = self.inputs.artifacts["confs"]
        wannier_centroid = self.inputs.artifacts["wannier_centroid"]

        train_dwann_step = Step(
            "train-dwann",
            PythonOPTemplate(
                DWannTrain, 
                image="registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6",
                python_packages = upload_python_packages
            ),
            artifacts={
                "confs": confs,
                "label": wannier_centroid
            },
            parameters = {
                "conf_fmt": conf_fmt,
                "dp_setting": dp_setting,
            },
            executor = train_executor
        )
        self.add(train_dwann_step)
        self.outputs.artifacts["dwann_model"]._from = train_dwann_step.outputs.artifacts["frozen_model"]