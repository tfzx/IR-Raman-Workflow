from typing import Dict, List, Tuple, Union
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
from dflow.python import (
    PythonOPTemplate,
    Slices,
    OP
)
import mlwf_op, pp_op
from mlwf_op.mlwf_steps import MLWFSteps
# from mlwf_op.prepare_input_op import Prepare
# from mlwf_op.run_mlwf_op import RunMLWF
# from mlwf_op.collect_wfc_op import CollectWFC
# from mlwf_op.qe_wannier90 import PrepareQeWann, RunQeWann
# from mlwf_op.collect_wannier90 import CollectWann

class DipoleSteps(Steps):
    def __init__(
            self,
            name: str,
            mlwf_template: MLWFSteps,
            wc_op: OP,
            wc_executor: Executor,
            upload_python_packages: List[Union[str, Path]] = None
        ):
        self._input_parameters = {
            "input_setting": InputParameter(type = dict, value = {}),
            "task_setting": InputParameter(type = dict, value = {}),
        }
        self._input_artifacts = {
            "confs": InputArtifact(),
            "pseudo": InputArtifact()
        }
        self._output_artifacts = {
            "backward": OutputArtifact(),
            "wannier_function_centers": OutputArtifact(),
            "wannier_centroid": OutputArtifact()
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
            upload_python_packages = mlwf_op.__path__ + pp_op.__path__
        self.build_steps(
            mlwf_template, 
            wc_op,
            wc_executor, 
            upload_python_packages
        )
        
    def build_steps(
            self, 
            mlwf_template: MLWFSteps,
            wc_op: OP, 
            wc_executor: Executor,
            upload_python_packages: List[Union[str, Path]]
        ):
        input_setting = self.inputs.parameters["input_setting"]
        task_setting = self.inputs.parameters["task_setting"]
        confs_artifact = self.inputs.artifacts["confs"]
        pseudo_artifact = self.inputs.artifacts["pseudo"]
        prep_polar = Step(
            
        )
        mlwf_step = Step(
            name = "cal-MLWF",
            template = mlwf_template,
            parameters = {
                "input_setting": input_setting,
                "task_setting": task_setting
            },
            artifacts = {
                "confs": confs_artifact,
                "pseudo": pseudo_artifact
            }
        )
        self.add(mlwf_step)
        wc_step = Step(
            "cal-wc",
            PythonOPTemplate(
                wc_op, 
                image="registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6",
                python_packages = upload_python_packages
            ),
            artifacts={
                "confs": confs_artifact,
                "wannier_function_centers": mlwf_step.outputs.artifacts["wannier_function_centers"]
            },
            executor = wc_executor
        )
        self.add(wc_step)
        self.outputs.artifacts["backward"]._from = mlwf_step.outputs.artifacts["backward"]
        self.outputs.artifacts["wannier_function_centers"]._from = mlwf_step.outputs.artifacts["wannier_function_centers"]
        self.outputs.artifacts["wannier_centroid"]._from = wc_step.outputs.artifacts["wannier_centroid"]
