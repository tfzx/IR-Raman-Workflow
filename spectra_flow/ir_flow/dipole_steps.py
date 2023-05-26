from typing import Dict, List, Tuple, Union
from pathlib import Path
from dflow import (
    OutputParameter,
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
import spectra_flow
from spectra_flow.base_workflow import BasicSteps
from spectra_flow.mlwf.mlwf_steps import MLWFSteps
from spectra_flow.post.wannier_centroid_op import CalWC
# from mlwf_op.prepare_input_op import Prepare
# from mlwf_op.run_mlwf_op import RunMLWF
# from mlwf_op.collect_wfc_op import CollectWFC
# from mlwf_op.qe_wannier90 import PrepareQeWann, RunQeWann
# from mlwf_op.collect_wannier90 import CollectWann

class DipoleSteps(BasicSteps):
    @classmethod
    def get_inputs(cls) -> Tuple[Dict[str, InputParameter], Dict[str, InputArtifact]]:
        return {
            "input_setting": InputParameter(type = dict, value = {}),
            "task_setting": InputParameter(type = dict, value = {}),
            "conf_fmt": InputParameter(type = dict, value = {})
        }, {
            "confs": InputArtifact(),
            "pseudo": InputArtifact(),
            "cal_dipole_python": InputArtifact(optional = True)
        }
    
    @classmethod
    def get_outputs(cls) -> Tuple[Dict[str, OutputParameter], Dict[str, OutputArtifact]]:
        return {}, {
            "wannier_function_centers": OutputArtifact(),
            "wannier_centroid": OutputArtifact()
        }

    def __init__(
            self,
            name: str,
            mlwf_template: MLWFSteps,
            cal_executor: Executor,
            upload_python_packages: List[Union[str, Path]] = None
        ):
        super().__init__(name)
        if not upload_python_packages:
            upload_python_packages = spectra_flow.__path__
        self.build_steps(
            mlwf_template, 
            cal_executor, 
            upload_python_packages
        )
    
    def build_steps(
            self, 
            mlwf_template: MLWFSteps,
            cal_executor: Executor,
            upload_python_packages: List[Union[str, Path]]
        ):
        input_setting = self.inputs.parameters["input_setting"]
        task_setting = self.inputs.parameters["task_setting"]
        conf_fmt = self.inputs.parameters["conf_fmt"]
        confs_artifact = self.inputs.artifacts["confs"]
        pseudo_artifact = self.inputs.artifacts["pseudo"]
        cal_dipole_python = self.inputs.artifacts["cal_dipole_python"]
        mlwf_step = Step(
            name = "cal-MLWF",
            template = mlwf_template,
            parameters = {
                "input_setting": input_setting,
                "task_setting": task_setting,
                "conf_fmt": conf_fmt
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
                CalWC, 
                image="registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6",
                python_packages = upload_python_packages
            ),
            parameters = {
                "conf_fmt": conf_fmt
            },
            artifacts={
                "confs": confs_artifact,
                "cal_dipole_python": cal_dipole_python,
                "wannier_function_centers": mlwf_step.outputs.artifacts["wannier_function_centers"]
            },
            executor = cal_executor
        )
        self.add(wc_step)
        self.outputs.artifacts["wannier_function_centers"]._from = mlwf_step.outputs.artifacts["wannier_function_centers"]
        self.outputs.artifacts["wannier_centroid"]._from = wc_step.outputs.artifacts["wannier_centroid"]
