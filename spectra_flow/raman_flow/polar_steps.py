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
    OP
)
import spectra_flow
from spectra_flow.mlwf.mlwf_steps import MLWFSteps
from spectra_flow.raman_flow.prep_polar import PrepPolar
from spectra_flow.raman_flow.post_polar import CalPolar
# from mlwf_op.prepare_input_op import Prepare
# from mlwf_op.run_mlwf_op import RunMLWF
# from mlwf_op.collect_wfc_op import CollectWFC
# from mlwf_op.qe_wannier90 import PrepareQeWann, RunQeWann
# from mlwf_op.collect_wannier90 import CollectWann

class PolarSteps(Steps):
    def __init__(
            self,
            name: str,
            mlwf_template: MLWFSteps,
            wc_op: OP,
            base_exexutor: Executor,
            cal_executor: Executor,
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
            "wannier_centroid": OutputArtifact(),
            "polarizability": OutputArtifact()
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
            upload_python_packages = []
        upload_python_packages += spectra_flow.__path__
        self.build_steps(
            mlwf_template, 
            wc_op,
            base_exexutor,
            cal_executor, 
            upload_python_packages
        )
        
    def build_steps(
            self, 
            mlwf_template: MLWFSteps,
            wc_op: OP, 
            base_exexutor: Executor,
            cal_executor: Executor,
            upload_python_packages: List[Union[str, Path]]
        ):
        input_setting = self.inputs.parameters["input_setting"]
        task_setting = self.inputs.parameters["task_setting"]
        confs_artifact = self.inputs.artifacts["confs"]
        pseudo_artifact = self.inputs.artifacts["pseudo"]
        prep_polar = Step(
            name = "prep-polar",
            template = PythonOPTemplate(
                PrepPolar, 
                python_packages = upload_python_packages
            ),
            parameters = {
                "input_setting": input_setting
            },
            executor = base_exexutor
        )
        self.add(prep_polar)
        mlwf_step = Step(
            name = "cal-MLWF",
            template = mlwf_template,
            parameters = {
                "input_setting": prep_polar.outputs.parameters["input_setting"],
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
            executor = cal_executor
        )
        self.add(wc_step)
        post_polar = Step(
            name = "post-polar",
            template = PythonOPTemplate(
                CalPolar,
                python_packages = upload_python_packages
            ),
            parameters = {
                "input_setting": input_setting
            },
            artifacts = {
                "wannier_centroid": wc_step.outputs.artifacts["wannier_centroid"]
            },
            executor = cal_executor
        )
        self.add(post_polar)
        self.outputs.artifacts["backward"]._from = mlwf_step.outputs.artifacts["backward"]
        self.outputs.artifacts["wannier_function_centers"]._from = mlwf_step.outputs.artifacts["wannier_function_centers"]
        self.outputs.artifacts["wannier_centroid"]._from = wc_step.outputs.artifacts["wannier_centroid"]
        self.outputs.artifacts["polarizability"]._from = post_polar.outputs.artifacts["polarizability"]
