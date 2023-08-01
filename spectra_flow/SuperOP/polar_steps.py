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
from dflow.python import (
    PythonOPTemplate,
    OP
)
import spectra_flow
from spectra_flow.base_workflow import SuperOP
from spectra_flow.SuperOP.mlwf_steps import MLWFSteps
from spectra_flow.SuperOP.dipole_steps import DipoleSteps
from spectra_flow.SuperOP.prep_polar import PrepPolar
from spectra_flow.SuperOP.post_polar import PostPolar
# from mlwf_op.prepare_input_op import Prepare
# from mlwf_op.run_mlwf_op import RunMLWF
# from mlwf_op.collect_wfc_op import CollectWFC
# from mlwf_op.qe_wannier90 import PrepareQeWann, RunQeWann
# from mlwf_op.collect_wannier90 import CollectWann

class PolarSteps(SuperOP):
    @classmethod
    def get_inputs(cls) -> Tuple[Dict[str, InputParameter], Dict[str, InputArtifact]]:
        return {
            "polar_setting": InputParameter(type = dict, value = {}),
            "mlwf_setting": InputParameter(type = dict, value = {}),
            "task_setting": InputParameter(type = dict, value = {}),
            "conf_fmt": InputParameter(type = dict, value = {})
        }, {
            "confs": InputArtifact(),
            "pseudo": InputArtifact(),
            "cal_dipole_python": InputArtifact(optional = True)
        }
    
    @classmethod
    def get_outputs(cls) -> Tuple[Dict[str, OutputParameter], Dict[str, OutputArtifact]]:
        return {
            "final_conf_fmt": OutputParameter()
        }, {
            "labeled_confs": OutputArtifact(),
            "failed_confs": OutputArtifact(),
            "wannier_function_centers": OutputArtifact(),
            "wannier_centroid": OutputArtifact(),
            "polarizability": OutputArtifact()
        }

    def __init__(
            self,
            name: str,
            mlwf_template: MLWFSteps,
            base_exexutor: Executor,
            cal_executor: Executor,
            upload_python_packages: Optional[List[Union[str, Path]]] = None
        ):
        super().__init__(name)
        if not upload_python_packages:
            upload_python_packages = list(spectra_flow.__path__)
        self.build_steps(
            mlwf_template, 
            base_exexutor,
            cal_executor, 
            upload_python_packages
        )
        
    def build_steps(
            self, 
            mlwf_template: MLWFSteps,
            base_exexutor: Executor,
            cal_executor: Executor,
            upload_python_packages: List[Union[str, Path]]
        ):
        polar_setting = self.inputs.parameters["polar_setting"]
        mlwf_setting = self.inputs.parameters["mlwf_setting"]
        task_setting = self.inputs.parameters["task_setting"]
        conf_fmt = self.inputs.parameters["conf_fmt"]
        confs_artifact = self.inputs.artifacts["confs"]
        pseudo_artifact = self.inputs.artifacts["pseudo"]
        cal_dipole_python = self.inputs.artifacts["cal_dipole_python"]
        prep_polar = Step(
            name = "prep-polar",
            template = PythonOPTemplate(
                PrepPolar,  # type: ignore
                python_packages = upload_python_packages # type: ignore
            ),
            parameters = {
                "polar_setting": polar_setting,
                "mlwf_setting": mlwf_setting
            },
            key = "prepare-polar",
            executor = base_exexutor
        )
        self.add(prep_polar)
        dipole_template = DipoleSteps(
            "Cal-Dipole",
            mlwf_template,
            cal_executor,
            upload_python_packages
        )
        dipole_step = Step(
            "cal-dipole",
            dipole_template,
            parameters = {
                "mlwf_setting": prep_polar.outputs.parameters["mlwf_setting"],
                "task_setting": task_setting,
                "conf_fmt": conf_fmt
            },
            artifacts = {
                "confs": confs_artifact,
                "pseudo": pseudo_artifact,
                "cal_dipole_python": cal_dipole_python,
            }
        )
        self.add(dipole_step)
        final_confs = dipole_step.outputs.artifacts["labeled_confs"]
        failed_confs = dipole_step.outputs.artifacts["failed_confs"]
        final_conf_fmt = dipole_step.outputs.parameters["final_conf_fmt"]

        post_polar = Step(
            name = "post-polar",
            template = PythonOPTemplate(
                PostPolar, # type: ignore
                python_packages = upload_python_packages # type: ignore
            ),
            parameters = {
                "polar_setting": polar_setting,
                "conf_fmt": final_conf_fmt
            },
            artifacts = {
                "confs": final_confs,
                "wannier_centroid": dipole_step.outputs.artifacts["wannier_centroid"]
            },
            key = "post-polar",
            executor = cal_executor
        )
        self.add(post_polar)
        self.outputs.artifacts["failed_confs"]._from = failed_confs
        self.outputs.parameters["final_conf_fmt"].value_from_parameter = final_conf_fmt
        self.outputs.artifacts["wannier_function_centers"]._from = dipole_step.outputs.artifacts["wannier_function_centers"]
        self.outputs.artifacts["wannier_centroid"]._from = dipole_step.outputs.artifacts["wannier_centroid"]
        self.outputs.artifacts["labeled_confs"]._from = post_polar.outputs.artifacts["labeled_confs"]
        self.outputs.artifacts["polarizability"]._from = post_polar.outputs.artifacts["polarizability"]
