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
)
from dflow.step import Step
import spectra_flow
from spectra_flow.base_workflow import SuperOP
from spectra_flow.MD.deepmd_lmp_op import DpLmpSample

class MDSteps(SuperOP):
    @classmethod
    def get_inputs(cls) -> Tuple[Dict[str, InputParameter], Dict[str, InputArtifact]]:
        return {
            "global": InputParameter(type = dict, value = {}),
            "init_conf_fmt": InputParameter(type = dict, value = {}),
        }, {
            "init_conf": InputArtifact(),
            "dp_model": InputArtifact()
        }
    
    @classmethod
    def get_outputs(cls) -> Tuple[Dict[str, OutputParameter], Dict[str, OutputArtifact]]:
        return {
            "sys_fmt": OutputParameter()
        }, {
            "sampled_system": OutputArtifact(),
            "lammps_log": OutputArtifact(),
        }

    def __init__(self, 
            name: str,
            md_executor: Executor,
            upload_python_packages: Optional[List[Union[str, Path]]] = None
        ):
        super().__init__(name)
        if not upload_python_packages:
            upload_python_packages = list(spectra_flow.__path__)
        self.build_steps(
            md_executor, 
            upload_python_packages
        )

    def build_steps(
            self, 
            md_executor: Executor,
            upload_python_packages: List[Union[str, Path]]
        ):
        global_config = self.inputs.parameters["global"]
        init_conf_fmt = self.inputs.parameters["init_conf_fmt"]
        init_conf = self.inputs.artifacts["init_conf"]
        dp_model = self.inputs.artifacts["dp_model"]
        
        deepmd_lammps = Step(
            "lammps",
            PythonOPTemplate(
                DpLmpSample,  # type: ignore
                image="registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6",
                python_packages = upload_python_packages # type: ignore
            ),
            artifacts = {
                "init_conf": init_conf,
                "dp_model": dp_model
            },
            parameters = {
                "global": global_config,
                "conf_fmt": init_conf_fmt
            },
            executor = md_executor
        )
        self.add(deepmd_lammps)

        self.outputs.artifacts["sampled_system"]._from = deepmd_lammps.outputs.artifacts["sampled_system"]
        self.outputs.parameters["sys_fmt"].value_from_parameter = deepmd_lammps.outputs.parameters["sys_fmt"]
        self.outputs.artifacts["lammps_log"]._from = deepmd_lammps.outputs.artifacts["lammps_log"]