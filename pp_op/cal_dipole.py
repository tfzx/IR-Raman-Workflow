from typing import Dict, List, Tuple, Union
from pathlib import Path
import dpdata, numpy as np
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
    Slices
)
from dflow.utils import (
    set_directory
)
from mlwf_op.qe_prepare import PrepareQe
from mlwf_op.qe_run import RunMLWFQe
from mlwf_op.collect_wfc_op import CollectWFC

class cal_dipole(Steps):
    def __init__(
            self,
            name,
            prepare_executor: Executor,
            run_executor: Executor,
        ):
        self._input_parameters = {
            "input_setting": InputParameter(type = dict, value = {}),
            "machine_setting": InputParameter(type = dict, value = {}),
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
        self.build_steps(prepare_executor, run_executor)
        
    def build_steps(self, prepare_executor, run_executor):
        input_setting = self.inputs.parameters["input_setting"]
        machine_setting = self.inputs.parameters["machine_setting"]
        confs_artifact = self.inputs.artifacts["confs"]
        prepare = Step(
            "prepare",
            PythonOPTemplate(
                PrepareQe, 
                image="registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6"
            ),
            artifacts={
                "confs": confs_artifact,
                "pseudo": self.inputs.artifacts["pseudo"]
            },
            parameters={
                "input_setting": input_setting,
                "group_size": machine_setting["group_size"]
            },
            executor = prepare_executor
        )
        self.add(prepare)
        run = Step(
            "Run",
            PythonOPTemplate(
                RunMLWFQe, 
                image = "registry.dp.tech/dptech/prod-13467/wannier-qe:7.0",
                slices = Slices(
                    # "int('{{item}}')",
                    input_artifact = ["task_path"],
                    input_parameter = ["frames"],
                    output_artifact = ["backward"],
                    sub_path = True
                ),
            ),
            parameters = {
                "name": input_setting["name"],
                "backward_list": machine_setting["backward_list"],
                "backward_dir_name": machine_setting["backward_dir_name"],
                "commands": input_setting["commands"],
                "frames": prepare.outputs.parameters["frames_list"]
            },
            artifacts = {
                "task_path": prepare.outputs.artifacts["task_path"]
            },
            key = "run-MLWF-{{item}}",
            executor = run_executor
        )
        self.add(run)
        collect = Step(
            "collect-wfc",
            PythonOPTemplate(
                CollectWFC, 
                image="registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6"
            ),
            artifacts={
                "confs": confs_artifact,
                "backward": run.outputs.artifacts["backward"]
            },
            parameters={
                "name": input_setting["name"],
            },
            executor = prepare_executor
        )
        self.add(collect)
        pass
    