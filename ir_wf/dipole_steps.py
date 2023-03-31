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
    Slices,
    OP
)
from dflow.utils import (
    set_directory
)
import mlwf_op, pp_op
# from mlwf_op.prepare_input_op import Prepare
# from mlwf_op.run_mlwf_op import RunMLWF
# from mlwf_op.collect_wfc_op import CollectWFC
# from mlwf_op.qe_wannier90 import PrepareQeWann, RunQeWann
# from mlwf_op.collect_wannier90 import CollectWann
from pp_op.wannier_centroid_op import CalWC

class DipoleSteps(Steps):
    def __init__(
            self,
            name,
            prepare_op: OP,
            run_op: OP,
            collect_op: OP,
            prepare_executor: Executor,
            run_executor: Executor,
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
            prepare_op, 
            run_op, 
            collect_op, 
            prepare_executor, 
            run_executor, 
            cal_executor, 
            upload_python_packages
        )
        
    def build_steps(
            self, 
            prepare_op: OP, 
            run_op: OP, 
            collect_op: OP, 
            prepare_executor: Executor, 
            run_executor: Executor, 
            cal_executor: Executor,
            upload_python_packages: List[Union[str, Path]]
        ):
        input_setting = self.inputs.parameters["input_setting"]
        task_setting = self.inputs.parameters["task_setting"]
        confs_artifact = self.inputs.artifacts["confs"]
        prepare = Step(
            "prepare",
            PythonOPTemplate(
                prepare_op, 
                image="registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6",
                python_packages = upload_python_packages
            ),
            artifacts={
                "confs": confs_artifact,
                "pseudo": self.inputs.artifacts["pseudo"]
            },
            parameters={
                "input_setting": input_setting,
                "task_setting": task_setting
            },
            executor = prepare_executor
        )
        self.add(prepare)
        run = Step(
            "run",
            PythonOPTemplate(
                run_op, 
                image = "registry.dp.tech/dptech/prod-13467/wannier-qe:7.0",
                slices = Slices(
                    # "int('{{item}}')",
                    input_artifact = ["task_path"],
                    input_parameter = ["frames"],
                    output_artifact = ["backward"],
                    sub_path = True
                ),
                python_packages = upload_python_packages
            ),
            parameters = {
                "input_setting": input_setting,
                "task_setting": task_setting,
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
            "collect",
            PythonOPTemplate(
                collect_op, 
                image="registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6",
                python_packages = upload_python_packages,
            ),
            artifacts={
                "confs": confs_artifact,
                "backward": run.outputs.artifacts["backward"]
            },
            parameters={
                "input_setting": input_setting,
            },
            executor = prepare_executor
        )
        self.add(collect)
        cal = Step(
            "calculate",
            PythonOPTemplate(
                CalWC, 
                image="registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6",
                python_packages = upload_python_packages
            ),
            artifacts={
                "confs": confs_artifact,
                "wannier_function_centers": collect.outputs.artifacts["wannier_function_centers"]
            },
            executor = cal_executor
        )
        self.add(cal)
        self.outputs.artifacts["backward"]._from = run.outputs.artifacts["backward"]
        self.outputs.artifacts["wannier_function_centers"]._from = collect.outputs.artifacts["wannier_function_centers"]
        self.outputs.artifacts["wannier_centroid"]._from = cal.outputs.artifacts["wannier_centroid"]
        return
