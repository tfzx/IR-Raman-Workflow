from typing import Dict, List, Tuple, Union, Optional
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
import spectra_flow

class MLWFSteps(Steps):
    def __init__(self, 
            name: str, 
            prepare_op: OP,
            run_op: OP,
            collect_op: OP, 
            prepare_executor: Executor,
            run_executor: Executor,
            collect_executor: Executor = None,
            upload_python_packages: List[Union[str, Path]] = None
        ) -> None:
        
        self._input_parameters = {
            "mlwf_setting": InputParameter(type = dict, value = {}),
            "task_setting": InputParameter(type = dict, value = {}),
            "conf_fmt": InputParameter(type = dict, value = {})
        }
        self._input_artifacts = {
            "confs": InputArtifact(),
            "pseudo": InputArtifact(),
            "cal_dipole_python": InputArtifact(optional = True)
        }
        self._output_artifacts = {
            "backward": OutputArtifact(),
            "wannier_function_centers": OutputArtifact()
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
            upload_python_packages = spectra_flow.__path__
        if not collect_executor:
            collect_executor = prepare_executor
        self.build_steps(
            prepare_op, 
            run_op,
            collect_op,
            prepare_executor,
            run_executor,
            collect_executor,
            upload_python_packages
        )
    
    def build_steps(
            self, 
            prepare_op: OP, 
            run_op: OP, 
            collect_op: OP, 
            prepare_executor: Executor, 
            run_executor: Executor, 
            collect_executor: Executor,
            upload_python_packages: List[Union[str, Path]]
        ):
        mlwf_setting = self.inputs.parameters["mlwf_setting"]
        task_setting = self.inputs.parameters["task_setting"]
        confs_artifact = self.inputs.artifacts["confs"]
        cal_dipole_python = self.inputs.artifacts["cal_dipole_python"]
        conf_fmt = self.inputs.parameters["conf_fmt"]
        prepare = Step(
            "prepare",
            PythonOPTemplate(
                prepare_op, 
                image="registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6",
                python_packages = upload_python_packages
            ),
            artifacts={
                "confs": confs_artifact,
                "pseudo": self.inputs.artifacts["pseudo"],
                "cal_dipole_python": cal_dipole_python
            },
            parameters={
                "mlwf_setting": mlwf_setting,
                "task_setting": task_setting,
                "conf_fmt": conf_fmt
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
                "mlwf_setting": prepare.outputs.parameters["mlwf_setting"],
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
                "mlwf_setting": prepare.outputs.parameters["mlwf_setting"],
                "conf_fmt": conf_fmt
            },
            executor = collect_executor
        )
        self.add(collect)
        self.outputs.artifacts["backward"]._from = run.outputs.artifacts["backward"]
        self.outputs.artifacts["wannier_function_centers"]._from = collect.outputs.artifacts["wannier_function_centers"]