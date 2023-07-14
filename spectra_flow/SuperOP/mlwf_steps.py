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
    OutputParameter,
    OutputArtifact,
    argo_sequence,
    argo_len
)
from dflow.python import (
    PythonOPTemplate,
    Slices,
    OP
)
import spectra_flow
from spectra_flow.base_workflow import SuperOP

class MLWFSteps(SuperOP):
    @classmethod
    def get_inputs(cls) -> Tuple[Dict[str, InputParameter], Dict[str, InputArtifact]]:
        return {
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
            "backward": OutputArtifact(),
            "final_confs": OutputArtifact(),
            "failed_confs": OutputArtifact(),
            "wannier_function_centers": OutputArtifact()
        }
    
    def __init__(self, 
            name: str, 
            prepare_op: OP,
            run_op: OP,
            collect_op: OP, 
            prepare_executor: Optional[Executor],
            run_executor: Optional[Executor],
            collect_executor: Optional[Executor] = None,
            upload_python_packages: Optional[List[Union[str, Path]]] = None,
            parallelism: Optional[int] = None,
            continue_on_error: bool = False,
            continue_on_failed: bool = True,
            continue_on_num_success: Optional[int] = None,
            continue_on_success_ratio: Optional[float] = None,
        ) -> None:
        super().__init__(
            name, 
            # parallelism = parallelism
        )
        if not upload_python_packages:
            upload_python_packages = []
        upload_set = set(upload_python_packages)
        upload_set.update(spectra_flow.__path__)
        upload_python_packages = list(upload_set)
        if not collect_executor:
            collect_executor = prepare_executor
        self.build_steps(
            prepare_op, 
            run_op,
            collect_op,
            prepare_executor,
            run_executor,
            collect_executor,
            upload_python_packages,
            parallelism,    # TODO: add parallelism at workflow level.
            continue_on_error,
            continue_on_failed,
            continue_on_num_success,
            continue_on_success_ratio
        )
    
    def build_steps(
            self, 
            prepare_op: OP, 
            run_op: OP, 
            collect_op: OP, 
            prepare_executor: Optional[Executor], 
            run_executor: Optional[Executor], 
            collect_executor: Optional[Executor],
            upload_python_packages: List[Union[str, Path]],
            parallelism: Optional[int] = None,
            continue_on_error: bool = False,
            continue_on_failed: bool = True,
            continue_on_num_success: Optional[int] = None,
            continue_on_success_ratio: Optional[float] = None,
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
                python_packages = upload_python_packages # type: ignore
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
            key = "prepare-mlwf",
            executor = prepare_executor
        )
        self.add(prepare)
        run = Step(
            "run",
            PythonOPTemplate(
                run_op, 
                image = "registry.dp.tech/dptech/prod-13467/wannier-qe:7.0",
                slices = Slices(
                    input_artifact = ["task_path"],
                    input_parameter = ["frames"],
                    output_artifact = ["backward"],
                    sub_path = True
                ),
                python_packages = upload_python_packages # type: ignore
            ),
            parameters = {
                "mlwf_setting": prepare.outputs.parameters["mlwf_setting"],
                "task_setting": task_setting,
                "frames": prepare.outputs.parameters["frames_list"]
            },
            artifacts = {
                "task_path": prepare.outputs.artifacts["task_path"]
            },
            key = "run-mlwf-{{item.order}}",
            executor = run_executor,
            parallelism = parallelism,
            continue_on_error = continue_on_error,
            continue_on_failed = continue_on_failed,
            continue_on_num_success = continue_on_num_success,
            continue_on_success_ratio = continue_on_success_ratio,
        )
        self.add(run)
        collect = Step(
            "collect",
            PythonOPTemplate(
                collect_op, 
                image="registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6",
                python_packages = upload_python_packages, # type: ignore
            ),
            artifacts={
                "confs": confs_artifact,
                "backward": run.outputs.artifacts["backward"]
            },
            parameters={
                "mlwf_setting": prepare.outputs.parameters["mlwf_setting"],
                "conf_fmt": conf_fmt
            },
            key = "collect-wfc",
            executor = collect_executor
        )
        self.add(collect)
        self.outputs.artifacts["backward"]._from = run.outputs.artifacts["backward"]
        self.outputs.artifacts["final_confs"]._from = collect.outputs.artifacts["final_confs"]
        self.outputs.artifacts["failed_confs"]._from = collect.outputs.artifacts["failed_confs"]
        self.outputs.parameters["final_conf_fmt"].value_from_parameter = collect.outputs.parameters["final_conf_fmt"]
        self.outputs.artifacts["wannier_function_centers"]._from = collect.outputs.artifacts["wannier_function_centers"]