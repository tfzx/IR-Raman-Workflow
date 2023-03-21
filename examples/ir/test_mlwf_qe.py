from typing import Dict, Union
from pathlib import Path
import time, json
from dflow.plugins.dispatcher import DispatcherExecutor
from dflow import (
    Step, 
    Workflow,
    upload_artifact,
    download_artifact
)
from dflow.python import (
    PythonOPTemplate,
    Slices,
    upload_packages
)
import mlwf_op
from mlwf_op.qe_prepare import PrepareQe
from mlwf_op.qe_run import RunMLWFQe

upload_packages += mlwf_op.__path__

def test_prep_run(input_setting: Dict[str, Union[str, dict]], machine_setting: dict):
    prepare_excutor = DispatcherExecutor(
            machine_dict={
                "batch_type": "Bohrium",
                "context_type": "Bohrium",
                "remote_profile": {
                    "input_data": {
                        "job_type": "container",
                        "platform": "ali",
                        "scass_type" : "c8_m16_cpu",
                        "image_name": "registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6",
                    },
                },
            },
        )
    run_executor = DispatcherExecutor(
            machine_dict={
                "batch_type": "Bohrium",
                "context_type": "Bohrium",
                "remote_profile": {
                    "input_data": machine_setting["params"],
                },
            },
        )
    wf = Workflow("prepare-run-test")
    prepare = Step("prepare",
                   PythonOPTemplate(PrepareQe, image="registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6"),
                   artifacts={
                        "confs": upload_artifact("./data"),
                        "pseudo": upload_artifact("./pseudo")
                    },
                   parameters={
                        "input_setting": input_setting,
                        "group_size": machine_setting["group_size"]
                   },
                   executor = prepare_excutor
                  )
    wf.add(prepare)
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
    wf.add(run)
    wf.submit()
    while wf.query_status() in ["Pending", "Running"]:
        time.sleep(1)

    assert(wf.query_status() == "Succeeded")
    return wf

def bohrium_login():
    from dflow import config, s3_config
    from dflow.plugins import bohrium
    from dflow.plugins.bohrium import TiefblueClient
    from getpass import getpass
    config["host"] = "https://workflows.deepmodeling.com"
    config["k8s_api_server"] = "https://workflows.deepmodeling.com"
    bohrium.config["username"] = input("Bohrium username: ")
    bohrium.config["password"] = getpass("Bohrium password: ")
    bohrium.config["project_id"] = input("Project ID: ")
    s3_config["repo_key"] = "oss-bohrium"
    s3_config["storage_client"] = TiefblueClient()

with open("./input_setting.json", "r") as f:
    input_setting = json.load(f)
    
with open("./machine_setting.json", "r") as f:
    machine_setting = json.load(f)

bohrium_login()

wf = test_prep_run(input_setting, machine_setting)
step = wf.query_step("prepare")[0]
download_artifact(step.outputs.artifacts["task_path"], path="./back")