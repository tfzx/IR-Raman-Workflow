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
from mlwf_op.collect_wfc_op import CollectWFC

upload_packages += mlwf_op.__path__

def test_prep_run(input_setting: Dict[str, Union[str, dict]], task_setting: dict, machine_setting: dict):
    prepare_executor = DispatcherExecutor(
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
    confs_artifact = upload_artifact("./data")
    prepare = Step(
        "prepare",
        PythonOPTemplate(PrepareQe, image="registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6"),
        artifacts={
            "confs": confs_artifact,
            "pseudo": upload_artifact("./pseudo")
        },
        parameters={
            "input_setting": input_setting,
            "task_setting": task_setting
        },
        executor = prepare_executor
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
    wf.add(run)
    collect = Step(
        "collect-wfc",
        PythonOPTemplate(CollectWFC, image="registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6"),
        artifacts={
            "confs": confs_artifact,
            "backward": run.outputs.artifacts["backward"]
        },
        parameters={
            "input_setting": input_setting,
        },
        executor = prepare_executor
    )
    wf.add(collect)
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
    with open("./account_config.json", "r") as f:
        account = json.load(f)
    # bohrium.config["username"] = input("Bohrium username: ")
    # bohrium.config["password"] = getpass("Bohrium password: ")
    # bohrium.config["project_id"] = input("Project ID: ")
    bohrium.config.update(account)
    s3_config["repo_key"] = "oss-bohrium"
    s3_config["storage_client"] = TiefblueClient()

with open("./input_setting.json", "r") as f:
    input_setting = json.load(f)

with open("./task_setting.json", "r") as f:
    task_setting = json.load(f)

with open("./machine_setting.json", "r") as f:
    machine_setting = json.load(f)

bohrium_login()


wf = test_prep_run(input_setting, task_setting, machine_setting)
step = wf.query_step("prepare")[0]
run = wf.query_step("Run")[0]
collect = wf.query_step("collect-wfc")[0]
download_artifact(run.outputs.artifacts["backward"], path="./back")
download_artifact(collect.outputs.artifacts["wannier_function_centers"], path="./back")