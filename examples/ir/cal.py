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
from ir_wf.dipole_steps import DipoleSteps


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
cal_executor = DispatcherExecutor(
    machine_dict={
        "batch_type": "Bohrium",
        "context_type": "Bohrium",
        "remote_profile": {
            "input_data": {
                "job_type": "container",
                "platform": "ali",
                "scass_type" : "c16_m32_cpu",
                "image_name": "registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6",
            },
        },
    },
)
steps = DipoleSteps(
    name = "calculate-dipole-qe",
    prepare_executor = prepare_executor,
    run_executor = run_executor,
    cal_executor = cal_executor
)

cal_dipole_step = Step(
    name = "calculate-dipole-step",
    template = steps,
    parameters = {
        "input_setting": input_setting,
        "task_setting": task_setting
    },
    artifacts = {
        "confs": upload_artifact("./data"),
        "pseudo": upload_artifact("./pseudo")
    }
)

wf = Workflow("cal-dipole-workflow")
wf.add(cal_dipole_step)
wf.submit()
while wf.query_status() in ["Pending", "Running"]:
    time.sleep(1)
assert(wf.query_status() == "Succeeded")

step = wf.query_step("calculate-dipole-step")[0]
download_artifact(step.outputs.artifacts["backward"], path="./back")
download_artifact(step.outputs.artifacts["wannier_function_centers"], path="./back")
download_artifact(step.outputs.artifacts["wannier_centroid"], path="./data")