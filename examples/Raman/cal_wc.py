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
import spectra_flow
from spectra_flow.post.wannier_centroid_op import CalWC
upload_packages += spectra_flow.__path__

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

bohrium_login()

cal_wc_excutor = DispatcherExecutor(
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
wf = Workflow("calculate-wc")
cal = Step(
    "calculate",
    PythonOPTemplate(CalWC, image="registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6"),
    artifacts={
        "confs": upload_artifact("./data"),
        "wannier_function_centers": upload_artifact("./back/wfc.raw")
    },
    executor = cal_wc_excutor
)
wf.add(cal)

wf.submit()
while wf.query_status() in ["Pending", "Running"]:
    time.sleep(1)

assert(wf.query_status() == "Succeeded")

step = wf.query_step("calculate")[0]
download_artifact(step.outputs.artifacts["wannier_centroid"], path="./data")