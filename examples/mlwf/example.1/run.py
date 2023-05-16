import time
from dflow import (
    Step, 
    Workflow,
    upload_artifact,
    download_artifact
)
from spectra_flow.ir_flow.dipole_steps import DipoleSteps
from spectra_flow.mlwf.qe_wannier90 import PrepareQeWann, RunQeWann, CollectWann
from spectra_flow.post.wannier_centroid_op import CalWC
from spectra_flow.mlwf.mlwf_steps import MLWFSteps
from spectra_flow.utils import load_json, bohrium_login, get_executor

if __name__ == "__main__":
    bohrium_login(load_json("../../account.json"))
    dipole_config = load_json("dipole.json")
    machine_config = load_json("../machine.json")

    base_executor = get_executor(machine_config["executors"]["base"])
    run_executor = get_executor(machine_config["executors"]["run"])
    cal_executor = get_executor(machine_config["executors"]["cal"])

    mlwf_template = MLWFSteps(
        name = "MLWF-Qe",
        prepare_op = PrepareQeWann, # use OPs in mlwf.qe_wannier90
        run_op = RunQeWann,
        collect_op = CollectWann,
        prepare_executor = base_executor,
        run_executor = run_executor
    )

    dipole_template = DipoleSteps(
        name = "Dipole",
        mlwf_template = mlwf_template,
        wc_op = CalWC,
        wc_executor = cal_executor
    )

    cal_dipole_step = Step(
        name = "Dipole-Step",
        template = dipole_template,
        parameters = {
            "input_setting": dipole_config["input_setting"],
            "task_setting": dipole_config["task_setting"],
            "conf_fmt": {
                "type_map": ["O", "H"],
                "fmt": "deepmd/raw"
            }
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

    step = wf.query_step("Dipole-Step")[0]
    download_artifact(step.outputs.artifacts["backward"], path="./back")
    download_artifact(step.outputs.artifacts["wannier_function_centers"], path="./back")
    download_artifact(step.outputs.artifacts["wannier_centroid"], path="./back")