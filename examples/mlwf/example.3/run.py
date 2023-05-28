import time
from dflow import (
    Step, 
    Workflow,
    upload_artifact,
    download_artifact
)
from spectra_flow.raman_flow.polar_steps import PolarSteps
from spectra_flow.mlwf.ef_qe_wann import PrepareEfQeWann, RunEfQeWann, CollectEfWann
from spectra_flow.mlwf.mlwf_steps import MLWFSteps
from spectra_flow.utils import load_json, bohrium_login, get_executor

if __name__ == "__main__":
    bohrium_login(load_json("../../account.json"))
    dipole_config = load_json("dipole.json")
    polar_config = load_json("polar.json")
    machine_config = load_json("../machine.json")

    base_executor = get_executor(machine_config["executors"]["base"])
    run_executor = get_executor(machine_config["executors"]["run"])
    cal_executor = get_executor(machine_config["executors"]["cal"])

    mlwf_template = MLWFSteps(
        name = "MLWF-Qe",
        prepare_op = PrepareEfQeWann, # use OPs in mlwf.qe_wannier90
        run_op = RunEfQeWann,
        collect_op = CollectEfWann,
        prepare_executor = base_executor,
        run_executor = run_executor
    )

    polar_template = PolarSteps(
        name = "Polar",
        mlwf_template = mlwf_template,
        base_exexutor = base_executor,
        cal_executor = cal_executor
    )

    import spectra_flow.post.cal_dipole
    cal_polar_step = Step(
        name = "Polar-Step",
        template = polar_template,
        parameters = {
            "polar_setting": polar_config,
            "input_setting": dipole_config["input_setting"],
            "task_setting": dipole_config["task_setting"],
            "conf_fmt": {
                "type_map": ["O", "H"],
                "fmt": "deepmd/raw"
            }
        },
        artifacts = {
            "confs": upload_artifact("./data"),
            "pseudo": upload_artifact("./pseudo"),
            # Optional, upload the python file to calculate wannier centroid.
            "cal_dipole_python": upload_artifact(spectra_flow.post.cal_dipole.__file__)
        }
    )

    wf = Workflow("cal-polar-workflow")
    wf.add(cal_polar_step)
    wf.submit()
    while wf.query_status() in ["Pending", "Running"]:
        time.sleep(1)
    assert(wf.query_status() == "Succeeded")

    step = wf.query_step("Polar-Step")[0]
    download_artifact(step.outputs.artifacts["wannier_function_centers"], path="./back")
    download_artifact(step.outputs.artifacts["wannier_centroid"], path="./back")
    download_artifact(step.outputs.artifacts["polarizability"], path="./back")