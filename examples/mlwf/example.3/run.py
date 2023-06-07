import time
from dflow import (
    Step, 
    Workflow,
    upload_artifact,
    download_artifact
)
from spectra_flow.SuperOP.polar_steps import PolarSteps
from spectra_flow.mlwf.qe_wannier90 import PrepareQeWann, RunQeWann, CollectWann
from spectra_flow.SuperOP.mlwf_steps import MLWFSteps
from spectra_flow.utils import load_json, bohrium_login, get_executor

if __name__ == "__main__":
    bohrium_login(load_json("../../account.json"))
    dipole_config = load_json("dipole.json")
    polar_config = load_json("polar.json")
    machine_config = load_json("../machine.json")

    base_executor = get_executor(machine_config["executors"]["base"])
    run_executor = get_executor(machine_config["executors"]["run"])
    cal_executor = get_executor(machine_config["executors"]["cal"])
    # use OPs in mlwf.qe_wannier90
    mlwf_template = MLWFSteps(
        name = "MLWF-Qe",
        prepare_op = PrepareQeWann,     # type: ignore
        run_op = RunQeWann,             # type: ignore
        collect_op = CollectWann,       # type: ignore
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
            "mlwf_setting": dipole_config["mlwf_setting"],
            "task_setting": dipole_config["task_setting"],
            "conf_fmt": {
                "type_map": ["O", "H"],
                "fmt": "deepmd/raw"
            }
        },
        artifacts = {
            "confs": upload_artifact("./data"),     # type: ignore
            "pseudo": upload_artifact("./pseudo"),  # type: ignore
            # Optional, upload the python file to calculate wannier centroid.
            "cal_dipole_python": upload_artifact("cal_dipole.py")   # type: ignore
        }
    )

    wf = Workflow("cal-polar-workflow") # type: ignore
    wf.add(cal_polar_step)
    wf.submit()
    while wf.query_status() in ["Pending", "Running"]:
        time.sleep(1)
    assert(wf.query_status() == "Succeeded")

    step = wf.query_step("Polar-Step")[0]
    download_artifact(step.outputs.artifacts["wannier_function_centers"], path="./back")    # type: ignore
    download_artifact(step.outputs.artifacts["wannier_centroid"], path="./back")            # type: ignore
    download_artifact(step.outputs.artifacts["polarizability"], path="./back")              # type: ignore