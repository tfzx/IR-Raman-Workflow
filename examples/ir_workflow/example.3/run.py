import time, matplotlib.pyplot as plt, numpy as np
from dflow import Workflow, download_artifact
from spectra_flow.ir_flow.ir import build_ir
from spectra_flow.utils import bohrium_login, load_json
from pathlib import Path

if __name__ == "__main__":
    acc_p = Path("../../account.json")
    if acc_p.exists():
        bohrium_login(load_json(acc_p))
    else:
        bohrium_login()

    # 4 steps: dipole, train, predict, cal_ir
    run_config = {
        "start_steps": "train",
        "end_steps": "train"
    }
    ir_step = build_ir(load_json("parameters.json"), load_json("../machine.json"), run_config)
    wf = Workflow("ir-workflow")
    wf.add(ir_step)
    wf.submit()
    while wf.query_status() in ["Pending", "Running"]:
        time.sleep(1)
    assert(wf.query_status() == "Succeeded")
    lcurve = download_artifact(wf.query_step("water")[0].outputs.artifacts["lcurve"], path = "back")[0]
    print(Path(lcurve).read_text())