import time, matplotlib.pyplot as plt, numpy as np
from dflow import Workflow, download_artifact
from spectra_flow.ir_flow.ir2 import build_ir
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
        "start_steps": "predict",
        "end_steps": "cal_ir"
    }
    ir_step = build_ir("ir-example1", load_json("parameters.json"), load_json("../machine.json"), run_config, debug = True)

    wf = Workflow("ir-workflow")
    wf.add(ir_step)
    wf.submit()
    while wf.query_status() in ["Pending", "Running"]:
        time.sleep(1)
    assert(wf.query_status() == "Succeeded")
    ir_path = download_artifact(wf.query_step("ir-example1")[0].outputs.artifacts["ir"], path = "back")[0]
    ir = np.loadtxt(ir_path)
    plt.plot(ir[:, 0], ir[:, 1] / 1000, label = r'$D_2O$, calculated', scalex = 1.5, scaley= 2.2)
    plt.xlim((0, 4000.))
    plt.ylim((0, 15))
    plt.xlabel(r'Wavenumber($\rm cm^{-1}$)', fontdict = {'size': 12})
    plt.ylabel(r'$n(\omega)\alpha(\omega) (10^3 cm^{-1})$', fontdict = {'size': 12})
    plt.legend()
    plt.show()