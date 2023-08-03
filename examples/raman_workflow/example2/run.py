import time, matplotlib.pyplot as plt, numpy as np
from dflow import Workflow, download_artifact
from spectra_flow.raman_flow.raman import build_raman
from spectra_flow.utils import bohrium_login, load_json
from pathlib import Path

def show_raman(raman_iso, raman_aniso):
    plt.plot(raman_iso[:, 0], raman_iso[:, 1], label = r'$D_2O$, calculated', scalex = 1.5, scaley= 2.2)
    plt.xlim((2000, 3000.))
    plt.ylim((0, 140))
    plt.xlabel(r'Wavenumber($\rm cm^{-1}$)', fontdict = {'size': 12})
    plt.ylabel(r'$R_{iso}(\omega) (a.u.)$', fontdict = {'size': 12})
    plt.title("Isotropic Raman Spectra")
    plt.legend()
    plt.show()

    plt.plot(raman_aniso[:, 0], raman_aniso[:, 1], label = r'$D_2O$, calculated', scalex = 1.5, scaley= 2.2)
    plt.xlim((2000, 3000.))
    plt.ylim((0, 60))
    plt.xlabel(r'Wavenumber($\rm cm^{-1}$)', fontdict = {'size': 12})
    plt.ylabel(r'$R_{aniso}(\omega) (a.u.)$', fontdict = {'size': 12})
    plt.title("Anisotropic Raman Spectra")
    plt.legend()
    plt.show()

    plt.plot(raman_aniso[:, 0], raman_aniso[:, 1] * raman_aniso[:, 0] / 2000, label = r'$D_2O$, calculated', scalex = 1.5, scaley= 2.2)
    plt.xlim((0, 2500.))
    plt.ylim((0, 12))
    plt.xlabel(r'Wavenumber($\rm cm^{-1}$)', fontdict = {'size': 12})
    plt.ylabel(r'$R_{aniso}(\omega)\cdot\omega (a.u.)$', fontdict = {'size': 12})
    plt.title("Low-frequency Raman Spectra")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    acc_p = Path("../../account.json")
    if acc_p.exists():
        bohrium_login(load_json(acc_p))
    else:
        bohrium_login()

    # 4 steps: dipole, train, predict, cal_ir
    run_config = {
        "start_step": "cal_raman",
        "end_step": "cal_raman"
    }
    raman_step = build_raman("raman-example1", load_json("parameters.json"), load_json("../machine.json"), run_config, debug = True)

    wf = Workflow("raman-workflow") # type: ignore
    wf.add(raman_step)
    wf.submit()
    while wf.query_status() in ["Pending", "Running"]:
        time.sleep(1)
    assert(wf.query_status() == "Succeeded")

    raman_path = download_artifact(wf.query_step("raman-example1")[0].outputs.artifacts["raman_iso"], path = "back")[0] # type: ignore
    raman_iso = np.loadtxt(raman_path)
    raman_path = download_artifact(wf.query_step("raman-example1")[0].outputs.artifacts["raman_aniso"], path = "back")[0] # type: ignore
    raman_aniso = np.loadtxt(raman_path)
    show_raman(raman_iso, raman_aniso)