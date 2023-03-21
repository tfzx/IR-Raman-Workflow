from typing import Dict, Union
from pathlib import Path
import dpdata, time
from dflow.plugins.dispatcher import DispatcherExecutor
from dflow import (
    Step, 
    Workflow,
    upload_artifact,
    download_artifact
)
from dflow.python import (
    PythonOPTemplate,
)
from mlwf_op.prepare_input_op import Prepare
from mlwf_op.inputs import QePwInputs, QePw2Wannier90Inputs, Wannier90Inputs, complete_qe, complete_wannier90, complete_pw2wan

class PrepareQe(Prepare):
    def __init__(self):
        super().__init__()

    def init_inputs(self, input_setting: Dict[str, Union[str, dict]], confs: dpdata.System):
        self.name = input_setting["name"]
        self.run_nscf = input_setting["dft_params"]["cal_type"] == "scf+nscf"
        input_scf, kpoints_scf = complete_qe(
            input_setting["dft_params"]["qe_params"], 
            "scf", 
            input_setting["dft_params"]["k_grid"], 
            confs
        )
        self.scf_writer = QePwInputs(input_scf, kpoints_scf, input_setting["dft_params"]["atomic_species"], confs)
        if self.run_nscf:
            input_nscf, kpoints_nscf = complete_qe(
                input_setting["dft_params"]["qe_params"], 
                "nscf", 
                input_setting["dft_params"]["k_grid"], 
                confs
            )
            self.nscf_writer = QePwInputs(input_nscf, kpoints_nscf, input_setting["dft_params"]["atomic_species"], confs)
        input_pw2wan = complete_pw2wan(input_setting["dft_params"]["pw2wan_params"], self.name)
        self.pw2wan_writer = QePw2Wannier90Inputs(input_pw2wan)
        wan_params, proj, kpoints = complete_wannier90(
            input_setting["wannier90_params"]["wan_params"], 
            input_setting["wannier90_params"]["projections"],
            input_setting["dft_params"]["k_grid"]
        )
        self.wannier90_writer = Wannier90Inputs(wan_params, proj, kpoints, confs)
        return super().init_inputs(input_setting, confs)

    def write_one_frame(self, frame: int):
        Path("scf.in").write_text(self.scf_writer.write(frame))
        if self.run_nscf:
            Path("nscf.in").write_text(self.nscf_writer.write(frame))
        Path(f"{self.name}.pw2wan").write_text(self.pw2wan_writer.write(frame))
        Path(f"{self.name}.win").write_text(self.wannier90_writer.write(frame))
        return super().write_one_frame(frame)


def test_prepare(input_setting: Dict[str, Union[str, dict]], machine_setting: dict):
    dispatcher_executor = DispatcherExecutor(
            machine_dict={
                "batch_type": "Bohrium",
                "context_type": "Bohrium",
                "remote_profile": {
                    "input_data": {
                        "job_type": "container",
                        "platform": "ali",
                        "scass_type" : "c8_m16_cpu"
                    },
                },
            },
        )
    wf = Workflow("prepare-test")
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
                   executor = dispatcher_executor
                  )
    wf.add(prepare)
    wf.submit()
    while wf.query_status() in ["Pending", "Running"]:
        time.sleep(1)

    assert(wf.query_status() == "Succeeded")
    return wf