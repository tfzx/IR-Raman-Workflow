from typing import Dict, List, Tuple, Union
from pathlib import Path
import dpdata, numpy as np, shutil
from dflow.python import (
    OP, 
    OPIO, 
    Artifact, 
    OPIOSign, 
    BigParameter,
)
from dflow.utils import (
    set_directory,
    run_command
)
from spectra_flow.utils import read_conf

class DpLmpSample(OP):
    def __init__(self) -> None:
        super().__init__()
    
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "ir_setting": BigParameter(Dict),
            "sample_setting": BigParameter(Dict),
            "input_conf": Artifact(Path),
            "conf_fmt": BigParameter(dict),
            "dp_model": Artifact(Path)
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "sampled_system": Artifact(Path),
            "lammps_log": Artifact(Path),
        })

    @OP.exec_sign_check
    def execute(
            self,
            op_in: OPIO,
    ) -> OPIO:
        conf_path: Path = op_in["conf_path"]
        conf_fmt = op_in["conf_fmt"]
        input_conf = read_conf(conf_path, conf_fmt)
        ir_setting = op_in["ir_setting"]
        sample_setting = op_in["sample_setting"]
        dp_model = op_in["dp_model"]
        lammps_dir, in_file_path = self.prepare_lmp(ir_setting, sample_setting, input_conf, dp_model)
        log, smp_sys = self.run_lammps(lammps_dir, in_file_path)
        return OPIO({
            "sampled_system": lammps_dir / smp_sys,
            "lammps_log": lammps_dir / log
        })
    
    def prepare_lmp(self, ir_setting: Dict, sample_setting: Dict, input_conf: dpdata.System, dp_model: Path):
        mass_l = sample_setting["mass"]
        temperature = ir_setting["temperature"]
        dt = ir_setting["dt"]
        nstep = ir_setting["nstep"]
        in_file = [
            "units           metal",
            "boundary        p p p",
            "atom_style      atomic",
            "neighbor        2.0 bin",
            "neigh_modify    every 1 delay 0 check yes",
            "read_data	     ./input.lmp"
        ]
        in_file += [f"mass {i + 1:d} {mass_l[i]:8.5f}" for i in range(len(input_conf["atom_names"]))]
        in_file += [
            "pair_style      deepmd dp_model.pb",
            "pair_coeff      * *",
            f"velocity       all create {temperature} 18234589",
            f"fix            1 all nvt temp {temperature} {temperature} {100 * dt}",
            "dump            1 all custom 1 water.dump id type x y z vx vy vz",
            f"timestep       {dt}",
            "thermo_style    custom step vol temp pe ke etotal press pxx pyy pzz pxy pxz pyz",
            "thermo          1",
            "thermo_modify   flush yes",
            f"run            {nstep}"
        ]
        lammps_dir = Path("lammps")
        lammps_dir.mkdir()
        with set_directory(lammps_dir):
            input_conf.to("lammps/lmp", "input.lmp", frame_idx = 0)
            shutil.copy(dp_model, "dp_model.pb")

            in_file_path = Path("in.water")
            in_file_path.write_text("\n".join(in_file))
        return lammps_dir, in_file_path

    def run_lammps(self, lammps_dir: Path, in_file_path: Path):
        with set_directory(lammps_dir):
            ret, out, err = run_command(f"export OMP_NUM_THREADS=32 && lmp -in {in_file_path.name}")
            log = Path("lmp.log")
            smp_sys = Path("sampled_system.npz")
            log.write_text(out)
            sys = dpdata.System("water.dump", fmt = "lammps/dump")
            np.savez_compressed(smp_sys, coords = sys["coords"], cells = sys["cells"], atom_types = sys["atom_types"])
        return log, smp_sys
