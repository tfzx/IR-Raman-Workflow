from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from mlwf_op.run_mlwf_op import RunMLWF
from dflow.utils import (
    set_directory,
    run_command
)
import shutil, subprocess, sys, shlex

class RunMLWFQe(RunMLWF):
    def __init__(self) -> None:
        super().__init__()

    def init_cmd(self, commands: Dict[str, str]):
        self.pw_cmd = commands.get("pw", "pw.x")
        self.pw2wan_cmd = commands.get("pw2wannier", "pw2wannier90.x")
        self.wannier_cmd = commands.get("wannier90", "wannier90.x")
        self.wannier90_pp_cmd = commands.get("wannier90_pp", "wannier90.x")
    
    def run_one_frame(self, backward_dir_name: str, backward_list: List[str]) -> Path:
        run_command(" ".join([self.pw_cmd, "-input", "scf.in"]))
        if Path("nscf.in").exists():
            run_command(" ".join([self.pw_cmd, "-input", "nscf.in"]))
        run_command(" ".join([self.wannier90_pp_cmd, "-pp", self.name]))
        run_command(" ".join(["mpirun -n 2 pw2wannier90.x"]), input=Path(f"{self.name}.pw2wan").read_text())
        run_command(" ".join([self.wannier_cmd, self.name]))
        run_command("rm -rf out")
        backward_dir = Path(backward_dir_name)
        backward_dir.mkdir()
        for f in backward_list:
            for p in Path(".").glob(f):
                if p.is_file():
                    shutil.copy(p, backward_dir)
                else:
                    shutil.copytree(p, backward_dir / p.name)
        return backward_dir