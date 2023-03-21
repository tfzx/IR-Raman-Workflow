from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from mlwf_op.run_mlwf_op import RunMLWF
from dflow.utils import (
    set_directory
)
import shutil, subprocess, sys, shlex


def run_command(
    cmd: Union[List[str], str],
    raise_error: bool = True,
    input: Optional[str] = None,
    try_bash: bool = False,
    interactive: bool = True,
    **kwargs,
) -> Tuple[int, str, str]:
    """
    Run shell command in subprocess

    Parameters:
    ----------
    cmd: list of str, or str
        Command to execute
    raise_error: bool
        Wheter to raise an error if the command failed
    input: str, optional
        Input string for the command
    try_bash: bool
        Try to use bash if bash exists, otherwise use sh
    **kwargs:
        Arguments in subprocess.Popen

    Raises:
    ------
    AssertionError:
        Raises if the error failed to execute and `raise_error` set to `True`

    Return:
    ------
    return_code: int
        The return code of the command
    out: str
        stdout content of the executed command
    err: str
        stderr content of the executed command
    """
    if isinstance(cmd, str):
        cmd = cmd.split()
    elif isinstance(cmd, list):
        cmd = [str(x) for x in cmd]

    if try_bash:
        arg = "-ic" if interactive else "-c"
        script = "if command -v bash 2>&1 >/dev/null; then bash %s " % arg + \
            shlex.quote(" ".join(cmd)) + "; else " + " ".join(cmd) + "; fi"
        cmd = [script]
        kwargs["shell"] = True

    sub = subprocess.Popen(
        args=cmd,
        stdin=None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        **kwargs
    )
    if input is not None:
        sub.stdin.write(bytes(input, encoding=sys.stdout.encoding))
    out = b""
    while sub.poll() is None:
        line = sub.stdout.readline()
        line_show = line.strip().decode(sys.stdout.encoding)
        if len(line_show) > 0:
            sub.stdout.flush()
            print(line_show)
            out += line
    # out, err = sub.communicate()
    return_code = sub.poll()
    # sub.stdout.seek(0)
    out = out.decode(sys.stdout.encoding)
    # err = err.decode(sys.stdout.encoding)
    if raise_error:
        assert return_code == 0, "Command %s failed: \n%s" % (cmd, "")
    return return_code, out

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
        run_command(" ".join([self.pw2wan_cmd, "<", f"{self.name}.pw2wan"]))
        run_command(" ".join([self.wannier_cmd, self.name]))
        run_command("rm -rf out")
        backward_dir = Path(backward_dir_name)
        backward_dir.mkdir()
        for f in backward_list:
            for p in Path(".").glob(f):
                if p.is_file():
                    shutil.copyfile(p, backward_dir)
                else:
                    shutil.copytree(p, backward_dir / p.name)
        return backward_dir