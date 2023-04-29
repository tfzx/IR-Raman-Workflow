from typing import Dict, Optional, Tuple, Union
import numpy as np
from copy import deepcopy
from tempfile import TemporaryFile
import dpdata
import abc
from spectra_flow.mlwf.utils import kmesh, complete_by_default

class QeInputs(abc.ABC):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def write(frame: int) -> str:
        pass

    @classmethod
    def write_configuration(cls, conf: dpdata.System, atoms: np.ndarray = None):
        if atoms is None:
            atoms = np.array(conf["atom_names"]).reshape(-1, 1)[conf["atom_types"]]
        cells = np.reshape(conf["cells"], (3, 3))
        coords = np.reshape(conf["coords"], (-1, 3))
        with TemporaryFile("w+") as f:
            f.write("\nCELL_PARAMETERS { angstrom }\n")
            np.savetxt(f, cells, fmt = "%15.8f")
            f.write("\nATOMIC_POSITIONS { angstrom }\n")
            atomic_positions = np.concatenate([atoms, np.char.mod("%15.8f", coords)], axis = 1)
            np.savetxt(f, atomic_positions, fmt = "%s")
            f.seek(0)
            conf_str = f.read()
        return conf_str
    
    @classmethod
    def write_parameters(cls, input_params: dict, atomic_species: dict = None, 
                          kpoints: dict = None, optional_input: str = None):
        with TemporaryFile("w+") as f:
            for key, val in input_params.items():
                if val is None or len(val) == 0:
                    f.writelines([f"&{key}\n", "/\n"])
                    continue
                f.write(f"&{key}\n")
                assert isinstance(val, dict)
                for key2, val2 in val.items():
                    if isinstance(val2, bool):
                        val2 = "." + str(val2).lower() + "."
                    elif isinstance(val2, str):
                        val2 = f"'{val2}'"
                    elif val2 is None:
                        val2 = "'none'"
                    f.write(f"    {key2} = {val2},\n")
                f.write("/\n")
            if atomic_species is not None:
                f.write("\nATOMIC_SPECIES\n")
                for atom, info in atomic_species.items():
                    f.write(f"{atom} {info['mass']} {info['pseudo']}\n")
            if kpoints is not None:
                kpoint_type: str = kpoints["type"].strip()
                f.write("\nK_POINTS { " + kpoint_type + " }\n")
                if kpoint_type == "crystal":
                    k_points = kpoints["k_points"]
                    f.write(f"{k_points.shape[0]}\n")
                    np.savetxt(f, k_points, fmt = "%15.8f")
                elif kpoint_type == "automatic":
                    nk1, nk2, nk3 = kpoints["k_grid"]
                    sk1, sk2, sk3 = kpoints.get("offset", (0, 0, 0))
                    f.write(f"{nk1} {nk2} {nk3} {sk1} {sk2} {sk3}\n")
            if optional_input is not None:
                f.write(optional_input)
                f.write("\n")
            f.seek(0)
            params_str = f.read()
        return params_str

def complete_qe(input_params: Dict[str, dict], calculation: Optional[str] = None, 
                k_grid: Optional[Tuple[int, int, int]] = None, 
                confs: Optional[dpdata.System] = None):
    input_params_default: Dict[str, dict] = {}
    if calculation:
        input_params_default["control"] = {
            "calculation"   : calculation
        }
    if confs:
        input_params_default["system"] = {
            "ntyp"  : len(confs["atom_names"]),
            "nat"   : confs.get_natoms()
        }
    input_params = complete_by_default(input_params, input_params_default, if_copy = True)
    kpoints = None
    if k_grid:
        calculation = input_params["control"]["calculation"]
        if calculation == "scf":
            kpoints = {
                "type": "automatic",
                "k_grid": k_grid
            }
        else:
            kpoints = {
                "type": "crystal",
                "k_grid": kmesh(*k_grid)
            }
    return input_params, kpoints

class QeParamsConfs(QeInputs):
    def __init__(self, input_params: Dict[str, dict], kpoints: Dict[str, Union[str, np.ndarray]], 
                 atomic_species: dict, confs: dpdata.System, optional_input: str = None) -> None:
        super().__init__()
        """
        Automatically complemant the system info("ntyp" and "nat") to input parmeters, and generate the kpoints by calculation type. 
        If cal = "scf", then kpoints will be "automatic". Otherwise kpoints will be "crystal" type. 
        """
        self.params_str = self.write_parameters(input_params, atomic_species, kpoints, optional_input)
        self.atoms = np.array(confs["atom_names"]).reshape(-1, 1)[confs["atom_types"]]
        self.confs = confs

    def write(self, frame: int):
        return "\n".join([self.params_str, self.write_configuration(self.confs[frame], self.atoms)])

def complete_pw2wan(input_params: Dict[str, dict], name: str, prefix: str = "mlwf", outdir: str = "out"):
    input_params = deepcopy(input_params)
    input_params["inputpp"].update({
        "outdir" : outdir,
        "prefix" : prefix,
        "seedname" : name,
    })
    return input_params

class QeParams(QeInputs):
    def __init__(self, input_params: Dict[str, dict]) -> None:
        super().__init__()
        self.params_str = self.write_parameters(input_params)
    
    def write(self, frame: int):
        return self.params_str

def complete_wannier90(wan_params: dict, proj: Optional[dict], k_grid: Tuple[int, int, int]):
    wan_params = deepcopy(wan_params)
    wan_params["mp_grid"] = "{}, {}, {}".format(*k_grid)
    if proj and len(proj) == 0:
        proj = None
    kpoints = kmesh(*k_grid)[:, :3]
    return wan_params, proj, kpoints

class Wannier90Inputs:
    def __init__(self, wan_params: dict, proj: Optional[dict], kpoints: np.ndarray, confs: dpdata.System) -> None:
        self.params_str = self.write_parameters(wan_params, kpoints, proj)
        self.atoms = np.array(confs["atom_names"]).reshape(-1, 1)[confs["atom_types"]]
        self.confs = confs

    def write(self, frame: int):
        return "\n".join([self.params_str, self.write_configuration(self.confs[frame], self.atoms)])

    @classmethod
    def write_configuration(cls, conf: dpdata.System, atoms: np.ndarray = None):
        if atoms is None:
            atoms = np.array(conf["atom_names"]).reshape(-1, 1)[conf["atom_types"]]
        cells = np.reshape(conf["cells"], (3, 3))
        coords = np.reshape(conf["coords"], (-1, 3))
        with TemporaryFile("w+") as f:
            f.write("\nbegin unit_cell_cart\n")
            f.write("Ang\n")
            np.savetxt(f, cells, fmt = "%15.8f")
            f.write("end unit_cell_cart\n")
            f.write("\nbegin atoms_cart\n")
            atomic_positions = np.concatenate([atoms, np.char.mod("%15.8f", coords)], axis = 1)
            np.savetxt(f, atomic_positions, fmt = "%s")
            f.write("end atoms_cart\n")
            f.seek(0)
            conf_str = f.read()
        return conf_str

    @classmethod
    def write_parameters(cls, wan_params: Dict[str, object], kpoints: np.ndarray, proj: Optional[Dict[str, str]]):
        with TemporaryFile("w+") as f:
            for key, val in wan_params.items():
                f.write(f"{key} = {val}\n")

            if proj:
                f.write("\nbegin projections\n")
                for atom, option in proj.items():
                    f.write(f"    {atom}: {option}\n")
                f.write("end projections\n")

            f.write("\nbegin kpoints\n")
            np.savetxt(f, kpoints, fmt = "%15.8f")
            f.write("end kpoints\n")
            f.seek(0)
            params_str = f.read()
        return params_str
            