from typing import Dict, List, Optional, Tuple, Union, Callable
import numpy as np
from copy import deepcopy
from tempfile import TemporaryFile
import dpdata
import abc
from spectra_flow.utils import kmesh, complete_by_default, recurcive_update

class QeInputs(abc.ABC):
    names_order = []
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def write(self, frame: int) -> str:
        pass

    @classmethod
    def write_namelists(cls, input_params: Dict[str, Optional[dict]]):
        namelists = {}
        for key in input_params:
            if key.lower() in cls.names_order:
                namelists[key.lower()] = input_params[key]  # convert to the uppercase.

        with TemporaryFile("w+") as f:
            for key in cls.names_order:
                if key in namelists:
                    val = namelists[key]
                else:
                    continue
                if not val:
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
            f.seek(0)
            namelists_str = f.read()
        return namelists_str

    @classmethod
    def write_species(cls, atomic_species: Optional[dict] = None):
        if atomic_species is None:
            return ""
        with TemporaryFile("w+") as f:
            f.write("ATOMIC_SPECIES\n")
            for atom, info in atomic_species.items():
                f.write(f"{atom} {info['mass']} {info['pseudo']}\n")
            f.seek(0)
            species_str = f.read()
        return species_str

    @classmethod
    def write_conf(cls, conf: dpdata.System, atom_names: Optional[np.ndarray] = None):
        """
        Description
        -----
        Write the cards `CELL_PARAMETERS` and `ATOMIC_POSITIONS`.
        The unit is angstrom.

        Parameters
        ------
        conf: `dpdata.System`.
            The configuration (one frame).

        atoms: `np.ndarray`, optional.
            The name of atoms. If not provided, this will be read from `conf["atom_names"]`.
        
        Return
        -----
        conf_str: `str`
            A string containing the cards `CELL_PARAMETERS` and `ATOMIC_POSITIONS`.
        """
        if atom_names is None:
            atom_names = np.array(conf["atom_names"]).reshape(-1, 1)[conf["atom_types"]] # type: ignore
        else:
            atom_names = atom_names.reshape(-1, 1)
        cells = np.reshape(conf["cells"], (3, 3)) # type: ignore
        coords = np.reshape(conf["coords"], (-1, 3)) # type: ignore
        with TemporaryFile("w+") as f:
            f.write("CELL_PARAMETERS { angstrom }\n")
            np.savetxt(f, cells, fmt = "%15.8f")
            f.write("\n")
            f.write("ATOMIC_POSITIONS { angstrom }\n")
            atomic_positions = np.concatenate([atom_names, np.char.mod("%15.8f", coords)], axis = 1) # type: ignore
            np.savetxt(f, atomic_positions, fmt = "%s")
            f.write("\n")
            f.seek(0)
            conf_str = f.read()
        return conf_str
    
    @classmethod
    def write_kpoints(cls, kpoints: Optional[dict] = None):
        if kpoints is None:
            return ""
        with TemporaryFile("w+") as f:
            kpoint_type: str = kpoints["type"].strip()
            f.write(f"K_POINTS { kpoint_type }\n")
            if kpoint_type == "crystal":
                k_points = kpoints["k_points"]
                f.write(f"{k_points.shape[0]}\n")
                np.savetxt(f, k_points, fmt = "%15.8f")
            elif kpoint_type == "automatic":
                nk1, nk2, nk3 = kpoints["k_grid"]
                sk1, sk2, sk3 = kpoints.get("offset", (0, 0, 0))
                f.write(f"{nk1} {nk2} {nk3} {sk1} {sk2} {sk3}\n")
            else:
                raise NotImplementedError(f"Unsupported kpoint type {kpoint_type}.")
            f.write("\n")
            f.seek(0)
            kpoints_str = f.read()
        return kpoints_str

    # @classmethod
    # def write_parameters(cls, input_params: dict, atomic_species: Optional[dict] = None, 
    #                       kpoints: Optional[dict] = None, optional_input: Optional[str] = None):
    #     with TemporaryFile("w+") as f:
    #         for key, val in input_params.items():
    #             if val is None or len(val) == 0:
    #                 f.writelines([f"&{key}\n", "/\n"])
    #                 continue
    #             f.write(f"&{key}\n")
    #             assert isinstance(val, dict)
    #             for key2, val2 in val.items():
    #                 if isinstance(val2, bool):
    #                     val2 = "." + str(val2).lower() + "."
    #                 elif isinstance(val2, str):
    #                     val2 = f"'{val2}'"
    #                 elif val2 is None:
    #                     val2 = "'none'"
    #                 f.write(f"    {key2} = {val2},\n")
    #             f.write("/\n")
    #         if atomic_species is not None:
    #             f.write("\nATOMIC_SPECIES\n")
    #             for atom, info in atomic_species.items():
    #                 f.write(f"{atom} {info['mass']} {info['pseudo']}\n")
    #         if kpoints is not None:
    #             kpoint_type: str = kpoints["type"].strip()
    #             f.write(f"\nK_POINTS { kpoint_type }\n")
    #             if kpoint_type == "crystal":
    #                 k_points = kpoints["k_points"]
    #                 f.write(f"{k_points.shape[0]}\n")
    #                 np.savetxt(f, k_points, fmt = "%15.8f")
    #             elif kpoint_type == "automatic":
    #                 nk1, nk2, nk3 = kpoints["k_grid"]
    #                 sk1, sk2, sk3 = kpoints.get("offset", (0, 0, 0))
    #                 f.write(f"{nk1} {nk2} {nk3} {sk1} {sk2} {sk3}\n")
    #             else:
    #                 raise NotImplementedError("Unsupported kpoint type!")
    #         if optional_input is not None:
    #             f.write(optional_input)
    #             f.write("\n")
    #         f.seek(0)
    #         params_str = f.read()
    #     return params_str

class QePwInputs(QeInputs):
    names_order = ["control", "system", "electrons", "ions", "cell", "fcp", "rism"]
    def __init__(self, input_params: Dict[str, Optional[dict]], kpoints: Optional[dict], 
                 atomic_species: dict, confs: dpdata.System, optional_input: Optional[str] = None) -> None:
        super().__init__()
        self.namelists_str = self.write_namelists(input_params)
        self.species_str = self.write_species(atomic_species)
        self.kpoints_str = self.write_kpoints(kpoints)
        self.optional_input = optional_input if optional_input is not None else ""
        self.atoms = np.array(confs["atom_names"]).reshape(-1, 1)[confs["atom_types"]] # type: ignore
        self.confs = confs

    def write(self, frame: int):
        return "\n".join([
            self.namelists_str, 
            self.species_str, 
            self.write_conf(self.confs[frame], self.atoms),
            self.kpoints_str, 
            self.optional_input
        ])

class QeCPInputs(QeInputs):
    names_order = ["control", "system", "electrons", "ions", "cell", "wannier"]
    def __init__(self, input_params: Dict[str, Optional[dict]], kpoints: Optional[dict], 
                 atomic_species: dict, confs: dpdata.System, optional_input: Optional[str] = None) -> None:
        super().__init__()
        self.namelists_str = self.write_namelists(input_params)
        self.species_str = self.write_species(atomic_species)
        self.kpoints_str = self.write_kpoints(kpoints)
        self.optional_input = optional_input if optional_input is not None else ""
        self.atoms = np.array(confs["atom_names"]).reshape(-1, 1)[confs["atom_types"]] # type: ignore
        self.confs = confs

    def write(self, frame: int):
        return "\n".join([
            self.namelists_str, 
            self.species_str, 
            self.write_conf(self.confs[frame], self.atoms),
            self.kpoints_str, 
            self.optional_input
        ])

class QePw2wanInputs(QeInputs):
    names_order = ["inputpp"]
    def __init__(self, input_params: Dict[str, dict]) -> None:
        super().__init__()
        self.namelists_str = self.write_namelists(input_params)
    
    def write(self, frame: int):
        return self.namelists_str

class Wannier90Inputs:
    def __init__(
            self, 
            wan_params: dict, 
            proj: Optional[Union[Dict[str, str], List[str]]], 
            kpoints: np.ndarray, 
            confs: dpdata.System, 
            rewrite_atoms: Optional[Callable[[dpdata.System], np.ndarray]] = None,
            rewrite_proj: Optional[Callable[[dpdata.System], Dict[str, str]]] = None
        ) -> None:
        self.params_str = self.write_parameters(wan_params)
        self.kpoints_str = self.write_kpoints(kpoints)
        if rewrite_proj is not None:
            self.rewrite_proj = rewrite_proj
        elif proj:
            self.proj_str = self.write_proj(proj)
        else:
            self.proj_str = ""
        if rewrite_atoms is None:
            self.atoms = np.array(confs["atom_names"]).reshape(-1, 1)[confs["atom_types"]] # type: ignore
        else:
            self.rewrite_atoms = rewrite_atoms
        self.confs = confs

    def write(self, frame: int):
        conf = self.confs[frame]
        if hasattr(self, "atoms"):
            atoms = self.atoms
        else:
            atoms = self.rewrite_atoms(conf).reshape(-1, 1)
        if hasattr(self, "proj_str"):
            proj_str = self.proj_str
        else:
            proj_str = self.write_proj(self.rewrite_proj(conf))
        return "\n".join([self.params_str, proj_str, self.kpoints_str, self.write_conf(conf, atoms)])

    @classmethod
    def write_conf(cls, conf: dpdata.System, atoms: Optional[np.ndarray] = None):
        if atoms is None:
            atoms = np.array(conf["atom_names"]).reshape(-1, 1)[conf["atom_types"]] # type: ignore
        cells = np.reshape(conf["cells"], (3, 3)) # type: ignore
        coords = np.reshape(conf["coords"], (-1, 3)) # type: ignore
        with TemporaryFile("w+") as f:
            f.write("\nbegin unit_cell_cart\n")
            f.write("Ang\n")
            np.savetxt(f, cells, fmt = "%15.8f")
            f.write("end unit_cell_cart\n")
            f.write("\nbegin atoms_cart\n")
            atomic_positions = np.concatenate([atoms, np.char.mod("%15.8f", coords)], axis = 1) # type: ignore
            np.savetxt(f, atomic_positions, fmt = "%s")
            f.write("end atoms_cart\n")
            f.seek(0)
            conf_str = f.read()
        return conf_str

    @classmethod
    def write_parameters(cls, wan_params: Dict[str, object]):
        with TemporaryFile("w+") as f:
            for key, val in wan_params.items():
                f.write(f"{key} = {val}\n")
            f.seek(0)
            params_str = f.read()
        return params_str
    
    @classmethod
    def write_kpoints(cls, kpoints: np.ndarray):
        with TemporaryFile("w+") as f:
            f.write("\nbegin kpoints\n")
            np.savetxt(f, kpoints, fmt = "%15.8f")
            f.write("end kpoints\n")
            f.seek(0)
            kpoints_str = f.read()
        return kpoints_str

    @classmethod
    def write_proj(cls, proj: Union[Dict[str, str], List[str]]):
        with TemporaryFile("w+") as f:
            f.write("\nbegin projections\n")
            if isinstance(proj, dict):
                if "units" in proj:
                    f.write(proj["units"] + "\n")
                    del proj["units"]
                for site, option in proj.items():
                    f.write(f"    {site}: {option}\n")
            elif isinstance(proj, list):
                f.write("\n".join(proj))
                f.write("\n")
            else:
                raise TypeError(f"Invalid type {type(proj)} for 'proj'!")
            f.write("end projections\n")
            f.seek(0)
            proj_str = f.read()
        return proj_str




def complete_qe(input_params: Dict[str, Optional[dict]], calculation: Optional[str] = None, 
                k_grid: Optional[Tuple[int, int, int]] = None, 
                confs: Optional[dpdata.System] = None):
    """
    Copy and complete the system info ("ntyp" and "nat") to input parmeters, and generate the kpoints by calculation type. 
    If `calculation` = "scf", then kpoints will be "automatic". Otherwise kpoints will be "crystal" type. 
    """
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
        calculation = input_params["control"]["calculation"] # type: ignore
        if calculation == "scf":
            kpoints = {
                "type": "automatic",
                "k_grid": k_grid
            }
        else:
            kpoints = {
                "type": "crystal",
                "k_points": kmesh(*k_grid)
            }
    return input_params, kpoints

def complete_pw2wan(input_params: Dict[str, dict], name: str, prefix: str = "mlwf", outdir: str = "out"):
    """
    Copy the params and complete `outdir`, `prefix` and `seedname`.
    """
    input_params = deepcopy(input_params)
    input_params["inputpp"].update({
        "outdir" : outdir,
        "prefix" : prefix,
        "seedname" : name,
    })
    return input_params

def complete_wannier90(wan_params: dict, proj: Optional[dict], k_grid: Tuple[int, int, int]):
    """
    Copy the `wan_params` and complete the `mp_grid`, generate the projection and kpoints.
    Check whether `num_wann <= num_bands`, if they are provided.
    """
    wan_params = deepcopy(wan_params)
    wan_params["mp_grid"] = "{}, {}, {}".format(*k_grid)
    if "num_bands" in wan_params and "num_wann" in wan_params:
        if wan_params["num_wann"] > wan_params["num_bands"]:
            raise AssertionError(
                f"num_wann = {wan_params['num_wann']}, which is greater than num_bands = {wan_params['num_bands']}"
            )
    if proj is not None and len(proj) == 0:
        proj = None
    kpoints = kmesh(*k_grid)[:, :3]
    return wan_params, proj, kpoints




def get_qe_writers(
        confs: dpdata.System,
        scf_grid: Tuple[int, int, int],
        nscf_grid: Tuple[int, int, int],
        scf_params: Dict[str, Optional[dict]],
        nscf_params: Optional[Dict[str, dict]],
        atomic_species: Dict[str, Dict[str, Union[str, float]]],
        run_nscf: bool = True
    ):
    input_scf, kpoints_scf = complete_qe(scf_params, "scf", scf_grid, confs)
    scf_writer = QePwInputs(input_scf, kpoints_scf, atomic_species, confs)
    nscf_writer = None; input_nscf = None
    if run_nscf:
        _nscf_params = deepcopy(scf_params)
        if nscf_params is not None:
            recurcive_update(_nscf_params, nscf_params)
        _nscf_params["control"]["restart_mode"] = "from_scratch"  # type: ignore
        input_nscf, kpoints_nscf = complete_qe(_nscf_params, "nscf", nscf_grid, confs)
        nscf_writer = QePwInputs(input_nscf, kpoints_nscf, atomic_species, confs)
    return scf_writer, nscf_writer, input_scf, input_nscf

def get_pw_w90_writers(
        seed_name: str,
        confs: dpdata.System,
        pw2wan_params: Dict[str, dict], 
        w90_params: Dict[str, dict],
        kgrid: Tuple[int, int, int],
        input_scf: Dict[str, Optional[dict]], 
        input_nscf: Optional[Dict[str, Optional[dict]]],
        rewrite_atoms: Callable[[dpdata.System], np.ndarray],
        rewrite_proj: Callable[[dpdata.System], Dict[str, str]]
    ):
    pw2wan_writer = get_pw2wan_writer(seed_name, pw2wan_params, input_scf)
    w90_writer = get_w90_writer(confs, w90_params, input_scf, input_nscf, kgrid, rewrite_atoms, rewrite_proj)
    return pw2wan_writer, w90_writer

def get_pw2wan_writer(
        seed_name: str,
        pw2wan_params: Dict[str, dict], 
        input_scf: Dict[str, Optional[dict]], 
    ):
    input_pw2wan = complete_pw2wan(
        pw2wan_params, 
        seed_name, 
        input_scf["control"]["prefix"], # type: ignore
        input_scf["control"]["outdir"]  # type: ignore
    )
    return QePw2wanInputs(input_pw2wan)

def get_w90_writer(
        confs: dpdata.System,
        w90_params: Dict[str, dict],
        scf_params: Dict[str, Optional[dict]],
        nscf_params: Optional[Dict[str, Optional[dict]]],
        kgrid: Tuple[int, int, int],
        rewrite_atoms: Optional[Callable[[dpdata.System], np.ndarray]],
        rewrite_proj: Optional[Callable[[dpdata.System], Dict[str, str]]]
    ):
    wan_params = w90_params["wan_params"]
    if nscf_params is not None:
        if "system" in nscf_params and "nbnd" in nscf_params["system"]: # type: ignore
            wan_params["num_bands"] = nscf_params["system"]["nbnd"] # type: ignore
    else:
        if "system" in scf_params and "nbnd" in scf_params["system"]: # type: ignore
            wan_params["num_bands"] = scf_params["system"]["nbnd"] # type: ignore
    wan_params, proj, kpoints = complete_wannier90(
        wan_params, 
        w90_params.get("projections", {}),
        kgrid
    )
    if w90_params.get("rewrite_atoms", False):
        if rewrite_atoms is None:
            print(f"[WARNING] 'rewrite_atoms' is True but cannot import the method!")
            print("Use default atom names.")
    else:
        rewrite_atoms = None
    if w90_params.get("rewrite_proj", False):
        if rewrite_proj is None:
            print(f"[WARNING] 'rewrite_proj' is True but cannot import the method!")
            print("Use projections defined in wannier90_params.")
    else:
        rewrite_proj = None
    return Wannier90Inputs(
        wan_params, proj, kpoints, 
        confs, rewrite_atoms, rewrite_proj
    )