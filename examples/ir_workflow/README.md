# Structure of file `global.json`

`global.json`:
```json
{
    "config": {
        "global": {},
        "dipole": {
            "dft_type": "'qe' or 'qe_cp'",
            "mlwf_setting": {},
            "task_setting": {}
        },
        "deep_model": {         # The parameters of deep wannier model.
            "train_inputs": {},
            "amplif": 1.0
        }
    },
    "uploads": {
        "frozen_model": {
            "deep_potential": "path to dp model",
            "deep_wannier": "path to dw model"
        },
        "system": {
            "system_name": {    # train_confs, init_conf, sampled_system
                "path": "path to system's file",
                "fmt": "format of the system. All dpdata-supported formats and additionally includes 'numpy/npz'"
            },
        },
        "other": {
            "name": "path. pseudo, train_label and total_dipole."
        }
    }
}
```

For `global["config"]["global"]`, the default parameters is:
```json
"global": {
    "name": "system",
    "calculation": "ir",
    "type_map": [],         # Reauired!
    "mass_map": [],         # Required! The mass of each atom. 
    "read_buffer": 50000,   # Optional. The buffer size while reading the lammps dump file.
    "dt": 0.0003,
    "nstep": 10000,         # The number of steps of sampleing
    "window": 1000,         # The number of 't' for correlation function.
    "temperature": 300,
    "width": 240            # The width of Gaussian filter. Smooth factor.
}
```