from math import ceil
from typing import Dict, List, Tuple, Union
from pathlib import Path
import abc
from dflow.python import (
    OP, 
    OPIO, 
    Artifact, 
    OPIOSign, 
    BigParameter
)
from dflow.utils import (
    set_directory
)


class PreparePolar(OP, abc.ABC):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "input_setting": BigParameter(dict)
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "input_setting_list": BigParameter(List[dict])
        })

    @OP.exec_sign_check
    def execute(
            self,
            op_in: OPIO,
    ) -> OPIO:
        input_setting: Dict[str, Union[str, dict]] = op_in["input_setting"]

        input_setting_list = self.generate_inputs(input_setting)
        return OPIO({
            "input_setting_list": input_setting_list
        })

    @abc.abstractmethod
    def generate_inputs(self, input_setting: Dict[str, Union[str, dict]]) -> List[Dict[str, Union[str, dict]]]:
        pass
