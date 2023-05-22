from typing import Dict, List, Optional, Tuple, Union
from dflow import (
    Step,
    Steps,
    Executor,
    Inputs,
    Outputs,
    InputParameter,
    InputArtifact,
    OutputParameter,
    OutputArtifact
)
from dflow.io import Inputs, Outputs
from dflow.python import (
    PythonOPTemplate,
    OP
)
from dflow.step import Step
import abc, numpy as np

class BasicSteps(Steps, abc.ABC):
    def __init__(self,
            name: Optional[str] = None,
            steps: List[Union[Step, List[Step]]] = None,
            memoize_key: Optional[str] = None,
            annotations: Dict[str, str] = None,
            parallelism: Optional[int] = None,
            ) -> None:
        self._input_parameters, self._input_artifacts = self.get_inputs()
        self._output_parameters, self._output_artifacts = self.get_outputs()
        super().__init__(
            name,
            inputs = Inputs(
                parameters = self._input_parameters,
                artifacts = self._input_artifacts
            ),
            outputs = Outputs(
                parameters= self._output_parameters,
                artifacts = self._output_artifacts
            ),
            steps = steps,
            memoize_key = memoize_key,
            annotations = annotations,
            parallelism = parallelism
        )
    
    @classmethod
    @abc.abstractmethod
    def get_inputs(cls) -> Tuple[Dict[str, InputParameter], Dict[str, InputArtifact]]:
        pass
    
    @classmethod
    @abc.abstractmethod
    def get_outputs(cls) -> Tuple[Dict[str, OutputParameter], Dict[str, OutputArtifact]]:
        pass


class AdaptFlow(Steps, abc.ABC):
    steps_list: List[BasicSteps] = []
    def __init__(self, name: str, run_list: List[BasicSteps], debug: bool = False) -> None:
        print("\n------------- Init AdaptFlow --------------\n")
        print("All steps:", [s.__name__ for s in run_list])
        self.debug = debug
        total_io = self.get_total_io(run_list)
        steps_inputs_parameters, steps_inputs_artifacts, steps_l = self.prebuild(total_io)
        self.all_step: Dict[BasicSteps, Step] = {}
        print("\n----------- Init Inputs/Outputs -----------\n")
        print("Input parameters:", list(self._input_parameters.keys()))
        print("Input artifacts:", list(self._input_artifacts.keys()))
        print("Output parameters:", list(self._output_parameters.keys()))
        print("Output artifacts:", list(self._output_artifacts.keys()))
        super().__init__(
            name = name,
            inputs = Inputs(
                parameters = self._input_parameters,
                artifacts = self._input_artifacts
            ),
            outputs = Outputs(
                parameters= self._output_parameters,
                artifacts = self._output_artifacts
            )
        )
        print("\n--------------- Build Steps ---------------\n")
        self.templates = self.build_templates(run_list)
        for step_l in steps_l:
            print(f"build {[s.__name__ for s in step_l]}")
        self.build(steps_inputs_parameters, steps_inputs_artifacts, steps_l)
        self.set_outputs(steps_l)
    
    def get_total_io(self, run_list: List[BasicSteps] = steps_list) -> Dict[BasicSteps, Dict[str, Tuple[BasicSteps, str]]]:
        io_dict = self.get_io_dict()
        total_io = {steps: {} for steps in run_list}
        pre_set = set(run_list + [None])
        for steps in run_list:
            for in_key in self.inputs_dict[steps]:
                for pre_steps, out_key in io_dict[steps][in_key]:
                    if pre_steps in pre_set:
                        total_io[steps][in_key] = (pre_steps, out_key)
                        break
                else:
                    raise RuntimeError(f"Cannot build Super OP {self}! Steps {steps} miss one input '{in_key}'!")
        return total_io
    
    def prebuild(self, total_io: Dict[BasicSteps, Dict[str, Tuple[BasicSteps, str]]]):
        run_list = list(total_io.keys())
        num_steps = len(run_list)
        run_dict = {run_list[i]: i for i in range(num_steps)}
        inputs_type = self.inputs_type
        inputs_dict = self.inputs_dict

        inc_mat = np.zeros((num_steps, num_steps), dtype = bool)
        steps_inputs_parameters = {}
        steps_inputs_artifacts = {}
        input_parameters = {}
        input_artifacts = {}

        for steps in run_list:
            steps_inputs_parameters[steps] = {}
            steps_inputs_artifacts[steps] = {}
            for in_key, (pre_steps, out_key) in total_io[steps].items():
                if in_key in inputs_type[steps]["parameters"]:
                    steps_inputs_parameters[steps][in_key] = (pre_steps, out_key)
                    is_para = True
                elif in_key in inputs_type[steps]["artifacts"]:
                    steps_inputs_artifacts[steps][in_key] = (pre_steps, out_key)
                    is_para = False
                else:
                    raise RuntimeError(f"Steps {steps} has no inputs named '{in_key}'!")
                if pre_steps is None:
                    if is_para:
                        input_parameters[out_key] = inputs_dict[steps][in_key]
                    else:
                        input_artifacts[out_key] = inputs_dict[steps][in_key]
                else:
                    inc_mat[run_dict[steps], run_dict[pre_steps]] = True

        self._input_parameters = input_parameters
        self._input_artifacts = input_artifacts

        steps_l: List[List[BasicSteps]] = []
        mask = np.ones((num_steps, ), dtype = bool)
        while mask.any():
            new_mask = np.sum(inc_mat[:, mask], axis = -1) > 0
            if (new_mask == mask).all():
                raise RuntimeError("There are loops in io map!!!")
            steps_l.append([run_list[i] for i in np.nonzero(mask & (~new_mask))[0]])
            mask = new_mask
        
        self._output_parameters, self._output_artifacts = self.get_outputs(steps_l)
        
        return steps_inputs_parameters, steps_inputs_artifacts, steps_l
    
    def build(self, steps_inputs_parameters, steps_inputs_artifacts, steps_l: List[List[BasicSteps]]):
        for step_l in steps_l:
            builded_step: List[Step] = []
            for s in step_l:
                p = {}
                a = {}
                for in_key, (pre_steps, out_key) in steps_inputs_parameters[s].items():
                    if pre_steps is None:
                        p[in_key] = self.inputs.parameters[out_key]
                    else:
                        p[in_key] = self.all_step[pre_steps].outputs.parameters[out_key]
                for in_key, (pre_steps, out_key) in steps_inputs_artifacts[s].items():
                    if pre_steps is None:
                        a[in_key] = self.inputs.artifacts[out_key]
                    else:
                        a[in_key] = self.all_step[pre_steps].outputs.artifacts[out_key]
                builded_step.append(self.build_steps(s, p, a))
                if self.debug:
                    self.show_step(s, p, a)
                self.all_step[s] = builded_step[-1]
            self.add(builded_step)

    def get_outputs(self, steps_l: List[List[BasicSteps]]):
        output_parameters: Dict[str, OutputParameter] = {}
        output_artifacts: Dict[str, OutputArtifact] = {}
        for steps in steps_l[-1]:
            p, a = steps.get_outputs()
            output_parameters.update(p)
            output_artifacts.update(a)
        return output_parameters, output_artifacts

    def set_outputs(self, steps_l: List[List[BasicSteps]]):
        for steps in steps_l[-1]:
            p, a = steps.get_outputs()
            s = self.all_step[steps]
            for key in p:
                self.outputs.parameters[key] = s.outputs.parameters[key]
            for key in a:
                self.outputs.artifacts[key]._from = s.outputs.artifacts[key]

    @classmethod
    def get_substeps_inputs(cls):
        inputs_dict: Dict[BasicSteps, Dict[str, Union[InputParameter, InputArtifact]]] = {}
        for steps in cls.steps_list:
            p, a = steps.get_inputs()
            p.update(a)
            inputs_dict[steps] = p
        return inputs_dict

    @property
    def inputs_dict(self):
        if not hasattr(self, "_inputs_dict"):
            self._inputs_dict = self.get_substeps_inputs()
        return self._inputs_dict

    @classmethod
    def get_substeps_outputs(cls):
        outputs_dict: Dict[BasicSteps, Dict[str, Union[OutputParameter, OutputArtifact]]] = {}
        for steps in cls.steps_list:
            p, a = steps.get_outputs()
            p.update(a)
            outputs_dict[steps] = p
        return outputs_dict

    @property
    def outputs_dict(self):
        if not hasattr(self, "_outputs_dict"):
            self._outputs_dict = self.get_substeps_outputs()
        return self._outputs_dict
    
    @classmethod
    def get_inputs_type(cls) -> Dict[BasicSteps, Dict[str, List[str]]]:
        inputs_type = {}
        for steps in cls.steps_list:
            inputs_type[steps] = {}
            p, a = steps.get_inputs()
            inputs_type[steps]["parameters"] = list(p.keys())
            inputs_type[steps]["artifacts"] = list(a.keys())
        return inputs_type

    @property
    def inputs_type(self):
        if not hasattr(self, "_inputs_type"):
            self._inputs_type = self.get_inputs_type()
        return self._inputs_type
    
    @classmethod
    @abc.abstractmethod
    def get_io_dict(cls) -> Dict[BasicSteps, Dict[str, List[Tuple[BasicSteps, str]]]]:
        """
            The default implementation is trivial.
        """
        io_dict = {}
        inputs_dict = cls.get_substeps_inputs()
        for steps, inputs in inputs_dict.items():
            io_dict[steps] = {key: [(None, key)] for key in inputs}
        return io_dict

    def build_steps(self, steps: BasicSteps, parameters: dict, artifacts: dict) -> Step:
        return Step(
            steps.__name__,
            self.templates[steps],
            parameters = parameters,
            artifacts = artifacts
        )
    
    @abc.abstractmethod
    def build_templates(self, run_list: List[BasicSteps]) -> Dict[BasicSteps, Steps]:
        return {steps: steps() for steps in run_list}
    
    def show_step(self, steps: BasicSteps, parameters: dict, artifacts: dict):
        print(steps.__name__)
        print(parameters)
        print(artifacts)



if __name__ == "__main__":
    from spectra_flow.utils import bohrium_login, load_json
    bohrium_login(load_json("../examples/account.json"))
    class A(BasicSteps):

        @classmethod
        def get_inputs(cls):
            return {
                "A.1": InputParameter(type = dict, value = {}),
                "A.2": InputParameter(type = dict, value = {})
            }, {
                "A.3": InputArtifact(),
                "A.4": InputArtifact()
            }
        
        @classmethod
        def get_outputs(cls):
            return {
                "A.5": OutputParameter(),
                "A.6": OutputParameter()
            }, {
                "A.7": OutputArtifact(),
                "A.8": OutputArtifact()
            }
    
    class B(BasicSteps):

        @classmethod
        def get_inputs(cls):
            return {
                "B.1": InputParameter(),
                "B.2": InputParameter()
            }, {
                "B.3": InputArtifact(),
                "B.4": InputArtifact()
            }
        
        @classmethod
        def get_outputs(cls):
            return {
                "B.5": OutputParameter(),
                "B.6": OutputParameter()
            }, {
                "B.7": OutputArtifact(),
                "B.8": OutputArtifact()
            }
    
    class C(BasicSteps):

        @classmethod
        def get_inputs(cls):
            return {
                "C.1": InputParameter(),
                "C.2": InputParameter()
            }, {
                "C.3": InputArtifact(),
                "C.4": InputArtifact()
            }
        
        @classmethod
        def get_outputs(cls):
            return {
                "C.5": OutputParameter(),
                "C.6": OutputParameter()
            }, {
                "C.7": OutputArtifact(),
                "C.8": OutputArtifact()
            }

    class ABflow(AdaptFlow):
        steps_list = [A, B, C]
        def __init__(self, name: str, run_list: List[BasicSteps], debug: bool = False) -> None:
            super().__init__(name, run_list, debug = debug)
        
        @classmethod
        def get_io_dict(cls) -> Dict[BasicSteps, Dict[str, List[Tuple[BasicSteps, str]]]]:
            io_dict = super().get_io_dict()
            io_dict[B]["B.1"].insert(1, (A, "A.5"))
            io_dict[B]["B.2"].insert(1, (A, "A.6"))
            io_dict[C]["C.1"].insert(0, (A, "A.6"))
            io_dict[C]["C.3"].insert(0, (A, "A.7"))
            io_dict[C]["C.1"].insert(0, (B, "B.5"))
            io_dict[C]["C.3"].insert(0, (B, "B.8"))
            return io_dict
        
        def build_templates(self, run_list: List[BasicSteps]) -> Dict[BasicSteps, Steps]:
            return super().build_templates(run_list)
    
    flow = ABflow("ab", [A, B, C], debug = True)