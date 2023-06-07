from copy import copy
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union, Type
from dflow import (
    Step,
    Steps,
    Inputs, 
    Outputs,
    Executor,
    OPTemplate,
    InputParameter,
    InputArtifact,
    OutputParameter,
    OutputArtifact,
)
from dflow.python import (
    OP,
    Artifact, 
    Parameter,
    BigParameter,
    OPIOSign, 
    PythonOPTemplate,
)
from dflow.step import Step
import abc, numpy as np


class SuperOP(Steps, abc.ABC):
    def __init__(self,
            name: Optional[str] = None,
            steps: Optional[List[Union[Step, List[Step]]]] = None,
            memoize_key: Optional[str] = None,
            annotations: Optional[Dict[str, str]] = None,
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
            steps = steps, # type: ignore
            memoize_key = memoize_key,
            annotations = annotations, # type: ignore
            parallelism = parallelism
        )
    
    @property
    def input_parameters(self):
        return self._input_parameters

    @property
    def input_artifacts(self):
        return self._input_artifacts
    
    @property
    def output_parameters(self):
        return self._output_parameters
    
    @property
    def output_artifacts(self):
        return self._output_artifacts
    
    @classmethod
    @abc.abstractmethod
    def get_inputs(cls) -> Tuple[Dict[str, InputParameter], Dict[str, InputArtifact]]:
        pass
    
    @classmethod
    @abc.abstractmethod
    def get_outputs(cls) -> Tuple[Dict[str, OutputParameter], Dict[str, OutputArtifact]]:
        pass

StepKeyPair = Tuple[Union[str, None], str]
StepType = Union[Type[SuperOP], Type[OP]]

class StepKey:
    def __init__(self, step_name: Union[str, None] = None) -> None:
        self.step = step_name

    def __getattribute__(self, __name: str) -> StepKeyPair:
        return (super().__getattribute__("step"), __name)

def get_inputs(cls: StepType):
    if issubclass(cls, SuperOP):
        return cls.get_inputs()
    if issubclass(cls, OP):
        p: Dict[str, InputParameter] = {}
        a: Dict[str, InputArtifact] = {}
        for key, val in cls.get_input_sign().items():
            if isinstance(val, (Parameter, BigParameter)):
                p[key] = InputParameter()
            elif isinstance(val, Artifact):
                a[key] = InputArtifact(optional = val.optional)
            else:
                raise NotImplementedError(f"Unknown input type: {type(val)}")
        return p, a
    raise NotImplementedError(f"The type {type(cls)} is not a subclass of 'BasicSteps' or 'OP'!")

def get_outputs(cls: StepType):
    if issubclass(cls, SuperOP):
        return cls.get_outputs()
    if issubclass(cls, OP):
        p: Dict[str, OutputParameter] = {}
        a: Dict[str, OutputArtifact] = {}
        for key, val in cls.get_output_sign().items():
            if isinstance(val, (Parameter, BigParameter)):
                p[key] = OutputParameter()
            elif isinstance(val, Artifact):
                a[key] = OutputArtifact()
            else:
                raise NotImplementedError(f"Unknown output type: {type(val)}")
        return p, a
    raise NotImplementedError(f"The type {type(cls)} is not a subclass of 'BasicSteps' or 'OP'!")

def is_optional(_inputs):
    return isinstance(_inputs, InputArtifact) and _inputs.optional

class IODictHandler:
    def __init__(
            self, 
            io_dict: Dict[str, Dict[str, List[StepKeyPair]]],
            given_inputs: Optional[Iterable[str]] = None, 
            pri_source: Optional[str] = None, 
        ) -> None:
        if pri_source is None:
            if given_inputs is None:
                pri_source = "step"
            else:
                pri_source = "self"
        self.pri_source = pri_source
        self.io_from_self, self.io_from_step = self.divide_by_source(io_dict)
        self.given_inputs = given_inputs
    
    def get_source(self, step_name, in_key, _inputs, run_set) -> List[Union[StepKeyPair, None]]:
        if self.pri_source == "self":
            get_func = [self.from_self, self.from_step]
        else:
            get_func = [self.from_step, self.from_self]
        return [func(step_name, in_key, _inputs, run_set) for func in get_func]
    
    def from_self(self, step_name, in_key, _inputs, run_set):
        for self_in_key in self.io_from_self[step_name][in_key]:
            if is_optional(_inputs) or (self.given_inputs is None) or (self_in_key in self.given_inputs):
                return (None, self_in_key)
    
    def from_step(self, step_name, in_key, _inputs, run_set):
        for stepkey in self.io_from_step[step_name][in_key]:
            if stepkey[0] in run_set:
                return stepkey

    @classmethod
    def divide_by_source(cls, io_dict: Dict[str, Dict[str, List[StepKeyPair]]]):
        io_from_self: Dict[str, Dict[str, List[str]]] = {}
        io_from_step: Dict[str, Dict[str, List[StepKeyPair]]] = {}
        for step_name, io_map in io_dict.items():
            io_from_self[step_name] = {}
            io_from_step[step_name] = {}
            for in_key, stepkeys in io_map.items():
                io_from_self[step_name][in_key] = []
                io_from_step[step_name][in_key] = []
                for stepkey in stepkeys:
                    if stepkey[0] is None:
                        io_from_self[step_name][in_key].append(stepkey[1])
                    else:
                        io_from_step[step_name][in_key].append(stepkey)
        return io_from_self, io_from_step


class AdaptiveFlow(Steps, abc.ABC):
    all_steps: Dict[str, StepType] = {}
    steps_list: List[str] = []
    parallel_steps: List[List[str]] = []
    python_op_executor: Dict[str, Optional[Executor]] = {}
    def __init__(
            self, 
            name: str, 
            run_list: Iterable[str], 
            given_inputs: Optional[Iterable[str]] = None, 
            pri_source: Optional[str] = None, 
            with_parallel: bool = True, 
            debug: bool = False
        ) -> None:
        print(f"\n------------- Init {self.__class__.__name__} --------------\n")
        print("All running steps:", list(run_list))
        self.with_parallel = with_parallel
        self.debug = debug
        run_list = list(set(run_list))
        assert len(run_list) > 0, "Error: Empty run list!"
        total_io = self.cal_total_io(run_list, given_inputs, pri_source)
        steps_inputs_parameters, steps_inputs_artifacts, step_ll = self.prebuild(total_io = total_io)
        self.added_step: Dict[str, Step] = {}
        if debug:
            print("\n----------- Init Inputs/Outputs -----------\n")
            print("Input parameters:", list(self._input_parameters.keys()))
            print("Input artifacts:", list(self._input_artifacts.keys()))
            print("Optional Inputs:", [key for key, _inputs in self._input_artifacts.items() if _inputs.optional])
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
        if debug:
            self.check_templates()
        self.build(steps_inputs_parameters, steps_inputs_artifacts, step_ll)
        self.set_outputs(step_ll)
    
    @classmethod
    def run_from_inputs(cls, given_inputs: Iterable[str], pri_source: str = "self"):
        steps_list = cls.steps_list
        all_steps = cls.all_steps
        inputs_dict = cls.get_substeps_inputs()
        io_dict = cls.get_io_dict()
        given_inputs = set(given_inputs)
        num_steps = len(steps_list)
        steps_dict = {steps_list[i]: i for i in range(num_steps)}
        
        run_list: List[str] = []
        run_set: Set[str] = set()
        pre_mat = np.zeros((num_steps, num_steps), dtype = bool)
        io_handler = IODictHandler(io_dict, given_inputs, pri_source)

        for step_name in steps_list:
            for in_key, _inputs in inputs_dict[all_steps[step_name]].items():
                for stepkey in io_handler.get_source(step_name, in_key, _inputs, run_set):
                    if stepkey is not None:                             # if source is not None,
                        if stepkey[0] is not None:                      # check if it's from a pre_step
                            pre_mat[steps_dict[step_name], steps_dict[stepkey[0]]] = True
                        break                                           # break to check the next in_key
                else:                                                   # else (no source):
                    pre_mat[steps_dict[step_name]] = False
                    break                                               # this step cannot join to run_list
            else:                                                       # if all check pass
                run_list.append(step_name)                              # add this step to run_list
                run_set.add(step_name)

        # generate the run_step tree.
        mask = np.ones((num_steps, ), dtype = bool)
        parents: Dict[str, List[str]] = {}
        for end_step_name in run_list:
            mask.fill(False)
            mask[steps_dict[end_step_name]] = True
            new_mask = mask
            while new_mask.any():
                new_mask = (np.sum(pre_mat[new_mask, :], axis = 0) > 0) & (~mask)
                mask |= new_mask
            parents[end_step_name] = [steps_list[i] for i in np.nonzero(mask)[0]]
        return parents

    def cal_total_io(self, run_list: Iterable[str], given_inputs: Optional[Iterable[str]] = None, pri_source: Optional[str] = None):
        io_dict = self.get_io_dict()

        total_io: Dict[str, Dict[str, StepKeyPair]] = {steps: {} for steps in run_list}
        run_set = set(run_list)
        io_handler = IODictHandler(io_dict, given_inputs, pri_source)

        for step_name in run_list:
            steps = self.all_steps[step_name]
            for in_key, _inputs in self.inputs_dict[steps].items():
                for stepkey in io_handler.get_source(step_name, in_key, _inputs, run_set):
                    if stepkey is not None:
                        total_io[step_name][in_key] = stepkey
                        break
                else:
                    raise AssertionError(
                        f"Cannot build workflow {self.__class__.__name__}!\n" + 
                        f"Step {step_name}({steps.__name__}) miss one input '{in_key}'!"
                    )
        return total_io
    
    def prebuild(self, run_list: List[str] = steps_list, total_io: Optional[Dict[str, Dict[str, StepKeyPair]]] = None):
        if total_io is not None:
            run_list = list(total_io.keys())
        else:
            total_io = self.cal_total_io(run_list)
        all_steps = self.all_steps
        inputs_dict = self.inputs_dict

        steps_inputs_parameters: Dict[str, Dict[str, StepKeyPair]] = {step_name: {} for step_name in run_list}
        steps_inputs_artifacts: Dict[str, Dict[str, StepKeyPair]] = {step_name: {} for step_name in run_list}
        input_parameters: Dict[str, InputParameter] = {}
        input_artifacts: Dict[str, InputArtifact] = {}

        for step_name in run_list:
            steps = all_steps[step_name]
            for in_key, (pre_steps_name, out_key) in total_io[step_name].items():
                this_inputs = inputs_dict[steps][in_key]
                if isinstance(this_inputs, InputParameter):
                    steps_inputs_parameters[step_name][in_key] = (pre_steps_name, out_key)
                    if pre_steps_name is None:
                        input_parameters[out_key] = this_inputs
                elif isinstance(this_inputs, InputArtifact):
                    steps_inputs_artifacts[step_name][in_key] = (pre_steps_name, out_key)
                    if pre_steps_name is None:
                        input_artifacts[out_key] = this_inputs
                else:
                    raise AssertionError(f"Step {step_name}({steps.__name__}) has no inputs named '{in_key}'!")

        self._input_parameters = input_parameters
        self._input_artifacts = input_artifacts
        
        if self.debug:
            # Check components
            components = self.cal_components(total_io, run_list)
            if len(components) > 1:
                print("*" * 60)
                print(f"[WARNING]: run list can be separated into {len(components)} part: \n{components}")
                print("*" * 60)
        if len(self.parallel_steps) > 0:
            step_ll = self.level_paralell(run_list)
        else:
            step_ll = self.default_parallel(total_io, run_list)
        if not self.with_parallel:
            step_ll = [[s] for s in sum(step_ll, [])]
        self._output_parameters, self._output_artifacts = self.get_outputs(step_ll)
        
        return steps_inputs_parameters, steps_inputs_artifacts, step_ll
    
    def build(self, 
              steps_inputs_parameters: Dict[str, Dict[str, StepKeyPair]], 
              steps_inputs_artifacts: Dict[str, Dict[str, StepKeyPair]], 
              step_ll: List[List[str]]
        ):
        print_head = "\n" if self.debug else ""
        for step_l in step_ll:
            builded_step_l: List[Step] = []
            print(print_head + f"build {step_l}")
            for s in step_l:
                p = {}
                a = {}
                for in_key, (pre_steps_name, out_key) in steps_inputs_parameters[s].items():
                    if pre_steps_name is None:
                        p[in_key] = self.inputs.parameters[out_key]
                    else:
                        p[in_key] = self.added_step[pre_steps_name].outputs.parameters[out_key]
                for in_key, (pre_steps_name, out_key) in steps_inputs_artifacts[s].items():
                    if pre_steps_name is None:
                        a[in_key] = self.inputs.artifacts[out_key]
                    else:
                        a[in_key] = self.added_step[pre_steps_name].outputs.artifacts[out_key]
                step = self.build_step(s, p, a)
                builded_step_l.append(step)
                if self.debug:
                    self.show_step(s, p, a)
                self.added_step[s] = step
            if len(builded_step_l) == 1:
                self.add(builded_step_l[0])
            else:
                self.add(builded_step_l)

    def build_step(self, step_name: str, parameters: dict, artifacts: dict) -> Step:
        return Step(
            step_name.upper().replace("_", "-").replace(".", "-"),
            self.templates[step_name],
            parameters = parameters,
            artifacts = artifacts,
            executor = self.python_op_executor.get(step_name, None)
        )
    
    @abc.abstractmethod
    def build_templates(self, run_list: List[str]) -> Dict[str, OPTemplate]:
        templates = {}
        for step_name in run_list:
            step_type = self.all_steps[step_name]
            if issubclass(step_type, OP):
                templates[step_name] = PythonOPTemplate(step_type) # type: ignore
            else:
                templates[step_name] = step_type()
        return templates
    
    def level_paralell(self, run_list: List[str]):
        run_set = set(run_list)
        step_ll = [
            [step_name for step_name in step_l if step_name in run_set] for step_l in self.parallel_steps
        ]
        i = 0
        while i < len(step_ll):
            if len(step_ll[i]) == 0:
                del step_ll[i]
            else:
                i += 1
        return step_ll

    def default_parallel(self, total_io: Dict[str, Dict[str, StepKeyPair]], run_list: Optional[List[str]] = None):
        if run_list is None:
            run_list = list(total_io.keys())
        num_steps = len(run_list)
        inc_mat = self.cal_inc_mat(total_io, run_list)

        step_ll: List[List[str]] = []
        mask = np.ones((num_steps, ), dtype = bool)
        while mask.any():
            new_mask = np.sum(inc_mat[:, mask], axis = -1) > 0
            if (new_mask == mask).all():
                raise RuntimeError("There are loops in io map!!!")
            step_ll.append([run_list[i] for i in np.nonzero(mask & (~new_mask))[0]])
            mask = new_mask
        return step_ll

    def show_step(self, step_name: str, parameters: dict, artifacts: dict):
        print(f"\n{step_name} ({self.all_steps[step_name].__name__})")
        print(parameters)
        print(artifacts)
    
    def get_outputs(self, step_l: List[List[str]]):
        output_parameters: Dict[str, OutputParameter] = {}
        output_artifacts: Dict[str, OutputArtifact] = {}
        if len(step_l) > 0:
            for step_name in step_l[-1]:
                p, a = get_outputs(self.all_steps[step_name])
                output_parameters.update(p)
                output_artifacts.update(a)
        return output_parameters, output_artifacts

    def set_outputs(self, step_l: List[List[str]]):
        if len(step_l) == 0:
            return
        for step_name in step_l[-1]:
            p, a = get_outputs(self.all_steps[step_name])
            s = self.added_step[step_name]
            for key in p:
                self.outputs.parameters[key].value_from_parameter = s.outputs.parameters[key]
            for key in a:
                self.outputs.artifacts[key]._from = s.outputs.artifacts[key]

    @property
    def input_parameters(self):
        return self._input_parameters

    @property
    def input_artifacts(self):
        return self._input_artifacts
    
    @property
    def output_parameters(self):
        return self._output_parameters
    
    @property
    def output_artifacts(self):
        return self._output_artifacts

    @classmethod
    def get_substeps_inputs(cls):
        inputs_dict: Dict[StepType, Dict[str, Union[InputParameter, InputArtifact]]] = {}
        for steps in set(cls.all_steps.values()):
            p, a = get_inputs(steps)
            inputs_dict[steps] = {}
            inputs_dict[steps].update(p)
            inputs_dict[steps].update(a)
        return inputs_dict

    @property
    def inputs_dict(self):
        if not hasattr(self, "_inputs_dict"):
            self._inputs_dict = self.get_substeps_inputs()
        return self._inputs_dict

    @classmethod
    def get_substeps_outputs(cls):
        outputs_dict: Dict[StepType, Dict[str, Union[OutputParameter, OutputArtifact]]] = {}
        for steps in set(cls.all_steps.values()):
            p, a = get_outputs(steps)
            outputs_dict[steps] = {}
            outputs_dict[steps].update(p)
            outputs_dict[steps].update(a)
        return outputs_dict

    @property
    def outputs_dict(self):
        if not hasattr(self, "_outputs_dict"):
            self._outputs_dict = self.get_substeps_outputs()
        return self._outputs_dict
    
    @classmethod
    def get_inputs_type(cls) -> Dict[StepType, Dict[str, List[str]]]:
        inputs_type = {}
        for steps in set(cls.all_steps.values()):
            inputs_type[steps] = {}
            p, a = get_inputs(steps)
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
    def get_io_dict(cls) -> Dict[str, Dict[str, List[StepKeyPair]]]:
        """
            The default implementation is trivial.
            return: Dict[steps_name, Dict[in_key, List[StepKeyPair]]], where StepKeyPair = [pre_steps_name, out_key]
        """
        io_dict = {}
        inputs_dict = cls.get_substeps_inputs()
        for step_name, steps in cls.all_steps.items():
            io_dict[step_name] = {key: [(None, key)] for key in inputs_dict[steps]}
        return io_dict

    @classmethod
    def check_steps_list(cls):
        pre_set: Set[Union[str, None]] = set([None])
        inputs_dict = cls.get_substeps_inputs()
        io_dict = cls.get_io_dict()
        for step_name in cls.steps_list:
            steps = cls.all_steps[step_name]
            for in_key in inputs_dict[steps]:
                for pre_steps, _ in io_dict[step_name][in_key]:
                    if pre_steps not in pre_set:
                        sorted_steps = cls.sort_steps_list()
                        raise AssertionError(
                            f"The order of steps_list is not consistent with io map! Maybe {sorted_steps} ?"
                        )
            pre_set.add(step_name)

    @classmethod
    def check_steps(cls):
        assert set(cls.steps_list) == set(cls.all_steps.keys())
        if len(cls.parallel_steps) > 0:
            sumof_parallel = sum(cls.parallel_steps, [])
            assert len(sumof_parallel) == len(cls.steps_list)
            assert set(sumof_parallel) == set(cls.steps_list)

    @classmethod
    def check_io_dict(cls):
        inputs_dict = cls.get_substeps_inputs()
        io_dict = cls.get_io_dict()
        assert set(io_dict.keys()) == set(cls.steps_list)
        for step_name in cls.steps_list:
            steps = cls.all_steps[step_name]
            for in_key in inputs_dict[steps]:
                assert in_key in io_dict[step_name]
                assert len(io_dict[step_name][in_key]) > 0

    def check_templates(self):
        import re
        for step_name, template in self.templates.items():
            assert isinstance(template, OPTemplate)
            assert re.match(r"^[a-zA-Z0-9\-]*$", template.name) is not None, \
                f"Template name '{template.name}' of step '{step_name}' is invalid!"

    @classmethod
    def sort_steps_list(cls):
        steps_list = cls.steps_list
        num_steps = len(steps_list)
        steps_dict = {steps_list[i]: i for i in range(num_steps)}
        inc_mat = np.zeros((num_steps, num_steps), dtype = bool)
        inputs_dict = cls.get_substeps_inputs()
        io_dict = cls.get_io_dict()
        for step_name in steps_list:
            for in_key in inputs_dict[cls.all_steps[step_name]]:
                for pre_steps, _ in io_dict[step_name][in_key]:
                    if pre_steps is not None:
                        inc_mat[steps_dict[step_name], steps_dict[pre_steps]] = True
        
        sorted_steps_ll: List[List[str]] = []
        mask = np.ones((num_steps, ), dtype = bool)
        while mask.any():
            new_mask = np.sum(inc_mat[:, mask], axis = -1) > 0
            if (new_mask == mask).all():
                raise RuntimeError("There are loops in io map!!!")
            sorted_steps_ll.append([steps_list[i] for i in np.nonzero(mask & (~new_mask))[0]])
            mask = new_mask
        return sorted_steps_ll

    @classmethod
    def cal_inc_mat(cls, total_io: Dict[str, Dict[str, StepKeyPair]], run_list: List[str]):
        run_dict = {run_list[i]: i for i in range(len(run_list))}
        inc_mat = np.zeros((len(run_list), len(run_list)), dtype = bool)
        for step_name in run_list:
            for pre_step_name, _ in total_io[step_name].values():
                if pre_step_name is not None:
                    inc_mat[run_dict[step_name], run_dict[pre_step_name]] = True
        return inc_mat

    @classmethod
    def cal_components(cls, total_io: Dict[str, Dict[str, StepKeyPair]], run_list: Optional[List[str]] = None):
        if run_list is None:
            run_list = list(total_io.keys())
        num_steps = len(run_list)
        run_dict = {run_list[i]: i for i in range(num_steps)}
        inc_mat = cls.cal_inc_mat(total_io, run_list)
        inc_mat |= inc_mat.T

        mask = np.ones((num_steps, ), dtype = bool)
        components: List[List[str]] = []
        remained_list = copy(run_list)
        while remained_list:
            end_step = remained_list[-1]
            mask.fill(False)
            mask[run_dict[end_step]] = True
            new_mask = mask
            while new_mask.any():
                new_mask = (np.sum(inc_mat[new_mask, :], axis = 0) > 0) & (~mask)
                mask |= new_mask
            components.append([run_list[i] for i in np.nonzero(mask)[0]])
            for s in components[-1]:
                remained_list.remove(s)
        return components

if __name__ == "__main__":
    from spectra_flow.utils import bohrium_login, load_json
    bohrium_login(load_json("../examples/account.json"))
    class A(SuperOP):

        @classmethod
        def get_inputs(cls):
            return {
                "A.1": InputParameter(type = dict, value = {}),
                "A.2": InputParameter(type = dict, value = {})
            }, {
                "A.3": InputArtifact(),
                "A.4": InputArtifact(optional=True)
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
    
    class B(SuperOP):

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
    
    class C(SuperOP):

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

    class D(OP):
        @classmethod
        def get_input_sign(cls):
            return OPIOSign({
                "D.1": Parameter(str),
                "D.2": Artifact(str, optional=True),
            })

        @classmethod
        def get_output_sign(cls):
            return OPIOSign({
                "D.3": Artifact(str)
            })


    class ABflow(AdaptiveFlow):
        all_steps = {"a": A, "b": B, "c": C, "d": D}
        steps_list = ["a", "b", "c", "d"]
        
        @classmethod
        def get_io_dict(cls):
            io_dict = super().get_io_dict()
            io_dict["b"]["B.1"] += [("a", "A.5")]
            io_dict["b"]["B.2"] += [("a", "A.6")]
            io_dict["c"]["C.1"] += [ ("b", "B.5")]
            io_dict["c"]["C.3"] += [ ("b", "B.8")]
            io_dict["d"]["D.1"] += [ ("c", "C.5")]
            # io_dict["d"]["D.2"] += [ ("c", "C.8")]
            return io_dict
        
        def build_templates(self, run_list):
            return super().build_templates(run_list)
    
    ABflow.check_steps_list()
    inputs = [
        "A.1", "A.2", "A.3", 
        # "A.4", 
        # "B.1", 
        "B.2", 
        "B.3", 
        "B.4", 
        "C.2", 
        "C.4"
    ]
    run_list = ABflow.run_from_inputs(inputs)
    print(list(run_list.values()))
    for s in ["d", "c", "b", "a"]:
        if s in run_list:
            run_list = run_list[s]
            break
    flow = ABflow("ab", run_list, given_inputs = inputs, pri_source = "self", debug = True)
    