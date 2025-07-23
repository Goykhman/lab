import numpy

from numba import njit
from numba.core.types import unicode_type
from numba.typed import Dict

from numbox.core.any.any_type import AnyType
from numbox.core.configurations import default_jit_options
from numbox.core.work.loader_utils import load_array_row_into_dict
from numbox.utils.timer import timer

from common.large_graph_setup import NUM_OF_ENTITIES_DEFAULT, NUM_OF_PURE_INPUTS_DEFAULT
from common.utils import prepare_input_data


@njit(**default_jit_options)
def run_entity(total, loader_dict):
    total.load(loader_dict)
    total.calculate()
    return total.data


@timer
@njit(**default_jit_options)
def _run_numbox(total, data, loader_dict, num_of_entities=NUM_OF_ENTITIES_DEFAULT):
    total_data: numpy.ndarray = numpy.empty((num_of_entities,), dtype=numpy.float64)
    for i in range(num_of_entities):
        load_array_row_into_dict(data, i, loader_dict)
        total_data[i] = run_entity(total, loader_dict)
    return total_data


def run_numbox(node, num_of_entities, num_of_inputs=NUM_OF_PURE_INPUTS_DEFAULT):
    data = prepare_input_data(num_of_inputs, num_of_entities)
    loader_dict = Dict.empty(unicode_type, AnyType)
    total_data = _run_numbox(node, data, loader_dict, num_of_entities)
    return total_data
