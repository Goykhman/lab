import logging
import numpy

from hamilton import driver
from numbox.utils.timer import timer

from common import large_graph
from common.large_graph_setup import NUM_OF_ENTITIES_DEFAULT, NUM_OF_PURE_INPUTS_DEFAULT, top_node_name
from common.utils import prepare_input_data


logger = logging.getLogger(__name__)
logging.getLogger("hamilton.base").setLevel(logging.ERROR)


@timer
def _run_hamilton(input_data, num_of_entities: int = NUM_OF_ENTITIES_DEFAULT):
    outputs_: numpy.ndarray = numpy.empty((num_of_entities,), dtype=numpy.float64)
    for i in range(num_of_entities):
        initial_values = {field: input_data[i][field] for field in input_data.dtype.names}
        dr = driver.Driver(initial_values, large_graph)
        output = dr.execute([top_node_name])[top_node_name].values[0]
        outputs_[i] = output
    return outputs_


def run_hamilton(num_of_entities: int = NUM_OF_ENTITIES_DEFAULT):
    data = prepare_input_data(
        num_of_sources=NUM_OF_PURE_INPUTS_DEFAULT,
        num_of_entities=num_of_entities,
        prefix=""
    )
    outputs_ = _run_hamilton(data, num_of_entities)
    return outputs_
