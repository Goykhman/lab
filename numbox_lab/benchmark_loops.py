import numpy
import sys

from numba import njit
from numpy import allclose

from numbox.core.configurations import default_jit_options
from numbox.core.work.builder import Derived, End, make_graph
from numbox.core.work.work import Work
from numbox.utils.timer import Timer


numpy.random.seed(1)
timer = Timer(6)


def derive_z(x_, y_):
    return x_ * y_ + x_ / (y_ + 1e-15) + x_ ** 2 + y_ ** 2


@timer
@njit(**default_jit_options)
def scalar_run(
    x_s_node: Work,
    y_s_node: Work,
    z_s_node: Work,
    x_data_: numpy.ndarray,
    y_data_: numpy.ndarray,
    z_results_: numpy.ndarray,
    size_: int
):
    for i in range(size_):
        x_s_node.data = x_data_[i]
        y_s_node.data = y_data_[i]
        z_s_node.derived = False
        z_s_node.calculate()
        z_results_[i] = z_s_node.data


@timer
@njit(**default_jit_options)
def vector_run(z_v_node_):
    z_v_node_.calculate()
    return z_v_node_.data


def vector_calculation(x_data_, y_data_, size_):
    x_v = End(name="x_v", init_value=x_data_)
    y_v = End(name="y_v", init_value=y_data_)
    z_v = Derived(
        name="z_v",
        init_value=numpy.zeros((size_,), dtype=numpy.float64),
        derive=derive_z,
        sources=(x_v, y_v)
    )
    z_v_node = make_graph(z_v).z_v
    vector_result_ = vector_run(z_v_node)
    return vector_result_


def scalar_calculation(x_data_, y_data_, size_):
    x_s = End(name="x_s", init_value=0.0)
    y_s = End(name="y_s", init_value=0.0)
    z_s = Derived(name="z_s", init_value=0.0, derive=derive_z, sources=(x_s, y_s))
    graph_s = make_graph(x_s, y_s, z_s)
    scalar_result_ = numpy.empty((size_,), dtype=numpy.float64)
    scalar_run(graph_s.x_s, graph_s.y_s, graph_s.z_s, x_data_, y_data_, scalar_result_, size_)
    return scalar_result_


if __name__ == "__main__":
    """
    Observed
   
    size = 1000:
    
        WARNING:numbox.utils.timer:Execution of vector_run took 0.001561s
        WARNING:numbox.utils.timer:Execution of scalar_run took 0.001417s
        scalar time / vector time = 0.90
        
    size = 10000:
    
        WARNING:numbox.utils.timer:Execution of vector_run took 0.001358s
        WARNING:numbox.utils.timer:Execution of scalar_run took 0.001394s
        scalar time / vector time = 1.03
        
    size = 100000:
    
        WARNING:numbox.utils.timer:Execution of vector_run took 0.001438s
        WARNING:numbox.utils.timer:Execution of scalar_run took 0.003017s
        scalar time / vector time = 2.10
    """
    size = 1000
    x_data = numpy.random.random(size)
    y_data = numpy.random.random(size)

    vector_result = vector_calculation(x_data, y_data, size)
    scalar_result = scalar_calculation(x_data, y_data, size)
    assert allclose(scalar_result, vector_result)

    times = timer.times
    print(f"scalar time / vector time = {(times['scalar_run'] / times['vector_run']):.2f}", file=sys.stderr)
