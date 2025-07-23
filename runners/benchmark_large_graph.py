import sys

from numpy import allclose
from numbox.utils.timer import timer

from common import large_graph
from common.large_graph_setup import top_node_name
from hamilton_lab.make_large_graph import run_hamilton
from numbox_lab.make_large_graph import numbox_graph, node_prefix
from numbox_lab.benchmark_large_graph import run_numbox


def compare_hamilton_and_numbox():
    """
    For `num_of_entities = 1000`, observed (second run, after numbox has compiled and cached):
    WARNING:numbox.utils.timer:Execution of _run_numbox took 0.816s
    WARNING:numbox.utils.timer:Execution of _run_hamilton took 120.374s
    _run_hamilton time / _run_numbox time = 147.5
    """
    work_w_1990_name = f"{node_prefix}{top_node_name}"
    graph_, derived, end = numbox_graph(large_graph, work_w_1990_name)
    top_node_ = getattr(graph_, work_w_1990_name)
    top_node_.calculate()
    num_of_entities = 1000
    numbox_output = run_numbox(node=top_node_, num_of_entities=num_of_entities)
    hamilton_output = run_hamilton(num_of_entities)
    assert allclose(hamilton_output, numbox_output)
    print(
        f"_run_hamilton time / _run_numbox time = {(timer.times['_run_hamilton'] / timer.times['_run_numbox']):.1f}",
        file=sys.stderr
    )


if __name__ == "__main__":
    compare_hamilton_and_numbox()
