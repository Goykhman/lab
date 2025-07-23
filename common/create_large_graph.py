import numpy

from io import StringIO
from numba.core.types import float64
from numpy.random import randint, seed

from numbox.core.configurations import default_jit_options
from numbox.utils.highlevel import cres


seed(1)


NUM_OF_ENTITIES_DEFAULT = 1000
NUM_OF_PURE_INPUTS_DEFAULT = 1000


def make_func_header(num_sources, i_, j_, ty="numpy.ndarray"):
    func_name = f"w_{i_}"
    sources = ", ".join([f"w_{k_}: {ty}" for k_ in range(j_, j_ + num_sources)])
    func_header = f"""
def {func_name}({sources}) -> {ty}:"""
    return func_header


def py_create_pure_inputs(num_of_pure_inputs):
    all_lines = []
    for i in range(num_of_pure_inputs):
        all_lines.append(f"""
def w_{i}() -> float:
    return 0.0
""")
    return num_of_pure_inputs, all_lines


def numpy_create_pure_inputs(num_of_pure_inputs):
    all_lines = []
    for i in range(num_of_pure_inputs):
        all_lines.append(f"""
def w_{i}() -> numpy.ndarray:
    return 0.0
""")
    return num_of_pure_inputs, all_lines


def py_func_code_txt(num_sources, i_, j_):
    func_header = make_func_header(num_sources, i_, j_, ty="float")
    if num_sources == 1:
        return f"""{func_header}
    return (3.14 + w_{j_}) / 1.41"""
    elif num_sources == 2:
        return f"""{func_header}
    return w_{j_} * (w_{j_ + 1} - 1.41) + 3.14"""
    else:
        assert num_sources == 3, f"{num_sources} not supported"
        return f"""{func_header}
    return w_{j_} + w_{j_ + 1} * w_{j_ + 2} if w_{j_ + 2} >= 1 else w_{j_ + 2} * 3 if w_{j_ + 2} >= 0 else w_{j_} + 2.17 * w_{j_ + 1} + 3.14 * w_{j_ + 2}"""


def numpy_func_code_txt(num_sources, i_, j_):
    func_header = make_func_header(num_sources, i_, j_)
    if num_sources == 1:
        return f"""{func_header}
    return (3.14 + w_{j_}) / 1.41"""
    elif num_sources == 2:
        return f"""{func_header}
    return w_{j_} * (w_{j_ + 1} - 1.41) + 3.14"""
    else:
        assert num_sources == 3, f"{num_sources} not supported"
        return f"""{func_header}
    return numpy.where(w_{j_ + 2} >= 1, w_{j_} + w_{j_ + 1} * w_{j_ + 2}, numpy.where(w_{j_ + 2} >= 0, w_{j_ + 2} * 3, w_{j_} + 2.17 * w_{j_ + 1} + 3.14 * w_{j_ + 2}))"""


def make_derived_node(num_sources, i_, j_):
    derive = "calc_1_" if num_sources == 1 else "calc_2_" if num_sources == 2 else "calc_3_"
    sources = ", ".join([f"w_{k_}" for k_ in range(j_, j_ + num_sources)])
    sources = sources + ", " if "," not in sources else sources
    sources = f"({sources})"
    declaration = f'w_{i_} = ll_make_work("w_{i_}", 0.0, {sources}, {derive})'
    return declaration


def create_tree_level(lower_level_start: int, lower_level_end: int, derived_maker=numpy_func_code_txt):
    lines = []
    lower_level_index_ = lower_level_start
    node_index_ = lower_level_end
    lower_level_start_new = node_index_
    while lower_level_index_ < lower_level_end - 3:
        num_sources = randint(1, 4)
        lines.append(derived_maker(num_sources, node_index_, lower_level_index_))
        node_index_ += 1
        lower_level_index_ += num_sources
    if lower_level_index_ < lower_level_end:
        lines.append(derived_maker(lower_level_end - lower_level_index_, node_index_, lower_level_index_))
        node_index_ += 1
    lower_level_end_new = node_index_
    return lines, lower_level_start_new, lower_level_end_new


def create_graph(
    num_of_pure_inputs: int = NUM_OF_PURE_INPUTS_DEFAULT,
    inputs_maker=numpy_create_pure_inputs,
    derived_maker=numpy_func_code_txt, reseed=True
):
    if reseed:
        seed(1)
    s = 0
    e, all_lines = inputs_maker(num_of_pure_inputs)
    while e - s > 1:
        lines, s, e = create_tree_level(s, e, derived_maker)
        all_lines.extend(lines)
    return "\n".join(all_lines)


def write_py_large_graph():
    with open("./large_graph.py", "w") as f:
        print(create_graph(inputs_maker=py_create_pure_inputs, derived_maker=py_func_code_txt), file=f)


def write_numpy_large_graph():
    with open("./numpy_large_graph.py", "w") as f:
        print("import numpy\n", file=f)
        print(create_graph(inputs_maker=numpy_create_pure_inputs, derived_maker=numpy_func_code_txt), file=f)


if __name__ == "__main__":
    write_py_large_graph()
    write_numpy_large_graph()
