import numpy

numpy.random.seed(1)


def prepare_input_data(num_of_sources, num_of_entities, prefix="work_", reset_seed=True):
    """
    In numbox framework, derived (non-end) nodes typically have different names than
    the functions that define how these nodes are derived. Use non-trivial `prefix`.

    In hamilton framework, names of nodes are the same as the names of the functions
    that define their data values. Use `prefix=""`.

    `reset_seed = True` is desired when the same input data needs to be fed to more
    than one framework (e.g., hamilton and numbox) in order to match their outputs.
    """
    data_ty = numpy.dtype([(f"{prefix}w_{i}", numpy.float64) for i in range(num_of_sources)], align=True)
    data_: numpy.ndarray = numpy.empty((num_of_entities,), dtype=data_ty)
    for i in range(num_of_sources):
        value_ = 10 * numpy.random.rand(num_of_entities)
        data_[f"{prefix}w_{i}"] = value_
    if reset_seed:
        numpy.random.seed(1)
    return data_
