from inspect import getmembers, isfunction, signature
from numba import typeof
from numbox.core.work.builder import Derived, End, make_graph


def load_all_functions(module_):
    members = getmembers(module_)
    return [obj for name, obj in members if isfunction(obj)]


node_prefix = "work_"


def get_node_spec(n_, derived_, end_):
    if n_ in derived_:
        return derived_[n_]
    elif n_ in end_:
        return end_[n_]
    else:
        raise ValueError(f"Could not find {n_} among {sorted(derived_.keys())} or {sorted(end_.keys())}")


def create_node(func, derived_, end_, temp_registry_):
    """ This creates a return value as a default constructor output for the return
    type. Anything that a function defining end nodes might actually be returning is
    ignored. """
    sig = signature(func)
    ret = sig.return_annotation
    init_value = ret()
    params = tuple(sig.parameters.keys())
    node_name = f"{node_prefix}{func.__name__}"
    if len(params) != 0:
        derived_[node_name] = Derived(
            name=node_name,
            init_value=init_value,
            derive=func,
            sources=params,
            registry=temp_registry_,
            ty=typeof(init_value)
        )
    else:
        end_[node_name] = End(name=node_name, init_value=init_value)


def prune_derived_nodes(derived_, end_):
    for name_, node_ in derived_.items():
        sources = [get_node_spec(f"{node_prefix}{n}", derived_, end_) for n in node_.sources]
        derived_[name_] = Derived(
            name=node_.name, init_value=node_.init_value, derive=node_.derive, sources=sources, ty=node_.ty
        )


def numbox_graph(module_, *access_nodes):
    derived_ = {}
    end_ = {}
    funcs = load_all_functions(module_)
    temp_registry_ = {}
    for func in funcs:
        create_node(func, derived_, end_, temp_registry_)
    prune_derived_nodes(derived_, end_)
    access_nodes = [get_node_spec(n, derived_, end_) for n in access_nodes]
    return make_graph(*access_nodes), derived_, end_
