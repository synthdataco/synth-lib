"""ChampionMiner must satisfy bittensor's axon.attach introspection.

`bittensor.core.axon.Axon.attach` does not take the request type as an argument — it reads it off
the forward function:

    first_param = next(iter(signature(forward_fn).parameters.values()))
    param_class = first_param.annotation
    assert issubclass(param_class, Synapse)

That makes the *runtime type* of the annotation part of the miner's contract, which no type checker
and no serving test would otherwise catch. Adding `from __future__ import annotations` to
champion_miner.py turns it into the string "Simulation" and `attach` fails with
`TypeError: issubclass() arg 1 must be a class` — startup, before the first request, so every
deployed champion crash-loops.

These tests reproduce attach's introspection without a chain connection: no subtensor, no wallet, no
network. Instantiating a Miner would need all three, which is why the regression shipped.
"""

import inspect

from bittensor import Synapse  # type: ignore[import-untyped]
from synth.protocol import Simulation  # type: ignore[import-untyped]

from synth_lib.serving.champion_miner import ChampionMiner


def _first_param_as_attach_sees_it() -> inspect.Parameter:
    """The first parameter of a BOUND forward_miner, which is what axon.attach receives.

    Binding matters: on the unbound function the first parameter is `self`, so inspecting the class
    attribute directly would assert against the wrong annotation and pass either way.
    """
    bound = ChampionMiner.forward_miner.__get__(object(), ChampionMiner)
    return next(iter(inspect.signature(bound).parameters.values()))


def test_forward_miner_first_param_is_the_synapse_not_self():
    assert _first_param_as_attach_sees_it().name == "synapse"


def test_forward_miner_annotation_is_a_class_not_a_string():
    annotation = _first_param_as_attach_sees_it().annotation
    assert isinstance(annotation, type), (
        f"forward_miner's synapse annotation is {annotation!r} ({type(annotation).__name__}), not a class. "
        "Something re-enabled PEP 563 in champion_miner.py — axon.attach will raise "
        "TypeError: issubclass() arg 1 must be a class and the miner will not start."
    )
    assert annotation is Simulation


def test_attach_introspection_succeeds():
    """The exact assertion bittensor's axon.attach makes."""
    param_class = _first_param_as_attach_sees_it().annotation
    assert issubclass(param_class, Synapse)
    assert param_class.__name__ == "Simulation"  # attach uses this as the request name
