"""ARBITER: Attentive Relational Bidirectional Impartial Typed Encoder for Reasoning.

Official release package for ARBITER-272M.
"""

from .configuration_arbiter import ArbiterConfig
from .modeling_arbiter import ArbiterModel, SetTransformerHyperFiLMHead

__version__ = "1.1.1"
__all__ = ["ArbiterConfig", "ArbiterModel", "SetTransformerHyperFiLMHead"]
