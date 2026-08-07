"""NEXUS Knowledge Intelligence Engine.

The knowledge/ML subsystem of the NEXUS autonomous knowledge-discovery
platform. Responsible for ingestion, knowledge-graph construction,
hybrid retrieval, GraphRAG, uncertainty modelling, contradiction
detection, knowledge-gap analysis and investigation scoring.

The subsystem is transport- and provider-independent. Every external
dependency (embedding models, vector stores, graph databases, search
systems) is accessed exclusively through typed ports (interfaces)
defined in :mod:`nexus_knowledge.port`.
"""

__version__ = "0.1.0"
