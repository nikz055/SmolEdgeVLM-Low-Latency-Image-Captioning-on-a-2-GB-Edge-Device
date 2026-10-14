"""SmolEdgeVLM: low-latency image captioning on a 2 GB edge device."""
from .merge import TokenMerger, lattice_anchors, size_bias, unmerge
from .modules import PoolConnector, Restore

__all__ = ['TokenMerger', 'lattice_anchors', 'size_bias', 'unmerge', 'PoolConnector', 'Restore', 'SmolEdgeVLM']


def __getattr__(name):                      # import the transformers-based model only when it is used
    if name == 'SmolEdgeVLM':
        from .model import SmolEdgeVLM
        return SmolEdgeVLM
    raise AttributeError(name)
