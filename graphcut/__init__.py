"""GrabCut segmentation refinement: dual colour GMM + parallel Push-Relabel min-cut.

Optional stage. The shipping pipeline is MOG2 -> mask_refiner -> blur composite;
this package is the graph-cut refinement we measured against it.

    push_relabel_numba : parallel push-relabel max-flow (checkerboard push +
                         BFS global relabel). Verified against Boykov-Kolmogorov.
    fgd_gmm_numba      : Rother-2004 full-covariance 5-component colour GMM.
    morph_numba        : dilate / erode / largest connected component.
    grabcut_numba      : the pipeline that wires them to a MOG2 seed.
"""
from .grabcut_numba import GrabCutPipeline
from .push_relabel_numba import push_relabel

__all__ = ["GrabCutPipeline", "push_relabel"]
