"""
proteinoid_spikes
=================

Reference implementation for:

    Sharma, S., Mahmud, A., Tarabella, G., Mougoyannis, P., Adamatzky, A.
    "Graph-based Complexity and Computational Capabilities of Proteinoid
    Spike Systems." arXiv:2504.10362v2 (2025).

Modules map 1:1 onto the paper:

    data_loading        Section II.A    -- load the 5 raw spike trains
    transformations     Section III     -- F1, F2, multinodal graph
    complexity          Section IV      -- 8 metrics + meta-metric
    features            Section V.B     -- 16-dim feature vector
    classifier          Section V.A/C   -- feedforward ReLU network
    plotting            Figs. 3-9       -- paper figures
"""

__version__ = "1.0.0"