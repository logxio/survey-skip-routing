"""Independent structural ablations for the World Bank engine."""
from ortho_config import _flag
K1 = _flag('V68_K1', False)
K2 = _flag('V68_K2', True)
K3 = _flag('V68_K3', False)
# Five-fold arms fuse on five partitions before meeting two-fold trees.
V73_K1 = _flag('V73_K1', True)
# Treatment-group log-odds recalibration of mirror and follow-up items.
V73_K2 = _flag('V73_K2', True)
# Post-assessment catboost takes the deep and hgb share of its pass window.
V76_K1 = _flag('V76_K1', True)
# Final stack runs on every thread; 8 of the 18-second tail return to the passes.
V80_W1 = _flag('V80_W1', False)
# Child CatBoost takes non-follow-up, highest-loss items first.
V80_W2 = _flag('V80_W2', False)
# Final stack runs in forked worker processes on Linux; the 18-second tail stays.
V83_W1 = _flag('V83_W1', True)
# Psychometric IRT arm on every scale item, its folds spread over forked workers.
V83_W2 = _flag('V83_W2', False)
# Joint latent-class regression arm over the GIVEN columns and all 76 targets.
V87_W1 = _flag('V87_W1', True)
# wb_lca trait kind fit to convergence with exact softmax Newton steps; skip-gated children and
# jointly unanswered follow-up blocks take their parent's open probability inside every class.
V91_W1 = _flag('V91_W1', True)
# wb_lca trait kind with per-randomisation-group answer intercepts (implies the the configured option machinery), 8.5 s window.
V91_W2 = _flag('V91_W2', True)
# Output vectors water-filled for the scorer's mixing floor, strength chosen per item on its final OOF.
V97_W1 = _flag('V97_W1', False)

V101_W1 = _flag('V101_W1', False)
V101_W2 = _flag('V101_W2', False)
V101_W3 = _flag('V101_W3', False)
V101_W4 = _flag('V101_W4', False)

# Direct covariate trait and conditional follow-up posterior.
from ortho_config import V110_W1, V110_W2

#  CSR float32 Stage-2 kernel (off: the reference kernel).
V113_W1 = _flag("V113_W1", False)
# wb_lca on its own wave deadlines: class kinds and t1 to 43%, u1 to 76%, direct effects the rest.
V115_W1 = _flag("V115_W1", False)
# wb_lca Stage 1 and Stage 2 on one forked pool: trait fits first, each fold's direct
# effects as soon as Stage-1 tasks are dispatched, trait EM capped at trait_stop, class EM
# keeps its fit at the deadline; 24-second window ( : 14/14 cells 20+10 complete, +0.001894 nats/item).
V115_W3 = _flag("V115_W3", True)
# Direct-effect sweeps divide the curvature by the columns they update, not by every column.
V115_W2 = _flag("V115_W2", True)

# Refit the completed direct-effect union on all six available components.
V118_W1 = _flag('V118_W1', True)

# World Bank : budget restoration, anchors 5-fold shallow/two_stage, children logit window,  followup + raking + GIVEN->PREDICT mirrors.
from ortho_config import V124_W1, V124_W2

import os as _os
V127_W1 = _os.environ.get("SB_V127_W1", "1") == "1"
V127_W2 = _os.environ.get("SB_V127_W2", "1") == "1"
