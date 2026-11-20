"""Independent survey switches. Environment values: 1/0, true/false, yes/no, on/off."""
import os

def _flag(name, default):
    aliases = {"V118_W1": "SB_V68_V118_W1", "V101_W3": "SB_V68_V101_W3",
               "V116_K1": "SB_V64_V116_K1", "V116_K2": "SB_V64_V116_K2",
               "V122_K1": "SB_V64_V122_K1", "V122_K2": "SB_V64_V122_K2",
               "V124_W1": "SB_V68_V124_W1", "V124_W2": "SB_V68_V124_W2",
               "V125_K1": "SB_V64_V125_K1", "V125_K2": "SB_V64_V125_K2"}
    value = os.environ.get(aliases.get(name, name), os.environ.get(name))
    if value is None:
        return default
    value = value.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError("Invalid boolean for " + name + ": " + value)

V64_FAST = _flag("V64_FAST", True)
V64_DONOR = _flag("V64_DONOR", True)
V64_GATE = _flag("V64_GATE", True)
V65_K1_FAST_STAGES = _flag("V65_K1_FAST_STAGES", True)
V65_K2_RCT = _flag("V65_K2_RCT", True)
V65_K3_MULTITASK = _flag("V65_K3_MULTITASK", True)
V66_HIERARCHY = _flag("V66_HIERARCHY", True)
V66_GRAPH = _flag("V66_GRAPH", True)
V66_POPULATION = _flag("V66_POPULATION", True)
V72_K1 = _flag("V72_K1", True)
V72_K2 = _flag("V72_K2", True)
# UNICEF joint latent-class regression arm in place of the expiring latent profile.
V80_U1 = _flag("V80_U1", True)
# UNICEF child CatBoost folds scheduled longest-first across all eight threads.
V80_U2 = _flag("V80_U2", False)
# UNICEF latent classes with country-specific answer profiles and mixed class counts.
V83_U1 = _flag("V83_U1", True)
# UNICEF latent-class posteriors join every tree and logit feature matrix.
V83_U2 = _flag("V83_U2", False)
# UNICEF final stack in forked processes, with five-fold arms fused before the two-fold trees.
V89_U1 = _flag("V89_U1", True)
# UNICEF latent-class posteriors at several GIVEN-likelihood temperatures, mixed per item.
V89_U2 = _flag("V89_U2", False)
# UNICEF child items: class posteriors that also read the stacked anchor predictions.
V89_U3 = _flag("V89_U3", False)
# UNICEF nominal-response trait arms, pooled and with country target offsets.
V93_U1 = _flag("V93_U1", True)
# Trait kinds when the configured option is on: t pooled, tc country offsets, x adds direct GIVEN effects.
V93_U1_KINDS = tuple(k for k in os.environ.get("V93_U1_KINDS", "tc,tcx,t").split(",") if k)
# UNICEF trait arm tcxp: tcx plus a shrunk rank-one effect on the joint levels of GIVEN pairs (Country x Q01).
V99_U1 = _flag("V99_U1", True)
V99_U1_PAIRS = tuple(tuple(p.split(":")) for p in os.environ.get("V99_U1_PAIRS", "Country:Q01").split(",") if p)
# UNICEF compact stack: the trait anchor plus each learner family's out-of-fold champion, kept unless the full stack wins both halves.
V106_U1 = _flag("V106_U1", False)
# UNICEF trait arm tcxv beside tcxp: the pairs below, each pair table kept to its noise-gated rank (up to the configured option), one trait posterior refresh.
V106_U2 = _flag("V106_U2", True)
V106_U2_PAIRS = tuple(tuple(p.split(":")) for p in os.environ.get("V106_U2_PAIRS", "Country:Q01,Country:Q05").split(",") if p)
V106_U2_RANK = int(os.environ.get("V106_U2_RANK", "3"))
V106_U2_REFRESH = _flag("V106_U2_REFRESH", True)
# Seconds the UNICEF run keeps after its learners when the configured option is on.
V89_U1_TAIL = float(os.environ.get("V89_U1_TAIL", "10.0"))

SWITCHES = {name: globals()[name] for name in ('V64_FAST', 'V64_DONOR', 'V64_GATE', 'V65_K1_FAST_STAGES', 'V65_K2_RCT', 'V65_K3_MULTITASK', 'V66_HIERARCHY', 'V66_GRAPH', 'V66_POPULATION', 'V72_K1', 'V72_K2', 'V80_U1', 'V80_U2', 'V83_U1', 'V83_U2', 'V89_U1', 'V89_U2', 'V89_U3', 'V93_U1', 'V99_U1', 'V106_U1', 'V106_U2')}

V109_K1 = _flag("V109_K1", True)
V109_K2 = _flag("V109_K2", False)
V109_K3 = _flag("V109_K3", False)
V110_W1 = _flag("V110_W1", True)
V110_W2 = _flag("V110_W2", False)
SWITCHES.update({name: globals()[name] for name in ("V109_K1", "V109_K2", "V109_K3", "V110_W1", "V110_W2")})
# UNHCR trait fit: round-one closed-gate "0" cells of zero-on-skip items read as the skip.
V112_K1 = _flag("V112_K1", True)
# UNHCR rows with no visible twin: after the configured option, a stack over the current output and every arm fitted on twin-less visible rows.
V112_K2 = _flag("V112_K2", False)
SWITCHES.update({name: globals()[name] for name in ("V112_K1", "V112_K2")})

V114_K1 = _flag("V114_K1", True)
V114_K2 = _flag("V114_K2", True)
V114_K3 = _flag("V114_K3", False)
SWITCHES.update({name: globals()[name] for name in ("V114_K1", "V114_K2", "V114_K3")})
# UNICEF direct-effect sweeps divide the curvature by the columns they update, not by every column.
V115_U1 = _flag("V115_U1", False)
# On top of U1: each Country pair table sweeps alone, over Country x every other GIVEN column.
V115_U2 = _flag("V115_U2", False)
# Exact curvature on the main-effect sweeps only; pair sweeps keep the J step.
V115_U3 = _flag("V115_U3", False)
SWITCHES.update({name: globals()[name] for name in ("V115_U1", "V115_U2", "V115_U3")})

# Pooled panel trait fit uses one observation per matched household.
V116_K1 = _flag("V116_K1", True)
V116_K2 = _flag("V116_K2", False)
SWITCHES.update({name: globals()[name] for name in ("V116_K1", "V116_K2")})

# Fold-local latent routing experiments.
V122_K1 = _flag("V122_K1", True)
V122_K2 = _flag("V122_K2", True)
SWITCHES.update({name: globals()[name] for name in ("V122_K1", "V122_K2")})

# World Bank : per-survey budget floor, wb_lca budget restoration, anchors 5-fold shallow/two_stage, children logit window.
V124_W1 = _flag("V124_W1", True)
# World Bank :  nonlinear per-item followup beside shared followup, finite_pop_raking + attrition consensus, GIVEN->PREDICT mirror transitions.
V124_W2 = _flag("V124_W2", True)
SWITCHES.update({name: globals()[name] for name in ("V124_W1", "V124_W2")})

# UNHCR twin_base.bases computed in forked worker processes on Linux.
V125_K1 = _flag("V125_K1", True)
# UNHCR latent_classes wall seconds (up to 18.0 s) refunded to learner stage windows.
V125_K2 = _flag("V125_K2", True)
SWITCHES.update({name: globals()[name] for name in ("V125_K1", "V125_K2")})


import os as _os
V127_W1 = _os.environ.get("SB_V127_W1", "1") == "1"
V127_W2 = _os.environ.get("SB_V127_W2", "1") == "1"
V128_K1 = _os.environ.get("SB_V128_K1", "1") == "1"
V128_K2 = _os.environ.get("SB_V128_K2", "1") == "1"
V128_K3 = _os.environ.get("SB_V128_K3", "1") == "1"
V129_U1 = _os.environ.get("SB_V129_U1", "1") == "1"
V129_U2 = _os.environ.get("SB_V129_U2", "1") == "1"
V129_U3 = _os.environ.get("SB_V129_U3", "1") == "1"
