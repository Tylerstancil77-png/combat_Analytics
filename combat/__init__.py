"""
combat_analytics_v1 — biomechanics engine for sparring footage.

Package layout (clean-restructure target):
    combat.tracking  — swappable tracking cores behind a common interface
    combat.analytics — strike detection, calibration, event output (tracker-agnostic)
    combat.fighters  — fighter / hand / head state containers
    combat.overlay   — annotation/drawing helpers

The tracking core (pose / SAM2 / EdgeTAM / detector) is decoupled from the
analytics engine via the data contract in `combat.tracking.base`, so the
winning tracker can be dropped in without touching downstream analytics.
"""
