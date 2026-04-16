"""Tools for fitting constrained compositions from Heger02 progenitor data."""

from __future__ import annotations

__all__ = [
    "Heger02CompositionFitter",
    "ProfilePrediction",
    "ReducedNetwork",
    "SukhboldProfile",
    "build_artifact",
    "build_reduced_artifact",
    "compute_composition_ye",
    "verify_holdout_model",
    "load_artifact",
    "load_reduced_network",
    "plot_sukhbold_profile_comparison",
    "plot_sukhbold_two_panel_comparison",
    "plot_sukhbold_ye_comparison",
    "read_sukhbold_profile",
]


def __getattr__(name: str):
    if name == "Heger02CompositionFitter":
        from .fitter import Heger02CompositionFitter

        return Heger02CompositionFitter
    if name == "ProfilePrediction":
        from .fitter import ProfilePrediction

        return ProfilePrediction
    if name == "ReducedNetwork":
        from .reduced_network import ReducedNetwork

        return ReducedNetwork
    if name == "SukhboldProfile":
        from .sukhbold_profile import SukhboldProfile

        return SukhboldProfile
    if name == "build_artifact":
        from .heger02_dataset import build_artifact

        return build_artifact
    if name == "build_reduced_artifact":
        from .heger02_dataset import build_reduced_artifact

        return build_reduced_artifact
    if name == "compute_composition_ye":
        from .sukhbold_ye_plot import compute_composition_ye

        return compute_composition_ye
    if name == "verify_holdout_model":
        from .verification import verify_holdout_model

        return verify_holdout_model
    if name == "load_artifact":
        from .heger02_dataset import load_artifact

        return load_artifact
    if name == "load_reduced_network":
        from .reduced_network import load_reduced_network

        return load_reduced_network
    if name == "plot_sukhbold_profile_comparison":
        from .sukhbold_plot import plot_sukhbold_profile_comparison

        return plot_sukhbold_profile_comparison
    if name == "plot_sukhbold_two_panel_comparison":
        from .sukhbold_two_panel_plot import plot_sukhbold_two_panel_comparison

        return plot_sukhbold_two_panel_comparison
    if name == "plot_sukhbold_ye_comparison":
        from .sukhbold_ye_plot import plot_sukhbold_ye_comparison

        return plot_sukhbold_ye_comparison
    if name == "read_sukhbold_profile":
        from .sukhbold_profile import read_sukhbold_profile

        return read_sukhbold_profile
    raise AttributeError(name)
