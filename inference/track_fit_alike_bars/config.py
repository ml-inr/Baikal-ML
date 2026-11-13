"""Parameters of the fit that the BARS resource file sets.

Defaults are those of `resources/reco-muon_scl_DATA.rc` of BARS v2.3.2, which
the MC resource files share for every parameter used here (they differ only in
the channel masks and the neutrino-selection filter, neither reproduced).

Constants that BARS hard-codes rather than reads from the resource file live next
to the code that uses them, each with the BARS routine it comes from.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path


@dataclass(frozen=True)
class FitConfig:
    # BCausalityClique -- `Scanfit.*` keys, wired in programs/reco-muon/reco.cc
    clique_caus_window: float = 30.0      # Scanfit.CausWindow, ns
    clique_long_window: float = 137.0     # Scanfit.LongWindow, ns
    clique_min_size: int = 8              # Scanfit.MinCliqueSize
    clique_tolerance: int = 2             # Scanfit.Tolerance

    # BLocalTriggerSelect -- hard-coded in reco.cc
    lt_high_threshold: float = 3.0        # p.e.
    lt_low_threshold: float = 1.0         # p.e.
    lt_time_window: float = 100.0         # ns

    # BGlobalTriggerSelect and BOfflineTriggerCut
    min_strings: int = 2                  # OfflineTriggerCut.MinStringNum
    min_channels: int = 8                 # OfflineTriggerCut.MinChannelNum

    # BInitializeCausScanGrid and BScanFitClique
    scan_theta_step_deg: float = 6.0      # Scanfit.ThetaStep
    scan_phi_step_deg: float = 6.0        # Scanfit.PhiStep
    scan_caus_window: float = 30.0        # Scanfit.CausWindow, ns
    scan_min_clique_size: int = 7         # Scanfit.MinCliqueSize - 1 (reco.cc)
    scan_clique_gap: int = 1              # BScanFitClique::fCliqueGap default
    scan_zdist_min: float = 60.0          # BScanFitClique::fZdistMin default, m
    scan_time_sigma: float = 3.0          # BScanFitClique::fTimeSigma default, ns

    # BRecoMuonProducer
    prefit_type: int = 4                  # MethodConfig.PrefitType
    n_candidate_trials: int = 15          # reco.cc, with UseScanfitData

    @classmethod
    def from_bars_rc(cls, path: str | Path) -> "FitConfig":
        """Read the fit parameters from a BARS reco-muon resource file.

        Refuses a file that switches on a branch this port does not reproduce,
        rather than silently fitting something else.
        """
        values = _read_rc(Path(path))
        unsupported = {
            "MethodConfig.UsePrefilter": "0",
            "MethodConfig.UseMulticlusterGroupMask": "0",
            "MethodConfig.UseGroupMask": "0",
            "MethodConfig.UseScanfitMask": "1",
            "MethodConfig.UseScanfitData": "1",
            "MethodConfig.UseMuonProbTables": "0",
            "MethodConfig.UseMuonHistogram": "0",
            "MethodConfig.UseFixedSource": "0",
        }
        wrong = {k: values.get(k) for k, want in unsupported.items()
                 if values.get(k, want) != want}
        if wrong:
            raise NotImplementedError(f"{path}: not reproduced by this port: {wrong}")

        mapping = {
            "clique_caus_window": ("Scanfit.CausWindow", float),
            "clique_long_window": ("Scanfit.LongWindow", float),
            "clique_min_size": ("Scanfit.MinCliqueSize", int),
            "clique_tolerance": ("Scanfit.Tolerance", int),
            "min_strings": ("OfflineTriggerCut.MinStringNum", int),
            "min_channels": ("OfflineTriggerCut.MinChannelNum", int),
            "scan_theta_step_deg": ("Scanfit.ThetaStep", float),
            "scan_phi_step_deg": ("Scanfit.PhiStep", float),
            "scan_caus_window": ("Scanfit.CausWindow", float),
            "prefit_type": ("MethodConfig.PrefitType", int),
        }
        kwargs = {}
        for name, (key, cast) in mapping.items():
            if key in values:
                kwargs[name] = cast(float(values[key]))
        if "Scanfit.MinCliqueSize" in values:
            kwargs["scan_min_clique_size"] = int(float(
                values["Scanfit.MinCliqueSize"])) - 1
        if kwargs.get("prefit_type", 4) not in (2, 4):
            raise NotImplementedError(f"PrefitType {kwargs['prefit_type']}")
        return replace(cls(), **kwargs)

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def _read_rc(path: Path) -> dict[str, str]:
    """`key value` lines of a TEnv resource file; comments and bare keys dropped."""
    out = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        parts = line.split(None, 1)
        if len(parts) == 2:
            out[parts[0]] = parts[1].strip()
    return out
