import typing as tp

from dataclasses import dataclass, fields, asdict


@dataclass
class BaseFeaturePaths:
    def get_fields(self):
        return list(fields(self))

    def to_dict(self):
        return asdict(self)
    
    
@dataclass
class EventFeaturesPaths(BaseFeaturePaths):
    """
    Paths to features
    of event itself
    """

    # Number of pulses in event
    PulsesN: str = "BEvent./BEvent.fPulseN"


@dataclass
class MCEventFeaturesPaths(EventFeaturesPaths):
    """
    Paths to features
    of event itself in Mote-Carlo data
    """
    # Prime particle features
    PrimeTheta: str = "BMCEvent./BMCEvent.fPrimaryParticleTheta"
    PrimePhi: str = "BMCEvent./BMCEvent.fPrimaryParticlePhi"
    PrimeEn: str = "BMCEvent./BMCEvent.fPrimaryParticleEnergy"
    PrimeNuclN: str = "BMCEvent./BMCEvent.fNucleonN"
    # Aggregate features of Muons in events
    ResponseMuN: str = "BMCEvent./BMCEvent.fResponseMuonsN"
    FirstMuTime = "BMCEvent./BMCEvent.fFirstMuonTime"
    BundleEnReg: str = "BMCEvent./BMCEvent.fSumEnergyBundleReg"
    # Weight of event in MC
    EventWeight: str = "BMCEvent./BMCEvent.fEventWeight"
    
    
@dataclass
class ExpEventFeaturesPaths(EventFeaturesPaths):
    """
    Paths to features
    of event itself in experimental data. 
    Nothing, but number of pulses is known.
    """
    pass


@dataclass
class MuonsFeaturesPaths(BaseFeaturePaths):
    """
    Paths to features of
    individual muons in events
    """
    pass


@dataclass
class MCMuonsFeaturesPaths(MuonsFeaturesPaths):
    """
    Paths to features of
    individual muons in events in Monte-Carlo data.
    """
    RespMuTheta: str = "BMCEvent./BMCEvent.fTracks/BMCEvent.fTracks.fTheta"
    RespMuPhi: str = "BMCEvent./BMCEvent.fTracks/BMCEvent.fTracks.fPhi"
    RespMuTrackX: str = "BMCEvent.fTracks.fX"
    RespMuTrackY: str = "BMCEvent.fTracks.fY"
    RespMuTrackZ: str = "BMCEvent.fTracks.fZ"
    RespMuDelay: str = "BMCEvent./BMCEvent.fTracks/BMCEvent.fTracks.fDelay"
    RespMuEn: str = "BMCEvent./BMCEvent.fTracks/BMCEvent.fTracks.fMuonEnergy"
    

@dataclass
class ExpMuonsFeaturesPaths(MuonsFeaturesPaths):
    """
    Paths to features of
    individual muons in events.
    Nothing is known for experimental data.
    """
    pass


@dataclass
class PulsesFeaturesPaths(BaseFeaturePaths):
    """
    Paths to features of
    individual pulses in events
    """

    PulsesChID: str = "BEvent./BEvent.fPulses/BEvent.fPulses.fChannelID"
    PulsesAmpl: str = "BEvent./BEvent.fPulses/BEvent.fPulses.fAmplitude"
    PulsesTime: str = "BEvent./BEvent.fPulses/BEvent.fPulses.fTime"
 
    
@dataclass
class MCPulsesFeaturesPaths(PulsesFeaturesPaths):
    """
    Paths to features of
    individual pulses in events
    in Monte-Carlo data (magich number flag added).
    """
    PulsesFlg: str = (
        "MCEventMask./MCEventMask.BEventMask/MCEventMask.BEventMask.fOrigins/MCEventMask.BEventMask.fOrigins.fFlag"
    )
 
    
@dataclass
class ExpPulsesFeaturesPaths(PulsesFeaturesPaths):
    """
    Paths to features of
    individual pulses in events
    in experimental data.
    """
    pass


@dataclass
class BaseRootPaths(BaseFeaturePaths):
    geom_path: str = "BGeomTel./BGeomTel.BGeomTel/BGeomTel.BGeomTel.fOMs/BGeomTel.BGeomTel.fOMs.fPosition"
    data_header: tp.Optional[str] = None
    coords_header: tp.Optional[str] = None
    ev_paths: tp.Optional[EventFeaturesPaths] = None
    ind_mu_paths: tp.Optional[MuonsFeaturesPaths] = None
    pulses_paths: tp.Optional[MCPulsesFeaturesPaths] = None

@dataclass
class MCRootPaths(BaseRootPaths):
    data_header: str = "Events"
    coords_header: str = "ArrayConfig"
    ev_paths: EventFeaturesPaths = MCEventFeaturesPaths()
    ind_mu_paths: MuonsFeaturesPaths = MCMuonsFeaturesPaths()
    pulses_paths: PulsesFeaturesPaths = MCPulsesFeaturesPaths()
    
@dataclass
class ExpRootPaths(BaseRootPaths):
    data_header: str = "Events"
    coords_header: str = "Events"
    ev_paths: EventFeaturesPaths = ExpEventFeaturesPaths()
    ind_mu_paths: MuonsFeaturesPaths = ExpMuonsFeaturesPaths()
    pulses_paths: PulsesFeaturesPaths = ExpPulsesFeaturesPaths()
