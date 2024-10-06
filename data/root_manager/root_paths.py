from dataclasses import dataclass, fields, asdict

@dataclass
class BaseFeaturePaths:
    def __init__(self):
        pass
    def get_fileds(self):
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
    PulsesN: str = 'Events/BEvent./BEvent.fPulseN'
    # Prime particle features
    PrimeTheta: str = "Events/BMCEvent./BMCEvent.fPrimaryParticleTheta"
    PrimePhi: str = "Events/BMCEvent./BMCEvent.fPrimaryParticlePhi"
    PrimeEn: str = "Events/BMCEvent./BMCEvent.fPrimaryParticleEnergy"
    PrimeNuclN: str = "Events/BMCEvent./BMCEvent.fNucleonN"
    # Aggregate features of Muons in events
    ResponseMuN: str = "Events/BMCEvent./BMCEvent.fResponseMuonsN"
    FirstMuTime= "Events/BMCEvent./BMCEvent.fFirstMuonTime"
    BundleEnReg: str = "Events/BMCEvent./BMCEvent.fSumEnergyBundleReg"
    # Weight of event in MC 
    EventWeight: str = "Events/BMCEvent./BMCEvent.fEventWeight"
    

@dataclass
class MuonsFeaturesPaths(BaseFeaturePaths):
    """
    Paths to features of 
    individual muons in events
    """
    RespMuTheta: str = "Events/BMCEvent./BMCEvent.fTracks/BMCEvent.fTracks.fTheta"
    RespMuPhi: str = "Events/BMCEvent./BMCEvent.fTracks/BMCEvent.fTracks.fPhi"
    RespMuTrackX: str = "Events/BMCEvent.fTracks.fX"
    RespMuTrackY: str = "Events/BMCEvent.fTracks.fY"
    RespMuTrackZ: str = "Events/BMCEvent.fTracks.fZ"
    RespMuDelay: str = "Events/BMCEvent./BMCEvent.fTracks/BMCEvent.fTracks.fDelay"
    RespMuEn: str = "Events/BMCEvent./BMCEvent.fTracks/BMCEvent.fTracks.fMuonEnergy"
    
@dataclass
class PulsesFeaturesPaths(BaseFeaturePaths):
    """
    Paths to features of 
    individual pulses in events
    """
    PulsesChID: str = "Events/BEvent./BEvent.fPulses/BEvent.fPulses.fChannelID"
    PulsesAmpl: str = "Events/BEvent./BEvent.fPulses/BEvent.fPulses.fAmplitude"
    PulsesTime: str = "Events/BEvent./BEvent.fPulses/BEvent.fPulses.fTime"
    PulsesFlg: str = "Events/MCEventMask./MCEventMask.BEventMask/MCEventMask.BEventMask.fOrigins/MCEventMask.BEventMask.fOrigins.fFlag"
    
    
@dataclass
class RootPaths(BaseFeaturePaths):
    ev_paths: EventFeaturesPaths = EventFeaturesPaths()
    ind_mu_paths: MuonsFeaturesPaths = MuonsFeaturesPaths()
    pulses_paths: PulsesFeaturesPaths = PulsesFeaturesPaths()
    geom_path: str = "ArrayConfig/BGeomTel./BGeomTel.BGeomTel/BGeomTel.BGeomTel.fOMs/BGeomTel.BGeomTel.fOMs.fPosition"