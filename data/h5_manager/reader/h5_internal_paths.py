from dataclasses import dataclass, fields, asdict


@dataclass
class BaseFeaturePaths:
    def get_fields(self):
        return list(fields(self))

    def to_dict(self) -> dict[str, str]:
        return asdict(self)
    
    def to_dict_with_vars(self, particle_type: str, filenum: int) -> dict[str, str]:
        """Inserts prty type and root filenumber 
        to the path (for hdf5 file) instead of placeholders

        Args:
            particle_type (str): muatm, nuatm or nue2
            filenum (int): root file num

        Returns:
            dict[str, str]: schema to parse hdf5 file
        """
        def replace_placeholders(value):
            if isinstance(value, str):
                return value.replace("<ParticleType>", particle_type).replace("<FileNum>", str(filenum))
            if isinstance(value, dict):
                return {k: replace_placeholders(v) for k, v in value.items()}
            return value
        
        return {k: replace_placeholders(v) for k, v in self.to_dict().items()}


@dataclass
class EventFeaturesPaths(BaseFeaturePaths):
    """
    Paths to features
    of event itself
    """

    ## Number of pulses in event
    # PulsesN: str = "Events/BEvent./BEvent.fPulseN"
    # Prime particle features
    PrimeTheta: str = "<ParticleType>/prime_prty/part_<FileNum>/data.0"
    PrimePhi: str = "<ParticleType>/prime_prty/part_<FileNum>/data.1"
    PrimeEn: str = "<ParticleType>/prime_prty/part_<FileNum>/data.2"
    PrimeNuclN: str = "<ParticleType>/prime_prty/part_<FileNum>/data.3"
    # Aggregate features of Muons in events
    ResponseMuN: str = "<ParticleType>/prime_prty/part_<FileNum>/data.4"
    FirstMuTime = "<ParticleType>/muons_prty/aggregate/part_<FileNum>/data.0"
    BundleEnReg: str = "<ParticleType>/muons_prty/aggregate/part_<FileNum>/data.1"
    # Weight of event in MC
    EventWeight: str = "<ParticleType>/prime_prty/part_<FileNum>/data.5"
    # Event ID
    ev_id: str = "<ParticleType>/ev_ids/part_<FileNum>/data"


@dataclass
class MuonsFeaturesPaths(BaseFeaturePaths):
    """
    Paths to features of
    individual muons in events
    """

    RespMuTheta: str = "<ParticleType>/muons_prty/individ/part_<FileNum>/data.0"
    RespMuPhi: str = "<ParticleType>/muons_prty/individ/part_<FileNum>/data.1"
    RespMuTrackX: str = "<ParticleType>/muons_prty/individ/part_<FileNum>/data.2"
    RespMuTrackY: str = "<ParticleType>/muons_prty/individ/part_<FileNum>/data.3"
    RespMuTrackZ: str = "<ParticleType>/muons_prty/individ/part_<FileNum>/data.4"
    RespMuDelay: str = "<ParticleType>/muons_prty/individ/part_<FileNum>/data.5"
    RespMuEn: str = "<ParticleType>/muons_prty/individ/part_<FileNum>/data.6"
    # Auxilary. Not to be met in a final polars df
    EventStarts: str = "<ParticleType>/muons_prty/mu_starts/part_<FileNum>/data"
    # Event ID
    ev_id: str = "<ParticleType>/ev_ids/part_<FileNum>/data"


@dataclass
class PulsesFeaturesPaths(BaseFeaturePaths):
    """
    Paths to features of
    individual pulses in events
    """

    PulsesChID: str = "<ParticleType>/raw/channels/part_<FileNum>/data"
    PulsesAmpl: str = "<ParticleType>/raw/data/part_<FileNum>/data.0"
    PulsesTime: str = "<ParticleType>/raw/data/part_<FileNum>/data.1"
    Xrel: str = "<ParticleType>/raw/data/part_<FileNum>/data.2"
    Yrel: str = "<ParticleType>/raw/data/part_<FileNum>/data.3"
    Zrel: str = "<ParticleType>/raw/data/part_<FileNum>/data.4"
    t_res: str = "<ParticleType>/raw/t_res/part_<FileNum>/data"
    
    PulsesFlg: str = "<ParticleType>/raw/labels/part_<FileNum>/data"
    
    # Auxilary. Not to be met in a final polars df
    EventStarts: str = "<ParticleType>/raw/ev_starts/part_<FileNum>/data"
    
    # Event ID
    ev_id: str = "<ParticleType>/ev_ids/part_<FileNum>/data"


@dataclass
class H5Paths(BaseFeaturePaths):
    ev_paths: EventFeaturesPaths = EventFeaturesPaths()
    ind_mu_paths: MuonsFeaturesPaths = MuonsFeaturesPaths()
    pulses_paths: PulsesFeaturesPaths = PulsesFeaturesPaths()
    geom_path: str = "<ParticleType>/clusters_centers/data"
    

# UNIT TEST
if __name__=='__main__':
    import unittest
    from pprint import pprint

    class TestToDictWithVars(unittest.TestCase):
        def test_to_dict_with_vars(self):
            # Create an instance of H5Paths
            h5_paths = H5Paths()

            # Define the test particle type and file number
            particle_type = "test_particle"
            filenum = 42

            # Call the method under test
            result = h5_paths.to_dict_with_vars(particle_type, filenum)
            pprint(result, depth=3)

            # Check some key replacements in the result
            self.assertIn("ev_paths", result)
            self.assertIn("ind_mu_paths", result)
            self.assertIn("pulses_paths", result)
            self.assertIn("geom_path", result)

            # Verify replacements
            self.assertEqual(result["geom_path"], f"{particle_type}/clusters_centers/data")
            self.assertEqual(result["ev_paths"]["PrimeTheta"], f"{particle_type}/prime_prty/part_{filenum}/data.0")
            self.assertEqual(result["ind_mu_paths"]["RespMuTheta"], f"{particle_type}/muons_prty/individ/part_{filenum}/data.0")
            self.assertEqual(result["pulses_paths"]["PulsesAmpl"], f"{particle_type}/raw/data/part_{filenum}/data.0")

            # Check that all placeholders are replaced
            for key, value in result.items():
                if isinstance(value, str):
                    self.assertNotIn("<ParticleType>", value)
                    self.assertNotIn("<FileNum>", value)
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        self.assertNotIn("<ParticleType>", subvalue)
                        self.assertNotIn("<FileNum>", subvalue)

    unittest.main()
