from root_file_cfg import RootPathsConfig as PathsCfg

import numpy as np
import uproot as ur
import awkward as ak

class RootReader:
    def __init__(self, path2root: str):
        self.path = path2root
        
    @staticmethod
    def _check_file(rf_path: str) -> bool:
        try:
            with ur.open(rf_path) as rf:
                ev_num = rf['Events/BEvent./BEvent.fPulseN'].num_entries
                return ev_num > 1 or (ev_num == 1 and rf['Events/BEvent./BEvent.fPulseN'].array(library="np", entry_start=0, entry_stop=1)[0] != 0)
        except Exception:
            return False
        
    