import numpy as np
import uproot as ur
import awkward as ak
from typing import Optional, Dict


class ClusterManager:
    def __init__(self, config: Dict):
        self.path_geometry = config['root_paths']['geometry']
        
    @staticmethod
    def _get_start_index(rf) -> int:
        test = rf['Events/BEvent./BEvent.fPulseN'].array(library="np", entry_start=0, entry_stop=1)[0]
        return 1 if test == 0 else 0

    def get_cluster_centers(self, rf_path: str) -> Optional[np.ndarray]:
        try:
            with ur.open(rf_path) as rf:
                # Extract geometry info from the ROOT file
                st = self._get_start_index(rf)
                coordinates = np.array(ak.unzip(rf[self.path_geometry].array()))[:, st:]

                num_clusters = coordinates.shape[-1] // 288
                coordinates = coordinates.reshape(*coordinates.shape[:-1], num_clusters, 288)

                cl_centers = np.mean(coordinates, axis=(1, -1))
                return cl_centers.T

        except Exception as e:
            print(f"Error extracting cluster centers from {rf_path}: {str(e)}")
            return None

    
