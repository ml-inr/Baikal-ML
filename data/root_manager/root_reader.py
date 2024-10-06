from typing import Optional

import numpy as np
import pandas as pd
import uproot as ur

from data.root_manager.root_paths import RootPaths, BaseFeaturePaths


class BadFile(Exception):
    pass


class RootReader:
    
    def _check_file(self) -> bool:
        try:
            ev_num = self.PulsesN.num_entries
            return ev_num > 1 or (ev_num == 1 and self.PulsesN.array(library="np", entry_start=0, entry_stop=1)[0] != 0)
        except Exception:
            return False
    
    def get_start_index(self):
        test = self.rf['Events/BEvent./BEvent.fPulseN'].array(library="np", entry_start=0, entry_stop=1)[0]
        return 1 if test == 0 else 0
    
    def __init__(self, root_file: ur.reading.ReadOnlyDirectory, paths: RootPaths = RootPaths()):
        self.rf = root_file
        self.paths = paths
        self.PulsesN = self.rf[self.paths.ev_paths.PulsesN]
        if not self._check_file():
            raise BadFile("no valid events in file")
        self.start = self.get_start_index()  
        self.events = None
        self.data = None
    
    def _get_dict_from_paths(self, paths: BaseFeaturePaths) -> dict:
        """
        Reads root data as an arrays of objects
        and converts to Python dict.
        Returns: dict[str][object]
        """
        result_dict = {}
        for name, path in paths.to_dict().items():
            value = self.rf[path].array(library="np")[self.start:].astype(object)
            result_dict[name] = value
        return result_dict

    @staticmethod
    def _flatten_df(df: pd.DataFrame) -> pd.DataFrame:
        df['N'] = df.iloc[:, 0].apply(len)
        df['ev_id'] = df.reset_index()['index'].apply(lambda x: [x]*df['N'].iloc[x])
        df = df.drop(columns=['N'])
        flat_dict = {feature: np.concatenate(df[feature].to_list()) for feature in df.columns.to_list()}
        return pd.DataFrame(flat_dict)
        
    def read_events_as_df(self) -> pd.DataFrame:
        """
        Reads aggregated features about events as a whole
        and converts to pd.Dataframe.
        
        Returns:
            pd.Datarame[
                PulsesN[int],
                PrimeTheta[float], PrimePhi[float], PrimeEn[float], PrimeNuclN[int],
                ResponseMuN[int], BundleEnReg[float], 
                EventWeight[float],
                RespMuTheta[np.array[float]], RespMuPhi[np.array[float]],
                RespMuTrackX[np.array[float]], RespMuTrackY[np.array[float]], RespMuTrackZ[np.array[float]],
                RespMuDelay[np.array[float]], RespMuEn[np.array[float]],
                ev_id[int]
                ]
        """
        result_dict = self._get_dict_from_paths(self.paths.ev_paths)
        df_ev = pd.DataFrame(result_dict)
        df_ev['ev_id'] = df_ev.index
        return df_ev
    
    def read_pulses_as_df(self) -> pd.DataFrame:
        """
        Reads pulses features from root as array per event ->
        Unnests arrays and converts to one pd.Dataframe ->
        orders frame by event_id and PulsesTime.
        
        Returns:
            pd.Datarame[
                PulsesChID[int], 
                PulsesAmpl[float], 
                PulsesTime[float], 
                PulsesFlg[int], 
                ev_id[int]
                ]
        """
        result_dict = self._get_dict_from_paths(self.paths.pulses_paths)
        df_flat = self._flatten_df(pd.DataFrame(result_dict))
        df_flat = df_flat.sort_values(['ev_id', 'PulsesTime'])
        return df_flat
    
    def read_ind_mu_as_df(self):
        result_dict = self._get_dict_from_paths(self.paths.ind_mu_paths)
        df_flat = self._flatten_df(pd.DataFrame(result_dict))
        return df_flat
        
    def get_events(self) -> np.ndarray[float]:
        """_summary_

        Returns:
            np.ndarray[float]: array of shape (n,2+...). 
            Contains info about events: [
                    event_id,
                    primary_particle_type, primary_angles, primary_energy, 
                    [triggered_clusters_id], [clusters_centers], [num_of_hits_per_cluster],
                    num_of_muons, [muons_vertices], [muons_energies], [muons_angles]
                ]
        """
        if events is not None:
            return events

            
    
    def get_data(self) -> np.ndarray[float]:
        """
        I want all coords to be relative to the center of TELESCOPE!
        
        Returns:
            np.ndarray[float]: array of shape (N,8). 
            Contains hits: [IsSignal, t, x, y, z, Q, event_id, cluster_id, string_id, OM_id].
        """
        ...


if __name__=='__main__':
    '''
    Usage example
    '''
    eventss = []
    datas = []
    for path in ['/home/albert/Baikal2/data/mock_MC_2020/muatm/root/all/66470.root']:
        with open(path) as rf:
            rr = RootReader(rf)
            events = rr.get_events()
            data = rr.get_data()
        eventss.append(events), datas.append(data)
    events = np.concatenate(eventss)
    data = np.concatenate(datas)
    ...
        
        