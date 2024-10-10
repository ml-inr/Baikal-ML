from typing import Optional

import numpy as np
import pandas as pd
import uproot as ur
import awkward as ak
from functools import lru_cache

from data.root_manager.root_paths import RootPaths, BaseFeaturePaths
from data.root_manager.constants import Constants as Cnst


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
    
    def __init__(self, root_file: ur.reading.ReadOnlyDirectory, 
                 paths: RootPaths = RootPaths(), prefix: str = ''):
        self.rf = root_file
        self.paths = paths
        self.prefix = prefix
        self.PulsesN = self.rf[self.paths.ev_paths.PulsesN]
        if not self._check_file():
            raise BadFile("no valid events in file")
        self.start = self.get_start_index()  
        self.events = None
        self.data = None
    
    # def _get_dict_from_paths(self, paths: BaseFeaturePaths) -> dict:
    #     """
    #     Reads root data as an arrays of objects
    #     and converts to Python dict.
    #     Returns: dict[str][object]
    #     """
    #     result_dict = {}
    #     for name, path in paths.to_dict().items():
    #         value = self.rf[path].array(library="np")[self.start:].astype(object)
    #         result_dict[name] = value
    #     return result_dict
    
    @lru_cache(maxsize=None)
    def _get_array(self, path):
        return self.rf[path].array(library="np", entry_start=self.start)
    # def _get_array(self, path):
    #     return self.rf[path].array(library="np")[self.start:].astype(object)
    
    def _get_dict_from_paths(self, paths: BaseFeaturePaths) -> dict:
        """
        Reads root data as an arrays of objects
        and converts to Python dict.
        Returns: dict[str][object]
        """
        result_dict = {}
        for name, path in paths.to_dict().items():
            value = self._get_array(path)
            result_dict[name] = value
        return result_dict
    
    def _add_prefix_to_id(self, ids: pd.Series):
        ids = self.prefix + ids.astype(str)
        return ids
    
    def _flatten_df(self, df: pd.DataFrame) -> pd.DataFrame:
        N = df.iloc[:, 0].apply(len).to_numpy()
        ev_ids = np.repeat(np.arange(len(df)), N)
        element_ids = np.concatenate([np.arange(n) for n in N])
        
        flat_dict = {feature: np.concatenate(df[feature].to_list()) for feature in df.columns.to_list()}
        flat_dict['ev_id'] = ev_ids
        flat_dict['element_id'] = element_ids
        
        return pd.DataFrame(flat_dict)

    # def _flatten_df(self,df: pd.DataFrame) -> pd.DataFrame:
    #     df['N'] = df.iloc[:, 0].apply(len)
    #     df['ev_id'] = df.reset_index()['index'].apply(lambda x: [x]*df['N'].iloc[x])
    #     df['element_id'] = df.reset_index()['index'].apply(lambda x: [*range(df['N'].iloc[x])])
    #     df = df.drop(columns=['N'])
    #     flat_dict = {feature: np.concatenate(df[feature].to_list()) for feature in df.columns.to_list()}
    #     return pd.DataFrame(flat_dict)
        
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
        df_ev['ev_id'] = self._add_prefix_to_id(df_ev['ev_id'])
        for column in ['PulsesN','PrimeNuclN','ResponseMuN']:
            df_ev[column] = df_ev[column].astype(np.int32)
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
                ev_id[int],
                is_signal[bool]
                ]
        """
        result_dict = self._get_dict_from_paths(self.paths.pulses_paths)
        df_flat = self._flatten_df(pd.DataFrame(result_dict))
        df_flat = df_flat.sort_values(['ev_id', 'PulsesTime'])
        
        df_flat['ev_id'] = self._add_prefix_to_id(df_flat['ev_id'])
        df_flat = df_flat.drop(columns=['element_id'])
        df_flat['PulsesChID'] = df_flat['PulsesChID'].astype(np.int16)
        df_flat['is_signal'] = (df_flat['PulsesFlg']!=0).astype(bool)
        df_flat['mu_local_id'] = (df_flat['PulsesFlg']%1_000_000-1).astype(np.int16)
        df_flat['cluster_id'] = (df_flat['PulsesChID']//Cnst.CHANNEL_DIVISOR).astype(np.int8)
        df_flat['string_id'] = (df_flat['PulsesChID']//Cnst.STRING_DIVISOR).astype(np.int8)
        return df_flat
    
    def read_ind_mu_as_df(self):
        result_dict = self._get_dict_from_paths(self.paths.ind_mu_paths)
        df_flat = self._flatten_df(pd.DataFrame(result_dict))
        df_flat['ev_id'] = self._add_prefix_to_id(df_flat['ev_id'])
        df_flat = df_flat.rename(columns={'element_id': 'mu_local_id'})
        return df_flat

    def read_OM_coords(self):
        coords_array = np.array(ak.unzip(self.rf[self.paths.geom_path].array()))[:,self.start]
        df = pd.DataFrame(coords_array.T, columns=['X','Y','Z'], dtype=np.float64)
        df = df.reset_index()
        df = df.rename(columns={'index': 'PulsesChID'})
        df['PulsesChID'] = df['PulsesChID'].astype(np.int16)
        df['cluster_id'] = df['PulsesChID']//Cnst.CHANNEL_DIVISOR
        df['string_id'] = df['PulsesChID']//Cnst.STRING_DIVISOR
        #   Add info about clusters centers
        cl_centers = df[['X','Y','Z', 'cluster_id']].groupby(['cluster_id']).mean()
        cl_centers = cl_centers.rename(columns={'X': 'Xc', 'Y': 'Yc', 'Z': 'Zc'})
        df = df.join(cl_centers, on='cluster_id')
        #   Add coords, relatively to the centers
        for coord in ['X','Y','Z']:
            df[f"{coord}rel"] = df[f"{coord}"]-df[f"{coord}c"]
        return df


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
        
        