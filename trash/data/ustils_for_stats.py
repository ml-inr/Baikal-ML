import logging
from dataclasses import dataclass

import polars as pl

try:
    from data.root_manager.processor import Processor
    from data.root_manager.settings import ProcessorConfig
except ImportError:
    from root_manager.processor import Processor
    from root_manager.settings import ProcessorConfig


class StatsCollector:

    def __init__(
        self,
        mu_paths: list[str],
        nuatm_paths: list[str],
        nu2_paths: list[str],
        proc_cfg: ProcessorConfig,
    ) -> None:
        self.proc_cfg = proc_cfg
        self.mu_paths = mu_paths
        self.nuatm_paths = nuatm_paths
        self.nu2_paths = nu2_paths
        pass

    @staticmethod
    def _calc_sums(
        df: pl.DataFrame, field_name: str, num_files_local: int, num_files_total: int
    ) -> tuple[float, float, float]:
        """
        Calculate the number of hits, sum, and squared sum for a given field in the DataFrame.

        Args:
            df (pl.DataFrame): Input Polars DataFrame.
            field_name (str): The field to calculate sums for.
            num_files_local (int): Number of local files processed.
            num_files_total (int): Total number of files.

        Returns:
            tuple[float, float, float]: Number of hits, sum, and squared sum.
        """
        num_hits = df[field_name].explode().shape[0] / num_files_local * num_files_total
        S = df[field_name].explode().sum() / num_files_local * num_files_total
        S2 = df[field_name].explode().pow(2).sum() / num_files_local * num_files_total
        logging.debug(
            "Calculated sums for field '%s': num_hits=%.2f, sum=%.2f, squared_sum=%.2f",
            field_name,
            num_hits,
            S,
            S2,
        )
        return num_hits, S, S2

    def _estimate_mean_and_std(
        self,
        df_mu: pl.DataFrame,
        df_nuatm: pl.DataFrame,
        df_nu2: pl.DataFrame,
        field_name: str,
        num_files_local: int,
    ) -> tuple[float, float]:
        """
        Estimate mean and standard deviation for a given field across multiple datasets.
        """
        logging.info(f"Estimating mean and std for field: {field_name}")
        mu_hits, S_mu, S2_mu = self._calc_sums(df_mu, field_name, num_files_local, len(self.mu_paths))
        nuatm_hits, S_nuatm, S2_nuatm = self._calc_sums(df_nuatm, field_name, num_files_local, len(self.nuatm_paths))
        nu2_hits, S_nu2, S2_nu2 = self._calc_sums(df_nu2, field_name, num_files_local, len(self.nu2_paths))

        total_hits = mu_hits + nuatm_hits + nu2_hits
        mean = (S_mu + S_nuatm + S_nu2) / total_hits
        std = ((S2_mu + S2_nuatm + S2_nu2) / total_hits - mean**2) ** 0.5

        logging.info("Field '%s': Estimated Mean = %.4f, Std = %.4f", field_name, mean, std)
        return mean, std

    def get_stats(self, start: int = 0, stop: int = 50) -> None:
        """
        Collect statistics for datasets, including mean and standard deviation estimates.
        Estimation is made on a slice of root files from start to stop. If length of path-list less then stop - start, reads all files.
        
        Arguments:
            start: where to start slice from paths
            stop: where to stop slice from paths
        """
        assert stop > start, "Stop index must be greater than start index."
        # if stop - start > min(len(self.mu_paths), len(self.nuatm_paths), len(self.nu2_paths)):
        #     logging.warning("Range is too large for the available files. Adjusting to smaller range.")
        #     start = 0
        #     stop = min(len(self.mu_paths), len(self.nuatm_paths), len(self.nu2_paths))

        # Load processed data
        logging.info("Processing datasets from indices %d to %d.", start, stop)
        df_mu = (mu_proc := Processor(self.mu_paths[start:stop], self.proc_cfg)).process()
        df_nuatm = (nuatm_proc := Processor(self.nuatm_paths[start:stop], self.proc_cfg)).process()
        df_nu2 = (nu2_proc := Processor(self.nu2_paths[start:stop], self.proc_cfg)).process()

        # Estimate statistics
        self.mu_num_estimated = df_mu.shape[0] / min((stop - start),len(self.mu_paths)-start) * len(self.mu_paths)
        self.nuatm_num_estimated = df_nuatm.shape[0] / min((stop - start),len(self.nuatm_paths)-start) * len(self.nuatm_paths)
        self.nu2_num_estimated = df_nu2.shape[0] / min((stop - start),len(self.nu2_paths)-start) * len(self.nu2_paths)

        self.mu_filter_koef = mu_proc.filter_koef
        self.nuatm_filter_koef = nuatm_proc.filter_koef
        self.nu2_filter_koef = nu2_proc.filter_koef

        self.mu_nu_ratio = self.mu_num_estimated / (self.nu2_num_estimated + self.nuatm_num_estimated)
        self.nuatm_nu2_ratio = self.nuatm_num_estimated / self.nu2_num_estimated

        logging.info("Calculating mean and std for Q and t")
        self.Q_mean, self.Q_std = self._estimate_mean_and_std(df_mu, df_nuatm, df_nu2, "PulsesAmpl", stop - start)
        self.t_mean, self.t_std = self._estimate_mean_and_std(df_mu, df_nuatm, df_nu2, "PulsesTime", stop - start)

        logging.info(
            "Collected dataset statistics successfully. Ratios - Mu/Nu: %.4f, NuAtm/Nu2: %.4f",
            self.mu_nu_ratio,
            self.nuatm_nu2_ratio,
        )

        # TODO: make saver to save it into a file. Should contains as input proc_cfg + paths as calculated stats

    def return_stats(self):
        return {k: v for k,v in vars(self).items() if not k.endswith('paths')}
    
    def print_stats(self):
        print(f"Particles numbers\n\tMu: {self.mu_num_estimated},\n\tNuatm: {self.nuatm_num_estimated},\n\tNue2: {self.nu2_num_estimated}\n")
        print(f"Ratios:\n\tMu to Nu: {self.mu_nu_ratio},\n\tNuatm to Nue2: {self.nuatm_nu2_ratio}\n")
        print(f"Ratios:\n\tMu to Nu: {self.mu_nu_ratio},\n\tNuatm to Nue2: {self.nuatm_nu2_ratio}\n")
        print(f"Normilization params:\n\tQ: mean={self.Q_mean}, std={self.Q_std},\n\tt: mean={self.t_mean}, std={self.t_std},\n")