import os

from config_reader import ConfigReader
from sqlite_database_handler import SQLiteDatabaseHandler
from data.root_manager.root_file_processor import RootFileProcessor
from data.root_manager.cluster_manager import ClusterManager


class MainProcessor:
    def __init__(self, config_file: str):
        config_reader = ConfigReader(config_file)
        self.config = config_reader.get_config()
        self.root_processor = RootFileProcessor(self.config)
        self.cluster_manager = ClusterManager(self.config)
        self.db_handler = SQLiteDatabaseHandler(self.config['output']['sqlite_db_name'])

    def process(self):
        root_files_dir = self.config['input']['MC_dir_path']
        root_files = [f for f in os.listdir(root_files_dir) if f.endswith('.root')]

        cluster_centers = None

        for root_file in root_files:
            rf_path = os.path.join(root_files_dir, root_file)

            if cluster_centers is None:
                cluster_centers = self.cluster_manager.get_cluster_centers(rf_path)

            processed_data = self.root_processor.process_root_file(rf_path, cluster_centers)

            if processed_data:
                self.db_handler.insert_data(processed_data)

        self.db_handler.close()


if __name__ == "__main__":
    config_path = f"{os.path.dirname(os.path.realpath(__file__))}/root2sqlite_config.yaml"  # Update this to your configuration file path
    print(config_path)
    processor = MainProcessor(config_path)
    processor.process()