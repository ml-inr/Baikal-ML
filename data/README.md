# Data Processing Module

This module provides a pipeline for processing data from `.root` files, converting them into structured formats for machine learning tasks. The process efficiently handles large datasets by reading, processing, and batching data. Below is an overview of the pipeline and usage examples.

## Pipeline Overview

### **INFO**: in purpose of Neural Network training only `BatchGenerator` class with its configs `BatchGeneratorConfig` is needed. Usage exaple is at step 5.

1. **Raw `.root` Files**:
    - The process starts with `.root` files containing raw data.

2. **RootReader** (`data/root_manager/reader/root_reader.py`):
    - Converts `.root` files into Polars DataFrames. It maps paths inside `.root` files to a DataFrame schema.
    - **Usage Example**:
      ```python
      from data.root_manager.reader.root_reader import RootReader
      
      root_reader = RootReader("path/to/file.root")
      df = root_reader.read_all_data_as_df()
      print(df.head())
      ```

3. **Processor** (`data/root_manager/processor.py`):
    - Utilizes `RootReader` to process multiple `.root` files in parallel.
    - Applies settings specified in `ProcessorConfig` (`data/root_manager/settings.py`).
    - **Usage Example**:
      ```python
      from data.root_manager.settings import ProcessorConfig
      from data.root_manager.processor import Processor
      
      # default settings. Examine `data/root_manager/settings.py` to learn more.
      config = ProcessorConfig() 
      processor = Processor(paths_to_roots=["file1.root", "file2.root"], config=config)
      processed_data = processor.process()
      print(processed_data.head())
      ```

4. **ChunkGenerator** (`data/root_manager/chunk_generator.py`):
    - Generates chunks of processed data, mixing information from different `.root` files.
    - **Usage Example**:
      ```python
      from data.root_manager.settings import ChunkGeneratorConfig
      from data.root_manager.chunk_generator import ChunkGenerator
      
      # default settings. Examine `data/root_manager/settings.py` to learn more.
      chunk_gen = ChunkGenerator(cfg=ChunkGeneratorConfig()) 
      for chunk in chunk_gen.get_chunks():
          print(chunk.head())
      ```

5. **BatchGenerator** (`data/batch_generator.py`):
    - Uses `ChunkGenerator` to create data batches. Supports data normalization and augmentation.
    - Returns inputs and targets for training as a torch.Tensor type.
    - **Usage Example**:
        ```python
        from data.settings import BatchGeneratorConfig
        from data.batch_generator import BatchGenerator

        # Read paths
        path_mu = "/net/62/home3/ivkhar/Baikal/data/initial_data/MC_2020/muatm/root/all/"
        path_nuatm = "/net/62/home3/ivkhar/Baikal/data/initial_data/MC_2020/nuatm/root/all/"
        path_nu2 = "/net/62/home3/ivkhar/Baikal/data/initial_data/MC_2020/nue2_100pev/root/all/"
        def explore_paths(p: str, start: int, stop: int):
            files = os.listdir(p)[start:stop]
            return sorted([f"{p}{file}" for file in files])
        mu_paths = explore_paths(path_mu, 0, 800)
        nuatm_paths = explore_paths(path_nuatm, 0, 1000)
        nu2_paths = explore_paths(path_nu2, 0, 60)
        all_paths = mu_paths + nuatm_paths + nu2_paths

        # Default settings. To learn more possibilities, examine `data.root_manager.settings` and `root_manager.settings`
        cfg = BatchGeneratorConfig()
        train_data = BatchGenerator(root_paths=all_paths, cfg=cfg)
        batches = train_data.get_batches()
        
        # create generator
        for batch in batches:
            inputs, targets = batch
            print(inputs.shape, targets.shape)
            break
        ```

## Configuration

The behavior of the pipeline can be customized using configuration classes found in `data/settings.py` and `data/root_manager/settings.py`. Users can define settings for normalization, augmentation, and processing as per their requirements.

## Installation
Ensure you have the required dependencies installed. You can use the provided scripts:
```bash
bash miniconda_setup.sh
bash install_venv.sh