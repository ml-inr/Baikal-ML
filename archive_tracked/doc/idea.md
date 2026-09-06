# Neural Network Research Project

## Project Overview

This research project focuses on developing neural networks for neutrino detection using the Baikal underwater neutrino telescope data, with emphasis on binary classification, modularity, reproducibility, and experimental flexibility.

## Project Goals

- **Data Management**: Build a robust data manager capable of reading Baikal detector `.root` files and saving data into configurable HDF5 datasets for training, testing, and validating neural network models
- **Neutrino Detection**: Develop and train neural network models for binary classification of neutrino-induced events vs atmospheric muon background
- **Time-Series Modeling**: Handle variable-length sequences of detector hits with temporal and spatial features
- **Framework Design**: Build a modular, reproducible research framework that enables systematic experimentation with different neural network architectures

## Project Structure

The project is organized using standard Python package structure under `src/`:

### `data_manager/`
Module responsible for ROOT file processing and HDF5 conversion. Handles the conversion from Baikal detector ROOT files to structured HDF5 datasets suitable for machine learning workflows with physics-specific processing (time residuals, clustering, coordinate transformations).

### `src/models/` 
Neural network architectures for time-series classification. Contains PyTorch models specialized for variable-length sequences and binary classification of neutrino events.

### `src/training/`
Training infrastructure including training loops, optimization algorithms, learning rate scheduling, and metrics calculation for binary classification tasks.

### `src/data/`
PyTorch dataset classes and data loading utilities. Handles variable-length sequences, batching with padding/masking, and data augmentation for neutrino detection.

### `inference/`
Model evaluation tools, inference utilities, and prediction pipelines for comprehensive model assessment and deployment.

## Configuration Management

All models and datasets should be perfectly configurable and reproducible. The configuration system needs to be designed to support:
- Experiment reproducibility
- Parameter sweeps and hyperparameter optimization
- Easy switching between different model configurations
- Dataset configuration for various preprocessing pipelines

*Note: The specific implementation approach for configuration management is still under consideration.*

## Technical Stack

### Core Dependencies
- **Python 3.10+**: Primary programming language
- **PyTorch**: Deep learning framework for model development and training
- **NumPy**: Fundamental numerical computing library
- **SciPy**: Scientific computing utilities

### Data Handling
- **PyROOT**: Interface for ROOT data analysis framework
- **PyHDF5**: HDF5 file format handling for efficient data storage
- **awkward**: Array manipulation for complex data structures
- **pandas**: Data manipulation and analysis

### Visualization and Development
- **matplotlib**: Plotting and visualization
- **jupyter**: Interactive development and experimentation

## Research Focus

The project emphasizes the following principles for neutrino detection:

- **Binary Classification**: Distinguish neutrino-induced events (signal) from atmospheric muon background using time-series neural networks
- **Variable-Length Sequences**: Handle detector hit sequences with 35-200+ hits per event using proper padding/masking techniques
- **Physics-Informed Features**: Utilize 5D hit features (amplitude, time, x/y/z coordinates) with proper coordinate transformations and time centering
- **Reproducible Experiments**: All experiments fully reproducible with seed management and YAML configuration tracking
- **KISS Principles**: Keep implementations simple, avoid premature optimization, focus on working solutions

## Development Environment

- **IDE**: Visual Studio Code for primary development
- **Interactive Research**: Jupyter notebooks within VSCode for testing new libraries, prototyping ideas, and conducting exploratory data analysis
- **Version Control**: Git for tracking code changes and experiment versions

## Next Steps

1. ✅ Implement the ROOT → HDF5 data conversion pipeline
2. ✅ Create PyTorch dataset for variable-length neutrino/muon sequences  
3. ✅ Establish YAML configuration management system
4. 🚀 **Current**: Design and train simple neural network models (MLPs, RNNs)
5. 🔄 Implement training loops with binary classification metrics
6. 📈 Set up experiment tracking and model evaluation pipelines