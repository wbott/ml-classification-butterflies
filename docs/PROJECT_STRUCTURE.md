# Project Structure

This document describes the organization of the butterfly classification project.

## Directory Structure

```
ml-classification-butterflies/
├── src/                          # Source code
│   ├── data/                     # Data handling modules
│   │   ├── dataset.py           # Dataset loading and preprocessing
│   │   └── augmentation.py      # Image augmentation utilities
│   ├── models/                   # Model definitions
│   │   ├── base_model.py        # Abstract base model class
│   │   ├── fnn.py               # Feedforward Neural Network
│   │   ├── cnn.py               # Custom CNN
│   │   └── vgg_transfer.py      # VGG16 transfer learning
│   ├── training/                 # Training utilities
│   │   ├── trainer.py           # Training loops and callbacks
│   │   └── evaluation.py       # Model evaluation metrics
│   └── utils/                    # Utility functions
│       ├── config.py            # Configuration management
│       └── visualization.py     # Plotting and visualization
├── data/                         # Data directory
│   ├── raw/                     # Raw dataset files
│   ├── processed/               # Preprocessed data
│   └── external/                # External data sources
├── models/                       # Trained model artifacts
│   ├── checkpoints/             # Training checkpoints
│   └── final/                   # Final trained models
├── notebooks/                    # Jupyter notebooks
│   ├── exploratory/            # Exploratory data analysis
│   ├── experiments/            # Experiment notebooks
│   └── ml-classification-butterflies.ipynb
├── config/                       # Configuration files
│   ├── model_config.yaml       # Model configurations
│   └── training_config.yaml    # Training parameters
├── scripts/                      # Utility scripts
│   ├── download_data.py        # Data download automation
│   ├── train_model.py          # Training script
│   └── setup.py                # Environment setup
├── tests/                        # Test suite
│   ├── test_model_performance.py
│   ├── test_data_validation.py
│   ├── test_image_processing.py
│   └── test_edge_cases.py
├── docs/                         # Documentation
│   ├── api/                    # API documentation
│   ├── tutorials/              # Usage tutorials
│   └── PROJECT_STRUCTURE.md   # This file
└── deployment/                   # Deployment files
    ├── docker/                 # Docker configurations
    └── web_app/               # Web application
```

## Module Descriptions

### src/data/
Contains data handling and preprocessing utilities:
- `dataset.py`: Main dataset class for loading and creating data generators
- `augmentation.py`: Custom image augmentation functions

### src/models/
Contains model definitions:
- `base_model.py`: Abstract base class with common functionality
- `vgg_transfer.py`: VGG16 transfer learning implementation
- `fnn.py`: Feedforward neural network (planned)
- `cnn.py`: Custom CNN architecture (planned)

### src/utils/
Utility functions and helpers:
- `config.py`: Configuration management using YAML files
- `visualization.py`: Plotting and visualization utilities (planned)

### config/
Configuration files in YAML format:
- `model_config.yaml`: Model architectures and hyperparameters
- `training_config.yaml`: Training-specific configurations (planned)

### scripts/
Executable scripts for common tasks:
- `train_model.py`: Main training script with CLI interface
- `download_data.py`: Automated data download (planned)
- `setup.py`: Environment setup script (planned)

## Usage Examples

### Training a Model
```bash
python scripts/train_model.py --model vgg16 --train-csv data/raw/Training_set.csv --epochs 50
```

### Running Tests
```bash
python run_tests.py
```

### Configuration
Edit `config/model_config.yaml` to modify model parameters and training settings.

## Benefits of This Structure

1. **Modularity**: Each component has a specific responsibility
2. **Reusability**: Models and utilities can be easily reused
3. **Testability**: Clear separation enables comprehensive testing
4. **Maintainability**: Easy to understand and modify
5. **Scalability**: Structure supports project growth
6. **Configuration-driven**: Easy to experiment with different parameters