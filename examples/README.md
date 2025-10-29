# OPTICUS Examples

This directory contains example notebooks demonstrating how to use the OPTICUS package for ice scattering length prediction.

## Available Examples

### 1. ViT Training Example (`vit_training_example.ipynb`)
- Complete workflow for training Vision Transformer models
- Data loading, model creation, training, and evaluation
- Comprehensive visualizations and analysis

### 2. ResNet Training Example (`resnet_training_example.ipynb`)
- Complete workflow for training ResNet models
- Alternative CNN-based approach
- Comparison with ViT models

### 3. Model Evaluation Example (`model_evaluation_example.ipynb`)
- Load trained models from checkpoints
- Evaluate on different data splits
- Generate detailed analysis and visualizations
- Statistical analysis and performance comparison

### 4. Configuration-based Training (`config_based_training_example.ipynb`)
- Use YAML configuration files for experiments
- Support for different model types and tasks
- Easy experiment management and reproducibility

## Quick Start

1. **Install OPTICUS**:
   ```bash
   cd /path/to/OPTICUS
   pip install -e .
   ```

2. **Update Data Paths**:
   - Open any example notebook
   - Update the `hdf5_path` in the configuration section
   - Point to your HDF5 data file

3. **Run Examples**:
   - Start with `vit_training_example.ipynb` for ViT training
   - Use `model_evaluation_example.ipynb` to evaluate trained models
   - Try `config_based_training_example.ipynb` for configuration management

## Data Format

The examples expect HDF5 files with the following structure:
- `images`: (N, 500, 500) numpy array of grayscale images
- `labels`: (N,) numpy array of scattering length values

## Configuration

Each example includes configuration sections where you can:
- Set data paths
- Adjust model parameters
- Configure training settings
- Specify output directories

## Output

The examples generate:
- Trained model checkpoints in `checkpoints/`
- Evaluation plots in `plots/`
- Performance metrics and statistics

## Tips

- Start with smaller datasets for testing
- Adjust batch size based on GPU memory
- Use early stopping to prevent overfitting
- Monitor validation loss during training
- Compare different model architectures

## Support

For questions or issues, please refer to the main README.md or create an issue in the repository.
