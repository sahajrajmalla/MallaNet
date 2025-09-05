# MallaNet: Residual Branch-Merge CNN with Homogeneous Filter Capsules for Devanagari Recognition

## Overview
MallaNet is a deep learning model designed for handwritten Devanagari character recognition, achieving a benchmark test accuracy of 99.71% on the Devanagari Handwritten Character Dataset (DHCD). This repository contains the implementation of MallaNet, a Residual Enhanced Branching and Merging Convolutional Neural Network with Homogeneous Filter Capsules (HFCs), extending the Branching and Merging Convolutional Network with Homogeneous Vector Capsules (BMCNNwHVCs). The model integrates optimized residual blocks, refined HFC layers, and a merging layer to capture multi-scale features and preserve spatial hierarchies, addressing the complexities of the Devanagari script’s 46 character classes.

This repository provides the complete codebase, including model implementation, training and evaluation scripts, hyperparameter tuning results, and visualizations, alongside the manuscript submitted for publication. MallaNet supports applications in optical character recognition (OCR) for regional scripts, facilitating document digitization and cultural preservation.

## Repository Structure
The repository is organized as follows:

```
.
├── data
│   ├── extracted        # Preprocessed dataset (resized to 32x32, normalized)
│   └── raw              # source of the dataset in a text file
├── experiments
│   ├── devanagari       # Experiment logs and results for Devanagari dataset (ensemble/hvc/one_model)
│   └── english          # Experiment logs for English MNIST dataset (ensemble/one_model)
├── models
│   └── best_model.pth   # Trained MallaNet model weights
├── notebooks
│   ├── MallaNet_colab.ipynb  # Jupyter notebook for training and evaluation
│   ├── plots            # Directory for storing generated plots
│   ├── trail_and_error  # Experimental notebooks for hyperparameter tuning
│   └── viz.ipynb        # Notebook for generating visualizations
├── plots
│   ├── accuracy_curves.png    # Training and validation accuracy curves
│   ├── config_comparison.png  # Comparison of hyperparameter configurations
│   ├── confusion_matrix.png   # Confusion matrix for test set
│   └── loss_curves.png        # Training and validation loss curves
├── results
│   ├── epoch_logs.csv        # Epoch-wise training and validation metrics
│   ├── hyperparam_results.csv # Hyperparameter tuning results
│   └── test_metrics.csv       # Per-class test metrics (precision, recall, F1-score)
├── src
│   ├── __init__.py      # Package initialization
│   ├── main.py          # Main script for training MallaNet
│   └── test.py          # Script for evaluating the trained model
├── LICENSE              # License file (MIT License)
├── README.md            # This file
└── requirements.txt     # Python dependencies
```

## Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA support (optional for training, recommended for performance)
- Google Colab with T4 GPU (for replication of training environment)

### Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/sahajrajmalla/MallaNet.git
   cd MallaNet
   ```

2. **Install Dependencies**:
   Install the required Python packages using the provided `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

   Typical dependencies include:
   - `torch` (PyTorch with CUDA support for GPU training)
   - `torchvision` (for dataset handling and transformations)
   - `numpy`, `pandas`, `scikit-learn` (for data processing and evaluation)
   - `matplotlib`, `seaborn` (for visualizations)
   - See `requirements.txt` for the complete list.

3. **Download the Dataset**:
   The Devanagari Handwritten Character Dataset (DHCD) is publicly available at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Devanagari+Handwritten+Character+Dataset). Download and extract it to the `data/raw/` directory, or use the provided preprocessing scripts to organize the dataset into `data/extracted/` (resized to 32x32 pixels, normalized to [-1, 1]).

4. **Optional: Pre-trained Model**:
   The pre-trained MallaNet model (`best_model.pth`) is provided in the `models/` directory. If you wish to train from scratch, follow the training instructions below.

## Usage

### Training MallaNet
To train the MallaNet model on the DHCD:
1. Ensure the dataset is preprocessed and available in `data/extracted/`.
2. Run the main training script:
   ```bash
   python src/main.py
   ```
   This script uses the optimal hyperparameters:
   - Learning rate: 0.0005
   - Batch size: 128
   - Dropout rate: 0.0
   - Label smoothing: 0.1
   - Optimizer: AdamW (weight decay: 0.0001)
   - Epochs: Up to 100 with early stopping
   - Learning rate scheduler: ReduceLROnPlateau (factor=0.5, patience=5)

   Training logs and metrics are saved to `results/epoch_logs.csv`, and the best model is saved as `models/best_model.pth`.

3. Alternatively, use the `MallaNet_colab.ipynb` notebook in Google Colab for an interactive training experience with a T4 GPU.

### Evaluating MallaNet
To evaluate the trained model on the DHCD test set:
```bash
python src/test.py
```
This script loads `best_model.pth` and computes test metrics (accuracy, precision, recall, F1-score), saving results to `results/test_metrics.csv`. The confusion matrix and F1-score visualizations are saved in the `plots/` directory.

### Visualizations
To generate visualizations (e.g., accuracy/loss curves, confusion matrix, F1-score bar chart):
1. Open `notebooks/viz.ipynb` in Jupyter or Google Colab.
2. Run the notebook to produce plots saved in the `plots/` directory.

### Hyperparameter Tuning
The repository includes results from a grid search over hyperparameters (learning rates, batch sizes, dropout rates, label smoothing values) in `results/hyperparam_results.csv`. To replicate or extend the tuning process, refer to the notebooks in `notebooks/trail_and_error/`.

## Model Architecture
MallaNet extends the BMCNNwHVCs framework with:
- **Residual Blocks**: Three convolutional blocks with residual connections (128, 256, 512 channels) to mitigate vanishing gradients.
- **Homogeneous Filter Capsule (HFC) Layers**: Three HFC layers capture multi-scale spatial hierarchies for the 46 Devanagari classes.
- **Merging Layer**: Combines logits from HFC layers using learnable weights for robust classification.
- Total parameters: 17,320,579.

The model achieves a test accuracy of 99.71% on the DHCD, surpassing prior benchmarks (e.g., 99.16% by Masrat et al., 98.47% by Acharya et al.).

## Dataset
The DHCD consists of 92,000 grayscale images (32x32 pixels) across 46 classes (10 digits, 36 consonants), split into 78,200 training and 13,800 testing images. Data augmentation (random rotations, affine transformations, Gaussian noise) enhances robustness to handwriting variability.

## Results
- **Test Accuracy**: 99.71%
- **Macro-Average F1-Score**: 99.71%
- **Test Loss**: 0.7033
- **Key Visualizations**:
  - Confusion matrix (`plots/confusion_matrix.png`)
  - F1-score bar chart (`plots/f1_score_bar_chart.png`)
  - Accuracy/loss curves (`plots/accuracy_curves.png`, `plots/loss_curves.png`)

Detailed per-class metrics are available in `results/test_metrics.csv`, and hyperparameter tuning results are in `results/hyperparam_results.csv`.

## Reproducing Results
To reproduce the results:
1. Set up the environment as described in the Installation section.
2. Preprocess the DHCD and place it in `data/extracted/`.
3. Run `src/main.py` for training or `src/test.py` for evaluation.
4. Use `notebooks/viz.ipynb` to generate visualizations.
5. Ensure a fixed random seed (42) for reproducibility.

## Citation
If you use MallaNet or this repository in your research, please cite:

```bibtex
@article{malla2025mallanet,
  title={MallaNet: Residual Branch-Merge CNN with Homogeneous Filter Capsules for Devanagari Recognition},
  author={Malla, Sahaj Raj},
  journal={TBD},
  year={2025},
  publisher={TBD}
}
```

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For questions or access to code during the review process, contact:
- Sahaj Raj Malla: [sm03200822@student.ku.edu.np](mailto:sm03200822@student.ku.edu.np)
- GitHub: [https://github.com/sahajrajmalla/MallaNet](https://github.com/sahajrajmalla/MallaNet)

## Acknowledgments
- The Devanagari Handwritten Character Dataset (DHCD) from the UCI Machine Learning Repository.
- Google Colab for providing computational resources (T4 GPU).
- PyTorch and related libraries for enabling efficient model development.