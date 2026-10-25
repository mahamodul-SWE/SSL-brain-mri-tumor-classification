# Notebook Run Guide

Notebook: `Brain_Tumor_SSL_SimSiam_SE_GradCAM.ipynb`

## 1. Prepare the environment

- Install **Anaconda/Miniconda** and open JupyterLab or Jupyter Notebook.
- Create and activate your project environment.
- Install a PyTorch/torchvision build compatible with your system (and CUDA, if using an NVIDIA GPU) using the official PyTorch installation selector.
- Run this in a notebook cell to install the remaining packages:

```python
%pip install kagglehub numpy pandas scikit-learn matplotlib opencv-python pillow tqdm
```

Restart the kernel after installation if prompted.

## 2. Prepare the dataset

The notebook expects the public **Brain Tumor MRI Dataset** with `Training` and `Testing` folders. Each contains class folders for `glioma`, `meningioma`, `pituitary`, and `notumor`.

In **Section 3 — Configuration**, choose one option:

- **Dataset already downloaded:** set `AUTO_DOWNLOAD = False` and change `LOCAL_DATASET_ROOT` to the path of your extracted dataset folder.
- **Download from Kaggle automatically:** set `AUTO_DOWNLOAD = True`. Kaggle authentication may be required.

Example local path:

```python
LOCAL_DATASET_ROOT = Path(r"D:\Your\Dataset\brain-tumor-mri-dataset")
```

## 3. Test first, then run the experiment

1. In the configuration cell, set `QUICK_RUN = True`.
2. Use **Restart Kernel and Run All Cells** to check that setup, dataset paths, and code work. This is only a short smoke test, not a final experiment.
3. Set `QUICK_RUN = False`.
4. Restart the kernel and choose **Run All Cells** again. The full training can take considerable time, especially without a CUDA-enabled GPU.

For the proposed method, retain these settings:

```python
RUN_SSL = True
USE_SE = True
CONSISTENCY_WEIGHT = 0.20
```

The notebook uses `NUM_WORKERS = 0`, a safe default for Windows/Jupyter.

## 4. Find the outputs

Results are saved under:

```text
outputs/
└── D_Proposed_SE_SimSiam_WSC/
    ├── checkpoints/
    ├── figures/
    ├── config.json
    ├── test_split.csv
    └── ...
```

The notebook saves the best classifier as `checkpoints/best_classifier.pth` and produces evaluation results, plots, per-image predictions, and Grad-CAM examples as the relevant cells run.

## Important notes

- Run cells **in order**, from top to bottom.
- Keep the original `Testing` split untouched for final evaluation; the notebook uses the `Training` folder to create train/validation splits.
- For ablation studies, change the experiment switches in the configuration cell, give each experiment a distinct `EXPERIMENT_NAME`, then restart and run all cells.
- This notebook is for academic research. The model and outputs are **not clinically validated** and must not be used for medical diagnosis or treatment decisions.
