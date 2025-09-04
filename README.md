# Lung Abnormality Segmentation & Classification from Chest X-rays

This repository contains code, notebooks, and resources for a deep learning project focused on detecting and segmenting lung abnormalities (COVID-19, Non-COVID infection, and healthy tissue) from chest X-ray images. The project explores both multi-class semantic segmentation and image-level classification approaches.

## Project Structure

<details>
<summary>Directory Tree</summary>
lung_abnormality_segmentation/ ├── config/ # Configuration files (e.g., config.yaml) ├── data/ │ ├── processed/ # Preprocessed datasets │ └── raw/ # Raw datasets (COVID-QU-Ex, ChestXray-Masks-and-Labels, etc.) ├── images/ │ ├── classification_model/ # Training/validation figures for classification │ └── segmentation_model/ # Training/validation figures for segmentation ├── logs/ │ ├── fit_classification/ # TensorBoard logs for classification runs │ ├── fit_multi_class_Iter_1/ # TensorBoard logs for segmentation (iteration 1) │ ├── fit_multi_class_Iter_2/ # TensorBoard logs for segmentation (iteration 2) │ └── fit_multiclass_segmentation/ # Additional segmentation logs ├── notebooks/ │ ├── 01_Introduction_&_Initial_Model_Details.ipynb │ ├── 02_Segmentation_Model_Analysis.ipynb │ ├── 03_Classification_Model_Analysis.ipynb │ └── .ipynb_checkpoints/ ├── saved_models/ │ ├── saved_models_classification/ # Classification model checkpoints │ ├── saved_models_classification_iter2/ # Classification model (iteration 2) │ ├── saved_models_classification_iter3/ # Classification model (iteration 3) │ └── saved_models_segmentation/ # Segmentation model checkpoints ├── src/ │ ├── data_loader.py │ ├── evaluate.py │ ├── main.py │ ├── model.py │ ├── predict_and_visualize.py │ ├── train.py │ ├── iter1/ # Segmentation experiment scripts │ └── iter2/ ├── src_classification/ │ ├── data_loader.py │ ├── evaluate.py │ ├── main.py │ ├── model.py │ ├── predict.py │ └── train.py ├── .gitignore ├── lung_abnormality_segmentation.code-workspace └── README.md
</details>

## Project Overview

- **Goal:** Detect and segment lung abnormalities in chest X-rays, focusing on COVID-19, Non-COVID infections, and healthy tissue.
- **Approaches:**
  - **Multi-class Segmentation:** Pixel-wise classification using U-Net with ResNet50 backbone.
  - **Image Classification:** Whole-image classification using ResNet50, following resource and performance limitations with segmentation.

## Key Components

- **Data:** Uses the COVID-QU-Ex dataset and others, with pixel-level masks and image-level labels.
- **Segmentation Pipeline (`src/`):**
  - Data loading, preprocessing, augmentation
  - U-Net + ResNet50 model definition
  - Training, evaluation, prediction, and visualization scripts
- **Classification Pipeline (`src_classification/`):**
  - Data loading and stratified splits
  - ResNet50-based classifier
  - Training, evaluation, prediction scripts

## Notebooks

- [`01_Introduction_&_Initial_Model_Details.ipynb`](notebooks/01_Introduction_&_Initial_Model_Details.ipynb): Project motivation, dataset details, and initial model setup.
- [`02_Segmentation_Model_Analysis.ipynb`](notebooks/02_Segmentation_Model_Analysis.ipynb): Iterative analysis of segmentation experiments, challenges, and results.
- [`03_Classification_Model_Analysis.ipynb`](notebooks/03_Classification_Model_Analysis.ipynb): Classification model development, evaluation, and comparison to segmentation.

## How to Run

1. **Install dependencies:**  
   Create a virtual environment and install required packages (see `requirements.txt` if available).

2. **Prepare data:**  
   Place raw datasets in `data/raw/` as described above.

3. **Train segmentation model:**  
    `python src.main.py --train`

- Evaluate or predict with `--evaluate` or `--predict`.

4. **Train classification model:**  
    `python src_classification.main.py --train`
- Evaluate or predict with `--evaluate` or `--predict`.

5. **View results:**  
- Visualizations and logs are in `images/` and `logs/`.
- Notebooks provide analysis and figures.

## Results & Recommendations

- **Segmentation:** Achieved good background and healthy lung segmentation, but struggled with COVID-19 and Non-COVID regions due to class imbalance and mask sparsity. Unfreezing more backbone layers increased resource usage without significant improvement.
- **Classification:** Provided robust detection of COVID-19 and Non-COVID cases at the image level, with higher accuracy and lower resource requirements.
- **Recommendation:** For practical deployment and further research, focus on classification or hybrid approaches unless more granular annotation data, particularly to differentiate non-covid infected lung tissue. Segmentation ideally be revisited when further data available and significant computational resource is feasible.

## License

This project is for educational and research purposes. Please check dataset licenses before use in production.

---

For more details, see the analysis notebooks in `notebooks/` and code in `src/` and `src_classification/`.