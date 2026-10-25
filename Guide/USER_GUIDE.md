# Brain Tumor MRI Classifier — User Guide

## 1. Purpose and intended users

This guide is for students, researchers, supervisors, thesis examiners, and authorized research users who need to run the Brain Tumor MRI Classifier research application.

**Do not use this application for clinical diagnosis, treatment decisions, emergency triage, or autonomous patient screening.** Use de-identified research images only.

This guide covers setup, single-image analysis, batch classification, result interpretation, exports, and troubleshooting. The application is intended for research and educational use only.

---

## 2. Project files and requirements

The project folder should contain at least:

```text
brain_tumor_webapp/
├── app.py
├── model.py
├── inference.py
├── gradcam.py
├── styles.py
├── requirements.txt
├── START_APP.bat
├── .streamlit/
│   └── config.toml
└── models/
    └── best_classifier.pth
```

The checkpoint must be located at:

`models/best_classifier.pth`

---

## 3. Installation and launch

### Option A - Windows batch launcher

The supplied `START_APP.bat` expects a Conda environment named `brain_ssl`.

1. Make sure Anaconda/Miniconda is installed.
2. Create or prepare the `brain_ssl` environment.
3. Install the dependencies from `requirements.txt`.
4. Double-click `START_APP.bat`.
5. The app should open in a browser, normally at `http://localhost:8501`.

### Option B - Terminal / Anaconda Prompt

From the project directory:

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

If Streamlit does not open the browser automatically, open:

`http://localhost:8501`

---

## 4. Interface and navigation

The application has three main tabs:

1. **Single scan** - detailed analysis of one MRI image
2. **Batch screening** - process many MRI images together
3. **Model card & safeguards** - inspect model identity, limitations, safeguards, and reproducibility information

The sidebar shows:

- Model readiness
- CPU or CUDA device
- GPU name when available
- Backbone and SSL method
- Input size
- Supported classes
- Input safeguards
- Checkpoint name, stored internal validation Macro-F1, hash prefix, and class order

---

## 5. Supported image formats and limits

Accepted raster formats:

- JPG / JPEG
- PNG
- BMP
- TIF / TIFF

Important limits:

- Maximum file size: 25 MB per image
- Minimum decoded image dimension: 32 x 32
- Maximum decoded image area: 40 megapixels
- Raw DICOM is not supported in the current validated research pipeline

Use de-identified MRI-style raster exports. Avoid screenshots with patient names, colored annotations, UI panels, or unrelated content.

---

## 6. Single-image analysis workflow

### Step 1 — Open the Single scan tab

Choose **Single scan**.

### Step 2 — Upload one image

Click **Browse files** or drag and drop a supported image.

After upload, the app validates the image and runs the input QC checks before inference.

### Step 3 — Inspect the uploaded image

The original image is shown with filename and native dimensions.

Open **See the spatial input used by the model** to inspect the black-border-cropped and 224 x 224 resized image. This is useful for checking whether important anatomy is cut off or distorted before the neural network receives the tensor.

### Step 4 — Read the input QC panel

The panel shows:

- QC status: Pass or Review
- Native image size
- Dynamic range
- RGB deviation

Open **View QC measurements** for more detail.

If a warning appears, inspect the input carefully. A QC warning does not prove the image is invalid; it means the image differs from simple research-domain expectations.

### Step 5 — Choose optional analysis controls

- **Generate Grad-CAM explanation** - on by default
- **Experimental MC-dropout uncertainty (20 passes)** - off by default

MC-dropout takes extra computation because the model runs repeatedly.

### Step 6 — Click Analyze MRI

The app runs deterministic inference and the selected optional features.

### Step 7 — Read the result cards

The result area contains four main cards:

#### Tumor status

- **Tumor Positive** means the predicted class is Glioma, Meningioma, or Pituitary.
- **Tumor Negative** means the predicted class is No Tumor.

This is a model summary, not a clinical diagnosis.

#### Predicted class

The class with the highest deterministic softmax score.

#### Model score

The highest softmax value, shown as a percentage. It is not a clinically calibrated probability.

#### Research review

- **Flagged** means input QC and/or deterministic uncertainty heuristics triggered.
- **Not flagged** means none of the current heuristics triggered.

Not flagged does not guarantee that the model is correct.

---

## 7. Interpreting the class-score profile

The probability chart/table shows the model softmax distribution across:

- Glioma
- Meningioma
- Pituitary
- No Tumor

The four values should sum to approximately 100%.

A dominant top class may still be wrong, especially when the image comes from a different scanner, sequence, preprocessing workflow, population, orientation, or dataset from the training domain.

---

## 8. Interpreting score bands

The app uses simple research display bands:

| Top softmax score | Band |
|---:|---|
| 80% or higher | High |
| 60% to below 80% | Moderate |
| Below 60% | Low |

These are interface heuristics only. Do not describe them as validated diagnostic confidence categories.

---

## 9. Uncertainty and research-review indicators

The app shows three useful deterministic indicators.

### Top-1 score

The top softmax score.

### Top-2 margin

The difference between the first and second class scores.

Example:

- Glioma: 52%
- Meningioma: 44%
- Top-2 margin: 8 percentage points

A small margin means the model is less separated between its two leading choices.

### Normalized entropy

A value describing how spread out the four class scores are. Higher entropy generally means a flatter/more ambiguous distribution.

### Automatic deterministic review conditions

The app adds a review reason when:

- Top score is below 60%, or
- Top-2 margin is below 15 percentage points, or
- Normalized entropy is above 0.75

The final review state also includes QC warnings.

---

## 10. Experimental MC-dropout analysis

When enabled, the app runs 20 stochastic predictions and displays:

- MC mean top score
- Predicted-class score standard deviation
- Predictive entropy
- Expected entropy
- Mutual information
- Variation ratio

### Simple interpretation

- More variation between passes can indicate model instability for that image.
- A larger variation ratio means the stochastic predictions disagree more often.
- Larger predictive uncertainty metrics can indicate ambiguity.

However, there are no clinically validated cutoffs in this application. Use MC-dropout only as an experimental research diagnostic.

---

## 11. Interpreting Grad-CAM

If enabled and successfully generated, three panels appear:

1. Original image
2. Grad-CAM heatmap
3. Attention overlay

You can download the overlay PNG.

### What Grad-CAM can tell you

It can help you visually inspect which image regions influenced the selected class score more strongly.

### What Grad-CAM cannot tell you

It cannot reliably provide:

- Tumor boundaries
- Segmentation masks
- Lesion volume
- Histopathological confirmation
- Causal reasoning
- Clinical localization proof

Do not treat the red/yellow heatmap area as a tumor outline.

---

## 12. Exporting the single-scan research report

Click **Download reproducible JSON report**.

The report contains:

- Prediction fields
- All class scores
- Input QC measurements and warnings
- Optional MC-dropout results
- Run UUID and UTC timestamp
- Input SHA-256
- Checkpoint SHA-256
- Device
- Model manifest
- Research interpretation notice

This JSON is useful for thesis experiments, audit trails, comparing model builds, and confirming which checkpoint was used.

---

## 13. Starting a new analysis

Click **New scan** to clear the previous single-scan session state and load another image.

---

## 14. Batch-classification workflow

### Step 1 — Open Batch screening

Choose the **Batch screening** tab.

### Step 2 — Upload multiple images

Select several supported raster images.

### Step 3 — Run classification

Click **Run batch classification**.

The app validates and processes each file independently. One bad file does not prevent valid files from being processed.

### Step 4 — Review the batch overview

Summary metrics include:

- Processed
- Tumor positive
- Tumor negative
- Research review
- Average top score

### Step 5 — Filter results

You can:

- Search filename
- Filter Tumor Positive / Tumor Negative
- Filter High / Moderate / Low score band
- Filter Review / Not flagged

### Step 6 — Inspect class distribution

A bar chart summarizes the number of images predicted in each class.

### Step 7 — Export results

Download:

- **Batch CSV** for per-image results
- **Batch summary JSON** for aggregate counts, batch audit, and model manifest

### Step 8 — Check failed files

If any image cannot be processed, open **View processing errors** to see filenames and error messages.

---

## 15. Important batch CSV columns

| Column | Meaning |
|---|---|
| `filename` | Uploaded filename |
| `tumor_status` | Tumor Positive / Tumor Negative model summary |
| `predicted_class` | Highest-score class |
| `confidence_percent` | Top softmax score x 100 |
| `confidence_level` | High / Moderate / Low research band |
| `second_best_class` | Runner-up class |
| `top2_margin_percent` | Difference between top two scores |
| `normalized_entropy` | Four-class distribution entropy normalized to 0-1 scale |
| `research_review` | Combined deterministic/QC review state |
| `prob_*` | Per-class probability values stored as fractions (0-1) |
| `input_qc` | Pass / Review |
| `qc_warning_count` | Number of QC warning rules triggered |
| `input_sha256` | SHA-256 fingerprint of the uploaded file bytes |

---

## 16. Model card and safeguards

Use this tab when you need to document the model for a thesis demonstration or research record.

It shows:

- Architecture
- Classes and order
- Input requirements
- Checkpoint name and SHA-256
- Stored internal validation Macro-F1
- Inference pipeline
- Intended and non-intended use
- Safeguards
- Limitations
- Reproducibility manifest

Download the model manifest JSON when you need a machine-readable record of the model, checkpoint, and software environment.

---

## 17. Common warnings and what to do

### Native resolution below 224 pixels

The app will upscale the image. Prefer a source image with sufficient native resolution when possible.

### Unusual aspect ratio

Check whether the image includes extra panels, cropped anatomy, multiple slices, or a screenshot layout.

### Low dynamic range

Check whether the image is extremely dark, washed out, compressed, or preprocessed differently from the training domain.

### RGB channel differences

Check for colored arrows, labels, heatmaps, screenshots, or non-MRI color images.

### Low softmax score

Treat the case as uncertain research output and inspect competing class scores.

### Small top-2 margin

The top two classes are close. Prioritize the case for analysis/review.

### High normalized entropy

The prediction distribution is relatively spread across classes. Treat the result cautiously.

---

## 18. Troubleshooting

### Model checkpoint missing

Make sure this file exists:

`models/best_classifier.pth`

### Checkpoint architecture mismatch

Use the checkpoint created for the matching Experiment D model architecture. A checkpoint from a different architecture/classifier may fail strict loading.

### Unsupported image format

Convert the research image to JPG, PNG, BMP, or TIFF.

### File is not a readable medical image

The file may be corrupted, mislabeled, or not a valid image despite its extension.

### Image resolution is too small

Use an image at least 32 x 32 pixels. For model-domain quality, 224 pixels or more on each side is preferable.

### Image dimensions are unusually large

Use a smaller de-identified research export below 40 megapixels.

### File is over 25 MB

Create a smaller research export while preserving the required image content.

### App runs on CPU

This is expected when CUDA is unavailable. The model should still run, but batch and MC-dropout processing will be slower.

### Grad-CAM fails

The deterministic prediction can still succeed. Record the warning and avoid treating missing Grad-CAM as a failed prediction.

---

## 19. Privacy checklist

Before loading an image:

- Remove patient names and IDs from the visible image.
- Remove dates or study identifiers when they can identify a patient.
- Avoid hospital/PACS screenshots containing demographics.
- Use research IDs when filenames are needed.
- Keep the application on a controlled local machine for sensitive research data.

Clearing image metadata does not remove identifiers burned into the pixels.

---

## 20. Research-use checklist

Before reporting a result:

- Confirm the correct checkpoint is loaded.
- Confirm the input is a de-identified raster MRI image.
- Inspect the model-input preview.
- Read QC warnings.
- Read all four class scores, not only the top class.
- Check top-2 margin and entropy.
- Treat any review flag as a reason for closer research inspection.
- Do not interpret Grad-CAM as segmentation.
- Save the JSON/CSV audit outputs for reproducibility.
- Do not describe softmax values as validated clinical probabilities.
- Do not use the app as a substitute for radiology or pathology.
