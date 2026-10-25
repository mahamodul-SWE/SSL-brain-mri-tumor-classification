# Brain Tumor MRI Classifier

A research-oriented web application for classifying brain MRI images into four categories: **Glioma, Meningioma, Pituitary, and No Tumor**. The project accompanies an academic thesis exploring deep-learning-based brain tumor classification and self-supervised learning.

> **Important medical disclaimer:** This application is an educational and research prototype. It has **not been clinically validated** and is not a certified medical device. Its predictions must not be used for diagnosis, treatment decisions, emergency triage, or patient screening. Always rely on qualified healthcare professionals for medical decisions.

## Live Demo

**Web app:** https://brain-tumor-classifier-simsiam.streamlit.app/

## Why This Project?

Brain MRI interpretation is an important part of brain tumor assessment, while manual image analysis can require specialist time and expertise. This thesis project investigates how deep learning can support automated classification of brain MRI images and explores a self-supervised-learning-based approach.

The goal is to evaluate the model in an academic setting and make its outputs easier to inspect—not to replace radiologists or provide clinical diagnoses.

## Web App Features

The app organizes its workflow into five layers:

1. **Input Safety** — checks file format, readability, file size, image dimensions, and decoded image area.
2. **Input Quality** — flags potential issues such as low resolution, unusual aspect ratio, low contrast, or unexpected color-channel differences.
3. **AI Prediction** — predicts one of the four supported classes and displays the class-score profile.
4. **Intelligence & Explainability** — provides review indicators, optional experimental MC-dropout uncertainty estimates, and Grad-CAM visualizations when available.
5. **Research & Reproducibility** — supports downloadable prediction reports and batch outputs for research documentation.

### Helpful Review Indicators

- **Input-quality warnings** help identify images that may need closer inspection before inference.
- **Prediction review flags** may be raised when the top score is below 60%, the top-two score margin is below 15 percentage points, or normalized entropy is above 0.75.
- **Grad-CAM** highlights image regions that influence a model score; it does not provide a tumor boundary or segmentation mask.
- **MC-dropout** is an experimental research diagnostic and has no clinically validated threshold in this application.

These checks are interface-level research safeguards, not proof that an image or prediction is clinically valid.

## How to Use the App

1. Open the [live web app](https://brain-tumor-classifier-simsiam.streamlit.app/).
2. Choose **Single scan** to analyze one image or **Batch screening** to process multiple images.
3. Upload a supported raster image: JPG/JPEG, PNG, BMP, or TIFF.
4. Review the input-quality information and prediction results.
5. If available, inspect the Grad-CAM visualization and download the research outputs.

Use de-identified research images only. Raw DICOM files are not supported by the documented pipeline.

## Research Outputs

Depending on the selected workflow, the app can provide:
- Predicted class and class-score distribution
- Input-quality warnings and research-review indicators
- Grad-CAM visualization, when successfully generated
- Optional experimental MC-dropout metrics
- JSON report and model-manifest information for reproducibility
- Batch CSV results and a batch summary JSON

The displayed softmax score is a model score, **not a clinically calibrated probability**. A high score does not guarantee a correct prediction.

## Limitations

- The model and web app have **not been clinically validated**.
- Results may be unreliable for images that differ from the model's training data, preprocessing, scanner, sequence, or acquisition conditions.
- Input-quality checks and review flags are heuristics; they do not establish diagnostic accuracy.
- Grad-CAM is an explanatory visualization, not proof of causal reasoning, tumor localization, or segmentation.
- Experimental uncertainty metrics should not be interpreted as validated clinical risk estimates.

## License

This project is provided under the **Educational Use Only** license. Use is limited to educational, academic, and non-commercial research purposes under the terms in the [`LICENSE`](LICENSE) file. Commercial use requires prior written permission from the copyright holder.

## Disclaimer

For research and educational use only. Do not use this application as a substitute for professional medical evaluation, diagnosis, or treatment.
