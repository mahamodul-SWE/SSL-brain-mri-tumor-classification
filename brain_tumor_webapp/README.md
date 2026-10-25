# 🧠 NeuroVision AI

A local **Streamlit research application** for four-class brain MRI raster-image classification using the supplied SimSiam + SE-ResNet18 checkpoint.

> ⚠️ **Research Prototype Only:** This application is **not clinically approved or validated** and must not be used for diagnosis, treatment decisions, emergency triage, or autonomous clinical screening.

## Features

### Core inference

- Single MRI image prediction
- Batch image screening
- Four-class classification: Glioma, Meningioma, Pituitary, No Tumor
- Softmax score profile and second-best class
- Optional Grad-CAM visualization
- CPU / CUDA inference
- No external inference API

### Added research safeguards

- File-content validation instead of trusting the filename extension
- 25 MB per-image processing guard and unusually-large-image guard
- Image metadata discarded from the in-memory RGB copy
- Heuristic input quality/domain checks:
  - native resolution
  - aspect ratio
  - intensity dynamic range
  - RGB/color-overlay deviation
  - post-crop measurements
- Shows the spatial 224×224 image seen by the model before normalization
- Deterministic uncertainty diagnostics:
  - top-2 probability margin
  - normalized output entropy
  - research review flag for ambiguous outputs
- Optional **experimental MC-dropout** uncertainty (20 stochastic passes)
- Per-run reproducibility trace:
  - UUID run ID
  - UTC timestamp
  - input SHA-256
  - model checkpoint SHA-256
  - device and software manifest
- Reproducible JSON report for a single scan
- Grad-CAM overlay PNG download
- Batch QC/review columns, filters, hashes and audit-aware exports
- Model card with intended use, exclusions, limitations, stored internal validation Macro-F1 and checkpoint fingerprint

## Important interpretation notes

The softmax score bands, top-2 margin, entropy thresholds, input-QC thresholds and MC-dropout values in this application are **research heuristics**. They have not been calibrated on an independent external clinical validation set. A case that is not flagged can still be wrong or out-of-distribution.

Grad-CAM is a qualitative attention visualization. It is **not** a segmentation mask and does not prove that the model is using clinically correct reasoning.

The checkpoint stores an internal validation Macro-F1 of approximately **0.975**. This value must not be presented as clinical performance or external validity. External patient-level validation is required before any clinical claim.

## Why raw DICOM is not accepted

This checkpoint and preprocessing pipeline were designed for RGB raster image inputs. Raw DICOM introduces additional requirements such as modality/sequence handling, windowing or intensity scaling, orientation, metadata de-identification, scanner/protocol variability and a dedicated validation study. Converting DICOM to raster inside the app without validating that pipeline could create hidden domain shift, so the current build intentionally accepts only JPEG/PNG/BMP/TIFF research exports.

## Run locally

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

Then open `http://localhost:8501`.

## Project structure

```text
app.py          -> Streamlit interface, audit/report logic, model card
model.py        -> SE-ResNet18 architecture
inference.py    -> validation, preprocessing, QC, prediction, uncertainty
                    and batch table helpers
gradcam.py      -> Grad-CAM generation
styles.py       -> local UI styling (no remote Google Font import)
models/         -> trained model checkpoint
```

## Recommended publication-grade next steps

1. Use strict patient-level train/validation/test splits and audit near-duplicate leakage.
2. Evaluate on at least one independent external dataset from a different source/scanner/protocol.
3. Calibrate probability outputs on held-out validation data (for example temperature scaling when justified).
4. Select any abstention/review thresholds from validation data instead of using generic heuristic cutoffs.
5. Report class-wise sensitivity, specificity, precision, recall, F1, AUROC, confusion matrices and confidence intervals.
6. Evaluate calibration and selective-prediction metrics, including ECE/Brier score and risk-coverage behavior when appropriate.
7. Test subgroup/domain robustness and failure cases rather than reporting only aggregate accuracy.
8. Treat explanation methods as supporting evidence, not proof of causal or clinically valid reasoning.

## Design references

The safeguards and transparency structure were informed by current medical-AI guidance and research themes, including:

- FDA / Health Canada / MHRA, *Transparency for Machine Learning-Enabled Medical Devices: Guiding Principles* (2024): https://www.fda.gov/medical-devices/software-medical-device-samd/transparency-machine-learning-enabled-medical-devices-guiding-principles
- FDA / IMDRF, *Good Machine Learning Practice for Medical Device Development: Guiding Principles*: https://www.fda.gov/medical-devices/software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles
- WHO, *Ethics and governance of artificial intelligence for health* guidance: https://www.who.int/publications/i/item/9789240029200

These references do **not** mean this software is compliant, certified, approved, or clinically validated.

## License / usage scope

This project is provided only for educational, academic, research and personal non-commercial use unless the project owner specifies otherwise.
