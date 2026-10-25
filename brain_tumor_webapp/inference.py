from __future__ import annotations

import hashlib
import io
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torchvision import transforms

from model import DISPLAY_NAMES, BrainTumorClassifier, normalize_class_names

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
ALLOWED_FORMATS = {"JPEG", "PNG", "BMP", "TIFF"}
MAX_IMAGE_PIXELS = 40_000_000


@dataclass(frozen=True)
class ImageQualityReport:
    width: int
    height: int
    aspect_ratio: float
    cropped_width: int
    cropped_height: int
    retained_area_percent: float
    intensity_dynamic_range: float
    laplacian_variance: float
    color_channel_deviation: float
    qc_status: str
    warnings: tuple[str, ...]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["warnings"] = list(self.warnings)
        return data


@dataclass(frozen=True)
class PredictionResult:
    filename: str
    predicted_class: str
    display_class: str
    tumor_status: str
    confidence: float
    probabilities: dict[str, float]
    confidence_level: str
    second_best_class: str
    second_best_probability: float
    top2_margin: float
    normalized_entropy: float
    research_review_recommended: bool
    review_reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["review_reasons"] = list(self.review_reasons)
        return data


@dataclass(frozen=True)
class MCDropoutReport:
    passes: int
    predicted_class: str
    display_class: str
    mean_confidence: float
    predicted_class_std: float
    predictive_entropy: float
    expected_entropy: float
    mutual_information: float
    variation_ratio: float
    mean_probabilities: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


def crop_black_border(pil_image: Image.Image, threshold: int = 10) -> Image.Image:
    """Match the black-border crop used in the training notebook."""
    rgb_image = pil_image.convert("RGB")
    image_array = np.asarray(rgb_image)
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    coordinates = cv2.findNonZero((gray > threshold).astype(np.uint8))
    if coordinates is None:
        return rgb_image
    x, y, width, height = cv2.boundingRect(coordinates)
    return rgb_image.crop((x, y, x + width, y + height))


EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Lambda(crop_black_border),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def preprocess_preview(image: Image.Image) -> Image.Image:
    """Return the spatial image seen by the network before tensor normalization."""
    return crop_black_border(image).convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
    )


def open_uploaded_image(file_obj: BinaryIO | io.BytesIO) -> Image.Image:
    """Read and validate an uploaded image without trusting its extension."""
    try:
        image = Image.open(file_obj)
        image.verify()
        file_obj.seek(0)
        image = Image.open(file_obj)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError("The uploaded file is not a readable medical image.") from exc

    if image.format not in ALLOWED_FORMATS:
        raise ValueError(f"Unsupported image format: {image.format}")
    if image.width < 32 or image.height < 32:
        raise ValueError("Image resolution is too small for inference.")
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ValueError(
            "Image dimensions are unusually large for this prototype. "
            "Please use a de-identified research export below 40 megapixels."
        )

    rgb = image.convert("RGB")
    # The application does not need EXIF/TIFF metadata for inference. Clearing it
    # reduces accidental propagation of metadata if the in-memory image is reused.
    rgb.info.clear()
    return rgb


def analyze_image_quality(image: Image.Image) -> ImageQualityReport:
    """Run lightweight, non-diagnostic input/domain sanity checks.

    The thresholds are heuristic research safeguards, not validated MRI quality
    criteria. Warnings should prompt inspection, not automated rejection.
    """
    rgb = image.convert("RGB")
    raw = np.asarray(rgb)
    height, width = raw.shape[:2]
    aspect_ratio = max(width, height) / max(1, min(width, height))

    cropped = crop_black_border(rgb).convert("RGB")
    crop_array = np.asarray(cropped)
    crop_height, crop_width = crop_array.shape[:2]
    retained_area = (crop_width * crop_height) / max(1, width * height) * 100.0

    gray = cv2.cvtColor(crop_array, cv2.COLOR_RGB2GRAY)
    p01, p99 = np.percentile(gray, [1, 99])
    dynamic_range = float(p99 - p01)
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    color_deviation = float(
        np.mean(crop_array.max(axis=2).astype(np.float32) - crop_array.min(axis=2))
    )

    warnings: list[str] = []
    if width < IMAGE_SIZE or height < IMAGE_SIZE:
        warnings.append(
            "Native resolution is below 224 px on at least one side; the model will upscale the image."
        )
    if aspect_ratio > 1.60:
        warnings.append(
            "Unusual aspect ratio relative to the mostly square training-style inputs; inspect for domain mismatch."
        )
    if dynamic_range < 30.0:
        warnings.append(
            "Low intensity dynamic range; the image may have weak contrast or unusual preprocessing."
        )
    if color_deviation > 10.0:
        warnings.append(
            "Substantial RGB channel differences detected; the model was designed for MRI-style raster inputs, not color overlays."
        )

    return ImageQualityReport(
        width=width,
        height=height,
        aspect_ratio=float(aspect_ratio),
        cropped_width=crop_width,
        cropped_height=crop_height,
        retained_area_percent=float(retained_area),
        intensity_dynamic_range=dynamic_range,
        laplacian_variance=laplacian_variance,
        color_channel_deviation=color_deviation,
        qc_status="Review" if warnings else "Pass",
        warnings=tuple(warnings),
    )


def _safe_torch_load(checkpoint_path: Path, map_location: str | torch.device):
    """Use weights_only when available; support older PyTorch versions."""
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location=map_location)


def load_trained_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[BrainTumorClassifier, list[str], dict]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = _safe_torch_load(checkpoint_path, map_location="cpu")
    metadata: dict = checkpoint if isinstance(checkpoint, dict) else {}
    class_names = normalize_class_names(metadata.get("class_names"))

    config = metadata.get("config", {}) if isinstance(metadata, dict) else {}
    use_se = bool(config.get("use_se", True))
    dropout = float(config.get("dropout", 0.30))

    model = BrainTumorClassifier(
        num_classes=len(class_names),
        dropout=dropout,
        use_se=use_se,
    )

    state_dict = (
        checkpoint.get("model_state_dict")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a valid model state dictionary.")

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint architecture mismatch. Use the Experiment D "
            "best_classifier.pth created by the supplied notebook."
        ) from exc

    model.to(device)
    model.eval()
    return model, class_names, metadata


def confidence_level(confidence: float) -> str:
    """Legacy research score bands; not clinically calibrated thresholds."""
    if confidence >= 0.80:
        return "High"
    if confidence >= 0.60:
        return "Moderate"
    return "Low"


def _normalized_entropy(probabilities: np.ndarray) -> float:
    probabilities = np.clip(probabilities.astype(np.float64), 1e-12, 1.0)
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return entropy / math.log(len(probabilities)) if len(probabilities) > 1 else 0.0


def predict_image(
    model: BrainTumorClassifier,
    image: Image.Image,
    filename: str,
    class_names: list[str],
    device: torch.device,
) -> PredictionResult:
    tensor = EVAL_TRANSFORM(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(tensor)
        probabilities_tensor = torch.softmax(logits, dim=1)[0]

    probabilities_array = probabilities_tensor.detach().cpu().numpy()
    order = np.argsort(probabilities_array)[::-1]
    predicted_index = int(order[0])
    second_index = int(order[1])
    predicted_class = class_names[predicted_index]
    second_class = class_names[second_index]
    confidence = float(probabilities_array[predicted_index])
    second_probability = float(probabilities_array[second_index])
    top2_margin = confidence - second_probability
    entropy = _normalized_entropy(probabilities_array)

    probabilities = {
        DISPLAY_NAMES.get(class_name, class_name.title()): float(probability)
        for class_name, probability in zip(class_names, probabilities_array.tolist())
    }
    tumor_status = "Tumor Negative" if predicted_class == "notumor" else "Tumor Positive"

    review_reasons: list[str] = []
    if confidence < 0.60:
        review_reasons.append("Low softmax score band (<60%).")
    if top2_margin < 0.15:
        review_reasons.append("Small top-2 probability margin (<15 percentage points).")
    if entropy > 0.75:
        review_reasons.append("High normalized output entropy (>0.75).")

    return PredictionResult(
        filename=filename,
        predicted_class=predicted_class,
        display_class=DISPLAY_NAMES.get(predicted_class, predicted_class.title()),
        tumor_status=tumor_status,
        confidence=confidence,
        probabilities=probabilities,
        confidence_level=confidence_level(confidence),
        second_best_class=DISPLAY_NAMES.get(second_class, second_class.title()),
        second_best_probability=second_probability,
        top2_margin=float(top2_margin),
        normalized_entropy=float(entropy),
        research_review_recommended=bool(review_reasons),
        review_reasons=tuple(review_reasons),
    )


def estimate_mc_dropout_uncertainty(
    model: BrainTumorClassifier,
    image: Image.Image,
    class_names: list[str],
    device: torch.device,
    passes: int = 20,
) -> MCDropoutReport:
    """Experimental MC-dropout uncertainty while keeping BatchNorm in eval mode."""
    if passes < 2:
        raise ValueError("MC-dropout requires at least 2 stochastic passes.")

    tensor = EVAL_TRANSFORM(image).unsqueeze(0).to(device)
    dropout_states: list[tuple[nn.Module, bool]] = []
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            dropout_states.append((module, module.training))
            module.train()

    samples: list[torch.Tensor] = []
    try:
        with torch.inference_mode():
            for _ in range(passes):
                samples.append(torch.softmax(model(tensor), dim=1)[0].detach().cpu())
    finally:
        for module, previous_state in dropout_states:
            module.train(previous_state)

    stacked = torch.stack(samples, dim=0).numpy().astype(np.float64)
    mean_probs = stacked.mean(axis=0)
    mean_probs = mean_probs / mean_probs.sum()
    predicted_index = int(np.argmax(mean_probs))

    predictive_entropy = _normalized_entropy(mean_probs)
    sample_entropies = np.array([_normalized_entropy(row) for row in stacked])
    expected_entropy = float(sample_entropies.mean())
    mutual_information = max(0.0, predictive_entropy - expected_entropy)

    votes = np.argmax(stacked, axis=1)
    vote_count = int(np.bincount(votes, minlength=len(class_names)).max())
    variation_ratio = 1.0 - (vote_count / passes)

    predicted_samples = stacked[:, predicted_index]
    mean_probabilities = {
        DISPLAY_NAMES.get(class_name, class_name.title()): float(probability)
        for class_name, probability in zip(class_names, mean_probs.tolist())
    }

    predicted_class = class_names[predicted_index]
    return MCDropoutReport(
        passes=passes,
        predicted_class=predicted_class,
        display_class=DISPLAY_NAMES.get(predicted_class, predicted_class.title()),
        mean_confidence=float(mean_probs[predicted_index]),
        predicted_class_std=float(predicted_samples.std(ddof=0)),
        predictive_entropy=float(predictive_entropy),
        expected_entropy=expected_entropy,
        mutual_information=float(mutual_information),
        variation_ratio=float(variation_ratio),
        mean_probabilities=mean_probabilities,
    )


def prediction_to_dataframe(result: PredictionResult) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Class": list(result.probabilities.keys()),
            "Probability": [value * 100 for value in result.probabilities.values()],
        }
    ).sort_values("Probability", ascending=False)


def batch_results_dataframe(results: Iterable[PredictionResult]) -> pd.DataFrame:
    rows: list[dict] = []
    for result in results:
        row = {
            "filename": result.filename,
            "tumor_status": result.tumor_status,
            "predicted_class": result.display_class,
            "confidence_percent": round(result.confidence * 100, 2),
            "confidence_level": result.confidence_level,
            "second_best_class": result.second_best_class,
            "top2_margin_percent": round(result.top2_margin * 100, 2),
            "normalized_entropy": round(result.normalized_entropy, 4),
            "research_review": "Review" if result.research_review_recommended else "Not flagged",
        }
        for class_name, probability in result.probabilities.items():
            row[f"prob_{class_name.lower().replace(' ', '_')}"] = round(probability, 6)
        rows.append(row)
    return pd.DataFrame(rows)
