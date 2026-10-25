from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import platform
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image

from gradcam import GradCAM, make_gradcam_overlay
from inference import (
    ImageQualityReport,
    MCDropoutReport,
    PredictionResult,
    analyze_image_quality,
    batch_results_dataframe,
    estimate_mc_dropout_uncertainty,
    load_trained_model,
    open_uploaded_image,
    predict_image,
    prediction_to_dataframe,
    preprocess_preview,
    sha256_bytes,
)
from model import DISPLAY_NAMES
from styles import APP_CSS

APP_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = APP_DIR / "models" / "best_classifier.pth"
IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
MAX_UPLOAD_MB = 25.0
MC_DROPOUT_PASSES = 20

st.set_page_config(
    page_title="Automated Brain MRI Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(APP_CSS, unsafe_allow_html=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource(show_spinner="Loading trained SE-ResNet18 model...")
def cached_load_model(checkpoint_path: str, modified_time: float, device_name: str):
    del modified_time
    return load_trained_model(Path(checkpoint_path), torch.device(device_name))


@st.cache_data(show_spinner=False)
def cached_file_sha256(path: str, modified_time: float) -> str:
    del modified_time
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_sha256() -> str:
    if not CHECKPOINT_PATH.exists():
        return "unavailable"
    return cached_file_sha256(str(CHECKPOINT_PATH), CHECKPOINT_PATH.stat().st_mtime)


def model_bundle():
    if not CHECKPOINT_PATH.exists():
        return None
    return cached_load_model(
        str(CHECKPOINT_PATH),
        CHECKPOINT_PATH.stat().st_mtime,
        str(get_device()),
    )


def confidence_css(level: str) -> str:
    return {
        "High": "status-good",
        "Moderate": "status-warn",
        "Low": "status-bad",
    }[level]


def init_state() -> None:
    defaults = {
        "single_upload_version": 0,
        "batch_upload_version": 0,
        "single_result": None,
        "single_elapsed_ms": None,
        "single_image": None,
        "single_quality": None,
        "single_mc": None,
        "single_audit": None,
        "single_cam_heatmap": None,
        "single_cam_overlay": None,
        "single_cam_error": None,
        "batch_df": None,
        "batch_errors": [],
        "batch_audit": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def new_single_scan() -> None:
    st.session_state.single_upload_version += 1
    for key in (
        "single_result",
        "single_elapsed_ms",
        "single_image",
        "single_quality",
        "single_mc",
        "single_audit",
        "single_cam_heatmap",
        "single_cam_overlay",
        "single_cam_error",
    ):
        st.session_state[key] = None


def new_batch() -> None:
    st.session_state.batch_upload_version += 1
    st.session_state.batch_df = None
    st.session_state.batch_errors = []
    st.session_state.batch_audit = None


def result_card(label: str, value: str, subtitle: str, style_class: str = "neutral") -> str:
    return f"""
    <div class="result-card {style_class}">
      <div class="result-label">{label}</div>
      <div class="result-value">{value}</div>
      <div class="result-sub">{subtitle}</div>
    </div>
    """


def upload_size_mb(uploaded_file) -> float:
    size = getattr(uploaded_file, "size", None)
    if size is None:
        size = len(uploaded_file.getvalue())
    return float(size) / (1024 * 1024)


def validate_upload_size(uploaded_file) -> None:
    size_mb = upload_size_mb(uploaded_file)
    if size_mb > MAX_UPLOAD_MB:
        raise ValueError(
            f"File is {size_mb:.1f} MB. This research prototype limits each image to "
            f"{MAX_UPLOAD_MB:.0f} MB for safer local processing."
        )


def package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def public_checkpoint_config(metadata: dict) -> dict:
    config = metadata.get("config", {}) if isinstance(metadata, dict) else {}
    allowed = (
        "experiment_name",
        "seed",
        "use_imagenet_init",
        "run_ssl",
        "use_se",
        "consistency_weight",
        "image_size",
        "val_size",
        "batch_size",
        "ssl_epochs",
        "head_epochs",
        "layer4_epochs",
        "full_epochs",
    )
    return {key: config[key] for key in allowed if key in config}


def model_manifest(class_names: list[str], metadata: dict) -> dict:
    best_f1 = metadata.get("best_validation_macro_f1") if isinstance(metadata, dict) else None
    return {
        "app_name": "Automated Brain MRI Scan ",
        "research_status": "Academic research prototype; not clinically validated or approved",
        "model": "SimSiam-pretrained SE-ResNet18 four-class classifier",
        "class_order": [DISPLAY_NAMES.get(x, x) for x in class_names],
        "input_size": "224x224 RGB raster image",
        "preprocessing": "black-border crop -> resize 224x224 -> ImageNet normalization",
        "checkpoint_filename": CHECKPOINT_PATH.name,
        "checkpoint_sha256": checkpoint_sha256(),
        "internal_validation_macro_f1": float(best_f1) if best_f1 is not None else None,
        "checkpoint_config": public_checkpoint_config(metadata),
        "software": {
            "python": platform.python_version(),
            "streamlit": package_version("streamlit"),
            "torch": package_version("torch"),
            "torchvision": package_version("torchvision"),
            "opencv_python": package_version("opencv-python"),
            "pillow": package_version("Pillow"),
        },
    }


def make_run_audit(uploaded_file, quality: ImageQualityReport) -> dict:
    file_bytes = uploaded_file.getvalue()
    return {
        "run_id": str(uuid.uuid4()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "filename": uploaded_file.name,
        "input_sha256": sha256_bytes(file_bytes),
        "file_size_mb": round(upload_size_mb(uploaded_file), 4),
        "input_qc_status": quality.qc_status,
        "checkpoint_sha256": checkpoint_sha256(),
        "device": str(get_device()),
    }


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def render_sidebar(bundle) -> None:
    with st.sidebar:
        st.markdown("#### Brain Tumor Classification Web App")
        st.caption("MSc Thesis Prototype by, Mahamodul Hasan")
        st.caption("Supervised by, Dr. S M Hasan Mahmud")
        st.divider()

        device = get_device()
        st.markdown("### System status")
        if bundle is None:
            st.error("Model checkpoint missing")
        else:
            st.success("Model ready")

        st.write(f"**Device:** `{device}`")
        if device.type == "cuda":
            st.write(f"**GPU:** {torch.cuda.get_device_name(0)}")
        st.write("**Backbone:** SE-ResNet18")
        st.write("**SSL:** SimSiam")
        st.write("**Input:** 224 × 224 RGB")

        st.divider()
        st.markdown("### Supported classes")
        for name in ["Glioma", "Meningioma", "Pituitary", "No Tumor"]:
            st.write(f"• {name}")

        st.divider()
        st.markdown("### Input safeguards")
        st.write(f"• Max {MAX_UPLOAD_MB:.0f} MB per image")
        st.write("• File-content validation")
        st.write("• Metadata discarded in-memory")
        st.write("• Heuristic MRI-style QC")
        st.caption("Use only de-identified research images.")

        if bundle is not None:
            _, class_names, metadata = bundle
            st.divider()
            st.markdown("### Checkpoint")
            st.caption(CHECKPOINT_PATH.name)
            best_f1 = metadata.get("best_validation_macro_f1") if isinstance(metadata, dict) else None
            if best_f1 is not None:
                st.metric("Internal val. Macro-F1", f"{float(best_f1):.3f}")
            st.caption("SHA-256: " + checkpoint_sha256()[:16] + "…")
            st.caption(
                "Class order: "
                + " → ".join(DISPLAY_NAMES.get(x, x) for x in class_names)
            )


def render_hero() -> None:
    st.markdown(
        """
        <section class="hero">
          <div class="hero-kicker">MSc Thesis Research Prototype</div>
          <div class="hero-title-row">
            <div>
              <h1>Brain MRI Tumor Classification</h1>
              <p>
                Four-class MRI research inference with input quality checks,
                uncertainty diagnostics, reproducibility trace, and Grad-CAM explainability.
              </p>
            </div>
            <div class="hero-mark">🧠</div>
          </div>
          <div class="hero-badges">
            <span class="hero-badge">SimSiam SSL</span>
            <span class="hero-badge">SE-ResNet18</span>
            <span class="hero-badge">Input QC</span>
            <span class="hero-badge">Uncertainty</span>
            <span class="hero-badge">Grad-CAM</span>
            <span class="hero-badge">Audit trace</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_quality_report(quality: ImageQualityReport) -> None:
    st.markdown("#### Input quality & domain checks")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("QC status", quality.qc_status)
    q2.metric("Native size", f"{quality.width}×{quality.height}")
    q3.metric("Dynamic range", f"{quality.intensity_dynamic_range:.1f}")
    q4.metric("RGB deviation", f"{quality.color_channel_deviation:.1f}")

    if quality.warnings:
        st.warning("Research QC review: " + " ".join(quality.warnings))
    else:
        st.success("No heuristic input-quality/domain warnings were triggered.")

    with st.expander("View QC measurements"):
        st.dataframe(
            pd.DataFrame(
                [
                    ["Aspect ratio", f"{quality.aspect_ratio:.3f}"],
                    ["Post-crop size", f"{quality.cropped_width}×{quality.cropped_height}"],
                    ["Retained area", f"{quality.retained_area_percent:.1f}%"],
                    ["Intensity P99-P1", f"{quality.intensity_dynamic_range:.2f}"],
                    ["Laplacian variance", f"{quality.laplacian_variance:.2f}"],
                    ["Mean RGB channel spread", f"{quality.color_channel_deviation:.2f}"],
                ],
                columns=["Measure", "Value"],
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "These checks are heuristic engineering safeguards, not validated MRI image-quality criteria. "
            "A Pass does not establish diagnostic suitability."
        )


def render_uncertainty(result: PredictionResult, mc_report: MCDropoutReport | None) -> None:
    st.markdown("#### Uncertainty & review diagnostics")
    u1, u2, u3, u4 = st.columns(4)
    u1.metric("Top-1 score", f"{result.confidence * 100:.2f}%")
    u2.metric("Top-2 margin", f"{result.top2_margin * 100:.2f} pp")
    u3.metric("Normalized entropy", f"{result.normalized_entropy:.3f}")
    u4.metric("2nd class", result.second_best_class)

    if result.research_review_recommended:
        st.warning(
            "Research review flag triggered: " + " ".join(result.review_reasons) +
            " These thresholds are heuristic and not clinically validated."
        )
    else:
        st.info(
            "No deterministic uncertainty heuristic was triggered. Softmax output can still be "
            "overconfident, especially on out-of-distribution MRI images."
        )

    if mc_report is not None:
        st.markdown("##### Experimental MC-dropout")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Stochastic passes", mc_report.passes)
        m2.metric("MC mean top score", f"{mc_report.mean_confidence * 100:.2f}%")
        m3.metric("Predictive entropy", f"{mc_report.predictive_entropy:.3f}")
        m4.metric("Variation ratio", f"{mc_report.variation_ratio:.3f}")
        st.caption(
            f"Predicted-class probability SD: {mc_report.predicted_class_std:.4f} • "
            f"mutual information: {mc_report.mutual_information:.4f}. "
            "MC-dropout is experimental here and has not been calibrated against an external validation set."
        )


def render_single_tab(model, class_names, device, metadata) -> None:
    st.subheader("Analyze one MRI image")
    st.caption(
        "Upload one de-identified raster MRI image. The app validates the file, shows what the model sees, "
        "and reports uncertainty-related research diagnostics alongside the prediction."
    )

    uploaded_file = st.file_uploader(
        "Upload a brain MRI image",
        type=IMAGE_TYPES,
        accept_multiple_files=False,
        help="Raster image only. Raw DICOM is intentionally not accepted by this training pipeline.",
        key=f"single_upload_{st.session_state.single_upload_version}",
    )

    if uploaded_file is None:
        st.markdown(
            '<div class="upload-help">Drag and drop a scan above, or click Browse files to begin.</div>',
            unsafe_allow_html=True,
        )
        return

    try:
        validate_upload_size(uploaded_file)
        image = open_uploaded_image(uploaded_file)
        quality = analyze_image_quality(image)
    except ValueError as exc:
        st.error(str(exc))
        return

    preview_col, action_col = st.columns([1.05, 1], gap="large")

    with preview_col:
        st.markdown("#### Uploaded scan")
        st.image(
            image,
            caption=f"{uploaded_file.name} • {image.width}×{image.height}",
            use_container_width=True,
        )
        with st.expander("See the spatial input used by the model"):
            st.image(
                preprocess_preview(image),
                caption="After black-border crop and 224×224 resize; normalization occurs after this step.",
                use_container_width=True,
            )

    with action_col:
        st.markdown("#### Inference controls")
        st.markdown(
            '<div class="control-panel">'
            "<b>Pipeline</b><br>"
            "Validate → QC → black-border crop → 224×224 resize → normalization → SE-ResNet18"
            "</div>",
            unsafe_allow_html=True,
        )
        generate_cam = st.toggle("Generate Grad-CAM explanation", value=True)
        run_mc_dropout = st.toggle(
            f"Experimental MC-dropout uncertainty ({MC_DROPOUT_PASSES} passes)",
            value=False,
            help="Adds stochastic inference passes. This is an experimental research diagnostic, not calibrated clinical uncertainty.",
        )
        b1, b2 = st.columns(2)
        with b1:
            predict_clicked = st.button(
                "Analyze MRI",
                type="primary",
                use_container_width=True,
            )
        with b2:
            st.button("New scan", use_container_width=True, on_click=new_single_scan)
        st.markdown(
            '<p class="small-note">Softmax and MC-dropout values are model diagnostics, not clinical probabilities. '
            "The app calls no external inference API.</p>",
            unsafe_allow_html=True,
        )

    render_quality_report(quality)

    if predict_clicked:
        start_time = time.perf_counter()
        with st.spinner("Running model inference and safeguards..."):
            result = predict_image(
                model=model,
                image=image,
                filename=uploaded_file.name,
                class_names=class_names,
                device=device,
            )
            mc_report = None
            if run_mc_dropout:
                mc_report = estimate_mc_dropout_uncertainty(
                    model=model,
                    image=image,
                    class_names=class_names,
                    device=device,
                    passes=MC_DROPOUT_PASSES,
                )

            cam_heatmap = None
            cam_overlay = None
            cam_error = None
            if generate_cam:
                try:
                    gradcam = GradCAM(model)
                    try:
                        cam, _ = gradcam.generate(image=image, device=device)
                    finally:
                        gradcam.close()
                    cam_heatmap, cam_overlay = make_gradcam_overlay(image, cam)
                except Exception as exc:  # Grad-CAM must never block the prediction.
                    cam_error = str(exc)

        st.session_state.single_result = result
        st.session_state.single_elapsed_ms = (time.perf_counter() - start_time) * 1000
        st.session_state.single_image = image
        st.session_state.single_quality = quality
        st.session_state.single_mc = mc_report
        st.session_state.single_audit = make_run_audit(uploaded_file, quality)
        st.session_state.single_cam_heatmap = cam_heatmap
        st.session_state.single_cam_overlay = cam_overlay
        st.session_state.single_cam_error = cam_error

    result: PredictionResult | None = st.session_state.single_result
    if result is None:
        return

    stored_quality: ImageQualityReport = st.session_state.single_quality
    mc_report: MCDropoutReport | None = st.session_state.single_mc
    elapsed_ms = st.session_state.single_elapsed_ms or 0
    combined_review = result.research_review_recommended or stored_quality.qc_status == "Review"

    st.divider()
    status_style = "negative" if result.tumor_status == "Tumor Negative" else "positive"
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        st.markdown(
            result_card(
                "Tumor status",
                result.tumor_status,
                "Derived from the four-class model output",
                status_style,
            ),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            result_card("Predicted class", result.display_class, "Highest deterministic softmax score"),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            result_card(
                "Model score",
                f"{result.confidence * 100:.2f}%",
                f"Total analysis time: {elapsed_ms:.0f} ms",
            ),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            result_card(
                "Research review",
                "Flagged" if combined_review else "Not flagged",
                "Based on input QC + uncertainty heuristics",
                "review" if combined_review else "negative",
            ),
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<span class="status-pill {confidence_css(result.confidence_level)}">'
        f"● {result.confidence_level} softmax score band</span>",
        unsafe_allow_html=True,
    )

    prob_col, detail_col = st.columns([1.25, 1], gap="large")
    probability_df = prediction_to_dataframe(result)

    with prob_col:
        st.markdown("#### Class probability profile")
        st.bar_chart(
            probability_df.set_index("Class"),
            horizontal=True,
            x_label="Softmax score (%)",
            color="#2563eb",
        )

    with detail_col:
        st.markdown("#### Prediction details")
        detail_df = probability_df.copy()
        detail_df["Probability"] = detail_df["Probability"].map(lambda x: f"{x:.2f}%")
        st.dataframe(detail_df, hide_index=True, use_container_width=True)

        audit = st.session_state.single_audit or {}
        downloadable = {
            "prediction": result.to_dict(),
            "input_quality": stored_quality.to_dict(),
            "mc_dropout": mc_report.to_dict() if mc_report is not None else None,
            "audit": audit,
            "model_manifest": model_manifest(class_names, metadata),
            "interpretation_notice": (
                "Research prototype only. Scores and flags are not calibrated clinical probabilities "
                "and must not substitute for radiologist review or pathology."
            ),
        }
        st.download_button(
            "Download reproducible JSON report",
            data=json.dumps(downloadable, indent=2),
            file_name=f"{Path(result.filename).stem}_research_report.json",
            mime="application/json",
            use_container_width=True,
        )

    render_uncertainty(result, mc_report)

    if st.session_state.single_cam_error:
        st.warning(
            "Prediction succeeded, but Grad-CAM could not be generated: "
            + st.session_state.single_cam_error
        )
    elif st.session_state.single_cam_overlay is not None:
        st.markdown("#### Grad-CAM model explanation")
        g1, g2, g3 = st.columns(3, gap="medium")
        with g1:
            st.image(st.session_state.single_image, caption="Original image", use_container_width=True)
        with g2:
            st.image(st.session_state.single_cam_heatmap, caption="Grad-CAM heatmap", use_container_width=True)
        with g3:
            st.image(st.session_state.single_cam_overlay, caption="Attention overlay", use_container_width=True)
            st.download_button(
                "Download overlay PNG",
                data=image_to_png_bytes(st.session_state.single_cam_overlay),
                file_name=f"{Path(result.filename).stem}_gradcam_overlay.png",
                mime="image/png",
                use_container_width=True,
            )
        st.info(
            "Grad-CAM is a qualitative visualization of model attention. It is not a tumor segmentation mask, "
            "does not prove causal reasoning, and should not be interpreted as clinical localization."
        )


def render_batch_tab(model, class_names, device, metadata) -> None:
    st.subheader("Batch screening")
    st.caption(
        "Run research inference across multiple raster MRI images with per-image QC and review flags. "
        "Ground-truth labels are not required."
    )

    uploaded_files = st.file_uploader(
        "Upload multiple MRI images",
        type=IMAGE_TYPES,
        accept_multiple_files=True,
        key=f"batch_upload_{st.session_state.batch_upload_version}",
    )

    if uploaded_files:
        st.markdown(
            f'<div class="selection-strip"><b>{len(uploaded_files)}</b> images selected</div>',
            unsafe_allow_html=True,
        )
        run_col, new_col = st.columns(2, gap="medium")
        with run_col:
            run_batch = st.button("▶ Run batch classification", type="primary", use_container_width=True)
        with new_col:
            st.button("🆕 New batch", use_container_width=True, on_click=new_batch)

        if run_batch:
            results: list[PredictionResult] = []
            quality_reports: list[ImageQualityReport] = []
            input_hashes: list[str] = []
            errors: list[dict] = []
            progress = st.progress(0, text="Preparing batch...")

            for index, uploaded in enumerate(uploaded_files, start=1):
                try:
                    validate_upload_size(uploaded)
                    file_bytes = uploaded.getvalue()
                    image = open_uploaded_image(uploaded)
                    quality = analyze_image_quality(image)
                    result = predict_image(
                        model=model,
                        image=image,
                        filename=uploaded.name,
                        class_names=class_names,
                        device=device,
                    )
                    results.append(result)
                    quality_reports.append(quality)
                    input_hashes.append(sha256_bytes(file_bytes))
                except Exception as exc:
                    errors.append({"filename": uploaded.name, "error": str(exc)})

                progress.progress(index / len(uploaded_files), text=f"Processed {index}/{len(uploaded_files)}")

            progress.empty()
            batch_df = batch_results_dataframe(results)
            if not batch_df.empty:
                batch_df["input_qc"] = [q.qc_status for q in quality_reports]
                batch_df["qc_warning_count"] = [len(q.warnings) for q in quality_reports]
                batch_df["research_review"] = [
                    "Review" if (r.research_review_recommended or q.qc_status == "Review") else "Not flagged"
                    for r, q in zip(results, quality_reports)
                ]
                batch_df["input_sha256"] = input_hashes

            st.session_state.batch_df = batch_df
            st.session_state.batch_errors = errors
            st.session_state.batch_audit = {
                "batch_id": str(uuid.uuid4()),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "checkpoint_sha256": checkpoint_sha256(),
                "device": str(device),
            }

    batch_df = st.session_state.batch_df
    errors = st.session_state.batch_errors

    if batch_df is None:
        if not uploaded_files:
            st.info("Choose multiple images to create a batch classification report.")
        return

    if batch_df.empty and errors:
        st.warning("No images were successfully processed.")
    elif not batch_df.empty:
        st.divider()
        st.markdown("### Batch overview")

        positives = int((batch_df["tumor_status"] == "Tumor Positive").sum())
        negatives = int((batch_df["tumor_status"] == "Tumor Negative").sum())
        review_count = int((batch_df["research_review"] == "Review").sum())
        avg_conf = float(batch_df["confidence_percent"].mean())

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Processed", len(batch_df))
        m2.metric("Tumor positive", positives)
        m3.metric("Tumor negative", negatives)
        m4.metric("Research review", review_count)
        m5.metric("Avg. score", f"{avg_conf:.1f}%")

        st.markdown("### Results")
        search_col, status_col, conf_col, review_col = st.columns([1.5, 1, 1, 1])
        with search_col:
            search = st.text_input(
                "Search filename", placeholder="Type a filename...", label_visibility="collapsed"
            )
        with status_col:
            status_filter = st.selectbox(
                "Status", ["All statuses", "Tumor Positive", "Tumor Negative"], label_visibility="collapsed"
            )
        with conf_col:
            conf_filter = st.selectbox(
                "Confidence", ["All score bands", "High", "Moderate", "Low"], label_visibility="collapsed"
            )
        with review_col:
            review_filter = st.selectbox(
                "Review", ["All review states", "Review", "Not flagged"], label_visibility="collapsed"
            )

        filtered = batch_df.copy()
        if search:
            filtered = filtered[filtered["filename"].str.contains(search, case=False, na=False)]
        if status_filter != "All statuses":
            filtered = filtered[filtered["tumor_status"] == status_filter]
        if conf_filter != "All score bands":
            filtered = filtered[filtered["confidence_level"] == conf_filter]
        if review_filter != "All review states":
            filtered = filtered[filtered["research_review"] == review_filter]

        display_columns = [
            "filename",
            "tumor_status",
            "predicted_class",
            "confidence_percent",
            "top2_margin_percent",
            "normalized_entropy",
            "input_qc",
            "research_review",
        ]
        st.caption(f"Showing {len(filtered)} of {len(batch_df)} results")
        st.dataframe(
            filtered[display_columns],
            hide_index=True,
            use_container_width=True,
            height=390,
        )

        chart_col, export_col = st.columns([1.15, 1], gap="large")
        with chart_col:
            st.markdown("#### Predicted class distribution")
            distribution = (
                batch_df["predicted_class"].value_counts().rename_axis("Class").reset_index(name="Count")
            )
            st.bar_chart(distribution.set_index("Class"), color="#0284c7")

        with export_col:
            st.markdown("#### Export")
            st.download_button(
                "Download batch CSV",
                data=batch_df.to_csv(index=False).encode("utf-8"),
                file_name="brain_mri_batch_predictions_with_qc.csv",
                mime="text/csv",
                use_container_width=True,
            )

            summary = {
                "processed": len(batch_df),
                "tumor_positive": positives,
                "tumor_negative": negatives,
                "research_review": review_count,
                "average_softmax_score_percent": round(avg_conf, 2),
                "input_qc": batch_df["input_qc"].value_counts().to_dict(),
                "score_bands": batch_df["confidence_level"].value_counts().to_dict(),
                "predicted_classes": batch_df["predicted_class"].value_counts().to_dict(),
                "audit": st.session_state.batch_audit,
                "model_manifest": model_manifest(class_names, metadata),
            }
            st.download_button(
                "Download batch summary JSON",
                data=json.dumps(summary, indent=2),
                file_name="brain_mri_batch_summary_with_audit.json",
                mime="application/json",
                use_container_width=True,
            )

        if review_count:
            st.warning(
                f"{review_count} image(s) were flagged by input-QC and/or uncertainty heuristics. "
                "This flag is for research review prioritization only."
            )

        if errors:
            st.warning(f"{len(errors)} file(s) could not be processed.")
            with st.expander("View processing errors"):
                st.dataframe(pd.DataFrame(errors), hide_index=True, use_container_width=True)

        st.button("🆕 Clear results & load new batch", use_container_width=True, on_click=new_batch)


def render_model_tab(bundle) -> None:
    model, class_names, metadata = bundle
    del model
    st.subheader("Model card, safeguards & usage")
    st.caption("Technical and governance information for research reproducibility and thesis demonstration.")

    manifest = model_manifest(class_names, metadata)
    best_f1 = manifest["internal_validation_macro_f1"]

    identity_col, perf_col = st.columns([1.25, 1], gap="large")
    with identity_col:
        st.markdown("#### Model identity")
        st.write("**Architecture:** SimSiam + SE-ResNet18 classifier")
        st.write("**Classes:** " + ", ".join(manifest["class_order"]))
        st.write("**Input:** 224×224 RGB raster image")
        st.write("**Checkpoint:** `best_classifier.pth`")
        st.code(manifest["checkpoint_sha256"], language="text")
        st.caption("The SHA-256 fingerprint lets exported reports be tied to the exact checkpoint file.")

    with perf_col:
        st.markdown("#### Reported checkpoint metric")
        if best_f1 is not None:
            st.metric("Internal validation Macro-F1", f"{best_f1:.3f}")
        else:
            st.info("No validation Macro-F1 was stored in the checkpoint metadata.")
        st.warning(
            "This stored validation score is not evidence of clinical validity or external generalization. "
            "Performance must be tested on independent patient-level external cohorts before any clinical claim."
        )

    st.markdown("#### Inference pipeline")
    st.code(
        "Raster MRI upload\n"
        "   ↓\n"
        "File validation + input QC\n"
        "   ↓\n"
        "Black-border crop\n"
        "   ↓\n"
        "Resize to 224×224 + ImageNet normalization\n"
        "   ↓\n"
        "SE-ResNet18 encoder + 4-class classifier\n"
        "   ↓\n"
        "Prediction + uncertainty diagnostics + optional Grad-CAM\n"
        "   ↓\n"
        "Reproducibility/audit export",
        language="text",
    )

    st.markdown("#### Intended use and exclusions")
    use_col, no_col = st.columns(2, gap="large")
    with use_col:
        st.success(
            "**Intended:** academic research, thesis demonstration, controlled experiments, "
            "and analysis of de-identified MRI-style raster images similar to the training domain."
        )
    with no_col:
        st.error(
            "**Not intended:** diagnosis, treatment decisions, emergency triage, autonomous screening, "
            "or direct use on raw DICOM/clinical PACS data without dedicated preprocessing and validation."
        )

    st.markdown("#### Added research safeguards")
    safeguards = pd.DataFrame(
        [
            ["Input integrity", "Content-based image validation, dimension guard, per-file size limit"],
            ["Privacy", "Image metadata discarded in-memory; app does not call an external inference API"],
            ["Domain/QC", "Resolution, aspect-ratio, contrast and color-overlay heuristic warnings"],
            ["Uncertainty", "Top-2 margin + normalized entropy + optional MC-dropout diagnostics"],
            ["Human review", "Research review flag prioritizes ambiguous/QC-warning cases; never auto-clears clinical risk"],
            ["Explainability", "Grad-CAM explicitly labeled qualitative and non-segmentation"],
            ["Traceability", "Input SHA-256, checkpoint SHA-256, UTC run ID/timestamp, software manifest"],
            ["Batch QA", "QC/review columns and filters are included in batch exports"],
        ],
        columns=["Area", "Safeguard"],
    )
    st.dataframe(safeguards, hide_index=True, use_container_width=True)

    st.markdown("#### Key limitations")
    st.warning(
        "Scanner vendor, MRI sequence, acquisition protocol, patient population, orientation, cropping, compression, "
        "burned-in labels, color overlays and other distribution shifts can change model behavior. "
        "The new QC and uncertainty indicators can miss out-of-distribution inputs and must not be treated as a guarantee."
    )
    

    st.markdown("#### Reproducibility manifest")
    st.json(manifest, expanded=False)
    st.download_button(
        "Download model manifest JSON",
        data=json.dumps(manifest, indent=2),
        file_name="neurovision_model_manifest.json",
        mime="application/json",
        use_container_width=True,
    )
    st.caption(
        "Design rationale follows modern medical-AI transparency/GMLP themes: intended-use clarity, human oversight, "
        "essential information, software engineering safeguards and traceability."
    )


init_state()
render_hero()

try:
    bundle = model_bundle()
except Exception as exc:
    bundle = None
    st.error(f"Model could not be loaded: {exc}")

render_sidebar(bundle)

if bundle is None:
    st.warning(
        "Place the trained **Experiment D** checkpoint at `models/best_classifier.pth`, then refresh this page."
    )
    st.code(
        "brain_tumor_webapp/\n"
        "├── app.py\n"
        "└── models/\n"
        "    └── best_classifier.pth",
        language="text",
    )
    st.stop()

model, class_names, checkpoint_metadata = bundle
device = get_device()

single_tab, batch_tab, model_tab = st.tabs(
    ["🔬 Single scan", "📚 Batch screening", "🛡️ Model card & safeguards"]
)

with single_tab:
    render_single_tab(model, class_names, device, checkpoint_metadata)

with batch_tab:
    render_batch_tab(model, class_names, device, checkpoint_metadata)

with model_tab:
    render_model_tab(bundle)

st.markdown(
    """
    <div class="footer-note">
      <strong>Research-use disclaimer:</strong> This application is an academic prototype and
      must not be used as a substitute for radiologist review, pathology, or clinical diagnosis.
      The quality checks, softmax bands, entropy, top-2 margin, MC-dropout and Grad-CAM outputs are
      research diagnostics only. Remove patient-identifying information before loading images into any research software.
    </div>
    """,
    unsafe_allow_html=True,
)
