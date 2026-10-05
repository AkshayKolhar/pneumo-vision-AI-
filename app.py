import hashlib
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

from utils.grad_cam import generate_gradcam, gradcam_images
from utils.inference import (
    exceeds_threshold,
    load_model,
    pneumonia_probability,
    preprocess_image,
)


ROOT = Path(__file__).resolve().parent
BASELINE_MODEL_PATH = ROOT / "models" / "best_pneumonia_model.pth"
CHALLENGER_MODEL_PATH = ROOT / "models" / "challenger_pneumonia_model.pth"
MODEL_PATH = (
    CHALLENGER_MODEL_PATH
    if CHALLENGER_MODEL_PATH.is_file()
    else BASELINE_MODEL_PATH
)
MODEL_NAME = "challenger" if MODEL_PATH == CHALLENGER_MODEL_PATH else "baseline"
COMPARISON_REPORT_PATH = ROOT / "models" / "model_comparison.json"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

st.set_page_config(
    page_title="LungLens | X-ray research demo",
    page_icon=":material/health_and_safety:",
    layout="wide",
)


@st.cache_resource
def get_model(model_path: str, modified_at: float, device_name: str):
    del modified_at
    return load_model(model_path, torch.device(device_name))


@st.cache_data
def checkpoint_sha256(model_path: str, modified_at: float) -> str:
    del modified_at
    return hashlib.sha256(Path(model_path).read_bytes()).hexdigest()


def get_matching_model_report():
    if not COMPARISON_REPORT_PATH.is_file() or not MODEL_PATH.is_file():
        return None
    report = json.loads(COMPARISON_REPORT_PATH.read_text(encoding="utf-8"))
    model_report = report.get("models", {}).get(MODEL_NAME)
    if model_report is None:
        return None
    digest = checkpoint_sha256(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    if model_report.get("checkpoint_sha256") != digest:
        return None
    return report, model_report


st.title("LungLens")
st.subheader("Chest X-ray classification · research prototype")
st.warning(
    "Research demo only — not a medical device. This model is not validated for "
    "clinical use and must not be used to diagnose, treat, or rule out disease."
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
matching_report = get_matching_model_report()
default_threshold = 0.5
if matching_report is not None and MODEL_NAME == "challenger":
    default_threshold = matching_report[1][
        "threshold_selected_on_validation"
    ]["threshold"]
with st.sidebar:
    st.markdown("### Demo controls")
    threshold = st.slider(
        "Pneumonia score threshold",
        min_value=0.05,
        max_value=0.95,
        value=float(default_threshold),
        step=0.000001,
        format="%.3f",
        help="This changes only the displayed flag; it does not recalibrate the model.",
    )
    st.caption(f"Active checkpoint: **{MODEL_NAME}**")
    st.caption(f"Inference device: **{device.type.upper()}**")
    st.caption("Uploaded image data is processed in memory and not saved by this app.")

prediction_tab, model_tab = st.tabs(["Analyze an image", "Model card"])

with prediction_tab:
    st.markdown(
        "Upload a chest X-ray image to view the model's pneumonia-class score. "
        "The score is not a diagnosis or a calibrated probability of disease."
    )
    upload = st.file_uploader(
        "Choose a chest X-ray image",
        type=["jpg", "jpeg", "png"],
        help="Accepted formats: JPG, JPEG, and PNG. Maximum size: 10 MB.",
    )

    if upload is not None:
        if upload.size > MAX_UPLOAD_BYTES:
            st.error("This file exceeds the 10 MB upload limit.")
        else:
            try:
                image = Image.open(BytesIO(upload.getvalue())).convert("L")
            except (Image.DecompressionBombError, UnidentifiedImageError, OSError):
                st.error("The uploaded file could not be read as an image.")
            else:
                image_bytes = upload.getvalue()
                image_digest = hashlib.sha256(image_bytes).hexdigest()
                st.image(
                    image,
                    caption="Uploaded image (converted to grayscale)",
                    alt="Uploaded chest X-ray preview",
                    width="stretch",
                )
                show_activation = st.checkbox(
                    "Show illustrative Grad-CAM activation map",
                    value=True,
                    help=(
                        "Highlights image regions that influenced the model's "
                        "selected class. This is not a clinical explanation."
                    ),
                )

                if not MODEL_PATH.is_file():
                    st.error(
                        "Model checkpoint not found. Train the model with "
                        "`python train.py` or place the checkpoint at "
                        "`models/best_pneumonia_model.pth`."
                    )
                elif st.button(
                    "Analyze image",
                    type="primary",
                    icon=":material/analytics:",
                ):
                    try:
                        model = get_model(
                            str(MODEL_PATH),
                            MODEL_PATH.stat().st_mtime,
                            str(device),
                        )
                        with st.spinner("Analyzing image..."):
                            score = pneumonia_probability(model, image, device)
                            heatmap = None
                            if show_activation:
                                input_tensor = preprocess_image(image).unsqueeze(0).to(device)
                                heatmap, _ = generate_gradcam(
                                    model,
                                    input_tensor,
                                    model.backbone.layer4[-1],
                                )
                        st.session_state["analysis_result"] = {
                            "image_digest": image_digest,
                            "pneumonia_score": score,
                            "heatmap": heatmap,
                        }
                    except (RuntimeError, ValueError, OSError) as error:
                        st.error(f"Could not run inference: {error}")

                result = st.session_state.get("analysis_result")
                if result and result["image_digest"] == image_digest:
                    score = float(result["pneumonia_score"])
                    normal_score = 1.0 - score
                    flagged = exceeds_threshold(score, threshold)

                    st.markdown("### Analysis results")
                    normal_metric, pneumonia_metric = st.columns(2)
                    normal_metric.metric("Normal-class model score", f"{normal_score:.1%}")
                    pneumonia_metric.metric(
                        "Pneumonia-class model score",
                        f"{score:.1%}",
                    )
                    st.progress(score, text="Pneumonia-class score (not calibrated)")
                    if flagged:
                        st.warning(
                            "The model score meets the selected threshold. "
                            "This is not a diagnosis."
                        )
                    else:
                        st.info(
                            "The model score is below the selected threshold. "
                            "This does not rule out pneumonia."
                        )
                    st.caption(
                        f"Display threshold: {threshold:.0%}. "
                        "Changing it does not improve model performance."
                    )

                    if result["heatmap"] is not None:
                        colored, overlay = gradcam_images(
                            image,
                            np.asarray(result["heatmap"]),
                        )
                        st.markdown("### Illustrative model activation")
                        st.caption(
                            "Grad-CAM shows image regions associated with this "
                            "model's output. It is approximate, can be misleading, "
                            "and does not identify disease or explain a clinical finding."
                        )
                        heatmap_col, overlay_col = st.columns(2)
                        with heatmap_col:
                            st.image(
                                colored,
                                caption="Grad-CAM activation map",
                                alt="Illustrative model activation heatmap",
                                width="stretch",
                            )
                        with overlay_col:
                            st.image(
                                overlay,
                                caption="Activation map over the X-ray",
                                alt="Illustrative activation heatmap overlaid on X-ray",
                                width="stretch",
                            )

with model_tab:
    st.markdown("### Intended use")
    st.write(
        "An educational computer-vision prototype for demonstrating transfer "
        "learning with a ResNet-50 on grayscale chest X-rays. It is intended "
        "for hackathon demos and experimentation, not patient care."
    )
    st.markdown("### Evaluation snapshot")
    if matching_report is not None:
        report, model_report = matching_report
        if MODEL_NAME == "challenger":
            operating_point = model_report["test_at_selected_threshold"]
            selected_threshold = model_report[
                "threshold_selected_on_validation"
            ]["threshold"]
            st.caption(
                f"Test metrics use threshold {selected_threshold:.3f}, selected "
                "on validation data to retain at least 98% validation recall. "
                "This is an exploratory operating point, not a clinical recommendation."
            )
        else:
            operating_point = model_report["test_at_0_50"]
            st.caption(
                "Baseline test metrics use its fixed 0.50 threshold. The original "
                "training split is unknown, so this checkpoint was not recalibrated."
            )
        accuracy, recall, precision, specificity = st.columns(4)
        accuracy.metric("Test accuracy", f"{operating_point['accuracy']:.1%}")
        recall.metric(
            "Pneumonia recall",
            f"{operating_point['pneumonia_recall']:.1%}",
        )
        precision.metric(
            "Pneumonia precision",
            f"{operating_point['pneumonia_precision']:.1%}",
        )
        specificity.metric(
            "Normal specificity",
            f"{operating_point['normal_specificity']:.1%}",
        )
        true_normal, false_pneumonia = operating_point["confusion_matrix"][0]
        st.write(
            f"Evaluated on {operating_point['samples']} held-out images. "
            f"Of the normal scans, {false_pneumonia} were incorrectly flagged "
            f"as pneumonia ({true_normal} correctly classified)."
        )
        st.caption(
            f"Model comparison uses threshold selection on the validation split "
            f"and reports the untouched test set ({report['test_split']})."
        )
    else:
        st.info(
            "Run `python compare_models.py` after training to generate a "
            "checkpoint-matched comparison report."
        )
    st.markdown("### Important limitations")
    st.markdown(
        "- The training/evaluation dataset is small and may not represent other "
        "hospitals, scanners, populations, or image protocols.\n"
        "- The displayed score is a softmax output, not a clinically calibrated "
        "risk estimate.\n"
        "- Grad-CAM is an approximate visualization of model activations, not "
        "a reliable explanation or disease-localization tool.\n"
        "- A below-threshold result cannot exclude disease; a clinician must "
        "interpret imaging with the full clinical context.\n"
        "- The model has not undergone prospective, external, or regulatory "
        "validation."
    )
    st.caption(
        "No images are intentionally written to disk by the demo. Avoid uploading "
        "identifiable patient information to a shared or public deployment."
    )

st.divider()
st.caption(
    "LungLens is an educational research prototype. It does not provide medical advice."
)
