from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image

from inference import EVAL_TRANSFORM, crop_black_border
from model import BrainTumorClassifier


class GradCAM:
    """Small dependency-free Grad-CAM implementation for the final ResNet block."""

    def __init__(self, model: BrainTumorClassifier) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.target_layer = self.model.encoder.layer4[-1].conv2
        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)

    def _save_gradient(self, gradient: torch.Tensor) -> None:
        self.gradients = gradient.detach()

    def _forward_hook(self, _module, _inputs, output: torch.Tensor) -> None:
        self.activations = output.detach()
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def close(self) -> None:
        self.forward_handle.remove()

    def generate(
        self,
        image: Image.Image,
        device: torch.device,
        class_index: int | None = None,
    ) -> tuple[np.ndarray, int]:
        tensor = EVAL_TRANSFORM(image).unsqueeze(0).to(device)
        self.model.zero_grad(set_to_none=True)
        logits = self.model(tensor)

        if class_index is None:
            class_index = int(logits.argmax(dim=1).item())

        score = logits[0, class_index]
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture tensors.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)[0]
        cam = torch.relu(cam)
        cam = cam - cam.min()
        if float(cam.max()) > 0:
            cam = cam / cam.max()
        return cam.detach().cpu().numpy(), class_index


def make_gradcam_overlay(
    image: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.42,
) -> tuple[Image.Image, Image.Image]:
    cropped = crop_black_border(image).convert("RGB").resize((224, 224))
    base = np.asarray(cropped).astype(np.float32)

    cam_resized = cv2.resize(cam, (224, 224))
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    overlay = np.clip((1 - alpha) * base + alpha * heatmap_rgb, 0, 255).astype(np.uint8)

    return Image.fromarray(heatmap_rgb.astype(np.uint8)), Image.fromarray(overlay)
