"""
Wrapper de inferencia sobre EfficientNetMultiTask.
El modelo se carga una sola vez; predict() acepta un PIL.Image y retorna un dict
con todos los resultados incluyendo el Grad-CAM en base64.
"""

import io
import base64
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import matplotlib.cm as cm

BINARY_CLASSES = ["Benigno", "Maligno"]
SUBTYPE_CLASSES = [
    "Adenosis", "Fibroadenoma", "Tumor Filodes", "Adenoma Tubular",
    "Carcinoma Ductal", "Carcinoma Lobular", "Carcinoma Mucinoso", "Carcinoma Papilar",
]
MAG_CLASSES = ["40X", "100X", "200X", "400X"]

BIO_CONTEXT = {
    "Adenosis":            "Proliferación benigna de los ácinos glandulares. Bajo riesgo, manejo conservador.",
    "Fibroadenoma":        "Tumor benigno más frecuente en mujeres jóvenes. Contiene estroma y epitelio fibroso.",
    "Tumor Filodes":       "Tumor fibroepitelial raro. Puede ser benigno, borderline o maligno. Requiere resección amplia.",
    "Adenoma Tubular":     "Adenoma bien diferenciado. Pronóstico excelente, resección curativa.",
    "Carcinoma Ductal":    "Tipo maligno más frecuente (~80%). Alta agresividad. Requiere tratamiento multimodal.",
    "Carcinoma Lobular":   "Segundo tipo más común. Multifocal. Difícil detección en mamografía.",
    "Carcinoma Mucinoso":  "Produce mucina extracelular. Pronóstico relativamente favorable vs ductal.",
    "Carcinoma Papilar":   "Proyecciones papilares en los ductos. Variante poco común, pronóstico intermedio.",
}

MEAN = [0.7861, 0.6214, 0.7632]
STD  = [0.1322, 0.1561, 0.1168]


class EfficientNetMultiTask(nn.Module):
    def __init__(self, n_binary=2, n_subtype=8, n_mag=4, dropout=0.3):
        super().__init__()
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.avgpool  = backbone.avgpool
        feat_dim = backbone.classifier[1].in_features
        self.binary_head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(feat_dim, 256),
            nn.SiLU(), nn.Dropout(dropout * 0.5), nn.Linear(256, n_binary))
        self.subtype_head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(feat_dim, 512),
            nn.SiLU(), nn.Dropout(dropout * 0.5), nn.Linear(512, n_subtype))
        self.mag_head = nn.Sequential(
            nn.Dropout(dropout * 0.5), nn.Linear(feat_dim, 128),
            nn.SiLU(), nn.Linear(128, n_mag))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.binary_head(x), self.subtype_head(x), self.mag_head(x)


class _GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        target = model.features[-1]
        target.register_forward_hook(self._save_act)
        target.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, inp, out):
        self.activations = out.detach()

    def _save_grad(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, tensor, head_out, target_class):
        self.model.zero_grad()
        head_out[0, target_class].backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        return cam.squeeze().cpu().numpy()


def _overlay(pil_img, cam, alpha=0.45):
    heatmap = (cm.jet(cam)[:, :, :3] * 255).astype(np.uint8)
    orig = np.array(pil_img.resize((224, 224)))
    return (alpha * heatmap + (1 - alpha) * orig).astype(np.uint8)


def _to_b64(arr: np.ndarray) -> str:
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class Predictor:
    def __init__(self, checkpoint_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EfficientNetMultiTask().to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self._gradcam = _GradCAM(self.model)
        self._tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def predict(self, pil_image: Image.Image) -> dict:
        original = pil_image.convert("RGB")
        tensor = self._tfm(original).unsqueeze(0).to(self.device)
        tensor.requires_grad_(True)

        out_b, out_s, out_m = self.model(tensor)

        prob_b = torch.softmax(out_b, dim=1)[0]
        prob_s = torch.softmax(out_s, dim=1)[0]
        prob_m = torch.softmax(out_m, dim=1)[0]

        pred_b = prob_b.argmax().item()
        pred_s = prob_s.argmax().item()
        pred_m = prob_m.argmax().item()

        cam = self._gradcam(tensor, out_b, pred_b)
        overlay_arr = _overlay(original, cam)
        original_arr = np.array(original.resize((224, 224)))

        subtype_name = SUBTYPE_CLASSES[pred_s]
        return {
            "binary":       BINARY_CLASSES[pred_b],
            "binary_conf":  round(prob_b[pred_b].item(), 4),
            "subtype":      subtype_name,
            "subtype_conf": round(prob_s[pred_s].item(), 4),
            "mag":          MAG_CLASSES[pred_m],
            "probs_binary": {BINARY_CLASSES[i]: round(prob_b[i].item(), 4) for i in range(2)},
            "probs_subtype": {SUBTYPE_CLASSES[i]: round(prob_s[i].item(), 4) for i in range(8)},
            "bio_context":  BIO_CONTEXT.get(subtype_name, ""),
            "original_b64": _to_b64(original_arr),
            "gradcam_b64":  _to_b64(overlay_arr),
        }
