
#PRDECTION COMPLETE CODE 
import cv2
import torch
import numpy as np
from PIL import Image
import gradio as gr
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pennylane as qml

# LOAD CHECKPOINT

checkpoint = torch.load(
    "qcnn_model.pth",
    map_location="cpu",
    weights_only=False
)
class_names = checkpoint["class_names"]
print("Loaded classes:", class_names)

NO_INDEX  = class_names.index("no")
YES_INDEX = class_names.index("yes")

# QUANTUM LAYER


n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (4, n_qubits, 3)}
q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

# MODEL


class HybridQCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.efficientnet_b0(pretrained=True)
        self.backbone.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.backbone.classifier = nn.Identity()
        self.fc1 = nn.Linear(1280, n_qubits)
        self.q_layer = q_layer
        self.fc2 = nn.Linear(n_qubits, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.tanh(self.fc1(x))
        x = self.q_layer(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# LOAD MODEL


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridQCNN(len(class_names))
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

# GRAD-CAM

class GradCAM:
    def __init__(self, model, layer):
        self.activations = None
        self.gradients = None
        layer.register_forward_hook(self._forward)
        layer.register_backward_hook(self._backward)

    def _forward(self, m, i, o):
        self.activations = o

    def _backward(self, m, gi, go):
        self.gradients = go[0]

    def generate(self, x, cls):
        model.zero_grad()
        out = model(x)
        out[:, cls].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        return cam

gradcam = GradCAM(model, model.backbone.features[-1])

# TRANSFORM


transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# PREDICT (FIXED LOGIC)


def predict(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = Image.fromarray(gray)

    x = transform(img).unsqueeze(0).to(device)
    x.requires_grad_(True)

    with torch.no_grad():
        out = model(x)
        probs = torch.exp(out)[0]

    pred_idx = torch.argmax(probs).item()
    conf = float(probs[pred_idx])

    output_img = image.copy()

    # ---------- NO ----------
    if pred_idx == NO_INDEX:
        return output_img, {
            "no": conf,
            "yes": 1.0 - conf
        }

    # ---------- YES ----------
    cam = gradcam.generate(x, YES_INDEX)
    cam = cam.squeeze().detach().cpu().numpy()
    cam = cv2.resize(cam, (image.shape[1], image.shape[0]))

    #  Percentile-based threshold 
    cam_norm = cam / cam.max()
    thr = np.percentile(cam_norm, 90)   
    mask = (cam_norm >= thr).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        x0, y0, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

        #  Adaptive shrink 
        H, W = image.shape[:2]
        area_ratio = (w * h) / (H * W)

        if area_ratio > 0.2:
            shrink = 0.4
        else:
            shrink = 0.75

        cx = x0 + w // 2
        cy = y0 + h // 2
        w = int(w * shrink)
        h = int(h * shrink)

        x0 = max(cx - w // 2, 0)
        y0 = max(cy - h // 2, 0)

        cv2.rectangle(
            output_img,
            (x0, y0),
            (x0 + w, y0 + h),
            (255, 0, 0),
            3
        )

    return output_img, {
        "yes": conf,
        "no": 1.0 - conf
    }


# GRADIO


gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs=[
        gr.Image(type="numpy", label="Tumor Localization"),
        gr.Label(label="Tumor Detection")
    ],
    title="Brain Tumour Detection using QCNN",
).launch()
