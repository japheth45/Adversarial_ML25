# image_helpers.py
from pathlib import Path
import torch
import torchvision
from PIL import Image

# --- Constants and Model Loading ------------------------------------------

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL  = None  # Lazy-loaded singleton

def get_model():
    """Loads a pre-trained ResNet-50 model and returns it with the device."""
    global _MODEL
    if _MODEL is None:
        _MODEL = torchvision.models.resnet50(weights="IMAGENET1K_V2").to(_DEVICE)
        _MODEL.eval()
    return _MODEL, _DEVICE

# --- Label and Image Utilities --------------------------------------------

LABELS = [l.strip() for l in Path("imagenet_labels.txt").read_text().splitlines()]

def label_to_index(name: str) -> int:
    """Converts a human-readable label name to its ImageNet index."""
    name = name.lower()
    matches = [i for i, l in enumerate(LABELS) if name in l.lower()]
    if not matches:
        raise ValueError(f"No label contains '{name}'")
    if len(matches) > 1:
        raise ValueError(f"Ambiguous – matches: {[LABELS[i] for i in matches]}")
    return matches[0]

_PREPROC = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_image(path: str):
    """Loads and preprocesses an image for the model."""
    img = Image.open(path).convert("RGB")
    return _PREPROC(img).unsqueeze(0).to(_DEVICE), img

# --- Classifier Function --------------------------------------------------

@torch.no_grad()
def classify(path: str, k: int = 5):
    """Classifies an image and returns the top k predictions."""
    model, dev = get_model()
    x, _ = load_image(path)
    probs = torch.softmax(model(x), dim=1)[0]
    conf, idx = torch.topk(probs, k)
    return [(LABELS[i], float(conf[j])) for j, i in enumerate(idx)]

# --- Adversarial Attack Utilities -----------------------------------------

def embed_crop_back(original_path: str,
                    adv_crop: Image.Image,
                    save_path: str,
                    fmt: str = "JPEG",
                    quality: int = 100) -> None:
    """
    Embeds a 224x224 adversarial crop back into the original-resolution picture.
    """
    orig = Image.open(original_path).convert("RGB")
    W, H = orig.size

    if W < H: # portrait
        newW, newH = 256, int(round(H * 256 / W))
    else: # landscape
        newH, newW = 256, int(round(W * 256 / H))
    resized = orig.resize((newW, newH), Image.BILINEAR)

    left = int(round((newW - 224) / 2))
    upper = int(round((newH - 224) / 2))
    box = (left, upper, left + 224, upper + 224)

    resized.paste(adv_crop, box)
    result = resized.resize((W, H), Image.NEAREST)

    if fmt.upper() == "PNG":
        result.save(save_path, format="PNG")
    else:
        result.save(save_path, format="JPEG", quality=quality, subsampling=0)

def classify_tensor(tensor_unnormalized, k=5):
    """Classifies an un-normalized image tensor in the [0,1] range."""
    model, dev = get_model()
    # Manually normalize the tensor before sending it to the model
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    tensor_normalized = (tensor_unnormalized.to(dev) - mean) / std

    # Get predictions
    probs = torch.softmax(model(tensor_normalized), dim=1)[0]
    conf, idx = torch.topk(probs, k)
    return [(LABELS[i], float(conf[j])) for j, i in enumerate(idx)]

def save_tensor_image(t, path):
    """Save a (1,C,H,W) float tensor in [0,1] to disk as an image."""
    from torchvision.transforms import ToPILImage
    img = ToPILImage()(t.squeeze(0).detach().cpu().clamp(0,1))
    img.save(path)

def linf(a, b):
    return float((a - b).abs().max().item())

def print_topk(preds, k=5):
    print(f"\nTop-{k} Predictions:")
    for i, (name, conf) in enumerate(preds[:k], 1):
        print(f"{i:>2}. {name:30s}  {conf:.2%}")
