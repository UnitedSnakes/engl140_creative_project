import json
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

IMG_INDEX = "4"

# Load pre-trained model using torchvision's updated interface
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval()


def preprocess_image(img_path):
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_t = transform(img_pil).unsqueeze(0)
    return img_t, width, height, img


def generate_grad_cam(img, feature_grads, feature_maps, target_layer):
    """
    Generate Grad-CAM heatmap based on gradients and feature maps of a specific layer.
    """
    grads_val = feature_grads[target_layer].cpu().data.numpy().squeeze()
    target = feature_maps[target_layer].cpu().data.numpy().squeeze()

    weights = np.mean(grads_val, axis=(1, 2))  # Get weights by global average pooling
    cam = np.zeros(target.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * target[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)  # Normalize
    return cam


def apply_cam_on_image(img, cam):
    """
    Apply CAM on the original image.
    """
    img = cv2.resize(img, (224, 224))
    cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam = np.float32(cam) / 255
    img = np.float32(img) / 255
    cam = cam + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


# Preprocess the image
img_tensor, width, height, original_img = preprocess_image(f"img/{IMG_INDEX}.jpg")

# Feature maps and gradients capture
feature_maps = {}
feature_grads = {}


def get_features_hook(module, input, output):
    feature_maps[id(module)] = output


def get_grads_hook(module, grad_in, grad_out):
    feature_grads[id(module)] = grad_out[0]


# Register hooks
for name, module in model.named_modules():
    if "layer1" in name or "layer2" in name or "layer3" in name or "layer4" in name:
        module.register_forward_hook(get_features_hook)
        module.register_backward_hook(get_grads_hook)

# Forward pass
output = model(img_tensor)
pred_class = output.argmax(dim=1).item()

# Backward pass
model.zero_grad()
class_loss = output[0, pred_class]
class_loss.backward()

# Generate and apply Grad-CAM
for name, module in model.named_modules():
    if "layer1" in name or "layer2" in name or "layer3" in name or "layer4" in name:
        cam = generate_grad_cam(original_img, feature_grads, feature_maps, id(module))
        cam_img = apply_cam_on_image(original_img, cam)
        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 2, 1)
        plt.imshow(cv2.resize(cam_img, (original_img.shape[1], original_img.shape[0])))
        # plt.subplot(1, 2, 2)
        # plt.imshow(original_img)
        if not os.path.exists(os.path.join("result", IMG_INDEX)):
            os.mkdir(os.path.join("result", IMG_INDEX))
        plt.savefig(os.path.join("result", IMG_INDEX) + os.path.sep + name + ".png")
        plt.close()
        # plt.show()

# Load Imagenet class index file and get the predicted label
with open("imagenet_class_index.json", "r") as f:
    imagenet_class_index = json.load(f)
imagenet_labels = {int(key): value[1] for key, value in imagenet_class_index.items()}
predicted_label = imagenet_labels.get(pred_class, "Unknown class")
print(f"Predicted label: {predicted_label}")
