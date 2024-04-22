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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(img)
    # plt.show()
    # input()
    height, width = img.shape[:2]
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    transform = transforms.Compose(
        [
            # transforms.Resize(256),
            transforms.Resize(224),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_t = transform(img_pil).unsqueeze(0)
    return img_t, width, height, img


def generate_grad_cam(feature_grads, feature_maps, target_layer):
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

def apply_cam_on_image(img, cam, mode):
    """
    Apply CAM on the original image.
    mode = '1': heatmap overlay; mode = '2': brightness intensity;
    """
    img = cv2.resize(img, (224, 224))  # Resize original image to match CAM size
    
    if mode == "1":
        heatmap = np.uint8(255 * cam)  # Scale CAM to 0-255 to apply color map
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Apply JET colormap
        heatmap = np.float32(heatmap) / 255  # Normalize heatmap
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = np.float32(img) / 255  # Normalize image
        combined = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)  # Blend images
        combined = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        return np.uint8(255 * combined)  # Convert back to 8-bit

    elif mode == "2":
        alpha_mask = np.clip(cam, 0, 1)  # Use CAM directly as the alpha mask
        
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # uncomment if background being black:
        # hsv_img[..., 2] *= alpha_mask
        
        # uncomment if background being white:
        
        # Multiply the brightness channel by the alpha_mask to retain the original brightness
        # where the mask is 1 and set it to 255 (white) where the mask is 0
        hsv_img[..., 2] = hsv_img[..., 2] * alpha_mask + (255 * (1 - alpha_mask))
        
        # Make sure the V channel is within bounds [0, 255]
        hsv_img[..., 2] = np.clip(hsv_img[..., 2], 0, 255)
        
        blended_image = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return blended_image
    
    elif mode == "3":
        new_img = cv2.imread("img/5.jpg")
        new_img = cv2.resize(new_img, (224, 224))
        new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB).astype(np.float32)
        # hsv_new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        alpha_mask = np.clip(cam, 0, 1)  # Use CAM directly as the alpha mask
        
        # Histogram Equalization (applies only to grayscale images)
        # Note: Ensure alpha_mask is scaled to 0-255 and converted to uint8 if you use this method
        alpha_mask_eq = cv2.equalizeHist(np.uint8(alpha_mask * 255))
        
        # Apply thresholding to convert to binary mask
        _, alpha_mask_binary = cv2.threshold(alpha_mask_eq, thresh=128, maxval=1, type=cv2.THRESH_BINARY)

        # Optionally, use THRESH_BINARY_INV to invert the threshold effect
        # _, alpha_mask_binary = cv2.threshold(alpha_mask_eq, thresh=128, maxval=1, type=cv2.THRESH_BINARY_INV)

        # Logarithmic Transformation
        # Use this transformation to spread out the lower values more than the higher values
        # alpha_mask_log = np.log1p(alpha_mask)  # log(1 + x), where x is the input array
        # alpha_mask_log = alpha_mask_log / np.max(alpha_mask_log)  # Normalize to 0-1 range

        # Exponential Transformation
        # This enhances higher values more significantly than lower values
        # alpha_mask_exp = np.exp(alpha_mask) - 1  # exp(x) - 1, where x is the input array
        # alpha_mask_exp = alpha_mask_exp / np.max(alpha_mask_exp)  # Normalize to 0-1 range

        # Power Law (Gamma) Transformation
        # Gamma less than 1 enhances dark regions, greater than 1 enhances bright regions
        # gamma = 10  # Gamma value greater than 1 to enhance brighter areas more
        # alpha_mask_gamma = np.power(alpha_mask, gamma)  # Apply gamma correction
        
        alpha_mask = alpha_mask_binary
        
        # uncomment if new heavy:
        blended_image = img.astype(np.float32) * alpha_mask[..., np.newaxis] + \
                        new_img.astype(np.float32) * (1 - alpha_mask[..., np.newaxis])
                        
        # uncomment if original heavy:
        # blended_image = img.astype(np.float32) * (1 - alpha_mask[..., np.newaxis]) + \
                        # new_img.astype(np.float32) * alpha_mask[..., np.newaxis]
        
        blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)
        
        return blended_image

    else:
        raise ValueError("Unsupported mode. Use '1' for heatmap overlay or '2' for brightness intensity.")

# Preprocess the image
img_tensor, width, height, original_img = preprocess_image(f"img/{IMG_INDEX}.jpg")

# plt.imshow(original_img)
# plt.show()
# input()

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

layer_names = []
cams = []

# Generate and apply Grad-CAM
for index, (name, module) in enumerate(model.named_modules()):
    if index % 9 != 0:
        continue
    if "layer1" in name or "layer2" in name or "layer3" in name or "layer4" in name:        
        cam = generate_grad_cam(feature_grads, feature_maps, id(module))
        cam_img = apply_cam_on_image(original_img, cam, mode="3")
        layer_names.append(name)
        cams.append(cv2.resize(cam_img, (original_img.shape[1], original_img.shape[0])))

# 16 * 16 subgraphs
plt.figure(figsize=(40, 40))
for i, (name, cam) in enumerate(zip(layer_names, cams)):
    plt.subplot(4, 4, i + 1)
    plt.imshow(cam)
    plt.title(name, fontsize=40)
    plt.axis('off')

# plt.savefig(os.path.join("result", f"{IMG_INDEX}_combined_heatmaps.png"))
# plt.savefig(os.path.join("result", f"{IMG_INDEX}_combined_color_intensities.png"))
plt.savefig(os.path.join("result", f"{IMG_INDEX}_combined_new_img.png"))
plt.close()

# Load Imagenet class index file and get the predicted label
with open("imagenet_class_index.json", "r") as f:
    imagenet_class_index = json.load(f)
imagenet_labels = {int(key): value[1] for key, value in imagenet_class_index.items()}
predicted_label = imagenet_labels.get(pred_class, "Unknown class")
print(f"Predicted label: {predicted_label}")

# blend_img_tensor, blend_width, blend_height, blend_original_img = preprocess_image("img/2.jpg")
