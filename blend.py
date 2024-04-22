
def blend_images_with_heatmap(img1, img2, heatmap):
    """
    Blends two images with a heatmap. The heatmap intensity affects the visibility of the second image.
    
    Parameters:
        img1 (numpy.ndarray): The first image (background image).
        img2 (numpy.ndarray): The second image (foreground image).
        heatmap (numpy.ndarray): The heatmap image indicating blending weights for the second image.
        
    Returns:
        numpy.ndarray: The blended image.
    """
    # Ensure the heatmap is normalized between 0 and 1
    heatmap_normalized = cv2.normalize(heatmap, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    # Resize heatmap to match the image size if it's not the same already
    if heatmap_normalized.shape[:2] != img1.shape[:2]:
        heatmap_normalized = cv2.resize(heatmap_normalized, (img1.shape[1], img1.shape[0]))

    # Convert heatmap to the same type as img1 and img2 for blending
    heatmap_normalized = np.float32(heatmap_normalized)

    # Apply the heatmap to the second image
    foreground_weighted = cv2.multiply(img2.astype(np.float32), heatmap_normalized[..., np.newaxis])

    # Calculate the weight for the first image (background weight)
    background_weight = 1.0 - heatmap_normalized

    # Apply the background weight
    background_weighted = cv2.multiply(img1.astype(np.float32), background_weight[..., np.newaxis])

    # Combine the weighted images
    blended_image = cv2.add(foreground_weighted, background_weighted)

    # Convert back to original data type
    blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)
    
    return blended_image




blend_images_with_heatmap(original_img)