import torch
import numpy as np
import os
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import matplotlib.pyplot as plt

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

np.random.seed(3)

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)
    ax.imshow(img)

# Load model only once
sam2_checkpoint = "./checkpoints/sam2.1_hiera_base_plus.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=32,          # Back to default, major impact on speed
    points_per_batch=128,        # Keep higher batch for efficiency
    pred_iou_thresh=0.7,         # Still lower than default but not too low
    stability_score_thresh=0.88, # More permissive than default, but not extreme
    stability_score_offset=0.7,  # Middle ground
    box_nms_thresh=0.75,         # Slightly higher for better large object detection
    crop_n_layers=0,             # Remove cropping which is computationally expensive
    min_mask_region_area=500,    # Focus on larger regions, but not extreme
    use_m2m=False,               # Disable for speed, major impact on runtime
)
"""
mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=24,          # Fewer points for faster processing
    pred_iou_thresh=0.7,         # Lower threshold for better large object detection
    stability_score_thresh=0.9,  # More permissive than default
    stability_score_offset=0.8,  # Less aggressive
)
"""

# Directory containing the images
data_dir = "/cluster/courses/cil/monocular_depth/data/train/"

# Get all PNG files (only RGB, not depth)
rgb_files = [f for f in os.listdir(data_dir) if f.endswith("_rgb.png")]
# Sort files to ensure consistent order and take only the first 20
rgb_files.sort()
rgb_files = rgb_files[:20]

print(f"Processing {len(rgb_files)} images...")
for img_file in rgb_files:
    # Extract image number for shorter output filename
    img_num = img_file.split("_")[1]
    
    # Load and process image
    image_path = os.path.join(data_dir, img_file)
    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))
    
    # Generate masks
    masks = mask_generator.generate(image)
    
    # Create shorter output filename
    output_filename = f"seg_{img_num}.png"
    
    # Plot and save
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close()  # Close the figure to free memory
    
    print(f"Processed {img_file} → {output_filename}")