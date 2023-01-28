import os
import sys
import random
import math
import re
import time
import math
import numpy as np
import warnings

import utils
import model as modellib
import visualize
from model import log
from elements import *
from skimage import io, img_as_float

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

N = 4000
N_train = 3600
print ('Training on ', N_train, 'samples')

config = ElementsConfig()
config.display()

# Training dataset
dataset_train = ElementsDataset()
dataset_train.load_elements(1, N_train, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()

print ('Train:', dataset_train.image_ids)

# Validation dataset
dataset_val = ElementsDataset()
dataset_val.load_elements(N_train+1, N, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()

print ('Validation:', dataset_val.image_ids)

# Output location
output_directory = '../HtmlGeneration/validation_data/'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

class InferenceConfig(ElementsConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode = "inference", config = inference_config, model_dir = MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
# model_path = model.find_last()[1]
model_path = os.path.join(ROOT_DIR, "logs/mask_rcnn_elements_1000.h5")

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name = True)

for i in range(0):
    # Test on a random image
    image_id = random.choice(dataset_val.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, 
        inference_config, image_id, use_mini_mask = False)

    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    # Display ground-truth
    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, dataset_train.class_names, figsize = (8, 8))

    results = model.detect([original_image], verbose = 1)
    r = results[0]
    
    # Display all results
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], dataset_val.class_names, r['scores'], ax = get_ax())

    # Display results with confidence above threshold - this can be set in config, but still useful for customization
    #visualize.display_instances(original_image, rois, masks, class_ids, dataset_val.class_names, scores, ax=get_ax())


# Applies detection mask on top of the input image
# This one won't work, need to apply independently for each mask
def apply_masks(image, masks, class_ids, image_id, color_m, color_b, alpha = 0.7):
    mask = np.zeros((masks.shape[0], masks.shape[1]))
    masks_location = output_directory + 'masks/' + str(image_id) + '/'
    image_location = output_directory + 'images/' + str(image_id) + '/'

    if not os.path.exists(masks_location):
        os.makedirs(masks_location)

    if not os.path.exists(image_location):
        os.makedirs(image_location)

    for d in range(masks.shape[2]):
        c_mask = masks[...,d]
        class_id = class_ids[d]
        #print ('class id', class_id)
        io.imsave(masks_location + str(d+1) + '_' + dataset_val.get_class_name(class_id) + '.png', c_mask)
        for row in range(masks.shape[0]):
            for col in range(masks.shape[1]):
                mask[row, col] = max(mask[row, col], masks[row, col, d])

    masked_image = np.copy(img_as_float(image))
    for c in range(3):
        masked_image[:, :, c] = np.where(mask != 0, masked_image[:, :, c] * (1 - alpha) + alpha * color_m[c], masked_image[:, :, c])
        masked_image[:, :, c] = np.where(mask == 0, masked_image[:, :, c] * (1 - alpha) + alpha * color_b[c], masked_image[:, :, c])

    #io.imsave(output_directory + str(image_id) + '_masked.png', masked_image)
    io.imsave(image_location + str(image_id) + '.png', image)
    return masked_image

# Computes the Jaccard score (Intersection over Union) between detection cover and ground-truth cover
def compute_jaccard(masks, gt_masks):

    print ('PRED shape', masks.shape)
    print ('GT shape', gt_masks.shape)

    mask = np.zeros((masks.shape[0], masks.shape[1]))
    for d in range(masks.shape[2]):
        for row in range(masks.shape[0]):
            for col in range(masks.shape[1]):
                mask[row, col] = max(mask[row, col], masks[row, col, d])

    gt_mask = np.zeros((gt_masks.shape[0], gt_masks.shape[1]))
    for d in range(gt_masks.shape[2]):
        for row in range(gt_masks.shape[0]):
            for col in range(gt_masks.shape[1]):
                gt_mask[row, col] = max(gt_mask[row, col], gt_masks[row, col, d])

    intersection = np.zeros(mask.shape)
    union = np.zeros(mask.shape)

    intersection[:,:] = np.where(np.logical_and(mask != 0, gt_mask != 0), np.ones(mask.shape), np.zeros(mask.shape))
    union[:,:] = np.where(np.logical_or(mask != 0, gt_mask != 0), np.ones(mask.shape), np.zeros(mask.shape))

    intersection_count = np.sum(intersection)
    union_count = np.sum(union)
    print (intersection_count, union_count)
    if intersection_count == union_count:
        return 1.0
    else:
        return intersection_count/union_count

# Compute VOC-Style mAP @ IoU=0.5
# Running validation on N images. Increase for better accuracy.
image_ids = np.random.choice(dataset_val.image_ids, 128)
print (image_ids)
#image_ids = dataset_val.image_ids # for whole validation set

APs = []
JACs = []
for image_id in image_ids:

    print ('Validating', image_id + N_train + 1)

    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val,
        inference_config, image_id, use_mini_mask = False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)

    # For ignoring background
    gt_class_id = gt_class_id[...,1:]
    gt_bbox = gt_bbox[1:,...]
    gt_mask = gt_mask[...,1:]

    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]

    if r['masks'].shape[0] == 0:
        # Weird behavior (mini-mask) when no detection found
        r['masks'] = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 0)).astype(np.float32)

    roi_masks = []
    for roi_id in range(len(r['rois'])):
        roi = r['rois'][roi_id]
        class_id = r['class_ids'][roi_id]

        top = roi[0]
        left = roi[1]
        bottom = roi[2]
        right = roi[3]
        #print ('ROI regions', top, left, bottom, right)
        mask = np.zeros((128,128))
        mask[top:bottom, left:right] = np.ones((bottom-top, right-left))
        roi_masks.append(mask)

    roi_masks = np.array(roi_masks)
    roi_masks = np.swapaxes(np.swapaxes(roi_masks, 0, 2), 0, 1)

    print ('\nNew prediction!')

    # Compute Average Precision (AP)
    AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
        r["rois"], r["class_ids"], r["scores"], roi_masks)

    if gt_mask.shape[2] == 0 and r['masks'].shape[2] == 0:
        # Equal to 1 when no detection in gt and pred
        AP = 1.0
    elif math.isnan(AP):
        # Equal to 1 when NaN
        AP = 0.0

    # Compute Jaccard Score (JAC)
    JAC = compute_jaccard(roi_masks, gt_mask)

    APs.append(AP)
    JACs.append(JAC)

    print (image_id + N_train + 1, 'detection AP', AP, 'segmentation JAC', JAC)
    mask = apply_masks(image, roi_masks, r['class_ids'], image_id + N_train + 1, [0.0, 0.5, 0.5], [0.5, 0.1, 0.1])
    

    
    

print('mAP: ', np.mean(APs))
print('mJAC: ', np.mean(JACs))