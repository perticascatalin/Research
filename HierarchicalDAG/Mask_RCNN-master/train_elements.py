import os
import sys
import random
import math
import re
import time
import numpy as np

import utils
import model as modellib
import visualize
from model import log
from elements import *

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

# Validation dataset
dataset_val = ElementsDataset()
dataset_val.load_elements(N_train, N, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()

image_ids = np.random.choice(dataset_train.image_ids, 4)
print (image_ids)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    print ('Class ids', class_ids)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

start = time.time()
# Create model in training mode
model = modellib.MaskRCNN(mode = "training", config = config, model_dir = MODEL_DIR)

# Which weights to start with?
init_with = "last"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name = True,
        exclude = ["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name = True)

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate = config.LEARNING_RATE, 
            epochs = 1, 
            layers = 'heads')

# Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate = config.LEARNING_RATE / 10,
            epochs = 2, 
            layers = "all")

stop = time.time()
print ('Training Time', stop - start)

# Save weights
model_path = os.path.join(MODEL_DIR, "mask_rcnn_elements.h5")
model.keras_model.save_weights(model_path)

class InferenceConfig(ElementsConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode = "inference", 
                          config = inference_config,
                          model_dir = MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
# model_path = model.find_last()[1]
model_path = os.path.join(ROOT_DIR, "logs/mask_rcnn_elements.h5")

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name = True)

for i in range(8):
    # Test on a random image
    image_id = random.choice(dataset_val.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, 
        inference_config, image_id, use_mini_mask = False)

    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                                dataset_train.class_names, figsize = (8, 8))

    results = model.detect([original_image], verbose = 1)

    r = results[0]
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                                dataset_val.class_names, r['scores'], ax = get_ax())


# Compute VOC-Style mAP @ IoU=0.5
# Running on N images. Increase for better accuracy.
image_ids = np.random.choice(dataset_val.image_ids, 16)

APs = []
for image_id in image_ids:

    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, 
        inference_config, image_id, use_mini_mask = False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)

    # For ignoring background
    gt_class_id = gt_class_id[...,1:]
    gt_bbox = gt_bbox[1:,...]
    gt_mask = gt_mask[...,1:]

    # Run object detection
    results = model.detect([image], verbose = 0)
    r = results[0]

    if r['masks'].shape[0] == 0:
        # Weird behavior (mini-mask) when no detection found
        r['masks'] = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 0)).astype(np.float32)

    # Compute AP
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])

    if gt_mask.shape[2] == 0 and r['masks'].shape[2] == 0:
        # Average precision equal to 0 when no detection in gt and pred
        AP = 1.0
    elif math.isnan(AP):
        AP = 0.0

    APs.append(AP)
    
print("mAP: ", np.mean(APs))