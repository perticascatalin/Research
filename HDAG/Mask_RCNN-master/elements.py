import matplotlib
import matplotlib.pyplot as plt
from skimage import io, img_as_float, img_as_int, img_as_ubyte
from skimage.color import rgba2rgb
from config import Config
import utils
import os
import random
import numpy as np

data_location = '../HtmlGeneration/resized_data/'

def get_ax(rows = 1, cols = 1, size = 8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize = (size*cols, size*rows))
    return ax

class ElementsConfig(Config):
    # Give the configuration a recognizable name
    NAME = "elements"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 11  # background + 11 elements

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

    DETECTION_MIN_CONFIDENCE = 0.95
    LEARNING_RATE = 0.004

class ElementsDataset(utils.Dataset):
    def load_elements(self, start, end, height, width):
        # Add classes
        self.add_class("elements", 1, "header")
        self.add_class("elements", 2, "left-column")
        self.add_class("elements", 3, "column-item")
        self.add_class("elements", 4, "col-item-title")
        self.add_class("elements", 5, "col-item-text")
        self.add_class("elements", 6, "right-grid")
        self.add_class("elements", 7, "grid-row")
        self.add_class("elements", 8, "grid-item")
        self.add_class("elements", 9, "grid-item-img")
        self.add_class("elements", 10, "grid-item-btn")
        self.add_class("elements", 11, "grid-item-text")

        # Add images
        for i in range(start, end+1):
            image_path = data_location + 'images/' + str(i) + '/'
            self.add_image("elements", image_id = i - start, path = image_path, width = width, height = height)

    def load_image(self, image_id):
        info = self.image_info[image_id]
        image_name = info['path'].split('/')[-2]
        print ('Loading image', image_name)
        image_location = info['path'] + image_name + '.png'
        image = io.imread(image_location)
        #print ('Image shape', image.shape)
        image = img_as_ubyte(rgba2rgb(image))
        return image

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "elements":
            return info["elements"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def get_masks(self, mask_location):
        #print ('Getting mask from', mask_location)
        masks = []
        class_ids = []
        files = [filename for filename in os.listdir(mask_location) if filename.endswith('.png')]
        for filename in files:
            mask = np.rint(img_as_float(io.imread(mask_location + filename)))
            classname = filename.split('_')[1].split('.')[0]
            masks.append(mask)
            class_id = self.get_class_id(classname)
            class_ids.append(class_id)
        
        background = np.zeros((128, 128))
        for d in range(len(masks)):
            background = np.maximum(background, masks[d])
        background = np.ones((128,128)) - background
        #print (background)
        #print ('Background has', np.sum(background), 'pixels')
        masks = [background] + masks
        class_ids = [0] + class_ids

        masks = np.reshape(np.array(masks), (len(masks), 128, 128))
        masks = np.swapaxes(np.swapaxes(masks, 0, 2), 0, 1)
        return masks, np.array(class_ids)

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        image_name = info['path'].split('/')[-2]
        print ('Loading mask', image_name)
        mask_location = info['path'].replace('images', 'masks')
        mask, class_ids = self.get_masks(mask_location)
        return mask, class_ids.astype(np.int32)

    def get_class_id(self, classname):
        for classtype in self.class_info:
            if classtype['name'] == classname:
                return classtype['id']
        # if not found consider background
        return 0

    def get_class_name(self, class_id):
        for classtype in self.class_info:
            if classtype['id'] == class_id:
                return classtype['name']
        # if not found consider background
        return "BG"