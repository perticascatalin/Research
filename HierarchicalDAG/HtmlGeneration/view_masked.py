from skimage import io, img_as_float
from skimage.color import rgba2rgb
import numpy as np
import os

# Apply masks for detected elements, returns masked image and masked
def apply_masks(image, masks, color_m, alpha=0.4):
	
	masked_image = np.copy(img_as_float(rgba2rgb(image)))
	masked = np.zeros(masked_image.shape)

	for (cls, mask) in masks:
		print (cls)
		for c in range(3):
			masked_image[:, :, c] = np.where(mask != 0, 
				masked_image[:, :, c] * (1 - alpha) + alpha * color_m[cls][c], masked_image[:, :, c])
			masked[:, :, c] = np.where(mask != 0, 
				masked[:, :, c] * (1 - alpha) + alpha * color_m[cls][c], masked[:, :, c])

	return masked_image, masked

header = [1.0, 0.0, 0.0]
left_column = [0.0, 1.0, 0.0]
column_item = [0.0, 0.0, 1.0]
col_item_title = [1.0, 1.0, 0.0]
col_item_text = [0.0, 1.0, 1.0]
right_grid = [1.0, 0.0, 1.0]
grid_row = [1.0, 1.0, 1.0]
grid_item = [0.5, 0.5, 0.0]
grid_item_image = [0.0, 0.5, 0.5]
grid_item_button = [0.5, 0.0, 0.5]
grid_item_text = [0.5, 0.5, 0.5]

colors = {}
colors['header'] = header
colors['left-column'] = left_column
colors['column-item'] = column_item
colors['col-item-title'] = col_item_title
colors['col-item-text'] = col_item_text
colors['right-grid'] = right_grid
colors['grid-row'] = grid_row
colors['grid-item'] = grid_item
colors['grid-item-img'] = grid_item_image
colors['grid-item-btn'] = grid_item_button
colors['grid-item-text'] = grid_item_text

# Which ID to view masked
img_id = str(3648)

# Read image
image = io.imread('./screenshots/images/' + img_id + '/' + img_id + '.png')

# Read masks
files = [file_name for file_name in os.listdir('./screenshots/masks/' + img_id + '/') if file_name.endswith('.png')]
files = sorted(files, key = lambda x: x.split('_')[1])
masks = [(file_name.split('_')[1].split('.')[0], io.imread('./screenshots/masks/' + img_id + '/' + file_name))
	for file_name in files]

# Apply masks
masked_image, masked = apply_masks(image, masks, colors)
io.imsave('masked_image.png', masked_image)
io.imsave('masked.png', masked)