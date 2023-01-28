from skimage import io
from skimage.transform import resize
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

IMG_HEIGHT = 128
IMG_WIDTH = 128

for index in range(3001, 4001):

	# Which ID to resize
	img_id = str(index)
	print ('Sample', img_id)

	# Read image
	image = io.imread('./screenshots/image/' + img_id + '/' + img_id + '.png')

	# Read masks
	files = [file_name for file_name in os.listdir('./screenshots/mask/' + img_id + '/') if file_name.endswith('.png')]
	files = sorted(files, key = lambda x: x.split('_')[1])
	masks = [(file_name.split('_')[1].split('.')[0], io.imread('./screenshots/mask/' + img_id + '/' + file_name))
		for file_name in files]

	output_directory = './resized_data/'

	# Save resized image
	resized_image = resize(image, (IMG_HEIGHT, IMG_WIDTH))
	output_location = output_directory + 'images/' + img_id + '/'
	if not os.path.exists(output_location):
		os.makedirs(output_location)
	location = output_location + img_id + '.png'
	io.imsave(location, resized_image)

	# Save resized masks
	output_location = output_directory + 'masks/' + img_id + '/'
	if not os.path.exists(output_location):
		os.makedirs(output_location)
	masks = [(mask[0], resize(mask[1], (IMG_HEIGHT, IMG_WIDTH))) for mask in masks]
	mask_id = 0
	for mask in masks:
		mask_id += 1
		location = output_location + str(mask_id) + '_' + mask[0] + '.png'
		io.imsave(location, mask[1])