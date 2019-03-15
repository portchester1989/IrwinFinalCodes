import csv
import cv2
import numpy as np
import openslide as op
import os
import random as rd
import shutil

from PIL import Image


def create_bbox(coords, box_w, box_h, img_w, img_h):
	x = coords[0]
	y = coords[1]
	h = coords[2]
	w = coords[3]
	x_start = x - rd.uniform(0, max(box_w - w, 0)) #relative x-location that 256 box will start
	y_start = y - rd.uniform(0, max(box_h - h, 0)) #relative y-location that 256 box will start
	if x - box_w < 0:
		x_start = 0
	if y - box_h < 0:
		y_start = 0
	if x + box_w > img_w:
		x_start = img_w - box_w
	if y + box_h > img_h:
		y_start = img_h - box_h
	h_new = min(box_h, h)
	w_new = min(box_w, w)
	box = [int(x_start), int(y_start), int(box_h), int(box_w), int(h_new), int(w_new)]
	return box

def process_images(img_dir, bbox_dir, region_dir, region_bbox_filepath):

	# Ensure region dir exists
	if os.path.isdir(region_dir):
		shutil.rmtree(region_dir, ignore_errors=True)
	os.mkdir(region_dir)

	# dir of bbox from svs
	plaques_tangles_dir = './plaques_tangles_dir'
	if os.path.isdir(plaques_tangles_dir):
		shutil.rmtree(plaques_tangles_dir, ignore_errors=True)
	os.mkdir(plaques_tangles_dir)

	# clean old csv
	if os.path.isfile('./' + region_bbox_filepath):
		os.remove(region_bbox_filepath)
		print("Removed old csv")

	num = 0
	bboxes = []

	for file in os.listdir(img_dir):
		# Handle Hidden files for MacOS
		if file[0] == '.':
			continue
		print(file)
		# get bounding box filepath for all plaques/tangles in image
		bbox_file_path =  bbox_dir +  '/' + file[:-4] + '.csv'
		# read Image
		image = op.OpenSlide(img_dir + '/' + file)
		img_w, img_h = image.dimensions
		print(img_w, img_h, "image dimensions")
		# get coords for image
		class_label = np.genfromtxt(bbox_file_path, delimiter=',', dtype=str)
		coords = np.genfromtxt(bbox_file_path, delimiter=',', dtype=np.int32)
		num_obj, _ = coords.shape
		for i in range(num_obj):
			class_label_i = 1 if class_label[i, 0] == 'tangle' else 0
			bbox = create_bbox(coords[i,1:], 256, 256, img_w, img_h)
			xy = (bbox[0], bbox[1])
			height_width =(coords[i, 2], coords[i, 3])
			region_from_coords = image.read_region(xy, 0, (256, 256))
			rgbImage = Image.fromarray(cv2.cvtColor(np.asarray(region_from_coords), cv2.COLOR_RGBA2RGB))
			rgbImage.save(region_dir + '/' + str(num).zfill(7) + '.png', "PNG")
			region_from_coords_native = image.read_region((coords[i, 1], coords[i, 2]), 0, (coords[i, 3], coords[i, 4]))
			region_from_coords_native.save(plaques_tangles_dir + '/' + str(num).zfill(7)+ '.png', "PNG")
			bbox_i = [coords[i, 1] - bbox[0], coords[i, 2] - bbox[1], bbox[4], bbox[5], class_label_i]
			bboxes.append(bbox_i)
			num += 1
		image.close()
	np_bboxes = np.asarray(bboxes, dtype=np.int32)
	np.savetxt(region_bbox_filepath, np_bboxes, delimiter=",")

process_images('./imgs', './bboxes', './region_dir', 'regions_bbox.csv')

