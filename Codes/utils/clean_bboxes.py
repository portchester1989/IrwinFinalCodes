import numpy as np
import os


def clean_bboxes(bbox_dir):
	for file in os.listdir(bbox_dir):
		filepath = bbox_dir + '/' + file
		if file[0] == '.':
			continue
		coords = np.genfromtxt(filepath, delimiter=",", dtype=None, skip_header=1)
		os.remove(filepath)
		np.savetxt(filepath, coords, fmt="", delimiter=",")
	print("Done Cleaning")

clean_bboxes('./bbox_trial')