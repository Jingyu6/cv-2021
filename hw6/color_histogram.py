import numpy as np

from math import floor

def color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin):

	# make sure that we only count in-picture pixels
	height, width, _ = frame.shape

	# Bbox inside of the picture
	bbox_width = xmax - xmin + 1
	bbox_height = ymax - ymin + 1
	x_c, y_c = (xmin + xmax) / 2, (ymin + ymax) / 2
	x_c = np.clip(x_c, bbox_width / 2, width - bbox_width / 2 - 1)
	y_c = np.clip(y_c, bbox_height / 2, height - bbox_height / 2 - 1)
	xmin, xmax = floor(x_c - bbox_width / 2), floor(x_c + bbox_width / 2)
	ymin, ymax = floor(y_c - bbox_width / 2), floor(y_c + bbox_height / 2)

	bin_span = floor(256 / hist_bin)
	histo = np.zeros((hist_bin, hist_bin, hist_bin), dtype=np.int32)
	for y in range(ymin, ymax + 1):
		for x in range(xmin, xmax + 1):
			# reverse the width and height
			rgb = frame[y][x]
			rgb_normalized = np.floor(rgb / bin_span).astype(np.int32)
			histo[rgb_normalized[0], rgb_normalized[1], rgb_normalized[2]] += 1
	histo = histo / np.sum(histo)

	return histo