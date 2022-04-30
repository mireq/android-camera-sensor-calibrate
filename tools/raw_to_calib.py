# -*- coding: utf-8 -*-
import rawpy
import argparse
import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt


PLANES = [
	(slice(0, None, 2), slice(0, None, 2)),
	(slice(1, None, 2), slice(0, None, 2)),
	(slice(0, None, 2), slice(1, None, 2)),
	(slice(1, None, 2), slice(1, None, 2)),
]


def show_image(image):
	p_l, p_h = np.nanpercentile(image, [1, 99])
	cmap = mpl.cm.seismic.copy()
	cmap.set_bad('#888888', 1.)
	dist = (p_h - p_l) * 0.02
	plt.imshow(image, cmap=cmap, vmin=p_l-dist, vmax=p_h+dist)
	plt.colorbar()
	plt.tight_layout()
	plt.show()


def select_color(raw, requested_color):
	color_slice = None
	for plane_idx, color in zip(raw.raw_pattern.flatten(), list(raw.color_desc.decode('utf-8'))):
		if color == requested_color:
			color_slice = PLANES[plane_idx]
	color_slice = PLANES[1]
	return raw.raw_image[color_slice[0],color_slice[1]]


def main():
	parser = argparse.ArgumentParser(description="Analyze raw")
	parser.add_argument('input_raw', type=argparse.FileType(mode='rb'), help="Input raw")
	parser.add_argument('vignette_map', type=argparse.FileType(mode='wb'), help="Vignette map")
	args = parser.parse_args()

	raw = rawpy.imread(args.input_raw)
	raw_image = raw.raw_image
	output_images = []
	for idx, plane in enumerate(PLANES):
		image = raw_image[plane]
		image -= raw.black_level_per_channel[idx]
		image = cv2.blur(image, (10, 10))
		image = cv2.resize(image, tuple(c//20 for c in reversed(image.shape)), interpolation=cv2.INTER_AREA)
		max_value = np.max(image)
		vignette_map = max_value / image
		#show_image(vignette_map)
		output_images.append(vignette_map)

	#np.save(vignette_map, output_images, allow_pickle=False)
	np.save('vignette_map.npy', output_images, allow_pickle=False)

	#image = select_color(raw, 'G').astype(np.float64)
	#image = cv2.blur(image, (10, 10))
	#image = cv2.resize(image, tuple(c//20 for c in reversed(image.shape)), interpolation=cv2.INTER_AREA)
	#max_value = np.max(image)
	#vignette_map = max_value / image
	#show_image(vignette_map)
	#np.save('../../output/poco_x3_uw/vignette_map.npy', vignette_map, allow_pickle=False)




if __name__ == "__main__":
	main()
