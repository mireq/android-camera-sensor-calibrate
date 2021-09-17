# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


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


def subtract_black_frame(image, metadata):
	if not 'black_frame' in metadata:
		return image
	black_frame = np.load(metadata['dir'] / metadata['black_frame'])
	return image - black_frame


def linearize(image, metadata):
	if not 'gamma' in metadata:
		return image
	with open(metadata['dir'] / metadata['gamma'], 'r') as fp:
		gamma_data = json.load(fp)
	gamma_data = np.array(gamma_data, dtype=np.float64)
	xp = gamma_data[:,1]
	fp = gamma_data[:,0]
	shape = image.shape
	image = np.interp(image.flatten(), xp, fp).reshape(shape)
	return image


def apply_vignette(image, metadata):
	if not 'vignette_map' in metadata:
		return image
	vignette = np.load(metadata['dir'] / metadata['vignette_map'])
	vignette = cv2.resize(vignette, tuple(reversed(image.shape)), interpolation=cv2.INTER_LANCZOS4)
	return image * vignette


def normalize_image(image, metadata):
	image = image.copy()
	white_level = metadata['white_level']
	blacks = metadata['black_level']
	for plane_slice, black_level in zip(PLANES, blacks):
		image[plane_slice] -= black_level
		image[plane_slice] /= (white_level - black_level)
	return image


def denormalize_image(image, metadata):
	image = image.copy()
	white_level = metadata['white_level']
	blacks = metadata['black_level']
	for plane_slice, black_level in zip(PLANES, blacks):
		image[plane_slice] *= (white_level - black_level)
		image[plane_slice] += black_level
	return np.clip(image, 0, white_level).astype(np.uint16)


def main():
	parser = argparse.ArgumentParser(description="Fix raw")
	parser.add_argument('camera_metadata', type=argparse.FileType(mode='r'), help="Camera metadata json")
	parser.add_argument('input_raw', type=argparse.FileType(mode='rb'), help="Input raw")
	parser.add_argument('output_raw', type=argparse.FileType(mode='wb'), help="Output raw")
	args = parser.parse_args()

	camera_metadata = json.load(args.camera_metadata)
	raw_data = args.input_raw.read()
	raw_header = raw_data[:-(camera_metadata['resolution'][0] * camera_metadata['resolution'][1] * camera_metadata['bytes_per_pixel'])]
	raw_data = raw_data[-(camera_metadata['resolution'][0] * camera_metadata['resolution'][1] * camera_metadata['bytes_per_pixel']):]
	image = np.frombuffer(raw_data, dtype=np.uint16).reshape((camera_metadata['resolution'][1], camera_metadata['resolution'][0])).copy().astype(np.float64)

	camera_metadata['dir'] = Path(args.camera_metadata.name).resolve().parent
	image = subtract_black_frame(image, camera_metadata)
	image = normalize_image(image, camera_metadata)
	image = linearize(image, camera_metadata)
	image = apply_vignette(image, camera_metadata)
	image = denormalize_image(image, camera_metadata)

	args.output_raw.write(raw_header)
	args.output_raw.write(image.tobytes())


if __name__ == "__main__":
	main()
