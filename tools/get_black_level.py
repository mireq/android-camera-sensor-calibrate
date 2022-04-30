# -*- coding: utf-8 -*-
import rawpy
import argparse
import numpy as np
import sys


PLANES = [
	(slice(0, None, 2), slice(0, None, 2)),
	(slice(1, None, 2), slice(0, None, 2)),
	(slice(0, None, 2), slice(1, None, 2)),
	(slice(1, None, 2), slice(1, None, 2)),
]


def main():
	parser = argparse.ArgumentParser(description="Get black level")
	parser.add_argument('input_raw', type=argparse.FileType(mode='rb'), help="Input raw")
	args = parser.parse_args()

	raw = rawpy.imread(args.input_raw)
	raw_data = raw.raw_image

	for plane_num, plane in enumerate(PLANES):
		plane_data = raw_data[plane]
		color = chr(raw.color_desc[raw.raw_color(plane[0].start, plane[1].start)])
		clip_value = np.percentile(plane_data, 0.99)
		print(clip_value)
		mean = np.mean(plane_data[plane_data <= clip_value])
		sys.stdout.write(f"{color} {mean}\n")




if __name__ == "__main__":
	main()
