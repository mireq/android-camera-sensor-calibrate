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
PERCENTILES = [0.1, 0.2, 0.5, 1, 99, 99.5, 99.8, 99.9]


def main():
	parser = argparse.ArgumentParser(description="Analyze raw")
	parser.add_argument('input_raw', type=argparse.FileType(mode='rb'), help="Input raw")
	args = parser.parse_args()

	raw = rawpy.imread(args.input_raw)
	raw_data = raw.raw_image

	for plane_num, plane in enumerate(PLANES):
		plane_data = raw_data[plane]
		color = chr(raw.color_desc[raw.raw_color(plane[0].start, plane[1].start)])
		percentiles = np.percentile(plane_data, PERCENTILES).astype(int)
		sys.stdout.write(f"{color} {percentiles}\n")




if __name__ == "__main__":
	main()
