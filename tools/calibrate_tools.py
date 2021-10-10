#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import csv
import functools
import itertools
import json
import math
import os
import socket
import subprocess
import sys
from collections import namedtuple
from fractions import Fraction
from pathlib import Path

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


light = None
camera = None


Resolution = namedtuple('Resolution', ['width', 'height'])
Range = namedtuple('Resolution', ['low', 'high'])
Planes = namedtuple('Planes', ['r', 'g1', 'g2', 'b'])


PLANES = [
	(slice(0, None, 2), slice(0, None, 2)),
	(slice(1, None, 2), slice(0, None, 2)),
	(slice(0, None, 2), slice(1, None, 2)),
	(slice(1, None, 2), slice(1, None, 2)),
]
PIXEL_PATTERN_PLANE_INDEXES = {
	'RGGB': (0, 1, 2, 3),
	'GRBG': (1, 0, 3, 2),
	'GBRG': (1, 3, 0, 2),
	'BGGR': (3, 1, 2, 0),
	'RGB': (0, 1, 3, 0),
}


OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / 'output'


def read_csv(fp, fields):
	field_names = [f[0] for f in fields]
	field_types = [f[1] for f in fields]

	reader = csv.reader(fp)
	row = next(reader)
	if row != field_names:
		raise RuntimeError(f"Wrong structure, expected {field_names}, got {row}")

	class DataFrameBase(object):
		def __str__(self):
			args = ', '.join(arg_name + '=' + repr(getattr(self, arg_name)) for arg_name in self.__slots__)
			return f'{self.__class__.__name__}({args})'

	def __init__(self, *args):
		for i, field in enumerate(field_names):
			setattr(self, field, args[i])

	DataFrame = type('DataFrame', (DataFrameBase,), {'__slots__': field_names, '__init__': __init__})

	columns = [[] for i in range(len(field_names))]
	for row in reader:
		for i, field_type, value in zip(itertools.count(), field_types, row):
			columns[i].append(field_type(value))

	np_columns = []
	for column, field_type in zip(columns, field_types):
		np_columns.append(np.fromiter(column, field_type))
	return DataFrame(*np_columns)


def select_color(images, requested_color):
	color_slice = None
	for plane_idx, color in zip(camera.pixel_pattern_index, camera.pixel_pattern):
		if color == requested_color:
			color_slice = PLANES[plane_idx]

	images = images[:,color_slice[0],color_slice[1]]
	return images


def show_image(image):
	p_l, p_h = np.nanpercentile(image, [1, 99])
	cmap = mpl.cm.seismic.copy()
	cmap.set_bad('#888888', 1.)
	dist = (p_h - p_l) * 0.02
	plt.imshow(image, cmap=cmap, vmin=p_l-dist, vmax=p_h+dist)
	plt.colorbar()
	plt.tight_layout()
	plt.show()


def drop_bad_pixels(images, percentile=10):
	p_l, p_h = np.percentile(images, [percentile, 100-percentile], axis=[1, 2])
	images = images.astype(np.float64)
	for img, l, h in zip(images, p_l, p_h):
		img[np.logical_or(img < l, img > h)] = np.nan
	return images


class Light(object):
	LEDC_APB_CLK_FREQ = 80*1000000
	LEDC_REF_CLK_FREQ = 1000000
	LEDC_TIMER_DIV_NUM_MAX = 0x3FFFF

	def __init__(self, args):
		self.args = args
		self.resolution = 10
		self.value = Fraction(1/2)

	@functools.cached_property
	def dev(self):
		#subprocess.run(['stty', '-F', self.args.tty, '2000000'], check=True)
		subprocess.run(['stty', '-F', self.args.tty, '115200'], check=True)
		return open(self.args.tty, 'w')

	def set_frequency(self, frequency):
		div_param = (self.LEDC_APB_CLK_FREQ << 8) // frequency;
		self.resolution = 20
		while div_param // (1 << self.resolution) < 256:
			self.resolution -= 1
			if self.resolution == 0:
				raise RuntimeError("Failed to set frequency")
		self.dev.write(f'r{self.resolution}\nf{frequency}\n')
		self.set_value(self.value)

	def set_value(self, value):
		self.value = Fraction(value)
		raw_value = int(self.max_value * self.value)
		self.dev.write(f'v{raw_value}\na\n')
		self.dev.flush()

	@property
	def max_value(self):
		return (1 << self.resolution)


class Camera(object):
	def __init__(self, args):
		self.args = args

	@functools.cached_property
	def dev(self):
		sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		sock.connect((self.args.host, self.args.port))
		return sock.makefile('rwb')

	@functools.cached_property
	def resolution(self):
		self.write_cmd('getResolution')
		return Resolution(*(int(val) for val in self.dev.readline().decode('utf-8').split()))

	@functools.cached_property
	def iso_range(self):
		self.write_cmd('getIsoRange')
		return Range(*(int(val) for val in self.dev.readline().decode('utf-8').split()))

	@functools.cached_property
	def exposure_range_raw(self):
		self.write_cmd('getExposureRange')
		return Range(*(int(val) for val in self.dev.readline().decode('utf-8').split()))

	@functools.cached_property
	def exposure_range(self):
		return Range(*(float(val) / 1000000000 for val in self.exposure_range_raw))

	@functools.cached_property
	def bytes_per_pixel(self):
		self.write_cmd('getBytesPerPixel')
		return int(self.dev.readline().decode('utf-8').strip())

	@functools.cached_property
	def pixel_pattern(self):
		self.write_cmd('getPixelPattern')
		return self.dev.readline().decode('utf-8').strip()

	@functools.cached_property
	def pixel_pattern_index(self):
		return PIXEL_PATTERN_PLANE_INDEXES.get(self.pixel_pattern, PIXEL_PATTERN_PLANE_INDEXES['RGB'])

	@functools.cached_property
	def plane_slices(self):
		return Planes(*(PLANES[idx] for idx in self.pixel_pattern_index))

	def write_cmd(self, cmd):
		self.dev.write(f"{cmd}\n".encode('utf-8'))
		self.dev.flush()

	def set_iso(self, iso):
		self.write_cmd(f'setIso {iso}')

	def set_exposure(self, exposure):
		exposure = int(exposure * 1000000000)
		self.write_cmd(f'setExposure {exposure}')

	def get_raw(self, x=None, y=None, w=None, h=None, count=None):
		resolution = self.resolution
		bytes_per_pixel = self.bytes_per_pixel
		x = 0 if x is None else x
		y = 0 if y is None else y
		w = resolution.width if w is None else w
		h = resolution.height if h is None else h
		count = 1 if count is None else count
		pixel_pattern_index = self.pixel_pattern_index
		self.write_cmd(f'getRaw {x} {y} {w} {h} {count}')
		metadata = {}
		field = ''
		while True:
			field, value = self.dev.readline().decode('utf-8').split(': ', 1)
			if field == 'data':
				break
			value = value.rstrip('\n')
			if field in {'color_correction_gains', 'black_level'}:
				value = [float(val) for val in value.split()]
			elif field in {'white_level'}:
				value = int(value)
			if field == 'color_correction_gains':
				unsorted_value = value
				value = [0, 0, 0, 0]
				for target_idx, src_idx in enumerate(pixel_pattern_index):
					value[target_idx] = unsorted_value[src_idx]
			metadata[field] = value
		data = self.dev.read(w * h * bytes_per_pixel * count)
		np_array = np.frombuffer(data, dtype=np.uint16).reshape((count, h, w)).copy()
		return np_array, metadata


def get_resolution(args):
	res = camera.resolution
	sys.stdout.write(f"{res.width}x{res.height}\n")
	sys.stdout.flush()


def get_iso_range(args):
	ran = camera.iso_range
	sys.stdout.write(f"{ran.low}:{ran.high}\n")
	sys.stdout.flush()


def get_exposure_range(args):
	ran = camera.exposure_range
	sys.stdout.write(f"{ran.low:.8f}:{ran.high:.8f}\n")
	sys.stdout.flush()


def get_bytes_per_pixel(args):
	sys.stdout.write(f"{camera.bytes_per_pixel}\n")
	sys.stdout.flush()


def get_pixel_pattern(args):
	sys.stdout.write(f"{camera.pixel_pattern}\n")
	sys.stdout.flush()


def add_common_camera_args(subparser):
	subparser.add_argument('--iso', type=int, help="ISO", default=100)
	subparser.add_argument('--exposure', type=float, help="Exposure time", default=0.1)
	return subparser


def set_common_camera_args(camera, args):
	camera.set_iso(args.iso)
	camera.set_exposure(args.exposure)


def show(args):
	set_common_camera_args(camera, args)
	image, metadata = camera.get_raw()
	image = image[0]
	image = image.astype(np.float32)
	for plane_slice, color_gain, black_level in zip(PLANES, metadata['color_correction_gains'], metadata['black_level']):
		image[plane_slice] -= black_level
		image[plane_slice] *= color_gain
	image = image.astype(np.uint16)
	image = cv2.cvtColor(image, getattr(cv2, f'COLOR_Bayer{camera.pixel_pattern[:2]}2BGR_EA'))
	image = image.astype(np.float32) * (256.0 / metadata['white_level'] + 1)
	np.clip(image, 0, 255, image)
	image = image.astype(np.uint8)
	plt.imshow(image)
	plt.tight_layout()
	plt.show()


def measure(args):
	set_common_camera_args(camera, args)
	if args.size:
		half_size = args.size // 2
		start_x = ((camera.resolution.width // 2 - half_size) // 2) * 2
		start_y = ((camera.resolution.height // 2 - half_size) // 2) * 2
		images, __ = camera.get_raw(start_x, start_y, args.size, args.size, 1)
	else:
		images, __ = camera.get_raw(count=1)
	images = select_color(images, args.color)
	selection = drop_bad_pixels(images, 30)
	image = images[0]

	mean = np.nanmean(selection)
	std = np.nanstd(selection)
	sys.stdout.write(f"Mean: {mean}, stdev: {std}")

	if args.show:
		show_image(image)


def measure_frequency_response(args):
	set_common_camera_args(camera, args)
	half_size = args.size // 2
	start_x = ((camera.resolution.width // 2 - half_size) // 2) * 2
	start_y = ((camera.resolution.height // 2 - half_size) // 2) * 2
	light.set_value(Fraction(1, 2))

	exp_ns = int(args.exposure * 1000000000)
	with open(OUTPUT_DIR / f'frequency_response_iso{args.iso}_exp{exp_ns}.csv', 'w') as csv_fp:
		writer = csv.writer(csv_fp)
		writer.writerow(['frequency', 'total_mean', 'total_std', 'time_std'])
		csv_fp.flush()

		for step in itertools.count():
			frequency = int(float(args.frequency_start) ** (1.0 + step * args.exponent_increment))
			if frequency > args.frequency_end:
				break

			light.set_frequency(frequency)
			images, __ = camera.get_raw(start_x, start_y, args.size, args.size, args.count)
			images = select_color(images, args.color)
			images = drop_bad_pixels(images)

			frame_mean = np.nanmean(images, axis=(1, 2))
			frame_std = np.nanstd(images, axis=(1, 2))
			total_mean = np.mean(frame_mean)
			total_std = np.mean(frame_std)
			time_std = np.std(frame_mean)
			writer.writerow([str(frequency), f'{total_mean:.8f}', f'{total_std:.8f}', f'{time_std:.8f}'])
			csv_fp.flush()


def show_response(args, x):
	#plt.ion()
	#while True:
	args.csv_file.seek(0)
	data = read_csv(args.csv_file, [(x, int), ('total_mean', float), ('total_std', float), ('time_std', float)])
	mpl.style.use('seaborn')
	if x == 'frequency':
		plt.xscale('log')
	plt.plot(getattr(data, x), data.total_mean, '')
	plt.fill_between(getattr(data, x), data.total_mean-data.total_std, data.total_mean+data.total_std, alpha=0.1, facecolor='C0')
	plt.fill_between(getattr(data, x), data.total_mean-data.time_std, data.total_mean+data.time_std, alpha=0.3, facecolor='C0')
	plt.tight_layout()
	plt.show()
	#plt.draw()
	#plt.pause(1)
	#plt.clf()


def show_frequency_response(args):
	show_response(args, 'frequency')


def show_gamma_response(args):
	show_response(args, 'input')


def measure_gamma(args):
	set_common_camera_args(camera, args)
	half_size = args.size // 2
	start_x = ((camera.resolution.width // 2 - half_size) // 2) * 2
	start_y = ((camera.resolution.height // 2 - half_size) // 2) * 2

	light.set_frequency(args.frequency)
	exp_ns = int(args.exposure * 1000000000)
	with open(OUTPUT_DIR / f'gamma_iso{args.iso}_exp{exp_ns}.csv', 'w') as csv_fp:
		writer = csv.writer(csv_fp)
		writer.writerow(['input', 'total_mean', 'total_std', 'time_std'])
		csv_fp.flush()

		for val in range(args.points + 1):
			light.set_value(Fraction(val, args.points))

			images, __ = camera.get_raw(start_x, start_y, args.size, args.size, args.count)
			images = select_color(images, args.color)
			images = drop_bad_pixels(images)

			frame_mean = np.nanmean(images, axis=(1, 2))
			frame_std = np.nanstd(images, axis=(1, 2))
			total_mean = np.mean(frame_mean)
			total_std = np.mean(frame_std)
			time_std = np.std(frame_mean)
			writer.writerow([str(val), f'{total_mean:.8f}', f'{total_std:.8f}', f'{time_std:.8f}'])

			csv_fp.flush()


def create_gamma_curve(args):
	output_filename = os.path.splitext(args.csv_file.name)[0] + '.json'
	data = read_csv(args.csv_file, [('input', int), ('total_mean', float), ('total_std', float), ('time_std', float)])
	black_level = args.black_level
	white_level = args.white_level
	if black_level is None:
		black_level = math.floor(np.min(data.total_mean))
	if white_level is None:
		white_level = math.ceil(np.max(data.total_mean))

	max_input = data.input[0]
	min_input = max_input
	usable_points = 0
	for num, value in enumerate(zip(data.input, data.total_mean), 1):
		input_value, output_value = value
		usable_points = num
		max_input = input_value
		if math.ceil(output_value) == white_level:
			break

	normalized_inputs = [input_value / max_input for input_value in data.input[:usable_points]]
	normalized_outputs = [(output_value - black_level) / (white_level - black_level) for output_value in data.total_mean[:usable_points]]
	normalized_inputs = np.array(normalized_inputs)
	normalized_outputs = np.array(normalized_outputs)

	monotonic = np.ediff1d(normalized_outputs, to_end=[1.0]) >= 0
	normalized_inputs = normalized_inputs[monotonic]
	normalized_outputs = normalized_outputs[monotonic]

	start_point_count = 3
	start_point_density = 1
	end_point_count = 3
	end_point_density = 1
	point_density = 10

	start_points_inputs = np.mean(normalized_inputs[:start_point_count].reshape(-1, start_point_density), axis=1)
	end_points_inputs = np.mean(normalized_inputs[-end_point_count:].reshape(-1, start_point_density), axis=1)
	start_points_outputs = np.mean(normalized_outputs[:start_point_count].reshape(-1, start_point_density), axis=1)
	end_points_outputs = np.mean(normalized_outputs[-end_point_count:].reshape(-1, start_point_density), axis=1)
	mid_points_inputs = np.mean(normalized_inputs[start_point_count:-end_point_count-((len(normalized_inputs)-start_point_count-end_point_count)) % point_density].reshape(-1, point_density), axis=1)
	mid_points_outputs = np.mean(normalized_outputs[start_point_count:-end_point_count-((len(normalized_outputs)-start_point_count-end_point_count)) % point_density].reshape(-1, point_density), axis=1)

	normalized_inputs = np.concatenate([start_points_inputs, mid_points_inputs, end_points_inputs])
	normalized_outputs = np.concatenate([start_points_outputs, mid_points_outputs, end_points_outputs])

	normalized_outputs[0] = 0.0
	normalized_outputs[-1] = 1.0

	with open(output_filename, 'w') as fp:
		json.dump(list(zip(normalized_inputs, normalized_outputs)), fp)


def write_camera_info(args):
	__, metadata = camera.get_raw(0, 0, 0, 0)
	metadata['black_level'] = [int(v) for v in metadata['black_level']]
	properties = {
		'resolution': camera.resolution,
		'iso_range': camera.iso_range,
		'exposure_range': camera.exposure_range,
		'bytes_per_pixel': camera.bytes_per_pixel,
		'pixel_pattern': camera.pixel_pattern,
		'black_level': metadata['black_level'],
		'white_level': metadata['white_level'],
	}
	json.dump(properties, args.json_file, indent='\t')


def generate_dark_frame(args):
	set_common_camera_args(camera, args)
	cumulative_image = None
	for __ in range(args.count):
		raw_image, metadata = camera.get_raw()
		raw_image = raw_image[0].astype(np.int32)
		for plane, black_level in zip(PLANES, metadata['black_level']):
			raw_image[plane] -= int(black_level)
		if cumulative_image is None:
			cumulative_image = raw_image.astype(np.int32)
		else:
			cumulative_image += raw_image
	cumulative_image = cumulative_image.astype(np.float32) / args.count
	np.save(args.output_file, cumulative_image, allow_pickle=False)
	show_image(cumulative_image)


def save_raw(args):
	set_common_camera_args(camera, args)
	raw_image, __ = camera.get_raw()
	np.save(args.output_file, raw_image, allow_pickle=False)
	show_image(raw_image[0])


def write_vignette(args):
	#images = np.load('../output/white.raw')
	#metadata = {'black_level': [16.0, 16.0, 16.0, 16.0], 'white_level': 1023}
	set_common_camera_args(camera, args)
	images, metadata = camera.get_raw()
	images = images.astype(np.float64)

	for plane_slice, black_level in zip(PLANES, metadata['black_level']):
		images[0][plane_slice] -= int(black_level)
		images[0][plane_slice] /= (metadata['white_level'] - black_level)

	gamma_data = np.array(json.load(args.gamma), dtype=np.float64)
	xp = gamma_data[:,1]
	fp = gamma_data[:,0]
	shape = images.shape
	images = np.interp(images.flatten(), xp, fp).reshape(shape)

	image = select_color(images, args.color)[0].astype(np.float64)
	image = cv2.blur(image, (10, 10))
	image = cv2.resize(image, tuple(c//50 for c in reversed(image.shape)), interpolation=cv2.INTER_AREA)
	max_value = np.max(image)
	vignette_map = max_value / image
	show_image(vignette_map)
	np.save(args.output_file, vignette_map, allow_pickle=False)


def main():
	global camera
	global light

	parser = argparse.ArgumentParser(description="Calibrate tools")
	parser.add_argument('--port', type=int, help="Port", default=8421)
	parser.add_argument('--host', type=str, help="Host", default='localhost')
	parser.add_argument('--tty', type=str, help="TTY", default='/dev/ttyUSB0')

	subparsers = parser.add_subparsers(help="Command", dest='command')
	subparsers.required = True

	subparser = subparsers.add_parser('get_resolution', help="Get resolution")
	subparser = subparsers.add_parser('get_bytes_per_pixel', help="Get bytes per pixel")
	subparser = subparsers.add_parser('get_iso_range', help="Get iso range")
	subparser = subparsers.add_parser('get_exposure_range', help="Get exposure range")
	subparser = subparsers.add_parser('get_pixel_pattern', help="Get pixel pattern")
	subparser = add_common_camera_args(subparsers.add_parser('show', help="Show image"))
	subparser = add_common_camera_args(subparsers.add_parser('measure', help="Measure single frame"))
	subparser.add_argument('--size', type=int, help="Capture area size", default=100)
	subparser.add_argument('--color', type=str, choices=["R", "G", "B"], help="Color component", default="G")
	subparser.add_argument('--show', help="Show image", action='store_true')
	subparser = add_common_camera_args(subparsers.add_parser('measure_frequency_response', help="Measure frequency response"))
	subparser.add_argument('--size', type=int, help="Capture area size", default=100)
	subparser.add_argument('--count', type=int, help="Number of frames captured", default=10)
	subparser.add_argument('--color', type=str, choices=["R", "G", "B"], help="Color component", default="G")
	subparser.add_argument('--frequency_start', type=int, help="Frequency start", default=10)
	subparser.add_argument('--frequency_end', type=int, help="Frequency end", default=100000)
	subparser.add_argument('--exponent_increment', type=float, help="Exponent increment", default=0.1)
	subparser = subparsers.add_parser('show_frequency_response', help="Show frequency response")
	subparser.add_argument('csv_file', type=argparse.FileType(mode='r'), help="CSV file")
	subparser = add_common_camera_args(subparsers.add_parser('measure_gamma', help="Measure gamma"))
	subparser.add_argument('--size', type=int, help="Capture area size", default=100)
	subparser.add_argument('--count', type=int, help="Number of frames captured", default=10)
	subparser.add_argument('--color', type=str, choices=["R", "G", "B"], help="Color component", default="G")
	subparser.add_argument('--points', type=int, help="Number of points", default=1024)
	subparser.add_argument('--frequency', type=int, help="Number of points", default=10301)
	subparser = subparsers.add_parser('show_gamma_response', help="Show frequency response")
	subparser.add_argument('csv_file', type=argparse.FileType(mode='r'), help="CSV file")
	subparser = subparsers.add_parser('create_gamma_curve', help="Create gamma curve")
	subparser.add_argument('csv_file', type=argparse.FileType(mode='r'), help="CSV file")
	subparser.add_argument('--black_level', type=int, help="Black level")
	subparser.add_argument('--white_level', type=int, help="White level")
	subparser = subparsers.add_parser('write_camera_info', help="Camera info")
	subparser.add_argument('json_file', type=argparse.FileType(mode='w'), help="JSON file")
	subparser = add_common_camera_args(subparsers.add_parser('generate_dark_frame', help="Generate dark frame"))
	subparser.add_argument('--count', type=int, help="Number of frames captured", default=100)
	subparser.add_argument('output_file', type=argparse.FileType(mode='wb'), help="Dark frame file")
	subparser = add_common_camera_args(subparsers.add_parser('save_raw', help="Save raw"))
	subparser.add_argument('output_file', type=argparse.FileType(mode='wb'), help="Raw file")
	subparser = add_common_camera_args(subparsers.add_parser('write_vignette', help="Write vignette map"))
	subparser.add_argument('gamma', type=argparse.FileType(mode='r'), help="Gamma json")
	subparser.add_argument('output_file', type=argparse.FileType(mode='wb'), help="Raw file")
	subparser.add_argument('--color', type=str, choices=["R", "G", "B"], help="Color component", default="G")

	args = parser.parse_args()
	camera = Camera(args)
	light = Light(args)

	os.makedirs(OUTPUT_DIR, exist_ok=True)

	globals()[args.command](args)


if __name__ == "__main__":
	main()
