# -*- coding: utf-8 -*-
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


fix_raw_config = {}
dirname = Path(__file__).parent



def fix_file(filename):
	env = os.environ.copy()
	env['LC_ALL'] = 'C'
	tags = subprocess.check_output(['exiv2', '-PE', 'pr', filename], env=env).decode('utf-8')
	parsed_tags = {}
	for line in tags.split('\n'):
		try:
			tag_name, __, __, value = line.split(None, 3)
			parsed_tags[tag_name] = value
		except ValueError:
			pass

	config_file = None
	for config_file, settings in fix_raw_config.items():
		image_settings = {key: parsed_tags.get(key) for key in settings.keys()}
		if image_settings == settings:
			break
	else:
		config_file = None
	if config_file is None:
		sys.stderr.write(f"Configuration for file {filename} not found\n")
		return

	sys.stdout.write(f"Changing {filename} using {config_file}\n")
	subprocess.check_output(['python', dirname / 'fix_raw.py', config_file, filename])


def main():
	suffix = '_correct.dng'

	parser = argparse.ArgumentParser(description="Fix raw")
	parser.add_argument('files', nargs='+', help="Input raw files")
	args = parser.parse_args()

	with open(Path.home() / '.config' / 'fix_raw.json', 'r') as fp:
		fix_raw_config.update(json.load(fp))

	for filename in args.files:
		fix_file(filename)


if __name__ == "__main__":
	main()
