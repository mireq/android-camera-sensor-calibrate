# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


BYTES_PER_PIXEL = 2


def load_raw_file(filename, width, height):
	with open(filename, 'rb') as fp:
		return np.fromfile(fp, dtype=np.int16).reshape((height, width))


def main():
	d = load_raw_file('/dev/shm/raw', 4208, 3120)
	#d = load_raw_file('/dev/shm/raw2', 100, 100)
	#mpl.rcParams['toolbar'] = 'None'
	plt.imshow(d, cmap='gray')
	plt.tight_layout()
	plt.show()
	#print(d)
	#cv2.imshow("ok", d)
	#cv2.waitKey(0)



if __name__ == "__main__":
	main()
