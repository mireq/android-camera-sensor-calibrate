# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


with plt.xkcd():
	plt.rcParams.update({'font.size': 22})

	fig = plt.figure()
	ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
	ax.spines.right.set_color('none')
	ax.spines.top.set_color('none')
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_xlim([-1, 20])
	ax.set_ylim([0, 10])

	data = np.ones(100)
	data[70:] -= np.arange(30)

	ax.annotate("Nízka frekvencia\nv porovnaní s expozíciou", xy=(1, 7), arrowprops=dict(arrowstyle='->'), xytext=(1, 4))
	ax.annotate("Začína prevládať\nindukčnosť / kapacita", xy=(18, 7), arrowprops=dict(arrowstyle='->'), xytext=(17, 2), ha='right')
	ax.fill_between([-1, 0, 1, 2, 3], [0, 0, 6, 7.5, 8], [16, 16, 10, 8.5, 8], alpha=0.1, facecolor='C0')


	ax.plot([v/3 for v in range(0, 9)] + list(range(3, 20)), [4, 9, 5, 8, 7.5, 7, 8.5, 7.7] + [8] * 16 + [7, 5], color='C0')
	ax.plot([17, 18, 19], [8, 9, 10], color='C0', linestyle='dashed')

	ax.set_xlabel("Frekvencia")
	ax.set_ylabel("Nameraná hodnota")
	fig.text(0.5, 0.9, "Očakávaný výstup", ha='center')
	plt.show()
