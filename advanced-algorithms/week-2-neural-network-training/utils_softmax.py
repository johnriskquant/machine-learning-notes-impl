"""Interactive visualization of softmax inputs and outputs."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider




dlc = dict(dlblue = '#0096ff', dlorange = '#FF9300', dldarkred='#C00000', dlmagenta='#FF40FF', dlpurple='#7030A0', dldarkblue =  '#0D5BDC', dlmedblue='#4285F4')

def plt_softmax(my_softmax):
	"""Display logits and their softmax probabilities with interactive sliders."""
	figure, axes = plt.subplots(1, 2, figsize=(8, 4))
	figure.subplots_adjust(bottom=0.35)

	logits = np.array([1.0, 2.0, 3.0, 4.0])
	sliders = []
	for index, value in enumerate(logits):
		slider_axes = figure.add_axes((0.15, 0.10 + index * 0.05, 0.30, 0.03))
		sliders.append(
			Slider(slider_axes, f"z{index}", 0.1, 10.0, valinit=value, valstep=0.1)
		)

	input_bars = axes[0].barh([f"z{i}" for i in range(4)], logits, height=0.6)
	axes[0].set_xlim(0, 10)
	axes[0].set_title("z input to softmax")

	probabilities = my_softmax(logits)
	output_bars = axes[1].barh(
		[f"a{i}" for i in range(4)],
		probabilities,
		height=0.6,
		color=dlc["dldarkred"],
	)
	axes[1].set_xlim(0, 1)
	axes[1].set_title("softmax(z)")

	def update(_value):
		current_logits = np.array([slider.val for slider in sliders])
		current_probabilities = my_softmax(current_logits)

		for bar, value in zip(input_bars, current_logits):
			bar.set_width(value)
		for bar, value in zip(output_bars, current_probabilities):
			bar.set_width(value)

		figure.canvas.draw_idle()

	for slider in sliders:
		slider.on_changed(update)

	plt.show()
