import numpy as np

from scipy.stats import norm
from scipy.special import softmax
from color_histogram import color_histogram
from chi2_cost import chi2_cost


def observe(
	particles, 
	frame, 
	bbox_height, 
	bbox_width, 
	hist_bin, 
	target_hist, 
	sigma_observe
):
	particles_w = np.ones((particles.shape[0], 1))
	for i, particle in enumerate(particles):
		p_hist = color_histogram(
				int(particle[0] - bbox_width / 2),
				int(particle[1] - bbox_height / 2),
				int(particle[0] + bbox_width / 2),
				int(particle[1] + bbox_height / 2),
				frame,
				hist_bin
			)

		chi2 = chi2_cost(p_hist, target_hist)
		particles_w[i] = norm.pdf(chi2, loc=0, scale=sigma_observe)

	# normalization
	particles_w /= np.sum(particles_w)

	return particles_w