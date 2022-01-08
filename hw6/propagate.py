import numpy as np

def propagate(particles, frame_height, frame_width, params):
	model = params['model']
	sigma_pos = params['sigma_position']
	sigma_vel = params['sigma_velocity']

	indices = np.arange(len(particles))
	propagated_particales = np.ones_like(particles)

	while len(indices) > 0:

		noise = np.random.standard_normal(particles[indices].shape)
		noise[:, :2] *= sigma_pos

		if model == 0:
			A = np.eye(2)
		else:
			A = np.eye(4)
			A[0][2] = 1
			A[1][3] = 1
			noise[:, 2:] *= sigma_vel

		propagated_particales[indices] = (A @ particles[indices].T).T + noise
		indices = [
			i for i in indices
			if (propagated_particales[i][0] < (params['bbox_width'] / 2)) 
			or (propagated_particales[i][0] >= (frame_width - params['bbox_width'] / 2))
			or (propagated_particales[i][1] < (params['bbox_height'] / 2)) 
			or (propagated_particales[i][1] >= (frame_height - params['bbox_height'] / 2))
		]

	return propagated_particales
