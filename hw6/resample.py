import numpy as np

def resample(particles, particles_w):
	N = particles.shape[0]
	sample_indices = np.random.choice(
		np.arange(N),
		size=N,
		replace=True,
		p=particles_w.flatten()
	)

	return particles[sample_indices], np.ones_like(particles_w, dtype=np.float32) / N