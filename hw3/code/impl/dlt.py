import numpy as np

def BuildProjectionConstraintMatrix(points2D, points3D):

  # TODO
  # For each correspondence, build the two rows of the constraint matrix and stack them

  num_corrs = points2D.shape[0]
  constraint_matrix = np.zeros((num_corrs * 2, 12))

  for i in range(num_corrs):
    # append 1 to X to make it homogeneous
    x_t, X_t = points2D[i], np.append(points3D[i], 1)

    constraint_matrix[i * 2][4:] = np.append(-X_t, x_t[1] * X_t)
    constraint_matrix[i * 2 + 1] = np.append(np.append(X_t, np.zeros(4)), -x_t[0] * X_t)

  return constraint_matrix