import gc
import itertools
import matplotlib.pyplot as plt
import numpy as np

from impl.vis import Plot3DPoints
from impl.sfm.corrs import Find2D3DCorrespondences, GetPairMatches, UpdateReconstructionState
from impl.sfm.geometry import DecomposeEssentialMatrix, EstimateEssentialMatrix, TriangulatePoints, TriangulateImage, EstimateImagePose
from impl.sfm.image import Image
from impl.sfm.io import ReadFeatureMatches, ReadKMatrix
from impl.sfm.vis import PlotImages, PlotWithKeypoints, PlotImagePairMatches, PlotCameras, AxisEqual3D

def main():

  np.set_printoptions(linewidth=10000, edgeitems=100, precision=3)

  data_folder = '../data'
  image_names = [
    '0000.png',
    '0001.png',
    '0002.png',
    '0003.png',
    '0004.png',
    '0005.png',
    '0006.png',
    '0007.png',
    '0008.png',
    '0009.png'
  ]
  # Read images
  images = {}
  for im_name in image_names:
    images[im_name] = (Image(data_folder, im_name))

  # Read the matches
  matches = {}
  for image_pair in itertools.combinations(image_names, 2):
    matches[image_pair] = ReadFeatureMatches(image_pair, data_folder)

  K = ReadKMatrix(data_folder)

  init_images = [3, 4]

  """
  # Visualize images and features
  # You can comment these lines once you verified that the images are loaded correctly

  # Show the images
  PlotImages(images)

  # Show the keypoints
  for image_name in image_names:
    PlotWithKeypoints(images[image_name])

  # Show the feature matches
  for image_pair in itertools.combinations(image_names, 2):
    PlotImagePairMatches(images[image_pair[0]], images[image_pair[1]], matches[(image_pair[0], image_pair[1])])
    gc.collect()
  """

  e_im1_name = image_names[init_images[0]]
  e_im2_name = image_names[init_images[1]]
  e_im1 = images[e_im1_name]
  e_im2 = images[e_im2_name]
  e_matches = GetPairMatches(e_im1_name, e_im2_name, matches)

  # TODO Estimate relative pose of first pair
  # Estimate Fundamental matrix
  E = EstimateEssentialMatrix(K, e_im1, e_im2, e_matches)

  # Extract the relative pose from the essential matrix.
  # This gives four possible solutions and we need to check which one is the correct one in the next step
  possible_relative_poses = DecomposeEssentialMatrix(E)

  # TODO
  # For each possible relative pose, try to triangulate points.
  # We can assume that the correct solution is the one that gives the most points in front of both cameras
  # Be careful not to set the transformation in the wrong direction

  max_trig_num = 0
  best_solution = possible_relative_poses[0] # set as default
  for poses in possible_relative_poses:
    e_im1.SetPose(np.eye(3), np.zeros(3))
    # since the essential matrix is from im2 to im1, we take the inverse
    e_im2.SetPose(poses[0].transpose(), -poses[0].transpose() @ poses[1])
    points3D, _, _ = TriangulatePoints(K, e_im1, e_im2, e_matches)
    trig_num = points3D.shape[0]
    if trig_num > max_trig_num:
      max_trig_num = trig_num
      best_solution = poses

  # TODO
  # Set the image poses in the images (image.SetPose(...))
  e_im2.SetPose(best_solution[0].transpose(), -best_solution[0].transpose() @ best_solution[1])

  # TODO Triangulate initial points
  points3D, im1_corrs, im2_corrs = TriangulatePoints(K, e_im1, e_im2, e_matches)

  # Add the new 2D-3D correspondences to the images
  e_im1.Add3DCorrs(im1_corrs, list(range(points3D.shape[0])))
  e_im2.Add3DCorrs(im2_corrs, list(range(points3D.shape[0])))

  # Keep track of all registered images
  registered_images = [e_im1_name, e_im2_name]

  for reg_im in registered_images:
    print(f'Image {reg_im} sees {images[reg_im].NumObserved()} 3D points')

  # Register new images + triangulate
  # Run until we can register all images
  while len(registered_images) < len(images):
    for image_name in images:
      if image_name in registered_images:
        continue

      # Find 2D-3D correspondences
      image_kp_idxs, point3D_idxs = Find2D3DCorrespondences(image_name, images, matches, registered_images)

      # With two few correspondences the pose estimation becomes shaky.
      # Keep this image for later
      if len(image_kp_idxs) < 50:
        continue

      print(f'Register image {image_name} from {len(image_kp_idxs)} correspondences')

      # Estimate new image pose
      R, t = EstimateImagePose(images[image_name].kps[image_kp_idxs], points3D[point3D_idxs], K)

      # Set the estimated image pose in the image and add the correspondences between keypoints and 3D points
      images[image_name].SetPose(R, t)
      images[image_name].Add3DCorrs(image_kp_idxs, point3D_idxs)

      # Triangulate new points wth all previously registered images
      image_points3D, corrs = TriangulateImage(K, image_name, images, registered_images, matches)

      # TODO
      # Update the 3D points and image correspondences
      points3D, images = UpdateReconstructionState(image_points3D, corrs, points3D, images)

      registered_images.append(image_name)

  # Visualize
  fig = plt.figure()
  ax3d = fig.add_subplot(111, projection='3d')
  Plot3DPoints(points3D, ax3d)
  PlotCameras(images, registered_images, ax3d)
  # AxisEqual3D(ax3d)
  
  # Delay termination of the program until the figures are closed
  # Otherwise all figure windows will be killed with the program
  plt.show(block=True)


if __name__ == '__main__':
  main()