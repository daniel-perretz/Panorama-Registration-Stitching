import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import map_coordinates
from scipy.ndimage.filters import convolve
from scipy.ndimage import label, center_of_mass, filters
import shutil
from imageio import imwrite
from imageio import imread
from imageio import imwrite
from skimage.color import rgb2gray
import sol4_utils


def harris_corner_detector(im):
    """
  Detects harris corners.
  Make sure the returned coordinates are x major!!!
  :param im: A 2D array representing an image.
  :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates
   of the ith corner points.
  """
    k = 0.04
    filter_x = [[1, 0, -1]]
    filter_y = np.transpose(filter_x)
    Ix_derivative = filters.convolve(im, filter_x)
    Iy_derivative = filters.convolve(im, filter_y)
    Ix_2 = sol4_utils.blur_spatial(Ix_derivative * Ix_derivative, 3)
    Iy_2 = sol4_utils.blur_spatial(Iy_derivative * Iy_derivative, 3)
    IxIy = sol4_utils.blur_spatial(Ix_derivative * Iy_derivative, 3)
    IyIx = sol4_utils.blur_spatial(Iy_derivative * Ix_derivative, 3)
    detM = Ix_2 * Iy_2 - IxIy * IyIx
    traceM = Ix_2 + Iy_2
    R = detM - k * traceM ** 2
    binray_img = non_maximum_suppression(R)
    rows = np.where(binray_img)[0]
    cols = np.where(binray_img)[1]

    return np.stack((cols, rows), axis=1)


def sample_descriptor(im, pos, desc_rad):
    """
  Samples descriptors at the given corners.
  :param im: A 2D array representing an image.
  :param pos: An array with shape (N,2), where pos[i,:] are the [x,y]
   coordinates of the ith corner point.
  :param desc_rad: "Radius" of descriptors to compute.
  :return: A 3D array with shape (N,K,K) containing the ith
    descriptor at desc[i,:,:].
  """

    zero_descriptor = np.zeros(49).reshape(7,7)
    K = 1+2*desc_rad
    patch_list = np.zeros((len(pos),K,K))
    index = 0
    for pos in pos:
        y_list = np.repeat(np.arange(pos[0]-3,(pos[0]+3)+1),K)
        x_list = np.tile(np.arange(pos[1]-3,(pos[1]+3)+1),K)
        patch = np.stack((x_list,y_list),axis=0)
        new_coor = map_coordinates(im,patch,order=1,
                                   prefilter=False).reshape(K,K)
        avg = np.mean(new_coor)
        norm = np.linalg.norm(new_coor-avg)

        if(norm == 0):
            d = zero_descriptor
            patch_list[index] = d
        else:
            d = (new_coor-avg)/norm
            patch_list[index] = d
        index+=1
    return patch_list

def find_features(pyr):
    """
  Detects and extracts feature points from a pyramid.
  :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
  :return: A list containing:
              1) An array with shape (N,2) of [x,y] feature location per row
              found in the image.
                 These coordinates are provided at the pyramid level pyr[0].
              2) A feature descriptor array with shape (N,K,K)
  """
    pos = spread_out_corners(pyr[0],7,7,13)
    patch_list = sample_descriptor(pyr[2],pos/4,3)
    return [pos, patch_list]


def match_features(desc1, desc2, min_score):
    """
  Return indices of matching descriptors.
  :param desc1: A feature descriptor array with shape (N1,K,K).
  :param desc2: A feature descriptor array with shape (N2,K,K).
  :param min_score: Minimal match score.
  :return: A list containing:
              1) An array with shape (M,) and dtype int of matching indices in
               desc1.
              2) An array with shape (M,) and dtype int of matching indices
              in desc2.
  """

    flatten1 = desc1.reshape((desc1.shape[0],desc1.shape[1]**2))
    flatten2 = desc2.reshape((desc2.shape[0], desc2.shape[1] **2))
    s_matrix = flatten1@flatten2.T

    return match_features_helper(s_matrix,min_score)




def match_features_helper(b,min_score):
    b_transpose = np.transpose(b)
    two_greatest_each_row = np.argpartition(b, -2, axis=1)[:, -2:]
    two_greatest_each_col = np.argpartition(b_transpose, -2, axis=1)[:, -2:]
    row_array = []
    col_array = []
    for row in range(len(b)):
        temp_row = b[row]
        biggest_index = two_greatest_each_row[row][1]
        smallest_index = two_greatest_each_row[row][0]
        if (temp_row[biggest_index] == b_transpose[biggest_index]
        [two_greatest_each_col[biggest_index][0]]
        and temp_row[biggest_index] > min_score):
            row_array.append(row)
            col_array.append(biggest_index)
            b[:,two_greatest_each_row[row][0]] = min_score

        elif(temp_row[biggest_index] == b_transpose[biggest_index]
        [two_greatest_each_col[biggest_index][1]]
        and temp_row[biggest_index] >min_score):
            row_array.append(row)
            col_array.append(biggest_index)
            b[:,two_greatest_each_row[row][1]] = min_score

        elif (temp_row[smallest_index] == b_transpose[smallest_index][
            two_greatest_each_col[smallest_index][0]]
        and temp_row[smallest_index] > min_score):

            row_array.append(row)
            col_array.append(smallest_index)
            b[:, two_greatest_each_row[row][0]] = min_score

        elif(temp_row[smallest_index] == b_transpose[smallest_index][
                  two_greatest_each_col[smallest_index][1]]
        and temp_row[smallest_index] > min_score):
            row_array.append(row)
            col_array.append(smallest_index)
            b[:, two_greatest_each_row[row][1]] = min_score

    return [np.array(row_array,dtype=int),np.array(col_array,dtype=int)]



def apply_homography(pos1, H12):
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates
    obtained from transforming pos1 using H12.
    """
    homo_coords = np.hstack((pos1,[[1]]*pos1.shape[0]))
    transformed_coords = homo_coords @ H12.T
    return transformed_coords[:, :2] / transformed_coords[:, 2:]

def ransac_homography(points1, points2, num_iter, inlier_tol,
                      translation_only=False):
    """
  Computes homography between two sets of points using RANSAC.
  :param pos1: An array with shape (N,2) containing N rows of [x,y]
  coordinates of matched points in image 1.
  :param pos2: An array with shape (N,2) containing N rows of [x,y]
  coordinates of matched points in image 2.
  :param num_iter: Number of RANSAC iterations to perform.
  :param inlier_tol: inlier tolerance threshold.
  :param translation_only: see estimate rigid transform
  :return: A list containing:
              1) A 3x3 normalized homography matrix.
              2) An Array with shape (S,) where S is the number of inliers,
                  containing the indices in pos1/pos2 of the maximal
                  set of inlier matches found.
  """

    N = len(points1)
    opt_inline = []
    for iter in range(num_iter):
        to_choose = 1 if translation_only else 2
        rand_numbers = np.random.choice(N, to_choose)

        #second step
        rigid_transform = estimate_rigid_transform(points1[rand_numbers],
                                             points2[rand_numbers],
                                                   translation_only)

        #third step
        P2_transformed = apply_homography(points1,rigid_transform)

        E = np.linalg.norm(P2_transformed-points2,axis=1)**2
        inlier_matches = np.nonzero(E<inlier_tol)

        if(len(opt_inline)==0):
            opt_inline = inlier_matches

        elif(len(inlier_matches[0]) > len(opt_inline[0])):
            opt_inline = inlier_matches

    opt_tranformaion =\
        estimate_rigid_transform(points1[opt_inline],
                                                points2[opt_inline],
                                 translation_only)

    return opt_tranformaion,opt_inline[0]



def display_matches(im1, im2, points1, points2, inliers):
    """
  Dispalay matching points.
  :param im1: A grayscale image.
  :param im2: A grayscale image.
  :parma pos1: An aray shape (N,2), containing N rows of [x,y] coordinates
   of matched points in im1.
  :param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates
   of matched points in im2.
  :param inliers: An array with shape (S,) of inlier matches.
  """
    x1_coors = points1[:,0]
    x2_coors = points2[:,0]
    y1_coors = points1[:,1]
    y2_coors = points2[:,1]
    outliers = np.arange(len(points1))
    outliers[inliers] = False
    outliers = outliers[outliers!=False]
    overlayed_images = np.hstack((im1,im2))
    plt.imshow(overlayed_images,cmap="gray")

    red_doot_x = np.stack((x1_coors,x2_coors+len(im1[0])))
    red_doot_y = np.stack((y1_coors, y2_coors))
    plt.plot(red_doot_x, red_doot_y,color='r', linestyle ='None',
             marker ='o',markersize = 1, lw = 1)

    outliers_x = np.stack((x1_coors[outliers],x2_coors[outliers]+len(im1[0])))
    outliers_y = np.stack((y1_coors[outliers], y2_coors[outliers]))
    plt.plot(outliers_x,outliers_y,mfc='b',c='b',lw=.4,ms=10)

    inliers_x = np.stack((x1_coors[inliers],x2_coors[inliers]+len(im1[0])))
    inliers_y = np.stack((y1_coors[inliers],y2_coors[inliers]))
    plt.plot(inliers_x,inliers_y,mfc='y',c='y',lw=.4,ms=10)

    plt.show()




def accumulate_homographies(H_succesive, m):
    """
  Convert a list of succesive homographies to a 
  list of homographies to a common reference frame.
  :param H_successive: A list of M-1 3x3 homography 
    matrices where H_successive[i] is a homography which transforms points
    from coordinate system i to coordinate system i+1.
  :param m: Index of the coordinate system towards which we would like to 
    accumulate the given homographies.
  :return: A list of M 3x3 homography matrices, 
    where H2m[i] transforms points from coordinate system i to coordinate
     system m
  """
    H2m = np.zeros((len(H_succesive)+1,3,3))
    for i in range(m,-1,-1):
        if i == m:
            H2m[i] = np.eye(3)
        else:
            H2m[i] = H2m[i+1] @ H_succesive[i]

    for i in range(m+1,len(H2m)):
            H2m[i] = \
                (H2m[i-1]) @\
                np.linalg.inv(H_succesive[i-1])

    for i in range(len(H2m)):
        H2m[i] = H2m[i]/H2m[i,2,2]

    return list(H2m)


def compute_bounding_box(homography, w, h):
    """
  computes bounding box of warped image under homography, without actually
   warping the image
  :param homography: homography
  :param w: width of the image
  :param h: height of the image
  :return: 2x2 array, where the first row is [x,y] of the top left corner,
   and the second row is the [x,y] of the bottom right corner
  """
    returned_array = np.zeros((2,2)).astype(int)
    zero_points = apply_homography(np.array([[0,0]]),homography)
    zero_h = apply_homography(np.array([[0, h]]), homography)
    zero_w = apply_homography(np.array([[w, 0]]), homography)
    w_h = apply_homography(np.array([[w, h]]), homography)

    transposed_stack = np.stack((zero_points,zero_h,zero_w,w_h)).T
    x_max = np.max(transposed_stack[0])
    x_min = np.min(transposed_stack[0])
    y_max = np.max(transposed_stack[1])
    y_min = np.min(transposed_stack[1])

    returned_array[0][0] = x_min
    returned_array[0][1] = y_min
    returned_array[1][0] = x_max
    returned_array[1][1] = y_max
    return returned_array






def warp_channel(image, homography):
    """
  Warps a 2D image with a given homography.2
  :param image: a 2D image.
  :param homography: homograhpy.
  :return: A 2d warped image.
  """
    bounding_box = compute_bounding_box(homography,len(image[0]),len(image))
    x_dist = np.arange(bounding_box[0][0],bounding_box[1][0])
    y_dist = np.arange(bounding_box[0][1],bounding_box[1][1])
    Xcoord,Ycoord = np.meshgrid(x_dist,y_dist)
    stack = np.stack((Xcoord.reshape(len(Xcoord)*len(Xcoord[0])),
                      Ycoord.reshape(len(Ycoord)*len(Ycoord[0]))))

    canvas_index = apply_homography(stack.T,np.linalg.inv(homography))
    transformed_img = map_coordinates(image,canvas_index.T[::-1],order=1,
                                   prefilter=False)

    return transformed_img.reshape((len(y_dist),len(x_dist)))



def warp_image(image, homography):
    """
  Warps an RGB image with a given homography.
  :param image: an RGB image.
  :param homography: homograhpy.
  :return: A warped image.
  """
    return np.dstack(
        [warp_channel(image[..., channel], homography) for channel in
         range(3)])


def filter_homographies_with_translation(homographies,
                                         minimum_right_translation):
    """
  Filters rigid transformations encoded as homographies by the amount
   of translation from left to right.
  :param homographies: homograhpies to filter.
  :param minimum_right_translation: amount of translation below which
  the transformation is discarded.
  :return: filtered homographies..
  """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if homographies[i][0, -1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
    """
  Computes rigid transforming points1 towards points2,
  using least squares method.
  points1[i,:] corresponds to poins2[i,:]. In every point,
  the first coordinate is *x*.
  :param points1: array with shape (N,2). Holds coordinates
  of corresponding points from image 1.
  :param points2: array with shape (N,2). Holds coordinates
  of corresponding points from image 2.
  :param translation_only: whether to compute translation
  only. False (default) to compute rotation as well.
  :return: A 3x3 array with the computed homography.
  """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2, :2] = rotation
    H[:2, 2] = translation
    return H


def non_maximum_suppression(image):
    """
  Finds local maximas of an image.
  :param image: A 2D array representing an image.
  :return: A boolean array with the same shape as the input image,
  where True indicates local maximum.
  """
    # Find local maximas.
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max() * 0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
    centers = np.stack(centers).round().astype(np.int)
    ret = np.zeros_like(image, dtype=np.bool)
    ret[centers[:, 0], centers[:, 1]] = True

    return ret


def spread_out_corners(im, m, n, radius):
    """
  Splits the image im to m by n rectangles and uses harris_corner_detector
   on each.
  :param im: A 2D array representing an image.
  :param m: Vertical number of rectangles.
  :param n: Horizontal number of rectangles.
  :param radius: Minimal distance of corner points from the boundary
   of the image.
  :return: An array with shape (N,2), where ret[i,:] are the [x,y]
  coordinates of the ith corner points.
  """
    corners = [np.empty((0, 2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n + 1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m + 1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j + 1], x_bound[i]:x_bound[i + 1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:, 0] > radius) & (
                corners[:, 0] < im.shape[1] - radius) &
             (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
    ret = corners[legit, :]
    return ret


class PanoramicVideoGenerator:
    """
  Generates panorama from a set of images.
  """

    def __init__(self, data_dir, file_prefix, num_images, bonus=False):
        """
    The naming convention for a sequence of images is file_prefixN.jpg,
    where N is a running number 001, 002, 003...
    :param data_dir: path to input images.
    :param file_prefix: see above.
    :param num_images: number of images to produce the panoramas with.
    """
        self.bonus = bonus
        self.file_prefix = file_prefix
        self.files = [
            os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i
            in range(num_images)]
        self.files = list(filter(os.path.exists, self.files))
        self.panoramas = None
        self.homographies = None
        print('found %d images' % len(self.files))

    def align_images(self, translation_only=False):
        """
    compute homographies between all images to a common coordinate system
    :param translation_only: see estimte_rigid_transform
    """
        # Extract feature point locations and descriptors.
        points_and_descriptors = []
        for file in self.files:
            image = sol4_utils.read_image(file, 1)
            self.h, self.w = image.shape
            pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
            points_and_descriptors.append(find_features(pyramid))

        # Compute homographies between successive pairs of images.
        Hs = []
        for i in range(len(points_and_descriptors) - 1):
            points1, points2 = points_and_descriptors[i][0], \
                               points_and_descriptors[i + 1][0]
            desc1, desc2 = points_and_descriptors[i][1], \
                           points_and_descriptors[i + 1][1]

            # Find matching feature points.
            ind1, ind2 = match_features(desc1, desc2, .7)
            points1, points2 = points1[ind1, :], points2[ind2, :]

            # Compute homography using RANSAC.
            H12, inliers = ransac_homography(points1, points2, 100, 6,
                                             translation_only)

            # Uncomment for debugging: display inliers and outliers
            # among matching points.
            # In the submitted code this function should be commented out!
            # display_matches(self.images[i], self.images[i+1],
            # points1 , points2, inliers)

            Hs.append(H12)

        # Compute composite homographies from the central coordinate system.
        accumulated_homographies = accumulate_homographies(Hs,
                                                           (len(Hs) - 1) // 2)
        self.homographies = np.stack(accumulated_homographies)
        self.frames_for_panoramas = filter_homographies_with_translation(
            self.homographies, minimum_right_translation=5)
        self.homographies = self.homographies[self.frames_for_panoramas]

    def generate_panoramic_images(self, number_of_panoramas):
            """
      combine slices from input images to panoramas.
      :param number_of_panoramas: how many different slices
      to take from each input image
      """
            if self.bonus:
                self.generate_panoramic_images_bonus(number_of_panoramas)
            else:
                self.generate_panoramic_images_normal(number_of_panoramas)

    def generate_panoramic_images_normal(self, number_of_panoramas):
        """
    combine slices from input images to panoramas.
    :param number_of_panoramas: how many different slices
     to take from each input image
    """
        assert self.homographies is not None

        # compute bounding boxes of all warped input images in
        # the coordinate system of the middle image (as given by
        # the homographies)
        self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
        for i in range(self.frames_for_panoramas.size):
            self.bounding_boxes[i] = compute_bounding_box(self.homographies[i],
                                                          self.w, self.h)

        # change our reference coordinate system to the panoramas
        # all panoramas share the same coordinate system
        global_offset = np.min(self.bounding_boxes, axis=(0, 1))
        self.bounding_boxes -= global_offset

        slice_centers = np.linspace(0, self.w, number_of_panoramas + 2,
                                    endpoint=True, dtype=np.int)[1:-1]
        warped_slice_centers = np.zeros(
            (number_of_panoramas, self.frames_for_panoramas.size))
        # every slice is a different panorama, it indicates
        # the slices of the input images from which the panorama
        # will be concatenated
        for i in range(slice_centers.size):
            slice_center_2d = np.array([slice_centers[i], self.h // 2])[None,
                              :]
            # homography warps the slice center to the
            # coordinate system of the middle image
            warped_centers = [apply_homography(slice_center_2d, h) for h in
                              self.homographies]
            # we are actually only interested in the x
            # coordinate of each slice center in the panoramas'
            # coordinate system
            warped_slice_centers[i] = np.array(warped_centers)[:, :,
                                      0].squeeze() - global_offset[0]

        panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(
            np.int) + 1

        # boundary between input images in the panorama
        x_strip_boundary = ((warped_slice_centers[:,
                             :-1] + warped_slice_centers[:, 1:]) / 2)
        x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                      x_strip_boundary,
                                      np.ones((number_of_panoramas, 1)) *
                                      panorama_size[0]])
        x_strip_boundary = x_strip_boundary.round().astype(np.int)

        self.panoramas = np.zeros(
            (number_of_panoramas, panorama_size[1], panorama_size[0], 3),
            dtype=np.float64)
        for i, frame_index in enumerate(self.frames_for_panoramas):
            # warp every input image once, and populate all panoramas
            image = sol4_utils.read_image(self.files[frame_index], 2)
            warped_image = warp_image(image, self.homographies[i])
            x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
            y_bottom = y_offset + warped_image.shape[0]

            for panorama_index in range(number_of_panoramas):
                # take strip of warped image and paste to current panorama
                boundaries = x_strip_boundary[panorama_index, i:i + 2]
                image_strip = warped_image[:,
                              boundaries[0] - x_offset: boundaries[
                                                            1] - x_offset]
                x_end = boundaries[0] + image_strip.shape[1]
                self.panoramas[panorama_index, y_offset:y_bottom,
                boundaries[0]:x_end] = image_strip

        # crop out areas not recorded from enough angles
        # assert will fail if there is overlap in field of view
        # between the left most image and the right most image
        crop_left = int(self.bounding_boxes[0][1, 0])
        crop_right = int(self.bounding_boxes[-1][0, 0])
        assert crop_left < crop_right, 'for testing your code ' \
                                       'with a few images do not crop.'
        print(crop_left, crop_right)
        self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

    def generate_panoramic_images_bonus(self, number_of_panoramas):
        """
    The bonus
    :param number_of_panoramas: how many different slices
    to take from each input image
    """
        pass

    def save_panoramas_to_video(self):
        assert self.panoramas is not None
        out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
        try:
            shutil.rmtree(out_folder)
        except:
            print('could not remove folder')
            pass
        os.makedirs(out_folder)
        # save individual panorama images to 'tmp_folder_for_panoramic_frames'
        for i, panorama in enumerate(self.panoramas):
            imwrite('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
        if os.path.exists('%s.mp4' % self.file_prefix):
            os.remove('%s.mp4' % self.file_prefix)
        # write output video to current folder
        os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
                  (out_folder, self.file_prefix))

    def show_panorama(self, panorama_index, figsize=(20, 20)):
        assert self.panoramas is not None
        plt.figure(figsize=figsize)
        plt.imshow(self.panoramas[panorama_index].clip(0, 1))
        plt.show()

