# Panorama-Registration-Stitching
This project focuses on the process of automatic "Stereo Mosaicking" using a sequence of images that scan a scene from left to right. The main steps involved in this algorithm are registration and stitching.

Registration: The geometric transformation between consecutive image pairs is determined. This is achieved by:

Detecting Harris feature points in each image.
Extracting descriptors, such as MOPS-like descriptors, from these feature points.
Matching the descriptors between the consecutive image pair.
Using the RANSAC algorithm to fit a rigid transformation (such as translation or rotation) that best aligns the images. The RANSAC algorithm helps to filter out outlier matches and find a transformation that agrees with a large set of inlier matches.
Stitching: The registered images are combined to create a sequence of panoramas. This involves:

Compensating for global motion, such as camera rotation and translation, to align the images.
Stitching together strips from the aligned images to form the panoramas.
The resulting panoramas reveal residual parallax (the apparent shift in position of objects due to the camera's viewpoint change) and other motions.
Overall, the process of stereo mosaicking involves aligning the images by finding the geometric transformation between them and then combining them to create seamless panoramas. This technique allows for a comprehensive view of the scene with reduced parallax and other motion artifacts.




