#!/usr/local/bin/python3

import cv2
import argparse
import os
import matplotlib.pyplot as plt

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-img_ext", "--image_extension", required=False, default='jpg', help="extension name. default is 'jpg'.")
ap.add_argument("-mask_ext", "--mask_extension", required=False, default='png', help="extension name. default is 'png'.")
args = vars(ap.parse_args())

# Arguments
masks_dir_path = 'masks_small'
images_dir_path = 'imgs_small'
# masks_dir_path = 'ml_code_challenge_masks'
# images_dir_path = 'ml_code_challenge_imgs'
mask_ext = args['mask_extension']
image_ext = args['image_extension']

masks = []
for f in os.listdir(masks_dir_path):
    if f.endswith(mask_ext):
        masks.append(f)

images = []
for f in os.listdir(images_dir_path):
    if f.endswith(image_ext):
        images.append(f)


for i in range(len(masks)):
    mask = masks[i]

    mask_path = os.path.join(masks_dir_path, mask)
    mask_frame = cv2.imread(mask_path)
    mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)

    plt.imshow(mask_frame)
    plt.show()

    image = images[i]

    image_path = os.path.join(images_dir_path, image)
    image_frame = cv2.imread(image_path)

    plt.imshow(image_frame)


    plt.show()
    if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
        break

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()


# print("The output video is {}".format(output))
