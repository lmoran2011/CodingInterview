# CodingInterview


# Senseye ML Engineer Code Challenge

## Challenge statement:

Provide a solution to the following segmentation problem. Data of close up images of human eyes and pixel level masks classifying parts of the eye anatomy are provided. 
The goal is to provide a method that will produce these masks as outputs when the eye images are used as inputs. 


Eye images are stored in the ml_code_challenge_imgs zip folder and masks are provided in the ml_code_challenge_masks zip folder.
There are 500 images and 500 masks. Images and masks can be matched to each other by using the file name within their respective folders, 
which follow the pattern "set_x_frame_x" (image files end in .jpg and mask files end in .png)
As a png, the mask images look blank upon first inspection, but they contain the class indicators covering parts of the 
eye anatomy as pixel values:

* 1 = sclera
* 2 = iris
* 3 = pupil
* 0 = background


Some examples of how to get this up and running with some common deep learning architectures are: 

* https://towardsdatascience.com/fastai-image-segmentation-eacad8543f6f

* https://towardsdatascience.com/a-keras-pipeline-for-image-segmentation-part-1-6515a421157d

* https://androidkt.com/tensorflow-keras-unet-for-image-image-segmentation/

* https://towardsdatascience.com/u-net-b229b32b4a71


Note: Masks look to be black pictures at first glance, to be able to see the pictures within, install the packages found in setup.py, and then run the visualize.py script. For this scrip to run, you must keep the uncompressed folder names for the masks and the images the same.

When you run the script it will show a mask, and when you close the window the mask is on, it will show the appropriate eye image for it. You can keep closing the windows to go through the masks and images.


#### We find it has been helpful for previous applicants to know what areas we'll be focusing on

* we want to see the organization and structure of your code and modeling process

* we want to see how you develop training and testing of your algorithm

* we want to see how you evaluate the performance of your model

* we are not interested in the final score or accuracy of any model; we are interested in your problem setup

* We understand that these types of problems can easily expand beyond their planned time, 
  you can leave a description in the comments of what you would do for some tasks in replacement 
  of fully fleshing them out the event they require more than reasonable time. 
  
  Include a very short summary of how you've organized the solution and how 
  to run your code. Thank you for your interest in Senseye!
