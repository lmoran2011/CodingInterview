# CodingInterview


# ML Engineer Code Challenge Summary
As UNets and Image Segmentation are new for me, I spent some time getting familiar with them and reading through the various articles (very helpful!). With the time constraint, I used one of the articles as a starter code. I adjusted it the code to work with the given data.

 I opted to use an already created UNet model(model.py) that was linked in the second article. If I have more time, I would like to tweak the unet model and add better documentation to my scripts. Also, I would like to complete the script that I used to split the images into training, validation, and testing data to also allow for a larger or different dataset. 



  ### Include a very short summary of how you've organized the solution and how to run your code.

 The Jupyter Notebook contains the original testing of the code and is where I created the folders and split up the data. The `run_unet_model.py` script is how you will be able to actually run the code. It imports and calls the UNet model(`model.py`) that I borrowed from online.
 
 To run the solution, you will probably need to update the file paths in the args parser of the script `run_unet_model.py`. Then, you should be able to `run run_unet_model.py` in the command line. 
