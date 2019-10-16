import os
import re
from PIL import Image
from sklearn.model_selection import train_test_split
import argparse

from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

import model
# from create_folders import add_frames, add_masks

import pickle

class my_unet():

  def __init__(self, frames, masks, model, batch_size = 16):

    self.model = model.unet()
    self.train_generator=None
    self.val_generator = None
    self.batch_size= batch_size


    self.img_train=None
    self.img_val= None
    self.img_test=None

    self.mask_train= None
    self.mask_val= None
    self.mask_test= None
    self._create_train_val_test_data(frames, masks)

  def _create_train_val_test_data(self, frame, mask):
    ''' Takes in the image frames and masks.
          Then, utilizes ScikitLearn's train test
          split to create the training, validation, and test data.
                      :params: frames- list containing the names of the image files
                              masks-  list containing the names of the mask files
                      :return: img_train, mask_train, img_val,
                              mask_val, img_test, mask_test: List containing
                                                              train-val-test split
                                                              of inputs '''
    img_train, self.img_test, mask_train, self.mask_test = train_test_split(frame, mask, test_size=0.1, random_state=230)
    self.img_train, self.img_val, self.mask_train, self.mask_val= train_test_split(img_train,mask_train, test_size=0.2, random_state=230)


  def data_generators(self, data_path):
    train_datagen =ImageDataGenerator(rescale= 1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)


    train_image_generator = train_datagen.flow_from_directory(f'{data_path}/train_frames/',color_mode="grayscale",batch_size=self.batch_size,class_mode=None)
    train_mask_generator = train_datagen.flow_from_directory(f'{data_path}/train_masks/',color_mode="grayscale",batch_size=self.batch_size,class_mode=None)


    val_image_generator = val_datagen.flow_from_directory(f'{data_path}/val_frames/',color_mode="grayscale",batch_size=self.batch_size,class_mode=None )
    val_mask_generator = val_datagen.flow_from_directory(f'{data_path}/val_masks/', color_mode="grayscale",batch_size=self.batch_size,class_mode=None)

    test_image_generator = test_datagen.flow_from_directory(f'{data_path}/test_frames/',color_mode="grayscale",batch_size=self.batch_size,class_mode=None )
    test_mask_generator = test_datagen.flow_from_directory(f'{data_path}/test_masks/', color_mode="grayscale",batch_size=self.batch_size,class_mode=None)

    self.train_generator = zip(train_image_generator, train_mask_generator)
    self.val_generator = zip(val_image_generator, val_mask_generator)
    self.test_generator = zip(test_image_generator, test_mask_generator)

  def run_unet(self, weights_dir, no_epochs=1):

    no_of_training_imgs= len(self.img_train)
    no_of_val_imgs= len(self.img_val)

    check_point = ModelCheckpoint(weights_dir, monitor='val_accuracy', verbose= 1, save_best_only=True, mode ='max')
    earlystopping = EarlyStopping(verbose=1,monitor='val_accuracy', min_delta=.01, patience = 3, mode = 'max')

    callbacks_list = [check_point, earlystopping]

    results = self.model.fit_generator(self.train_generator, epochs = no_epochs, steps_per_epoch=(no_of_training_imgs//self.batch_size),validation_data= self.val_generator,validation_steps= (no_of_val_imgs//self.batch_size),callbacks= callbacks_list )

    return results

  def evaluate_model(self):
    return self.model.evaluate_generator(self.test_generator, steps= len(self.img_test), verbose=1)

  def predict_new(self, x):
    return self.model.predict(x)


if __name__=="__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("-data", help="path to directory with data",
                        default="/Users/lauren/Desktop/CodingInterview")
  parser.add_argument("-frames", help="path to directory with images",
                        default="/Users/lauren/Desktop/CodingInterview/imgs_small")
  parser.add_argument("-masks", help="path to directory with masks",
                        default="/Users/lauren/Desktop/CodingInterview/masks_small")
  parser.add_argument("-weights", help="path to directory where weights will be saved",
                        default="/Users/lauren/Desktop/CodingInterview/weights/weights.h5")
  parser.add_argument("-batch_size", help="batch size for unet",
                        default=16)
  parser.add_argument("-epochs", help="number of epochs for unet",
                        default=1)


  args = parser.parse_args()

  all_frames = os.listdir(args.frames)
  all_masks = os.listdir(args.masks)

  new_unet= my_unet(all_frames, all_masks, model, args.batch_size)
  # new_unet.create_folders(args.masks, args.frames, args.data)
  new_unet.data_generators(args.data)
  trained_model = new_unet.run_unet(args.weights, args.epochs)

  loss, acc = new_unet.evaluate_model()
  print(f'Loss:{loss}')
  print(f'Accuracy:{acc}')

