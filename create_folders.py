import os
import re
from PIL import Image


  def _add_frames(self, dir_name, image, frame_path, data_path):
      path = "imgs_small/" + image
      img = Image.open(path)
      new_name = dir_name +"/frames/"+ image

      img.save(new_name)

  def _add_masks(self, dir_name, image, mask_path, data_path):
      path = 'masks_small/'+image
      img = Image.open(path)
      new_name = dir_name + '/masks/'+ image
      img.save(new_name)


  def _add_folders(self, folders, method, path, data_path):
    for folder in folders:
      array = folder[0]
      name = [folder[1]] *len(array)
      list(map(method, name, array, path, data_path))



  def create_folders(self,mask_path, frames_path, data_path):
    folders = ['train_frames', 'train_masks','val_frames', 'val_masks', 'test_frames', 'test_masks']

    for folder in folders:
        os.makedirs(data_path+'/'+folder+'/data')

    frame_folders= [(self.img_train, 'train_frames'),(self.img_val, 'val_frames'), (self.img_test, 'test_frames')]
    mask_folders= [(self.mask_train, 'train_masks'),(self.mask_val, 'val_masks'), (self.mask_test, 'test_masks')]

    self._add_folders( frame_folders, self._add_frames,frames_path, data_path)
    self._add_folders( mask_folders, self._add_masks, mask_path, data_path)