import os
import os.path as osp
import numpy as np
import random
import cv2
from torch.utils import data
import imageio
from glob import glob
import tifffile


def get_img_ids(root):
    img_list =[os.path.splitext(os.path.basename(s))[0]
              for s in glob(os.path.join(root, "RGB", '*'))]
    return img_list


class TrainDataSet(data.Dataset):
    def __init__(self, root, crop_size=(256, 256), mean=(128, 128, 128),
                 scale=False, mirror=False, ignore_label=255):
        self.root = root
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_list = get_img_ids(root)
        self.img_ids = [i_id.strip() for i_id in self.img_list]
        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.root, "RGB/%s.png" % name)
            label_file = osp.join(self.root, "NCLS/%s.tif" % name.replace("RGB", "NCLS"))
            depth_file = osp.join(self.root, "Depth/%s.tif" % name.replace("RGB", "DEP"))
            self.files.append({
                "img": img_file,
                "label": label_file,
                "depth": depth_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = imageio.imread(datafiles["img"])
        depth = tifffile.imread(datafiles["depth"])
        label = imageio.imread(datafiles["label"])
        size = image.shape
        name = datafiles["name"]

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)
        depth = np.asarray(depth, np.float32)
        image = image.transpose((2, 0, 1))

        return image.copy(), label.copy(), depth.copy(), np.array(size), name



class ValidDataSet(data.Dataset):
    def __init__(self, root, crop_size=(256, 256), mean=(128, 128, 128),
                 scale=False, mirror=False, ignore_label=255):
        self.root = root
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_list = get_img_ids(root)
        self.img_ids = [i_id.strip() for i_id in self.img_list]
        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.root, "RGB/%s.png" % name)
            label_file = osp.join(self.root, "NCLS/%s.tif" % name.replace("RGB", "NCLS"))
            depth_file = osp.join(self.root, "Depth/%s.tif" % name.replace("RGB", "DEP"))
            self.files.append({
                "img": img_file,
                "label": label_file,
                "depth": depth_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = imageio.imread(datafiles["img"])
        depth = tifffile.imread(datafiles["depth"])
        label = imageio.imread(datafiles["label"])
        size = image.shape
        name = datafiles["name"]

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)
        depth = np.asarray(depth, np.float32)
        image = image.transpose((2, 0, 1))
        depth = depth.transpose((2, 0, 1))

        return image.copy(), label.copy(), depth.copy(), np.array(size), name
