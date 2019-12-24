# -*- coding: utf-8 -*-
import cv2
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from augmentation import * 
import glob
from PIL import Image


DATA_CLASS= ('red','green','yellow','red_left','red_right','yellow_left','yellow_right',
            'green_left','green_right','red_forward','green_forward','yellow_forward','horizon_red',
            'horizon_green','horizon_yellow','off','traffic_sign','car','motor','bike','bus',
            'truck','suv','express','person')

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=[size,size], mode="nearest").squeeze(0)
    return image

def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(data.Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img


def base_transform(image, size, mean):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x

class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean), boxes, labels

class DataDetection(data.Dataset):
    def __init__(self,root='/home/wh/DF_1018',image_set='train_image',detect_label='train.txt',transform=None, train=True, multiscale=True,minsize=416,maxsize=672):
        self.root = root
        self.image_set = image_set
        self.detect_label = detect_label
        self.transform = transform
        self.multiscale = multiscale
        '''
        self.batch_count = 0
        self.img_size = None
        self.min_size = minsize
        self.max_size = maxsize
        '''
        file = open(osp.join(self.root, self.detect_label))
        lines = file.read().split('\n') 
        image_file_path = []
        segmentation_label_path = []
        bboxes = []
        for line in lines:
            l = line.split(' ')
            obj_nums = len(l) - 2
            if(obj_nums<=0):
                continue
            image_file_path.append(l[0].rstrip().lstrip()) # image
            segmentation_label_path.append(l[1].rstrip().lstrip) # seg
            bb = []
            for i in range(obj_nums):
                bb.append([int(j.rstrip().lstrip()) for j in l[2+i].split(',')])
            bboxes.append(bb)
        
        imgs_num = len(image_file_path)
        self.train = train
        if self.train:
            self.image_file_path = image_file_path[:int(0.9*imgs_num)]
            self.segmentation_label_path = segmentation_label_path[:int(0.9*imgs_num)]
            self.bboxes = bboxes[:int(0.9*imgs_num)]
        else:
            self.image_file_path = image_file_path[int(0.9 * imgs_num):]
            self.segmentation_label_path = segmentation_label_path[int(0.9*imgs_num):]
            self.bboxes = bboxes[int(0.9*imgs_num):]

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        return im, gt
    
    def __len__(self):
        return len(self.image_file_path)
    
    def pull_item(self, index):
        img_id = self.image_file_path[index]
        img = cv2.imread(osp.join(self.root,self.image_set,img_id))
        height, width, _ = img.shape
        target = self.bboxes[index] #[[xmin, ymin, xmax, ymax, label_ind, confidence], ... ] list
        # label_ind:[1~25]
        # scale target to 0～1
        scale = np.array([width, height, width, height])
        target = np.array(target)
        boxes = 1.0*target[:,:4]/scale 
        label = target[:,4]-1 # label_ind:[0~24]
        if self.transform is not None:
            img, boxes, label = self.transform(img, boxes, label)
            # to rgb
            img = img[:, :, (2, 1, 0)]
        target = np.hstack((np.expand_dims(label, axis=1),boxes))
        targets = np.zeros((len(boxes), 6))
        targets[:, 1:] = target 
        return torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(targets), height, width
        # img(tensor)
        # target(numpy) boxes scale to 0~1
        #               label scale to 0~24

    def collate_fn(self, batch):
        imgs, targets = list(zip(*batch))
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        '''
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        '''
        imgs = torch.stack([img for img in imgs])
        
        return imgs, targets

