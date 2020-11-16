# coding=utf-8
import json
import numpy as np
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import os
# -----------------
# txt数据格式：path，xmin,ymin,xmax,ymax
# 每一行表示一个image
# --------------------
filename = 'train.json'#修改成你的json文件名字
jsondir = '/hdd2/wh/coco2017/annotations/instances_train2017.json'
img_root = '/hdd2/wh/coco2017/train2017'
f = open(jsondir,encoding='utf-8')
res = f.read()
data = json.loads(res)
# 保存数据的文件夹
folder = filename.split('.')[0]+'_txt'
if not os.path.exists(folder):
    os.mkdir(folder)
CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
cat2label = {cat: i for i, cat in enumerate(CLASSES)}
# 首先得到数据的categories的关键字
category = data['categories']
category_id = {}
for category_per in category:
    id = category_per['id']
    cls = category_per['name']
    category_id[id] = cat2label[cls]
print(category_id)
file_write = folder + '/' + filename.split('.')[0] + '.txt'
# 开始遍历字典，对每一个图像生成xml文件
imageID_all =[]
imageID_all_info = {}
for images_attr in list(data.keys()):
    if  images_attr == 'images':
        # 遍历每一个图像
        for data_per in data[images_attr]:
            image_name = data_per['file_name']
            image_route = os.path.join(img_root,image_name) 
            image_width = data_per['width']
            image_height = data_per['height']
            image_id = data_per['id']
            imageID_all.append(image_id)
            imageID_all_info[image_id]={'width':image_width,'height':image_height,'path':image_route,'filename':image_name}
    elif images_attr == 'annotations':
        for imageID_per in imageID_all:
            image_path = imageID_all_info[imageID_per]['path']
            # 图像包含了多少个bounding box
            boundingBox_image = [j for j in data[images_attr] if j['image_id']==imageID_per]
            boundingBox_cord =''
            path_cord =''
            if len(boundingBox_image)==0:
                continue
            for boundingBox_per in boundingBox_image: 
                if boundingBox_per.get('ignore', False):
                    continue
                if boundingBox_per.get('iscrowd', False):
                    continue
                #dict_keys(['segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id'])
                id = boundingBox_per['category_id']
                label = category_id[id]
                # xmin,ymin,w,h->xmin,ymin,xmax,ymax
                x = int(boundingBox_per['bbox'][0])
                y = int(boundingBox_per['bbox'][1])
                w = int(boundingBox_per['bbox'][2])
                h = int(boundingBox_per['bbox'][3])
                if boundingBox_per['area'] <= 0 or w < 1 or h < 1:
                    continue
                inter_w = max(0, min(x + w, imageID_all_info[imageID_per]['width']) - max(x, 0))
                inter_h = max(0, min(y + h, imageID_all_info[imageID_per]['height']) - max(y, 0))
                if inter_w * inter_h == 0:
                    continue
                xmin = str(x)
                ymin = str(y)
                xmax= str(x+w)
                ymax=str(y+h)
                boundingBox_cord += xmin +','+ymin+','+xmax+','+ymax+','+str(label)+' '
            boundingBox_cord = boundingBox_cord.rstrip()
            boundingBox_cord += '\n'
            path_cord = image_path + ' '+ boundingBox_cord
            with open(file_write, 'a+') as f:
                f.write(path_cord)
