#!/bin/bash
import subprocess
import os
import numpy as np
import json
import random
import imageio

id_name={'toilet': 62, 'teddy bear': 78, 'cup': 42, 'bicycle': 2, 'kite': 34, 'carrot': 52, 'stop sign': 12, 'tennis racket': 39, 'donut': 55, 'snowboard': 32, 'sandwich': 49, 'motorcycle': 4, 'oven': 70, 'keyboard': 67, 'scissors': 77, 'airplane': 5, 'couch': 58, 'mouse': 65, 'fire hydrant': 11, 'boat': 9, 'apple': 48, 'sheep': 19, 'horse': 18, 'banana': 47, 'baseball glove': 36, 'tv': 63, 'traffic light': 10, 'chair': 57, 'bowl': 46, 'microwave': 69, 'bench': 14, 'book': 74, 'elephant': 21, 'orange': 50, 'tie': 28, 'clock': 75, 'bird': 15, 'knife': 44, 'pizza': 54, 'fork': 43, 'hair drier': 79, 'frisbee': 30, 'umbrella': 26, 'bottle': 40, 'bus': 6, 'bear': 22, 'vase': 76, 'toothbrush': 80, 'spoon': 45, 'train': 7, 'sink': 72, 'potted plant': 59, 'handbag': 27, 'cell phone': 68, 'toaster': 71, 'broccoli': 51, 'refrigerator': 73, 'laptop': 64, 'remote': 66, 'surfboard': 38, 'cow': 20, 'dining table': 61, 'hot dog': 53, 'car': 3, 'sports ball': 33, 'skateboard': 37, 'dog': 17, 'bed': 60, 'cat': 16, 'person': 1, 'skis': 31, 'giraffe': 24, 'truck': 8, 'parking meter': 13, 'suitcase': 29, 'cake': 56, 'wine glass': 41, 'baseball bat': 35, 'backpack': 25, 'zebra': 23}
voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
### get information from the object detected#########
os.system('chmod 775 ./detect.sh')
detect_p = subprocess.Popen(["./detect.sh"], stdout=subprocess.PIPE)
out= detect_p.communicate()

def get_str_btw(s, f, b):
    par = s.partition(f)
    return (par[2].partition(b))[0][:]
def get_str_list(strr):
    strr=strr[1:-1]
    return strr.split(', ')
def get_str_mat(strr):
    strr=strr[1:-1]
    return strr.split(',\\n        ')
    
out=str(out[0])
shape=get_str_btw(out,'getshape','getshape')
shape=get_str_btw(shape,'(',')')
shape=shape.split(',')
out=out.split('det_boxes')
#print(out)
out=out[1].split('det_')
# out=[" [tensor([[0.4905, 0.6039, 0.6430, 0.7903],\\n        [0.8581, 0.6562, 0.9443, 0.7295],\\n        [0.5051, 0.4887, 0.6225, 0.7442]], device='cuda:0',\\n       grad_fn=<CatBackward>)]\\n", "labels [tensor([ 2,  4, 15], device='cuda:0')]\\n", 'scores [tensor([0.9932, 0.2098, 0.9546], device=\'cuda:0\', grad_fn=<CatBackward>)]\\ndisplay-im6.q16: unable to open X server `\' @ error/display.c/DisplayImageCommand/432.\\n"']

det_boxes=get_str_btw(out[0], '[tensor(', ", device='cuda:0'")

det_boxes = det_boxes.replace('\'', '\"')
if 'n' in det_boxes:
    det_boxes = det_boxes.replace('\\', '')
    det_boxes = det_boxes.replace('n', '')
if 'grad' in det_boxes:
    det_boxes=det_boxes[:-24]
#print(det_boxes)

#det_boxes=json.loads(det_boxes)
#print(det_boxes,"test here")
det_boxes = eval(det_boxes)

#print("det_boxes:",det_boxes)
# print(out[1])
det_labels=get_str_btw(out[1], '[tensor(', ", device='cuda:0'")
det_labels = det_labels.replace('\'', '\"')
# det_labels=json.loads(det_labels)
# det_labels=np.array(det_labels)
# print(out[2])

det_labels=get_str_list(det_labels)
#print("det_labels:",det_labels)
det_scores=get_str_btw(out[2], '[tensor(', ", device='cuda:0'")
det_scores = det_scores.replace('\'', '\"')
if 'grad' in det_scores:
    det_scores=det_scores[:-28]
#print(det_scores)
det_scores=json.loads(det_scores)
det_scores=np.array(det_scores)
#det_scores=get_str_list(det_scores)
#print("det_scores:",det_scores)

#### write into json file#########
wr_lis=[]
for i in range(len(det_scores)):
    if float(det_scores[i])<0.5:
        wr_lis.append(i)
        
#print(wr_lis)
#print(shape)
dic={}
dic["H"]=int(shape[0])
dic["W"]=int(shape[1])
name="COCO_val2014_"+"000000"+str(random.randint(100000, 999999))
dic["image_name"]=name+'.jpg'
detections=[]
for i in range(len(det_scores)):
    if i in wr_lis:
        continue
    new_dic={}
    new_dic["score"]=det_scores[i]
    det_boxes[i][0],det_boxes[i][1] = det_boxes[i][1],det_boxes[i][0]
    det_boxes[i][2],det_boxes[i][3] = det_boxes[i][3],det_boxes[i][2]

    new_dic["box_coords"]=det_boxes[i]
    if ']' in det_labels[i]:
        det_labels[i]=det_labels[i][:-4]
    new_dic["class_str"]=voc_labels[int(det_labels[i])-1]
    new_dic["class_no"]=id_name[voc_labels[int(det_labels[i])-1]]
    detections.append(new_dic)
dic['detections']=detections
#print(dic)
json_str = json.dumps(dic)

with open('../All_data/Object_Detections_simple/'+name+'.json', 'w') as json_file:
    json_file.write(json_str)

#savedir = '../VSGNet/All_data/test_simple'
#img_name = os.path.join(savedir, name+'.jpg')
#####这一块需要改的
#imageio.imwrite(img_name, img)


os.system('chmod 775 ./vsgnet.sh')
vsgnet_p = subprocess.Popen(["./vsgnet.sh"], stdout=subprocess.PIPE)
out= vsgnet_p.communicate()
#print(out)




