# -*- coding: utf-8 -*-
"""
File to test SORT tracker.
press 'w' to skip to next instance
hold on 'q' to skip to next cage

@author: AJ
"""

import os
import cv2
import glob
import torch
import numpy as np

from sort import *
from tqdm import tqdm
import colorsys

class pallet():
    def __init__(self, colorlabel, classes, alpha = 0.3, colorW = (255,255,255), classFilter = None):
        self.colorW = colorW
        self.alpha = alpha

        self.classes = classes
        self.colorlabel = colorlabel
        if type(classFilter) == type(None):
            self.classFilter = classes
        else:
            self.classFilter = classFilter

    def draw_tracks(self,image, tracked_objects):
        imgx = image.copy()
        for x1, y1, x2, y2, obj_id, obj_label in tracked_objects:
            if self.classes[int(obj_label)] not in self.classFilter:
                continue
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # if type(lastnObj) == bool:
            #     cv2.rectangle(imgx, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
            #     cv2.rectangle(image, (x1, y1), (x2, y2), self.colorW, 2)
            # else:
            cv2.rectangle(imgx, (x1, y1), (x2, y2), self.colorlabel[int(obj_label)], -1)
            cv2.rectangle(imgx, (int(x1), int(y2-35)), (int(x1+len(self.classes[int(obj_label)])*20+60),int(y2)), self.colorW, -1)
    
            if type(self.classes) != type(None):
                cv2.putText(image, '{}_{}'.format(self.classes[int(obj_label)],int(obj_id)), (int(x1), int(y2 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, self.colorlabel[int(obj_label)], 2)

        image = cv2.addWeighted(image, 1-alpha, imgx, alpha, gamma=0)
        del imgx
        return(image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sample TensorFlow XML-to-TFRecord converter")      
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--video_INfile', type=str, default='sample_vid.avi', help='file/URL, 0 for webcam')
    parser.add_argument('--OUTfile', default=None, help='file/URL, 0 for webcam')
    parser.add_argument('--OUTimage', default=None  , help='inference size (pixels)')
    parser.add_argument('--TRKframes', default=3  , help='inference size (pixels)')
    parser.add_argument('--TRKage', default=5  , help='inference size (pixels)')
    parser.add_argument('--batch', default=1  , help='inference batch size, default - 1')
    parser.add_argument('--objects', default=['person']  , help='List of object classes to be tracked')
    args = parser.parse_args()
        
    model = torch.hub.load('ultralytics/yolov5', args.weights.split('.')[0])  # or yolov5n - yolov5x6, custom

    try:
        video = cv2.VideoCapture(args.video_INfile)
        flag = True
    except:
        print('cannot find the video : %s'%args.video_INfile)
        exit()
    
    if type(args.OUTimage) is not type(None):
        os.makedirs(args.OUTimage, exist_ok=True)

    if type(args.OUTfile) is not type(None):
        out_clr = cv2.VideoWriter(args.OUTfile,cv2.VideoWriter_fourcc('M','J','P','G'), video.get(5), (int(video.get(3)),int(video.get(4))))
    

    w,h = int(video.get(3)),int(video.get(4))
    alpha = 0.4
    mot_tracker = Sort(min_hits=args.TRKframes, max_age = args.TRKage, classes=model.names)
    
    # color scheme
    N = len(model.names)
    HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    clr_val = list(RGB_tuples)
    clr_lst = [(i[0]*255,i[1]*255,i[2]*255)for i in clr_val]

    # object for drawing
    drawer= pallet(classes=model.names, colorlabel=clr_lst, classFilter = ['person'])

    count = 0
    tq = tqdm(total= video.get(7))
    while flag:
        imgBatch = []
        for i in range(args.batch):        
            flag,image = video.read()
            if flag:
                imgBatch.append(image)


        if len(imgBatch)==0:
            continue
        # prediction from the model
        results = model(imgBatch)


        
        for _img,_pred in zip(imgBatch, results.pred):        
            count+=1
            tq.update(1)
            
            if len(_pred) == 0:
                _pred = np.empty((0, 5))
            
            pred = [ [int(i[0]),int(i[1]),int(i[2]),int(i[3]),float(i[4]),int(i[5])] for i in _pred.cpu().detach().numpy()]
            tracked_objects = mot_tracker.update(np.array(pred)[:,:6])
                        
            _img = drawer.draw_tracks(_img, tracked_objects)

            if type(args.OUTfile) is not type(None):
                out_clr.write(_img)

            if type(args.OUTimage) is not type(None):
                cv2.imwrite('{}/frame_%d.jpg'.format(args.OUTimage,count), _img)

            cv2.imshow('Tracked Output', _img)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                print('Exiting SORT Tracker..')
                exit()

if type(args.OUTfile) is not type(None):
    out_clr.release()
cv2.destroyAllWindows()
video.release()

# python person_tracker.py --video_INfile location_to/video/input.mp4 --OUTfile  location_to/video/output.mp4 --OUTimage location_to/folder/to_save_image --TRKframes minimum_number_of_detection_to_track  --TRKage minimum_missed_detection_to_delete --batch batch_size --objects list_of_objects_tracked

# python person_tracker.py --OUTfile  ./test.avi