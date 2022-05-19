import torch
import time 
import json
import ctypes
import requests
import logging
import numpy as np

from torch.multiprocessing import Value
from torchvision import datasets, transforms, models
from PIL import Image


# model used to do inference
from torchvision.models  import resnet50,resnet101 

# logging function
log = torch.multiprocessing.log_to_stderr()
log.setLevel(logging.INFO)

# model Class
class Model:
    def __init__(self, PATH_TO_LABELS = None, GPU=0 , batchSize=24, detection_threshold = 0.05):
        self.GPU = GPU
        self.batchSize = batchSize

        self.detections_all = np.array([]).reshape(0,1000)
        self.detection_threshold = detection_threshold

        # Loading ImageNet labels from online
        if type(PATH_TO_LABELS) is type(None):
            self.label = json.loads(requests.get('https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json').content) 
        
        # Loading the ResNet50 model from pretrained ImageNet model
        self.model = models.resnet50(pretrained=True)   
        self.model = self.model.to(torch.device('cuda:%d'%self.GPU))
        self.model.eval()

        # For calculating time
        self.time = []
    
    # Prepreocessing pipeline
    def preprocess(self, batch, transform = transforms.ToTensor()):
        # converting image to Pillow and to tensors
        batch_tensor = [transform(Image.fromarray(np.uint8(img))) for img in batch]
        batch_tensor = torch.stack(batch_tensor, axis=0).to(torch.device('cuda:%d'%self.GPU))
        del batch
        return(batch_tensor) 
        
    # Batch detection function
    def batch_det(self, batch): 
        
        # initializing time
        tic = time.time()

        # preprocessing data batch and prediction
        batch_tensor = self.preprocess(batch)
        with torch.no_grad():
            prediction = self.model(batch_tensor)
        self.time.append(time.time() - tic)
        # converting predictions to labels
        detections_tensor = torch.argmax(prediction, axis=1).cpu().detach()
        detections = [self.label[str(i)][-1] for i in detections_tensor.numpy()]
        del batch_tensor
        return(detections)


# class for multiprocessing
class multiProc(object):
    # multiprocessing class initialization
    def __init__(self, PATH_TO_LABELS = None, batchSize=16):
        self.Proc = True
        self.batchSize = batchSize
        self.PATH_TO_LABELS = PATH_TO_LABELS

        # creating multiprocessing context
        self.mp = torch.multiprocessing.get_context('spawn')

        # Queues for data communication across flags 
        self.QIn = torch.multiprocessing.Queue()
        self.QOut = torch.multiprocessing.Queue()
        self.FRun = self.mp.Value(ctypes.c_bool, True)

        # count for processed batches
        self.count_QIn = 0
        self.run()

    # function that wraps predict function for multiprocessing
    def predict(self,  dev, PATH_TO_LABELS, batchSize):
        modelP = Model(PATH_TO_LABELS = PATH_TO_LABELS , GPU = dev , batchSize=batchSize)
        bFlag = True
        while(bFlag):
            # checking global variable for interuption
            bFlag = self.FRun.value
            while(not self.QIn.empty()):
                imageSet = self.QIn.get()
                (Im_k,Im_v) = imageSet.popitem()
                log.info("[GPU:%d] PRED : %d"%(dev, Im_k))
                # model prediction and queuing the results
                res = modelP.batch_det(Im_v)
                self.QOut.put({Im_k:res})
                del imageSet,Im_k,Im_v

        log.info('[GPU:%d] PROC completed...'%dev)
        print(self.FRun.value)
        del modelP
        torch.cuda.empty_cache()

        
    # starting multiprocessing with the function
    def run(self):
        try:
            # multiprocess 
            self.processes = []
            for i in range(torch.cuda.device_count()):
                p = self.mp.Process(target=self.predict, args=(i, self.PATH_TO_LABELS, self.batchSize))
                self.processes.append(p)

            for i,p in enumerate(self.processes):
                print("[GPU:%d] PROC started..."%i)
                p.start()
            
            return(True)
        # error handling
        except Exception as E:
            print("ERROR spinning multiprocess\n",E)
            return(False)

    # function to add list of images to the queue, frameNo serves as a unique ID for continous data
    def add_batches(self, frames, frameNo):
        count = frameNo
        for startIdx in range(0, len(frames), self.batchSize):
            count += 1
            endIdx = startIdx + self.batchSize

            if(endIdx)>= len(frames):                
                endIdx = len(frames)

            try:
                # adding list of images as a dictionary with a unique key index
                self.count_QIn += 1
                self.QIn.put({count:{'image':frames[startIdx:endIdx], 'ID':startIdx}})
                log.info("IMAGE ADD : %d"%count)

            except:
                print("**** Error adding images to Queue QIn *****")
                return(False)
        return(True)


    # method to recover the predictions from the output queue that is sorted
    def get_pred(self):
        # waiting for all the predictions to complete
        while (True):
            if (self.QOut.qsize() == self.count_QIn):
                print("[QIn] QSize = %d"%self.QIn.qsize())
                break
        # iterating through queue 
        dataDict = {}
        for i in range(self.QOut.qsize()):
            temp = self.QOut.get()
            self.count_QIn -= 1
            for key,val in temp.items():
                dataDict[key] = val

        # sorting the output dictonary
        dataDict = {key: val for key, val in sorted(dataDict.items(), key = lambda ele: ele[0])}
        dataOut = [val for key, val in dataDict.items()]
        return(dataDict, dataOut)

    # Terminates the process and updating the flag/lock
    def terminate(self):
            self.FRun.value = False
            print("Lock changed",self.FRun.value)
            time.sleep(0.5)
            for p in self.processes:
                p.terminate()
                p.join()
            return(True)

    # unit testing method for generating random image lists added to the queue
    def add_images(self, i, default = 100):
        for i in range(i, i+default):
            # using a key for unique identification, you can use uuid.uuid1() or random package for unique IDs
            self.QIn.put({i: 1*[np.ones((224,224, 3), dtype = np.uint8)]})
            self.count_QIn+=1
            log.info("IMAGE ADD : %d"%i)
            time.sleep(0.1)
        print(self.QOut.qsize(),self.QIn.qsize())

    # unit testing functionality for testing pipeline
    def add_data(self):
        # adding 100 random test images to the queue
        self.add_images(0, default = 100)
        print("[FINAL] QIn/QOut = %d/%d"%(self.QIn.qsize(),self.QOut.qsize()))        

        # collecting results from the queue
        dataDict, dataOut = self.get_pred()
        print("[FINAL] QIn/QOut = %d/%d"%(self.QIn.qsize(),self.QOut.qsize()))        

        # terminating the predict processess
        self.terminate()
        
        return(dataDict, dataOut)


if __name__ == '__main__':
    print('\n'.join(['%s'%torch.cuda.get_device_name(i)     for i in range(torch.cuda.device_count())]))
    proc = multiProc(batchSize=16)
    if proc.Proc:
        print(proc.add_data())
        
    # model = Model(GPU=0)
    # pred = model.batch_det(32*[np.ones((224,224,3))])
    # print(pred)
    exit()
