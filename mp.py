import torch
from torch import nn
import copy
from torchvision.models  import resnet50,resnet101 
import numpy as np
import time 
from torch.multiprocessing import Process, Value
# from multiprocessing import logging
import logging
# model used to do inference
import multiprocessing_logging
import ctypes

from torch.profiler import profile, record_function, ProfilerActivity
from torchvision import datasets, transforms, models
from PIL import Image
import json
import requests

multiprocessing_logging.install_mp_handler()

log = torch.multiprocessing.log_to_stderr()
log.setLevel(logging.INFO)

# from TestQ import MyQueue

class Model:
    def __init__(self, PATH_TO_MODEL = None, PATH_TO_LABELS = None, GPU=0 , batchSize=24, detection_threshold = 0.05):
        self.GPU = GPU
        self.PATH_TO_MODEL = PATH_TO_MODEL        
        self.batchSize = batchSize

        self.detections_all = np.array([]).reshape(0,1000)
        self.detection_threshold = detection_threshold

        if type(PATH_TO_LABELS) is type(None):
            self.label = json.loads(requests.get('https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json').content) 
        
        print(self.label)

        try:
            self.model = models.resnet50(pretrained=True)   
            self.model = self.model.to(torch.device('cuda:%d'%self.GPU))
            self.model.eval()

        except Exception as e:
            log.info("Error initializing GPU Processing and model || %s"%e)
        self.time = []
        print('GPU loaded...')
    
    def preprocess(self, batch, transform = transforms.ToTensor()):
        batch_tensor = [transform(Image.fromarray(np.uint8(img))) for img in batch]
        batch_tensor = torch.stack(batch_tensor, axis=0).to(torch.device('cuda:%d'%self.GPU))
        del batch
        return(batch_tensor) 
        

    def batch_det(self, batch, startIdx):
        # import ipdb;ipdb.set_trace()
        tic = time.time()
        batch_tensor = self.preprocess(batch)
        with torch.no_grad():
            prediction = self.model(batch_tensor)
        self.time.append(time.time() - tic)
        detections = torch.argmax(prediction, axis=1)
        
        return(detections)


# class running inference
class multiProc(object):
    def __init__(self, PATH_TO_MODEL = None, PATH_TO_LABELS = None, batchSize=16, detection_threshold = 0.05, detection_block_overlap_threshold = 0.01, detection_threshold_block = 0.1):
        self.Proc = True
        self.batchSize = batchSize
        self.PATH_TO_MODEL = PATH_TO_MODEL 
        self.PATH_TO_LABELS = PATH_TO_LABELS

        self.detection_threshold = detection_threshold

        self.mp = torch.multiprocessing.get_context('spawn')

        self.QIn = torch.multiprocessing.Queue()
        self.QOut = torch.multiprocessing.Queue()
        self.FRun = self.mp.Value(ctypes.c_bool, True)

        self.count_QOut = 0
        self.run()


    def predict(self,  dev, PATH_TO_MODEL, PATH_TO_LABELS, batchSize, detection_threshold):
        print("[GPU:%d] model loaded"%dev)
        modelP = Model(PATH_TO_MODEL = PATH_TO_MODEL, PATH_TO_LABELS = PATH_TO_LABELS , GPU = dev , batchSize=batchSize, detection_threshold = detection_threshold)
        print("[GPU:%d] model loaded"%dev)
        bFlag = True
        # time.sleep(3)
        while(bFlag):
            bFlag = self.FRun.value
            print("predict",bFlag)
            while(not self.QIn.empty()):
                imageSet = self.QIn.get()
                for idx,(Im_k,Im_v) in enumerate(imageSet.items()):
                    break
                log.info("[GPU:%d] PRED : %d"%(dev, Im_k))
                res = modelP.batch_det(Im_v,  startIdx = 0)
                
                self.QOut.put({Im_k:(Im_v[0][0][0],Im_v)})

                del imageSet,Im_k,Im_v

        log.info('[GPU:%d] PROC completed...'%dev)
        print(self.FRun.value)
        del modelP
        torch.cuda.empty_cache()

        
    
    def run(self):
        try:
            self.processes = []
            for i in range(torch.cuda.device_count()):
                p = self.mp.Process(target=self.predict, args=(i, self.PATH_TO_MODEL, self.PATH_TO_LABELS, self.batchSize, self.detection_threshold))
                self.processes.append(p)
                print("[GPU:%d] PROC initializing..."%i)

            for i,p in enumerate(self.processes):
                print("[GPU:%d] PROC started..."%i)
                p.start()
            
            return(True)

        except Exception as E:
            print("ERROR spinning multiprocess\n",E)
            return(False)
    
    def terminate(self):
            time.sleep(0.5)
            for p in self.processes:
                p.terminate()
                p.join()
                p.close()
            return(True)

    def add_batches(self, frames, frameNo):
        count = frameNo
        for startIdx in range(0, len(frames), self.batchSize):
            count += 1
            endIdx = startIdx + self.batchSize

            if(endIdx)>= len(frames):                
                endIdx = len(frames)

            try:
                self.count_QOut += 1
                self.QIn.put({count:{'image':frames[startIdx:endIdx], 'ID':startIdx}})
                log.info("IMAGE ADD : %d"%count)

            except:
                print("**** Error adding images to Queue QIn *****")
                return(False)
        return(True)

    def add_images(self, i, default = 100):
        for i in range(i, i+default):
            self.QIn.put({i: 2*[np.ones((224,224, 3), dtype = np.uint8)]})
            self.count_QOut+=1
            log.info("IMAGE ADD : %d"%i)
            time.sleep(0.1)
        print(self.QOut.qsize(),self.QIn.qsize())

    def add_data(self):
        self.add_images(0)

        print("[FINAL] QSize = %d"%self.QOut.qsize())
        
        count = 0
        print("completed Getting values 1-100")
        
        while (True):
            # print('Waiting for completeing prediction %d'%self.QOut.qsize())
            if (self.QOut.qsize() == self.count_QOut):
                print("[QIn] QSize = %d"%self.QIn.qsize())
                break
        print('All images predicted')


        self.add_images(100)
        print("[FINAL] QSize = %d"%self.QOut.qsize())
        


        print("completed Getting values 100-200")
        data = []
        dataDict = {}
        for i in range(self.QOut.qsize()):
            temp = self.QOut.get()
            for key,val in temp.items():
                dataDict[key] = val[0]
                data.append(val[1])
        print('Test %d'%len(dataDict))
        self.FRun.value = False
        print("Lock changed",self.FRun.value)
        
        self.terminate()

        print(dataDict)

if __name__ == '__main__':
    print('\n'.join(['%s'%torch.cuda.get_device_name(i)     for i in range(torch.cuda.device_count())]))
    # import ipdb;ipdb.set_trace()
    a = multiProc(batchSize=16)
    if a.Proc:
        a.add_data()
        
    # asd = Model(GPU=0)
    # pred = asd.batch_det(32*[np.ones((224,224,3))],1)
    # print(pred)
    exit()


