#!/usr/bin/env python2
import os
import sys
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, "..", ".."))

import math
from math import log10, floor
import txaio
txaio.use_twisted()

from autobahn.twisted.websocket import WebSocketServerProtocol, \
    WebSocketServerFactory
from twisted.internet import task, defer
from twisted.internet.ssl import DefaultOpenSSLContextFactory

from twisted.python import log
import Queue

import argparse
import cv2
import imagehash
import json
from PIL import Image
import numpy as np
import os
import StringIO
import urllib
import base64

from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import openface
import dlib
import time, datetime
import cv2gpu
import threading

import imgaug as ia
from imgaug import augmenters as iaa

from mem_top import mem_top
from pympler import muppy
from pympler import summary
import gc
import random
#gc.enable()
#gc.set_debug(gc.DEBUG_LEAK)

time_log_freq = 1000

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.3, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.
seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.2), # horizontally flip 50% of all images
        #iaa.Flipud(0.2), # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            rotate=(-15, 15), # rotate by -45 to +45 degrees
            shear=(-10, 10), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 1),
            [
                #sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.0)), # sharpen images
                #iaa.Emboss(alpha=(0, 1.0), strength=(0, 0.5)), # emboss images
                # search either for all edges or for directed edges,
                # blend the result with the original image using a blobby mask
                # iaa.SimplexNoiseAlpha(iaa.OneOf([
                #     iaa.EdgeDetect(alpha=(0.5, 1.0)),
                #     iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                # ])),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                # iaa.OneOf([
                #     iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                #     iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                # ]),
                #iaa.Invert(0.05, per_channel=True), # invert color channels
                #iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                #iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                # iaa.OneOf([
                #     #iaa.Multiply((0.5, 1.5), per_channel=0.5),
                #     iaa.FrequencyNoiseAlpha(
                #         exponent=(-4, 0),
                #         first=iaa.Multiply((0.5, 1.5), per_channel=False),
                #         second=iaa.ContrastNormalization((0.5, 2.0))
                #     )
                # ]),
                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                #iaa.Grayscale(alpha=(0.0, 1.0)),
                #sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
            ],
            random_order=True
        )
    ],
    random_order=True
)

start = time.time()

reload(sys)
sys.setdefaultencoding('utf-8')

modelDir = os.path.join(fileDir, '..', '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
# For TLS connections
tls_crt = os.path.join(fileDir, 'tls', 'server.crt')
tls_key = os.path.join(fileDir, 'tls', 'server.key')

parser = argparse.ArgumentParser()
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_5_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--unknown', type=bool, default=False,
                    help='Try to predict unknown people')
parser.add_argument('--port', type=int, default=9000,
                    help='WebSocket Port')
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

if args.verbose:
    print("Argument parsing and import libraries took {} seconds.".format(
        time.time() - start))
    start = time.time()


align = []
net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,
                              cuda=args.cuda)


number_of_cpu_net = 21
cpunet = []
for qq in range(0, number_of_cpu_net):                              
    cpunet.append(openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim, cuda=False))

face_cascade = cv2.CascadeClassifier('/root/openface/data/lbpcascades/lbpcascade_frontalface_improved.xml')
#face_cascade = cv2.CascadeClassifier('/root/openface/data/lbpcascades/lbpcascade_profileface.xml')
if face_cascade.empty():
    print("Can not load face cascade classifier xml!")


if cv2gpu.is_cuda_compatible():
    cv2gpu.init_gpu_detector('/root/openface/data/haarcascade_frontalface_default_cuda.xml')
    #cv2gpu.init_gpu_detector('/root/openface/data/lbpcascades/lbpcascade_frontalface_improved.xml')
    print("Using gpu haarcascade!")
else:
    cv2gpu.init_cpu_detector('/root/openface/data/haarcascade_frontalface_default.xml')
    print("Using cpu haarcascade only!")

if args.verbose:
    print("Loading the dlib and OpenFace models took {} seconds.".format(
        time.time() - start))
    start = time.time()

def round_to_1(x, sig=2):
    if x == 0:
        return 0
    else:
        return round(x, sig-int(floor(log10(abs(x))))-1)

def bb_centroid(bb):
    x = (bb.right()+bb.left())*0.5
    y = (bb.top()+bb.bottom())*0.5
    return np.array((x, y))

def utf8len(s):
    return len(s.encode('utf-8'))


class Face:

    def __init__(self, rep, identity):
        self.rep = rep
        self.identity = identity

    def __repr__(self):
        return "{{id: {}, rep[0:5]: {}}}".format(
            str(self.identity),
            self.rep[0:5]
        )


class OpenFaceServerProtocol(WebSocketServerProtocol):
    def __init__(self):
        super(OpenFaceServerProtocol, self).__init__()
        self.images = {} # is a map, the key is the hash of the image, the value is an instance of the class "Face"
        self.training = True
        self.unknowntraining = False
        self.unknowntraininglatestindex = 0
        self.unknowntraining_list = []
        self.people = []
        self.identity_ofppl = [] # w.r.t self.people, i.e. self.identity_ofppl[0] is the identity of self.people[0]
        self.svm = None
        self.knn = []
        self.ppl_oneclasssvm_clf = []
        self.doOneClassSVM = False

        #mean and std of different classes
        self.rep_of_each_class = {} # is a map, the key is the identity of the rep, the value is an array of rep of that identity
        self.mean = {} # is a map, the key is the identity of the rep, the value is the mean of the array of rep of that identity
        self.std = {}

        #This feature is not implemented !!! Dont use
        if args.unknown:
            self.unknownImgs = np.load("./examples/web/unknown.npy")
        
        #Store the bbs and the corresponding identity of last frame
        self.prev_bbs = []
        self.prev_identity = []
        self.prev_rep = []
        self.prev_score = []
        self.tracked_list_of_ppl = []
        self.rescuetimes = []

        self.num_of_cam = 2
        self.ReInitializedTrackingVars()
        #self.stop_all_tracking = False  # set to true when trainFace() is done again
        #self.has_prev = False

        #To store training alignedFace and process it altogether when training is finished (In order to speed up training)
        self.trainingnumber = 0
        self.trainingnumber_foreachide = {} # just for display purpose
        self.trainingIdentity = []
        self.trainingPhashs = []
        self.trainingAlignFaces = []
        self.trainingRep = []
        self.trainingContent = []

        self.counter=0
        self.lastframetime=[]


        #video stream
        self.rgbFrame = []
        self.gpu_mutex = threading.Lock()
        self.augment_mutex = threading.Lock() 
        # self.zzzjpg_mutex = threading.Lock()
        self.video_stream_path = []

        self.ui_size_x = 816 #816 #160 #384 #816 # 320
        self.ui_size_y = 459 #459 # 90 # 216 #459 #180

        self.queuesize = 5


        #self.video_stream_path.append('rtsp://admin:h0940232@172.18.9.100/Streaming/Channels/1')
        self.video_stream_path.append('rtsp://admin:h0940232@172.18.9.101/Streaming/Channels/1')
        self.video_stream_path.append('rtsp://admin:h0940232@172.18.9.100/Streaming/Channels/1')
        #self.video_stream_path.append('rtsp://172.18.9.99/axis-media/media.amp')

        self.cap = []     
        self.bufferframe = []  
        self.bufferframe_toprocess = []  
        self.processed_content = []  
        self.grabtime = []
        self.bufferframe_gray = []
        self.encodedimage = []
        self.processed_msg = Queue.Queue(maxsize=self.queuesize)
        self.learnimgqueue = Queue.Queue()
        self.tolearnimgqueue = Queue.Queue()
        self.prev_bbs_isused = []
        #self.augmentimgqueue = Queue.Queue()

        self.sending_trainingimg = False

        # s = '(\xef\xbd\xa1\xef\xbd\xa5\xcf\x89\xef\xbd\xa5\xef\xbd\xa1)\xef\xbe\x89'
        # s1 = s.decode('utf-8')
        # print s1
        #self.firstFrame = True      

        # self.cap = cv2.VideoCapture(self.video_stream_path)
        # if not self.cap.isOpened():
        #     print("cap is not open, try to release and open again")
        #     self.cap.release()
        #     self.cap.open(self.video_stream_path)
        #     if not self.cap.isOpened():
        #         print("cap is not open again, fail!!")
        
        # else:
        #     print("cap is opened successfully!")
        
        # #open a seperate thread to read the frames and save it to self.rgbFrame and zzz.jpg

        number_of_lrn_img_thread = 10
        for qq in range(0, number_of_lrn_img_thread):
            thread = threading.Thread(target=self.getLearnImgRep, args=(qq+1, ))
            thread.daemon = True                            
            thread.start()                                  

    def processImgToGray(self, cam_id):
        counter=0
        while True:
            counter=counter+1
            if not self.bufferframe[cam_id] is None:
                start = time.time()
                frame = self.bufferframe[cam_id].copy()
                if args.verbose and counter%time_log_freq==0:
                    print("cam id {} self.bufferframe[cam_id].copy() took {} seconds.".format(cam_id, time.time() - start)) 
                    start = time.time()

                self.bufferframe_gray[cam_id] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if args.verbose and counter%time_log_freq==0:
                    print("cam id {} cv2.cvtColor took {} seconds.".format(cam_id, time.time() - start)) 
            
            else:
                time.sleep(0.005)

    def processImgToEncode(self, cam_id):
        counter=0
        while True:
            counter=counter+1
            if not self.bufferframe[cam_id].empty():
                start = time.time()
                frame = self.bufferframe[cam_id].get()

                if self.bufferframe_toprocess[cam_id].full():
                    self.bufferframe_toprocess[cam_id].get()
                self.bufferframe_toprocess[cam_id].put(frame.copy())
                #print("cam id {} queue size is {}".format(cam_id, self.bufferframe[cam_id].qsize()))         

                #start = time.time()
                #print("self.rgbFrame.shape= {}".format(self.rgbFrame.shape))
                ratio_x = 1.0*self.ui_size_x/frame.shape[1]
                ratio_y = 1.0*self.ui_size_y/frame.shape[0]
                #sendimg = cv2.resize(frame, (0,0), fx=ratio_x, fy=ratio_y, interpolation=cv2.INTER_NEAREST)
                sendimg = cv2.resize(frame, (0,0), fx=ratio_x, fy=ratio_y)
                # if args.verbose and counter%20==0:
                #     print("cam id {} cv2.resize took {} seconds.".format(cam_id, time.time() - start)) 
                #     start = time.time()                

                # if args.verbose and counter%time_log_freq==0:
                #     print("cam id {} cv2.resize took {} seconds.".format(cam_id, time.time() - start))         

                # show the timestamp on the image for debugging
                # # 384x216
                #cv2.putText(sendimg, self.grabtime[cam_id], (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, 2)
                #cv2.putText(sendimg, str(datetime.datetime.now()), (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, 2)

                # # 640x360
                # #cv2.putText(sendimg, self.grabtime[cam_id], (100, 95), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, 2)
                # #cv2.putText(sendimg, str(datetime.datetime.now()), (100, 165), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, 2)

                #encode to jpeg format
                #encode param image quality 0 to 100. default:95
                #if you want to shrink data size, choose low image quality.
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 40]
                retval, buffer = cv2.imencode('.jpeg', sendimg, encode_param)
                quote = urllib.quote(base64.b64encode(buffer))
                content = 'data:image/jpeg;base64,' + quote

                # if args.verbose and counter%30==0:
                #     print("cam id {} cv2.imencode took {} seconds.".format(cam_id, time.time() - start))    

                self.encodedimage[cam_id] = content            

                #print("img size in bytes={}".format(utf8len(content)))
                #print("first few quote={}".format(quote[:160]))

                #if args.verbose:
                    #print("Save img took {} seconds.".format(time.time() - start))   


                # learnmsg = "None"
                # if not self.learnimgqueue.empty():
                #     learnmsg = self.learnimgqueue.get()
                
                # pcontent = []
                # for qx in range(0, len(self.video_stream_path)):
                #     if not self.processed_content[qx].empty():
                #         pcontent.append((qx, self.processed_content[qx].get_nowait()))

                # #print("debug x")

                # if len(self.encodedimage) == 2 and not self.encodedimage[0] is None and not self.encodedimage[1] is None:
                #     msg = {
                #         "type": "ANNOTATED",
                #         "content": pcontent,
                #         "fps": 10,
                #         "id": cam_id,
                #         "imageset": 2,
                #         "image": self.encodedimage,
                #         "dateAndtime": str(datetime.datetime.now()),
                #         "learnmsg": learnmsg
                #     }        

                #     #print("debug y")

                #     self.processed_msg.put_nowait(msg)       
                
                # elif not self.encodedimage[0] is None:
                #     msg = {
                #         "type": "ANNOTATED",
                #         "content": pcontent,
                #         "fps": 10,
                #         "id": cam_id,
                #         "imageset": 1,
                #         "image": self.encodedimage[0],
                #         "dateAndtime": str(datetime.datetime.now()),
                #         "learnmsg": learnmsg
                #     }    

                #     #print("debug z")   

                #     self.processed_msg.put_nowait(msg)        

                

            else:
                time.sleep(0.005)


    # A seperate thread to get latest rtsp frame for each camera
    def queryFrame(self, cam_id, path):
        self.cap[cam_id] = cv2.VideoCapture(path)
        print("capture of path [{}] is opened !".format(path))
        counter=0
        while True:
            counter=counter+1
            NeedReOpen=False
            if self.cap[cam_id].isOpened():

                start = time.time()
                #for kk in range(0,6):
                ret, frame = self.cap[cam_id].read()
                # ret = self.cap[cam_id].grab()
                # if args.verbose:
                #     print("cam id {} 1st grab img took {} seconds.".format(cam_id, time.time() - start))   
                #     start = time.time()   

                # ret = self.cap[cam_id].grab()
                # if args.verbose:
                #     print("cam id {} 2nd grab img took {} seconds.".format(cam_id, time.time() - start))   
                #     start = time.time()   

                # ret, frame = self.cap[cam_id].retrieve()

                self.grabtime[cam_id] = str(datetime.datetime.now())

                # if args.verbose:
                #     print("cam id {} retrieve img took {} seconds.".format(cam_id, time.time() - start))   
                #     start = time.time()                  

                # show the timestamp on the image for debugging

                #if args.verbose and counter%20==0:
                   #print("cam id {} cv2.imencode took {} seconds.".format(cam_id, time.time() - start))                   

                #print("img size in bytes={}".format(utf8len(content)))
                #print("first few quote={}".format(quote[:160]))


                #self.bufferframe_gray[cam_id] = cv2.cvtColor(self.bufferframe[cam_id], cv2.COLOR_BGR2GRAY)
                #ret = self.cap[cam_id].grab()
                
                # if args.verbose and counter%time_log_freq==0:
                #     print("cam id {} cap.read() took {} seconds.".format(cam_id, time.time() - start))   

                #if counter%3==0:
                    #ret, self.bufferframe[cam_id] = self.cap[cam_id].retrieve()

                if frame is None:
                    print("cam id {} self.bufferframe[cam_id] is None, try again. ret={}".format(cam_id, ret))
                    NeedReOpen = True

                elif not ret :
                    print("cam id {} not ret, try again. ret={}".format(cam_id, ret))
                    NeedReOpen = True
                
                else:
                    if self.bufferframe[cam_id].full():
                        self.bufferframe[cam_id].get()
                    self.bufferframe[cam_id].put_nowait(frame)


                    # frame = self.bufferframe[cam_id].get_nowait()
                    # ratio_x = 1.0*self.ui_size_x/frame.shape[1]
                    # ratio_y = 1.0*self.ui_size_y/frame.shape[0]
                    # sendimg = cv2.resize(frame, (0,0), fx=ratio_x, fy=ratio_y, interpolation=cv2.INTER_NEAREST)
                    # # if args.verbose and counter%20==0:
                    # #     print("cam id {} cv2.resize took {} seconds.".format(cam_id, time.time() - start)) 
                    # #     start = time.time()                

                    # # if args.verbose and counter%30==0:
                    # #     print("cam id {} cv2.resize took {} seconds.".format(cam_id, time.time() - start))         

                    # # show the timestamp on the image for debugging

                    # # 384x216
                    # cv2.putText(sendimg, self.grabtime[cam_id], (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, 2)
                    # cv2.putText(sendimg, str(datetime.datetime.now()), (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, 2)

                    # # 640x360
                    # #cv2.putText(sendimg, self.grabtime[cam_id], (100, 95), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, 2)
                    # #cv2.putText(sendimg, str(datetime.datetime.now()), (100, 165), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, 2)

                    # #encode to jpeg format
                    # #encode param image quality 0 to 100. default:95
                    # #if you want to shrink data size, choose low image quality.
                    # encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                    # retval, buffer = cv2.imencode('.jpeg', sendimg, encode_param)
                    # quote = urllib.quote(base64.b64encode(buffer))
                    # content = 'data:image/jpeg;base64,' + quote

                    # if args.verbose and counter%30==0:
                    #     print("cam id {} resize and imencode took {} seconds.".format(cam_id, time.time() - start))   
                    #     start = time.time() 

                    # self.encodedimage[cam_id] = content            

                    # #print("img size in bytes={}".format(utf8len(content)))
                    # #print("first few quote={}".format(quote[:160]))

                    # #if args.verbose:
                    #     #print("Save img took {} seconds.".format(time.time() - start))   

                    # if len(self.encodedimage) == 2 and not self.encodedimage[0] is None and not self.encodedimage[1] is None:
                    #     msg = {
                    #         "type": "ANNOTATED",
                    #         "content": [],
                    #         "fps": 10,
                    #         "id": cam_id,
                    #         "imageset": 2,
                    #         "image": self.encodedimage,
                    #         "dateAndtime": str(datetime.datetime.now())
                    #     }        
                    
                    # elif not self.encodedimage[0] is None:
                    #     msg = {
                    #         "type": "ANNOTATED",
                    #         "content": [],
                    #         "fps": 10,
                    #         "id": cam_id,
                    #         "imageset": 1,
                    #         "image": self.encodedimage[0],
                    #         "dateAndtime": str(datetime.datetime.now())
                    #     }        

                    # self.processed_msg.put_nowait(msg) 
                
            else:
                print("cam id {} self.cap[cam_id].isOpened() is false, try again".format(cam_id))
                NeedReOpen = True

            if NeedReOpen:
                self.cap[cam_id].release()
                self.cap[cam_id].open(path)
                print("cam id {} video capture is reopen!".format(cam_id))
                time.sleep(0.005)

        self.cap[cam_id].release()

    def getLearnImgRep(self, id):
        while True:
            #if self.trainingnumber > 0:
            if not self.tolearnimgqueue.empty():
                #self.sending_trainingimg = False
                
                index = self.tolearnimgqueue.get()
                #for zz in range(0, self.trainingnumber):
                ctx = self.trainingContent[index]

                if ctx is None:

                    start = time.time()
                    #self.sending_trainingimg = True
                    identity = self.trainingIdentity[index]
                    
                    #phash = self.trainingPhashs[index]
                    alignedFace = self.trainingAlignFaces[index]
                    phash = str(imagehash.phash(Image.fromarray(alignedFace)))

                    # to speed up by using some gpu resource
                    if self.gpu_mutex.acquire():
                        self.trainingRep[index] = net.forward(alignedFace)
                        self.gpu_mutex.release()
                    
                    # else:
                    #     self.trainingRep[index] = cpunet[id].forward(alignedFace)

                    self.images[phash] = Face(self.trainingRep[index], identity)
                    # TODO: Transferring as a string is suboptimal.
                    self.trainingContent[index] = [str(x) for x in alignedFace.flatten()]   
                    #self.trainingContent[index] = [str(x) for x in cv2.resize(alignedFace, (0,0), fx=0.5, fy=0.5).flatten()]   

                    msg = {
                        "type": "NEW_IMAGE",
                        "hash": phash,
                        "content": self.trainingContent[index],
                        "identity": identity,
                        "representation": self.trainingRep[index].tolist()
                    }

                    self.learnimgqueue.put(msg)


                        # print(phash)
                        # print(identity)
                        # print(self.trainingContent[zz])
                        # print(self.trainingRep[zz].tolist())
                        # print(json.dumps(msg))
                        #self.sendMessage(json.dumps(msg))
                        #print("One face (index: {}) of identity {} is processed and sent to client.".format(zz, identity))
                    #print("cam id {} cv2.cvtColor took {} seconds.".format(cam_id, time.time() - start)) 
                    print("One face (index: {}) of identity {} is processed at thread id {} using {} seconds.".format(index, identity, id, time.time() - start))


                # if self.sending_trainingimg:
                #     print("End a series of learning and start to trainface()")
                #     self.trainFace()
                #     self.sending_trainingimg = False

            else:
                time.sleep(0.1)

    def Determine_identity_by_rep(self, cam_id, rep):
        BestMatchSimilarityScore = 0

        if len(self.people) == 0:
            identity = -1  #unknown

        elif len(self.people) >= 1:

            # if self.svm:
            #     #when added person >1, the identity is the index return by svm
            #     identity = self.svm.predict([rep])[0]

            #     #also need to double confirm with the probability of each class
            #     prob = self.svm.predict_proba([rep])[0]
            #     print("[cam{}] prob of each class: {}".format(cam_id, ' '.join(str(e)[:6] for e in prob)))
            #     print("[cam{}] max prob is {} , of identity {} at class index {}".format(cam_id, max(prob), identity, np.argmax(prob)))

            #     #dont use this thrd, since for many ppl svm, the prob cannot attend that high
            #     #prob_thd = 0.95
            #     # if max(prob) < prob_thd:
            #     #     identity = -1
            #     #     print("Top prob < {}, not so sure is one of the trained person, treat as unknown".format(prob_thd))
            
            # else:
            #     identity = 0

            prob = 0
            if self.knn[cam_id]:

                start = time.time()
                #when added person >1, the identity is the index return by svm
                identity = self.knn[cam_id].predict([rep])[0]

                #also need to double confirm with the probability of each class
                prob = self.knn[cam_id].predict_proba([rep])[0]
                print("[cam{}] prob of each class: {}".format(cam_id, ' '.join(str(e)[:6] for e in prob)))
                print("[cam{}] max prob is {} , of identity {} at class index {}".format(cam_id, max(prob), identity, np.argmax(prob)))

                #dont use this thrd, since for many ppl svm, the prob cannot attend that high
                #prob_thd = 0.95
                # if max(prob) < prob_thd:
                #     identity = -1
                #     print("Top prob < {}, not so sure is one of the trained person, treat as unknown".format(prob_thd))
                print("[cam{}] knn predict took {} seconds.".format(id, time.time() - start))
            
            else:
                identity = 0            
            
            #identity = 0

            #double confirm with class mean and std to confirm
            # if not self.mean:
            #     self.getData()
            if identity >= 0:
                if self.mean and self.std:

                    #confirm with one class svm
                    if self.doOneClassSVM:
                        index = self.identity_ofppl.index(identity)
                        one_class_predict_result = self.ppl_oneclasssvm_clf[index].predict(rep)[0]
                        if one_class_predict_result < 0:
                            print("[cam{}] one class svm predict result is negative, not so sure is one of the trained person, treat as unknown".format(cam_id))
                            identity = -1   
                            return (identity, BestMatchSimilarityScore)  


                    #confirm with class center distance
                    # dist_list = []
                    # least_dist_identity = -1
                    # least_dist = 9999
                    # for ide in self.identity_ofppl:
                    #     diff = np.absolute(self.mean[ide]-rep)
                    #     dist_to_center = np.linalg.norm(diff)
                    #     dist_list.append(dist_to_center)
                    #     if dist_to_center < least_dist:
                    #         least_dist = dist_to_center
                    #         least_dist_identity = ide


                    # print("[cam{}] dist of each class: {}".format(cam_id, ' '.join(str(e)[:6] for e in dist_list)))
                    # print("[cam{}] Least dist identity is {} and the dist is {}".format(cam_id, least_dist_identity, least_dist))
                    #print("This class std is : {}".format(self.std[identity]))

                    #find out the closest rep and its identity
                    least_dist_img = 9999
                    least_dist_img_idx = -1
                    least_dist_diff = []

                    for img in self.images.values():
                        diff = np.absolute(img.rep-rep)
                        dist = np.linalg.norm(diff)
                        if dist < least_dist_img:
                            least_dist_img = dist
                            least_dist_img_idx = img.identity
                            least_dist_diff = diff

                    print("[cam{}] Least dist img's identity is {} and the dist is {}".format(cam_id, least_dist_img_idx, least_dist_img))
                    #identity = least_dist_img_idx

                    if least_dist_img_idx != identity:
                        print("[cam{}] knn identity != least_dist_img_idx, not so sure is one of the trained person, treat as unknown".format(cam_id))
                        identity = -1   
                        return (identity, BestMatchSimilarityScore)                          

                    #Best match: score 1, poorest match: score 0
                    BestMatchSimilarityScore = round_to_1(math.exp(-1*least_dist_img))

                    #BestMatchSimilarityScore = prob
                    print("[cam{}] BestMatchSimilarityScore is {}".format(cam_id, BestMatchSimilarityScore))

                    BestMatchSimilarityScoreThd = 0.4
                    if BestMatchSimilarityScore > BestMatchSimilarityScoreThd:

                        #check if diff > 6*std in any of the dimension
                        largest_ratio=0
                        for idx, val in enumerate(self.std[identity]):
                            #print("[cam{}] idx: {}, Diff: {}, std: {}, ratio: {}".format(cam_id, idx, diff[idx], val, diff[idx]/val))
                            ratio = least_dist_diff[idx]/val
                            if ratio > largest_ratio:
                                largest_ratio = ratio

                            ratio_thd = 2.1
                            if ratio > ratio_thd:
                                identity = -1
                                print("[cam{}] SDratio of dim{} = {} > {}*Std, not so sure is one of the trained person, treat as unknown".format(cam_id, idx, ratio, ratio_thd))
                                break

                        print("[cam{}] Largest ratio so far is {}".format(cam_id, largest_ratio))

                    else:

                        print("[cam{}] BestMatchSimilarityScore <= 0.4, not so sure is one of the trained person, treat as unknown".format(cam_id))
                        identity = -1

                else:
                    identity = -1
        
        return (identity, BestMatchSimilarityScore)

    def ReInitializedTrackingVars(self):

        self.prev_bbs = []
        self.prev_identity = []
        self.prev_rep = []
        self.prev_score = []
        self.tracked_list_of_ppl = []   
        self.rescuetimes = []   

        for q in range(0, self.num_of_cam):
            self.prev_bbs.append([])
            self.prev_identity.append([])
            self.prev_rep.append([])
            self.prev_score.append([])
            self.tracked_list_of_ppl.append([])
            self.rescuetimes.append({})

    def AugmentThread(self, id, identity, aug_num, img):

        start = time.time()
        images_aug = seq.augment_images([img]* min(aug_num, 50))
        print("augmentation {} took {} seconds.".format(id, time.time() - start))

        if self.augment_mutex.acquire():
            for img in images_aug:
                #phash = str(imagehash.phash(Image.fromarray(img)))
                #self.trainingPhashs.append(phash)
                self.trainingAlignFaces.append(img)
                self.trainingIdentity.append(identity)
                self.trainingRep.append(None)
                self.trainingContent.append(None)
                self.tolearnimgqueue.put(len(self.trainingContent)-1)                    
                self.trainingnumber += 1

                if identity in self.trainingnumber_foreachide:
                    self.trainingnumber_foreachide[identity] += 1
                else:
                    self.trainingnumber_foreachide[identity] = 1    

            self.augment_mutex.release()


    def matching(self, cam_id, bb, bbs_ref, identitylist):
        min_dist = 9999
        index = -1
        if len(bbs_ref)==0:
            print("[cam{}] len(bbs_ref)==0, tracking fail!".format(cam_id, bb_ratio))
            return (index, min_dist)


        # calculate the distance threshold, it should be linear to the size of the bb
        ratio = 0.7
        if index > -1 and self.unknowntraining and identitylist[index] in self.unknowntraining_list : # for unknown still under training, make it easier to track, avoid tracking lost
            ratio = 2.0

        dist_thd = max((bb.right()-bb.left())*ratio, 50.0)            

        bbc = bb_centroid(bb)
        match_count=0
        for idx, bbb in enumerate(bbs_ref):
            bbbc = bb_centroid(bbb)
            dist = np.linalg.norm(bbc-bbbc)

            #dist > 0.0 to avoid tracking a wrong detected background
            #In normal use case all the people should be walking, so impossible to have dist == 0.0
            if self.prev_bbs_isused[cam_id][idx] == 0 and dist < dist_thd and identitylist[idx] != -99:
                print("[cam{}] matching best index is updated to {}, dist is {}".format(cam_id, idx, dist))
                min_dist = dist
                index = idx
                match_count = match_count + 1
        

        #print("[cam{}] dist_thd: {}".format(cam_id, dist_thd))

        #Let tracking fail if more than one match
        if match_count >= 2:
            index = -1
            print("[cam{}] match_count = {}, possible matching boundingbox num > 1, tracking fail!".format(cam_id, match_count))            

        # calculate the bb size ratio for a double check, since there may be very small fake bb appear in the background
        if index >= 0:
            bb_ratio = (1.0*((bb.right()-bb.left()))/((bbs_ref[index].right()-bbs_ref[index].left())))
            #print("[cam{}] bb_ratio: {}".format(cam_id, bb_ratio))

            if bb_ratio < 0.6 or bb_ratio > 1.0/0.6:
                index = -1
                print("[cam{}] bb_ratio = {} exceed limit (<0.6 / > 1/0.6), tracking fail!".format(cam_id, bb_ratio))

        if min_dist > dist_thd: # or min_dist == 0.0:
            print("[cam{}] min_dist  = {} > {} = dist_thd, tracking fail!".format(cam_id, min_dist, dist_thd))
            index = -1

        # if IsUnknownTraining:
        #     if (bb.right()-bb.left()) < 50:
        #         index = -1
        #         print("bb width < 50, unknowntraining only accept big enough heads to be learnt")

        # if index > -1 and min_dist != 9999:
        #     bbs_ref_isused[index]=1

        return (index, min_dist)


    # def capFrame(self):
    #     while True:
    #         start = time.time()
    #         rgbFrame_org = None
    #         if self.cap.isOpened():
    #             ret, rgbFrame_org = self.cap.read()
    #             if not ret or rgbFrame_org is None:
    #                 print("not ret or rgbFrame_org is None, continue")
    #                 continue
    #         else:
    #             self.cap.release()
    #             self.cap.open(self.video_stream_path)
    #             if not self.cap.isOpened():
    #                 print("cap is not open again, fail!!")                  
    #                 return

    #         if self.rgbFrame_mutex.acquire():
    #             #self.rgbFrame = cv2.resize(rgbFrame_org, (0,0), fx=0.5, fy=0.5)
    #             self.rgbFrame = rgbFrame_org.copy()
    #             self.rgbFrame_mutex.release()

    #         if self.zzzjpg_mutex.acquire():
    #             cv2.imwrite('zzz.jpg', self.rgbFrame)
    #             self.zzzjpg_mutex.release()

    #         #print("one loop of capFrame took {} seconds.".format(time.time() - start))

    #         #time.sleep(0.01)

    def onConnect(self, request):
        print("Client connecting: {0}".format(request.peer))
        self.training = True

    def onOpen(self):
        print("WebSocket connection open.")

    def onMessage(self, payload, isBinary):
        raw = payload.decode('utf8')
        msg = json.loads(raw)
        self.counter += 1

        if self.counter < 30 or self.counter % 100 == 0:
            print("Frame {},  Received {} message of length {}.".format(self.counter, msg['type'], len(raw)))

        if msg['type'] == "ALL_STATE":
            print("loading data!")
            print("Altogether {} images and {} people.".format(len(msg['images']), len(msg['people'])))
            self.loadState(msg['images'], msg['training'], msg['people'], msg['people_ide'])
            print("==================================================================")

        elif msg['type'] == "NULL":
            self.sendMessage('{"type": "NULL"}')

        elif msg['type'] == "FRAME":
            #print("receive sendframeloop")
            #self.processFrame(msg['dataURL'], msg['identity'], msg['id'])
            #self.sendMessage('{"type": "PROCESSED"}')
            #cam_id = msg['id']
            learnmsg = msg['learnmsg']
            if learnmsg != "None":
                print("Learn unknown person msg is received")
                newpersonname = msg['learnmsg']['name']
                isWithAug = msg['learnmsg']['IsAug']
                img_list_dataurl = msg['learnmsg']['img_list_dataurl']                
                self.learn_unknown_person(newpersonname, isWithAug, img_list_dataurl)

            
            statemsg = msg['statemsg']
            if statemsg != "None":
                print("loading data!")
                print("Altogether {} images and {} people.".format(len(msg['statemsg']['images']), len(msg['statemsg']['people'])))
                thread = threading.Thread(target=self.loadState, args=(msg['statemsg']['images'], msg['statemsg']['training'], msg['statemsg']['people'], msg['statemsg']['people_ide']))                          
                thread.start()
                print("==================================================================")

            #Get all the learnt images and so all to the client at one time
            learnmsgs = []
            while not self.learnimgqueue.empty():
                learnmsgs.append(self.learnimgqueue.get())
            
            pcontent = []
            for qx in range(0, len(self.video_stream_path)):
                if not self.processed_content[qx].empty():
                    pcontent.append((qx, self.processed_content[qx].get_nowait()))

            #print("debug x")

            if len(self.encodedimage) == 2 and not self.encodedimage[0] is None and not self.encodedimage[1] is None:
                msg = {
                    "type": "ANNOTATED",
                    "content": pcontent,
                    "fps": round_to_1(1/self.lastframetime[0]),
                    #"id": cam_id,
                    "imageset": 2,
                    "image": self.encodedimage,
                    "dateAndtime": str(datetime.datetime.now()),
                    "learnmsg": learnmsgs
                }        

                #print("debug y")
                if self.processed_msg.full():
                    self.processed_msg.get()
                self.processed_msg.put_nowait(msg)       
            
            elif not self.encodedimage[0] is None:
                msg = {
                    "type": "ANNOTATED",
                    "content": pcontent,
                    "fps": round_to_1(1/self.lastframetime[0]),
                    #"id": cam_id,
                    "imageset": 1,
                    "image": self.encodedimage[0],
                    "dateAndtime": str(datetime.datetime.now()),
                    "learnmsg": learnmsgs
                }    

                #print("debug z")   

                if self.processed_msg.full():
                    self.processed_msg.get()
                self.processed_msg.put_nowait(msg)      



            if not self.processed_msg.empty():
                msg = self.processed_msg.get_nowait()

                #while True:
                    #if not self.sending_trainingimg:
                self.sendMessage(json.dumps(msg))
                #print("valid frames info to client")
                    #     break
                    
                    # else:
                    #     time.sleep(0.5)
                    #     print("frame process is pending since server is sending training img to client")  

            else:
                time.sleep(0.005)
                msg = {
                    "type": "ANNOTATED",
                    "content": "None",
                    #"id": cam_id,
                    "learnmsg": []
                }        
                self.sendMessage(json.dumps(msg))   
                #print("Null frame to client")             
        
        elif msg['type'] == "START":
            print("Receive client msg to start the server") 
            if len(self.cap) > 0:
                print("Server already started! do nothing")
            else:
                for zzz in range(0, len(self.video_stream_path)):
                    path = self.video_stream_path[zzz]
                    #self.cap[-1].set(cv2.CV_CAP_PROP_BUFFERSIZE, 1); # internal buffer will now store only 1 frame
                    self.cap.append(None)
                    self.bufferframe.append(Queue.Queue(maxsize=self.queuesize))
                    self.bufferframe_toprocess.append(Queue.Queue(maxsize=self.queuesize))
                    self.processed_content.append(Queue.Queue(maxsize=self.queuesize))
                    self.grabtime.append(None)
                    self.bufferframe_gray.append(None)
                    self.encodedimage.append(None)
                    self.prev_bbs_isused.append(None)
                    self.lastframetime.append(0.025)
                    self.knn.append(KNeighborsClassifier(n_neighbors=30))
                    align.append(openface.AlignDlib(args.dlibFacePredictor))
                    #self.processed_msg.append(Queue.Queue())

                    #if self.cap[-1].isOpened():
                    #print("capture of path [{}] is opened successfully!".format(path))
                    capthread = threading.Thread(target=self.queryFrame, args=(zzz, path))
                    capthread.daemon = True
                    capthread.start()

                        # processimgthread_togray = threading.Thread(target=self.processImgToGray, args=(zzz, ))
                        # processimgthread_togray.daemon = True                            
                        # processimgthread_togray.start()    

                    processimgthread_toencode = threading.Thread(target=self.processImgToEncode, args=(zzz, ))
                    processimgthread_toencode.daemon = True                            
                    processimgthread_toencode.start()                            

                    processframethread = threading.Thread(target=self.processFrame, args=(zzz, ))
                    processframethread.daemon = True                            
                    processframethread.start()                                  

                    # else:

                    #     self.cap[-1].release()
                    #     self.cap[-1].open(path)

                    #     if self.cap[-1].isOpened():
                    #         print("capture of path [{}] is opened successfully in 2nd try!".format(path))
                    #         threading.Thread(target=self.queryFrame, daemon=True, args=(zzz, path)).start()    

                    #     else:
                    #         print("capture of path [{}] cannot open again, fail!".format(path))               

            print("==================================================================")

        elif msg['type'] == "TRAINING":
            self.training = msg['val']
            self.ReInitializedTrackingVars() # stop all tracking
            if not self.training:
                #leave training, so train SVM for the latest image sets
                self.trainFace()
                self.trainingnumber = 0 # reset to zero
                self.trainingnumber_foreachide = {}
                self.trainingIdentity = []
                self.trainingPhashs = []
                self.trainingAlignFaces = []
                self.trainingRep = []
                self.trainingContent = []
                self.learnimgqueue = Queue.Queue()
                self.tolearnimgqueue = Queue.Queue()
                #self.unknowntraining = False
                self.unknowntraining_list = []

            print("self.training = {}".format(self.training))
            print("==================================================================")

        elif msg['type'] == "ADD_PERSON":
            newpersonname = msg['val'].encode('ascii', 'ignore')
            self.people.append(newpersonname)
            new_identity = msg['ide']
            self.identity_ofppl.append(new_identity)
            self.ppl_oneclasssvm_clf.append(svm.OneClassSVM(nu=0.05))
            print("A new person is added! [{}], with identity {}".format(newpersonname, new_identity))
            print("==================================================================")

        elif msg['type'] == "EDIT_PERSON_NAME":
            identity = msg['identity']
            idx = self.identity_ofppl.index(identity)
            oldpersonname = self.people[idx]
            newpersonname = msg['val'].encode('ascii', 'ignore')
            self.people[idx] = newpersonname
            print("A person name is changed! [{} -> {}], whose identity is {}".format(oldpersonname, newpersonname, identity))
            print("==================================================================")            

        elif msg['type'] == "UPDATE_IDENTITY":
            h = msg['hash'].encode('ascii', 'ignore')
            if h in self.images:
                self.images[h].identity = msg['idx']
                print("An image identity is updated to {} !".format(msg['idx']))      

                if not self.training:
                    self.trainFace()

                print("==================================================================")

            else:
                print("Image not found.")
        
        elif msg['type'] == "LEARN_UNKNOWN_PERSON":
            newpersonname = msg['name']
            isWithAug = msg['IsAug']
            img_list_dataurl = msg['img_list_dataurl']

            self.learn_unknown_person(newpersonname, isWithAug, img_list_dataurl)
            # identity = 0
            # if self.identity_ofppl:
            #     identity = max(self.identity_ofppl)+1
            # self.people.append(newpersonname)
            # self.identity_ofppl.append(identity)
            # self.ppl_oneclasssvm_clf.append(svm.OneClassSVM(nu=0.05))

            # msg = {
            #     "type": "NEW_PERSON",
            #     "val": newpersonname,
            #     "identity": identity
            # }
            # self.sendMessage(json.dumps(msg))    

            # decoded_imgs = []

            # head = "data:image/jpeg;base64,"
            # for dataurl in img_list_dataurl:
            #     assert(dataurl.startswith(head))

            #     data = dataurl[len(head):]
            #     data2 = urllib.unquote(data)

            #     imgdata = base64.b64decode(data2)
            #     imgF = StringIO.StringIO()
            #     imgF.write(imgdata)
            #     imgF.seek(0)
            #     pil_image = Image.open(imgF)
            #     photo = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            #     decoded_imgs.append(photo)


            # # Add the original photo first, since it will be shown as display
            # for img in decoded_imgs:

            #     phash = str(imagehash.phash(Image.fromarray(img)))
            #     self.trainingPhashs.append(phash)
            #     self.trainingAlignFaces.append(img)
            #     self.trainingIdentity.append(identity)
            #     self.trainingRep.append(None)
            #     self.trainingContent.append(None)
            #     self.tolearnimgqueue.put(len(self.trainingContent)-1)                    
            #     self.trainingnumber += 1

            #     if identity in self.trainingnumber_foreachide:
            #         self.trainingnumber_foreachide[identity] += 1
            #     else:
            #         self.trainingnumber_foreachide[identity] = 1      

            # aug_ratio = 1
            # aug_half_num = 100
            # if isWithAug and len(img_list_dataurl)<aug_half_num :
            #     aug_ratio = (aug_half_num*2)/len(img_list_dataurl)
            
            # #start = time.time()
            # augmentation_num = aug_ratio-1
            # if augmentation_num > 0:
            #     for zz in range(0, len(decoded_imgs)):

            #         processframethread = threading.Thread(target=self.AugmentThread, args=(zz, identity, augmentation_num, decoded_imgs[zz]))                         
            #         processframethread.start()      
            
            # #print("augmentation took {} seconds.".format(time.time() - start))
            # print("Leart unknown person with {} photo, its identity {}".format(len(img_list_dataurl), identity))
            # print("==================================================================")     

            # time.sleep(1)
            # thread = threading.Thread(target=self.trainFace, args=())
            # thread.start()                                  

        elif msg['type'] == "NEW_PHOTO":
            dataURL = msg['dataurl']
            #convert dataurl into an image

            head = "data:image/jpeg;base64,"
            assert(dataURL.startswith(head))
            data = dataURL[len(head):]
            data2 = urllib.unquote(data)
            imgdata = base64.b64decode(data2)

            imgF = StringIO.StringIO()
            imgF.write(imgdata)
            imgF.seek(0)
            pil_image = Image.open(imgF)

            photo = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR) 
            ratio_x = 96.0/photo.shape[1]
            ratio_y = 96.0/photo.shape[0]
            resizedphoto = cv2.resize(photo, (0,0),fx=ratio_x, fy=ratio_y)     

            identity = 0
            if self.identity_ofppl:
                identity = max(self.identity_ofppl)+1
            newpersonname = "Unknown" + str(identity)
            self.people.append(newpersonname)
            self.identity_ofppl.append(identity)
            self.ppl_oneclasssvm_clf.append(svm.OneClassSVM(nu=0.05))

            msg = {
                "type": "NEW_PERSON",
                "val": newpersonname,
                "identity": identity
            }
            self.sendMessage(json.dumps(msg))

            #augmentation to the single photo
            start = time.time()
            augmentation_num = 49
            images_aug = seq.augment_images([resizedphoto]*augmentation_num)
            images_aug.insert(0, resizedphoto)  # insert the original photo to the front
            print("augmentation took {} seconds.".format(time.time() - start))

            for img in images_aug:

                #phash = str(imagehash.phash(Image.fromarray(img)))
                #self.trainingPhashs.append(phash)
                self.trainingAlignFaces.append(img)
                self.trainingIdentity.append(identity)
                self.trainingRep.append(None)
                self.trainingContent.append(None)
                self.tolearnimgqueue.put(len(self.trainingContent)-1)                    
                self.trainingnumber += 1

                if identity in self.trainingnumber_foreachide:
                    self.trainingnumber_foreachide[identity] += 1
                else:
                    self.trainingnumber_foreachide[identity] = 1


            print("Processed a new photo and mark its identity {}".format(identity))
            print("==================================================================")        

            time.sleep(1)
            thread = threading.Thread(target=self.trainFace, args=())
            thread.start()          

        elif msg['type'] == "REMOVE_IMAGE":
            h = msg['hash'].encode('ascii', 'ignore')
            ide = msg['ide']
            if h in self.images:
                del self.images[h]
                print("An image is deleted! the identity of this image is {}".format(ide))      

                if not self.training:
                    self.trainFace()

                print("==================================================================")

            else:
                print("Image not found.")
        
        elif msg['type'] == "MERGE_PERSON":
            idx1 = msg['master']
            idx2 = msg['slave']
            idx1_name = msg['master_name']
            idx2_name = msg['slave_name']

            index = self.people.index(idx2_name)
            self.people.remove(idx2_name)
            self.identity_ofppl.remove(idx2)
            del self.ppl_oneclasssvm_clf[index]

            for key, img in self.images.iteritems():
                if img.identity == idx2:
                    self.images[key].identity = idx1          

            print("Person {} is merged to person {}, whose identity is {} and {}".format(idx2_name, idx1_name, idx2, idx1))      

            if not self.training:
                self.trainFace()            

            print("==================================================================")            

        elif msg['type'] == "DELETE_PERSON":
            ide = msg['val']
            personname = msg['name']

            index = self.people.index(personname)
            self.people.remove(personname)
            self.identity_ofppl.remove(ide)
            del self.ppl_oneclasssvm_clf[index]

            keylist = []
            for key, img in self.images.iteritems():
                if img.identity == ide:
                    keylist.append(key)

            for key in keylist:
                del self.images[key]

            print("Person {} is deleted, {} images of the person is deleted, whose identity is {}!".format(personname, len(keylist), ide))      

            if not self.training:
                self.trainFace()            

            print("==================================================================")

        # elif msg['type'] == 'REQ_TSNE':
        #     self.sendTSNE(msg['people'])

        else:
            print("Warning: Unknown message type: {}".format(msg['type']))


    def onClose(self, wasClean, code, reason):
        print("WebSocket connection closed: {0}".format(reason))

    def loadState(self, jsImages, training, jsPeople, jsIdentity):
        start = time.time()
        self.training = training

        for jsImage in jsImages:
            h = jsImage['hash'].encode('ascii', 'ignore')

            rep = jsImage['representation']
            if rep == "0":
                print("The image has no representation, need to recalculate!")

                dataURL = jsImage['image']
                head = "data:image/jpeg;base64,"
                assert(dataURL.startswith(head))
                imgdata = base64.b64decode(dataURL[len(head):])

                imgF = StringIO.StringIO()
                imgF.write(imgdata)
                imgF.seek(0)
                pil_image = Image.open(imgF)

                photo = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR) 
                ratio_x = 96.0/photo.shape[1]
                ratio_y = 96.0/photo.shape[0]
                resizedphoto = cv2.resize(photo, (0,0),fx=ratio_x, fy=ratio_y)                     
                rep = cpunet[0].forward(resizedphoto)

            self.images[h] = Face(np.array(rep),
                                  jsImage['identity'])

        for jsPerson in jsPeople:
            self.people.append(jsPerson.encode('ascii', 'ignore'))

        for jside in jsIdentity:
            self.identity_ofppl.append(jside)
            self.ppl_oneclasssvm_clf.append(svm.OneClassSVM(nu=0.05))

        print("List of trained person and their identity index:")
        for q in range(0, len(self.people)):
            print("[{}, {}]".format(self.people[q], self.identity_ofppl[q]))

        print("loadState took {} seconds.".format(time.time() - start))

        if not training:
            self.trainFace()

    def getData(self):
        start = time.time()
        X = []
        y = []
        EachClassX = [[] for _ in xrange(len(self.people))]

        for img in self.images.values():
            rep = img.rep
            X.append(rep)
            y.append(img.identity)
            index = self.identity_ofppl.index(img.identity)
            EachClassX[index].append(rep)
            
            if img.identity in self.rep_of_each_class:
                self.rep_of_each_class[img.identity].append(rep)
            else:
                self.rep_of_each_class[img.identity] = [rep]

            #print("class {} now has {} representation collected".format(img.identity, len(self.rep_of_each_class[img.identity])))

        numIdentities = len(set(y + [-1])) - 1
        if numIdentities == 0:
            print("numIdentities == 0, return None")
            return None

        if args.unknown:
            numUnknown = y.count(-1)
            numIdentified = len(y) - numUnknown
            numUnknownAdd = (numIdentified / numIdentities) - numUnknown
            if numUnknownAdd > 0:
                print("+ Augmenting with {} unknown images.".format(numUnknownAdd))
                for rep in self.unknownImgs[:numUnknownAdd]:
                    # print(rep)
                    X.append(rep)
                    y.append(-1)

        X = np.vstack(X)
        y = np.array(y)

        # calculate mean and std of different classes
        for key, value in self.rep_of_each_class.iteritems():
            #self.mean[key] = np.mean(value, axis=0)
            self.mean[key] = np.median(value, axis=0)
            self.std[key] = np.std(value, axis=0)
            print("Identity {} has calulcated its class mean and std".format(key))

        print("getData took {} seconds.".format(time.time() - start))
        return ((X, y), EachClassX)

    # def stop_all_tracking(self):
    #     self.prev_bbs = []
    #     self.prev_identity = []
    #     self.prev_rep = []
    #     self.prev_score = []
    #     self.tracked_list_of_ppl = []

    def learn_unknown_person(self, newpersonname, isWithAug, img_list_dataurl):
        identity = 0
        if self.identity_ofppl:
            identity = max(self.identity_ofppl)+1
        self.people.append(newpersonname)
        self.identity_ofppl.append(identity)
        self.ppl_oneclasssvm_clf.append(svm.OneClassSVM(nu=0.05))

        msg = {
            "type": "NEW_PERSON",
            "val": newpersonname,
            "identity": identity
        }
        self.sendMessage(json.dumps(msg))    

        decoded_imgs = []

        head = "data:image/jpeg;base64,"
        for dataurl in img_list_dataurl:
            assert(dataurl.startswith(head))

            data = dataurl[len(head):]
            data2 = urllib.unquote(data)

            imgdata = base64.b64decode(data2)
            imgF = StringIO.StringIO()
            imgF.write(imgdata)
            imgF.seek(0)
            pil_image = Image.open(imgF)
            photo = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            decoded_imgs.append(photo)


        # Add the original photo first, since it will be shown as display
        for img in decoded_imgs:

            #phash = str(imagehash.phash(Image.fromarray(img)))
            #self.trainingPhashs.append(phash)
            self.trainingAlignFaces.append(img)
            self.trainingIdentity.append(identity)
            self.trainingRep.append(None)
            self.trainingContent.append(None)
            self.tolearnimgqueue.put(len(self.trainingContent)-1)                    
            self.trainingnumber += 1

            if identity in self.trainingnumber_foreachide:
                self.trainingnumber_foreachide[identity] += 1
            else:
                self.trainingnumber_foreachide[identity] = 1      

        aug_ratio = 1
        aug_half_num = 150
        if isWithAug and len(img_list_dataurl)<aug_half_num :
            aug_ratio = (aug_half_num*2)/len(img_list_dataurl)
        
        #start = time.time()
        augmentation_num = aug_ratio-1
        if augmentation_num > 0:
            for zz in range(0, len(decoded_imgs)):

                processframethread = threading.Thread(target=self.AugmentThread, args=(zz, identity, augmentation_num, decoded_imgs[zz]))                         
                processframethread.start()      
        
        #print("augmentation took {} seconds.".format(time.time() - start))
        print("Leart unknown person with {} photo, its identity {}".format(len(img_list_dataurl), identity))
        print("==================================================================")     

        time.sleep(1)
        thread = threading.Thread(target=self.trainFace, args=())
        thread.start()       

    def trainFace(self):
        start = time.time()

        #self.stop_all_tracking()
        #print("Relearn faces,... Stop all tracking")
        print("trainFace()")

        self.rep_of_each_class.clear()
        self.mean.clear()
        self.std.clear()

        # make sure all of them are processed
        alldataloaded = False
        while not alldataloaded:
            alldataloaded = True

            if not self.tolearnimgqueue.empty():
                alldataloaded = False
                time.sleep(1)
                #break

        time.sleep(1)     
        d = self.getData()

        #print("process alignedFaces on the stack took {} seconds.".format(time.time() - start))
        #start = time.time()

        print("+ Training Face on {} labeled images.".format(len(self.images)))

        #a class must have >=5 samples to be a class
        true_num_of_identity=0
        for key, value in self.rep_of_each_class.iteritems():
            if len(value)>=5:
                print("Check: identity {} has {} >=5 training samples".format(key, len(value)))
                true_num_of_identity +=1

        if d is None:
            self.svm = None
            print("No data, skip svm")
            return
        else:
            start = time.time()

            # train the one class svm for each class
            if self.doOneClassSVM:
                EachClassX = d[1]
                for idx, classX in enumerate(EachClassX):
                    XX = np.vstack(classX)
                    self.ppl_oneclasssvm_clf[idx].fit(XX)
                    print("trained class {} with one class svm.".format(idx))


            # train the all class svm

            (X, y) = d[0]  # d = array(img.rep, img.identity)
            numIdentities = len(set(y + [-1]))
            if numIdentities <= 1: # no need svm if only one type of sample is present
                print("No need svm since there is only one person added")
                return

            if true_num_of_identity <= 1:
                print("No need svm since there is only one class has >=5 samples")
                return                

            param_grid = [
                {'C': [1, 10, 100, 1000],
                 'kernel': ['linear']},
                {'C': [1, 10, 100, 1000],
                 'gamma': [0.001, 0.0001],
                 'kernel': ['rbf']}
            ]

            #class sklearn.model_selection.GridSearchCV(estimator, param_grid, scoring=None, fit_params=None, 
            # SVC(C=1), Penalty parameter C of the error term.
            # cv : int, cross-validation generator or an iterable, optional
            # the training time of svm is quadratic with number of samples, so not suitable for large dataset


            #self.svm = GridSearchCV(SVC(C=1, probability=True), param_grid, cv=5).fit(X, y)
            self.svm = SVC(C=1.0, probability=True, kernel='linear').fit(X, y)

            print("train linearsvm took {} seconds.".format(time.time() - start))
            start = time.time()

            for zz in range(0, len(self.video_stream_path)):
                self.knn[zz].fit(X, y)

            #self.svm = LinearSVC.fit(X, y)

            print("train knn took {} seconds.".format(time.time() - start))

    #def processFrame(self, dataURL, identity, cam_id):
    def processFrame(self, cam_id):
        # if self.firstFrame:
        #     for path in self.video_stream_path:
        #         self.cap.append(cv2.VideoCapture(path))

        #         if self.cap[-1].isOpened():
        #             print("capture of path [{}] is opened successfully!", path)
        #         else:

        #             self.cap[-1].release()
        #             self.cap[-1].open(path)

        #             if self.cap[-1].isOpened():
        #                 print("capture of path [{}] is opened successfully in 2nd try!", path)         

        #             else:
        #                 print("capture of path [{}] cannot open again, fail!", path)   

        #     self.firstFrame=False

        #skip frame to achieve more smoothness
        #if self.counter % 2 ==0 or self.training:
        counter = 0
        while True:
            identity = -1
            framestart = time.time()
            start = time.time()
            NameAndBB = []
            counter = counter + 1

            # print memory summary in order to debug memory leakage
            #if counter%10 == 0:
                # gc.collect()
                # all_objects = muppy.get_objects()
                # print("Length of all object= {}".format(len(all_objects)))

                # sum1 = summary.summarize(all_objects)
                # summary.print_(sum1)
                # print mem_top()

                # print("Size of all variables===")
                # for var, obj in locals().items():
                #     print var, sys.getsizeof(obj)

            #print("Processing frame from cam {}".format(cam_id))

            ##
            # head = "data:image/jpeg;base64,"
            # assert(dataURL.startswith(head))
            # imgdata = base64.b64decode(dataURL[len(head):])

            # if args.verbose:
            #     print("Decode the image took {} seconds.".format(time.time() - start))
            #     start = time.time()

            # imgF = StringIO.StringIO()
            # imgF.write(imgdata)
            # imgF.seek(0)
            # pil_image = Image.open(imgF)
            ##

            # if self.cap[cam_id].isOpened():
            #     ret, self.rgbFrame = self.cap[cam_id].read()
            #     if not ret or self.rgbFrame is None:
            #         print("not ret or self.rgbFrame is None, try again. ret={}".format(ret))
            #         self.cap[cam_id].release()
            #         self.cap[cam_id].open(self.video_stream_path[cam_id])
            #         if not self.cap[cam_id].isOpened():
            #             print("cap is not open again, fail!!")                  
            #             return
            #         else:
            #             ret, self.rgbFrame = self.cap[cam_id].read()
            #             if not ret or self.rgbFrame is None:         
            #                 print("not ret or self.rgbFrame is None again, return to skip this frame. ret={}".format(ret))
            #                 return

            # else:
            #     self.cap[cam_id].release()
            #     self.cap[cam_id].open(self.video_stream_path[cam_id])
            #     if not self.cap[cam_id].isOpened():
            #         print("cap is not open again, fail!!")                  
            #         return
            #     else:
            #         ret, self.rgbFrame = self.cap[cam_id].read()
            #         if not ret or self.rgbFrame is None:         
            #             print("not ret or self.rgbFrame is None again, return to skip this frame. ret={}".format(ret))
            #             return

            # img.save("zzz.jpg", "JPEG")
            # if args.verbose:
            #     print("img.save the image took {} seconds.".format(time.time() - start))
            #     start = time.time()

            ##self.rgbFrame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            ##buf = np.fliplr(np.asarray(img))
            # buf = np.asarray(img)

            # if args.verbose:
            #     print("pil image to opencv mat took {} seconds.".format(time.time() - start))
            #     start = time.time()  

            #self.rgbFrame = np.zeros((720, 1280, 3), dtype=np.uint8)
            #self.rgbFrame = np.zeros((450, 800, 3), dtype=np.uint8)
            #rgbFrame = np.zeros((360, 640, 3), dtype=np.uint8)
            #rgbFrame = np.zeros((216, 384, 3), dtype=np.uint8), frame length 44370, total 55fps, load 4ms, write 3ms, face 8.3ms 
            #rgbFrame = np.zeros((234, 416, 3), dtype=np.uint8), frame length 51450, totoal 50fps, load 4.8ms, write 3.3ms, face 9.5ms

            #rgbFrame = np.zeros((252, 448, 3), dtype=np.uint8) # frame length 55282, totoal 48fps, load 5.4ms, write 3.6ms, face 9.6ms
            #self.rgbFrame[:, :, 0] = buf[:, :, 2]
            #self.rgbFrame[:, :, 1] = buf[:, :, 1]
            #self.rgbFrame[:, :, 2] = buf[:, :, 0]

            scale_factor = 1
            inv_scale = 1.0/scale_factor
            # rgbFrame = cv2.resize(rgbFrame_org, (0,0), fx=inv_scale, fy=inv_scale)
            # if args.verbose:
            #     print("resize the image took {} seconds.".format(time.time() - start))
            #     start = time.time()

            # rgbFrame_gray = cv2.cvtColor(rgbFrame, cv2.COLOR_BGR2GRAY)
            # if args.verbose:
            #     print("rgb to gray the image took {} seconds.".format(time.time() - start))
            #     start = time.time()

            #rgbFrame_gray = cv2.equalizeHist(rgbFrame_gray)
            #cv2.imwrite('zzz.png',rgbFrame_gray)
            

            # if not self.training:
            #     annotatedFrame_org = np.copy(buf)
            #     annotatedFrame = cv2.resize(annotatedFrame_org, (0,0), fx=0.5, fy=0.5)

            # cv2.imshow('frame', rgbFrame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     return

            #if args.verbose:
            #    print("equalizeHist the image took {} seconds.".format(time.time() - start))
            #    start = time.time()

            identities = []
            bbs = []

            # bbs = align.getAllFaceBoundingBoxes(rgbFrame)
            #bb = align.getLargestFaceBoundingBox(rgbFrame_gray)

            #Try using opencv blp face detection
            #minNeightbour: Parameter specifying how many neighbors each candidate rectangle should have to retain it.
            #faces = face_cascade.detectMultiScale(rgbFrame, 1.1, 2 ,cv2.CASCADE_SCALE_IMAGE,(20,20),(60,60))
            #print(len(faces))
            #convert faces to bb
            #for (x, y, w, h) in faces:
                #bbs.append(dlib.rectangle(x, y, x+w, y+h))



            # rgbFrame = None
            # if self.bufferframe_toprocess[cam_id].empty():
            #     print("[cam{}] self.bufferframe_toprocess is empty, skip frame".format(cam_id))
            #     # msg = {
            #     #     "type": "SKIP"
            #     # #     "content": "None",
            #     # #     "fps": 1,
            #     # #     "id": cam_id,
            #     # #     "image": None,
            #     # #     "dateAndtime": str(datetime.datetime.now())
            #     # }        
            #     # self.sendMessage(json.dumps(msg))
            #     #print("==================================================================")    
            #     time.sleep(1)  
            #     continue
            # else:
            rgbFrame = self.bufferframe_toprocess[cam_id].get(True)



            #time.sleep(1)  
            #thread = threading.Thread(target=self.encodeimage, args=(cam_id, rgbFrame))
            #thread.start()                                  # Start the execution

            # if args.verbose:
            #     print("copy self.bufferframe[cam_id] took {} seconds.".format(time.time() - start))
            #     start = time.time()

            #name_of_image = "zzz" + str(cam_id) + ".jpg"

            #cv2.imwrite(name_of_image, rgbFrame)
            #self.rgbFrame = cv2.imread('zzz.jpg')
            # if args.verbose:
            #     print("imwrite the image took {} seconds.".format(time.time() - start))
            #     start = time.time()

            # if self.training and identity == -1:
            #     if not self.unknowntraining:
            #         print("Now is training unknown people...")
            #     self.unknowntraining = True


            #this_dir = os.path.dirname(os.path.realpath(__file__))

            #if self.zzzjpg_mutex.acquire():
            #faces = cv2gpu.find_faces('/root/openface/' + name_of_image)
            #   img_str_in_char = cv2.imencode('.jpg', rgbFrame)[1].encode('utf-8')
            #img_str = np.array(cv2.imencode('.jpg', rgbFrame)[1]).tostring()
            #print(rgbFrame)



            rgbFrame_gray = None
            if len(rgbFrame.shape) == 3:
                rgbFrame_gray = cv2.cvtColor(rgbFrame, cv2.COLOR_BGR2GRAY)
            else:
                rgbFrame_gray = rgbFrame.copy()



            #(channel_b, channel_g, channel_r) = cv2.split(rgbFrame)

            #redImg = rgbFrame[:, :, 0]
            # set green and red channels to 0
            #redImg[:, :, 1] = 0
            #redImg[:, :, 2] = 0

            # if args.verbose:
            #     print("Extract red channel took {} seconds.".format(time.time() - start))
            #     start = time.time()

                
            if self.gpu_mutex.acquire():
                faces = cv2gpu.find_faces(rgbFrame_gray)
                self.gpu_mutex.release()


            #faces = face_cascade.detectMultiScale(rgbFrame_gray, 1.1, 4)
            #faces = []

            #faces = cv2gpu.find_faces(rgbFrame_gray)
                #self.zzzjpg_mutex.release()
            #faces = cv2gpu.find_faces('http://172.18.9.99/axis-cgi/jpg/image.cgi')
            #print(len(faces))

            #if on person specific training, only take the largest boundingbox, since the person going
            #to learn must stand in the front
            if self.training and len(faces) > 1:
                faces = max(faces, key=lambda rect: rect[2] * rect[3])
                faces = [faces]

            for (x, y, w, h) in faces:
                #print(x*scale_factor, y*scale_factor, w*scale_factor, h*scale_factor)
                bbs.append(dlib.rectangle(x*scale_factor, y*scale_factor, (x+w)*scale_factor, (y+h)*scale_factor))

            #bbs = [bb] if bb is not None else []

            if args.verbose and counter % time_log_freq == 0:
                print("[cam{}] Face detection took {} seconds.".format(cam_id, time.time() - start))
                start = time.time()

            # if len(bbs) > 1:
            #     print("Number of detected faces: ", len(bbs))

            identitylist = []
            replist = []
            scorelist = []
            BestMatchSimilarityScore = []
            self.prev_bbs_isused[cam_id] = [0] * (len(self.prev_bbs[cam_id])+1)
            #dist_thd = 20.0 # unit is pixel, the larger the more easy to got matching, save time, but more easy to have mismatch
            prob = [1.0]
            nn_processed_bbs = 0
            StopUpdatePrev = False
            name = None

            #if len(bbs) == 0:
                #print("[cam{}] No bbs is found in this frame!!".format(cam_id))

            #Main loop for each detected bounding boxes
            #bbs = []

            # since nn has quota, need to shuffle bbs to avoid keep nn the same person
            random.shuffle(bbs)

            for idxx, bb in enumerate(bbs):

                isNewlyDetect = False
                BestMatchSimilarityScore = 0
                bb_width = bb.right()-bb.left()
                BB_dataurl = None
                #print("[cam{}] BB:{} ->(left:{}, top:{}, x_width:{}, y_width:{})".format(cam_id, idxx+1, bb.left()*scale_factor, bb.top()*scale_factor,
                #bb_width*scale_factor, bb_width*scale_factor))
                
                if self.training:
                    if bb_width < 50:
                        print("[cam{}] bb width < 50, active training only accept big enough heads to be learnt".format(cam_id))
                        identitylist.append(-99) # -99 means its not going to be tracked
                        replist.append(None)
                        scorelist.append(0.0)
                        continue

                # use dlib to confirm again the bb is valid
                start = time.time()
                dlib_margin = 20
                rgbFrame_roi = rgbFrame[max(bb.top()-dlib_margin, 0):min(720-1, bb.bottom()+dlib_margin),max(bb.left()-dlib_margin, 0):min(1280-1, bb.right()+dlib_margin)]
                rgbFrame_roi_resize = None

                #resize the ROI otherwise dlib runs very slow
                if bb_width > 100:
                    resize_ratio = 0.7
                    if bb_width > 200:
                        resize_ratio = 0.4
                        if bb_width > 400:
                            resize_ratio = 0.25
                    rgbFrame_roi_resize = cv2.resize(rgbFrame_roi, (0,0), fx=resize_ratio, fy=resize_ratio)
                else:
                    rgbFrame_roi_resize = rgbFrame_roi.copy()

                dlib_bb = align[cam_id].getLargestFaceBoundingBox(rgbFrame_roi_resize)

                # if args.verbose and counter % 30 == 0:
                #     print("[cam{}] dlib confirmation took {} seconds.".format(cam_id, time.time() - start))
                #     start = time.time()

                if not dlib_bb:
                    #print("[cam{}] dlib confirmation fail!".format(cam_id))
                    identitylist.append(-99) # -99 means its not going to be tracked
                    replist.append(None)
                    scorelist.append(0.0)
                    continue

                # landmarks = align.findLandmarks(rgbFrame, bb)
                # if args.verbose:
                #     print("Find landmarks~ took {} seconds.".format(time.time() - start))
                #     start = time.time()

                BB_ROI = rgbFrame[bb.top():bb.bottom(),bb.left():bb.right()]
                ratio_x = 1.0*96/BB_ROI.shape[1]
                ratio_y = 1.0*96/BB_ROI.shape[0]
                BB_ROI_resize = cv2.resize(BB_ROI, (0,0), fx=ratio_x, fy=ratio_y)

                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                retval, buffer = cv2.imencode('.jpeg', BB_ROI_resize, encode_param)
                quote = urllib.quote(base64.b64encode(buffer))
                BB_dataurl = 'data:image/jpeg;base64,' + quote

                # Do tracking first, see if we can match with prev bbs, if yes, then 
                # for unknown, dont track more than 4 frames, as it may be wrongly classify into unknown, so give chance to correct
                #max_tracking_count_for_unknown = 1
                matchingresult = self.matching(cam_id, bb, self.prev_bbs[cam_id], self.prev_identity[cam_id])
                if (not self.training and len(self.prev_bbs[cam_id])>0 and 
                    len(self.prev_identity[cam_id])>0 and len(self.prev_rep[cam_id])>0 and matchingresult[0] >= 0 and 
                    self.prev_identity[cam_id][matchingresult[0]] >= 0 ):

                    identity = self.prev_identity[cam_id][matchingresult[0]]
                    self.prev_bbs_isused[cam_id][matchingresult[0]]=1
                    print("[cam{}] Tracking successful, matching index is {}, matching identity is {}, matching dist is {}, skip face landmark and nn net forward".format(cam_id, matchingresult[0], identity, matchingresult[1]))
                    #print("prev_identity: {}".format(' '.join(str(e) for e in self.prev_identity)))
                    #print("prev_rep: {}".format(' '.join(str(e) for e in self.prev_rep)))
                    
                    BestMatchSimilarityScore = self.prev_score[cam_id][matchingresult[0]]
                    rep = self.prev_rep[cam_id][matchingresult[0]]

                    if self.unknowntraining and identity == -1:

                        # double confirm whether it is real unmatch with any person in the database
                        # becoz it is possible that it is a miskake to recongize a real person into unknown
                        alignedFace = align[cam_id].align(args.imgDim, rgbFrame, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE) 
                        phash = str(imagehash.phash(Image.fromarray(alignedFace)))
                        if phash in self.images:
                            print("phash in self.image, skip training this")
                            identitylist.append(-99) # -99 means its not going to be tracked
                            replist.append(None)
                            scorelist.append(0.0)                        
                            continue  
                        rep = net.forward(alignedFace)

                        print("Double check if this rep cannot match with any record in the database...")
                        (ide, sscore) = self.Determine_identity_by_rep(cam_id, rep)

                        if ide == -1:
                            # add new person automatically since it detects a new bb
                            self.unknowntraininglatestindex += 1

                            identity = 0
                            if self.identity_ofppl:
                                identity = max(self.identity_ofppl)+1
                            newpersonname = "Unknown" + str(self.unknowntraininglatestindex)
                            self.people.append(newpersonname)
                            self.identity_ofppl.append(identity)
                            self.ppl_oneclasssvm_clf.append(svm.OneClassSVM(nu=0.05))

                            print("A new unknown person is detected -> {}, identity = {}".format(newpersonname, identity))

                            msg = {
                                "type": "NEW_PERSON",
                                "val": newpersonname,
                                "identity": identity
                            }
                            self.sendMessage(json.dumps(msg))
                            self.unknowntraining_list.append(identity)

                            if identity not in identities:
                                identities.append(identity)   

                        else:      
                            print("Fail! it is not a stable unknown, will not trigger automatic learning")
                            identitylist.append(-99) # -99 means its not going to be tracked
                            replist.append(None)
                            scorelist.append(0.0)                        
                            continue  
                            
                    # if it is the already learnt person
                    if not identity in self.unknowntraining_list:
                        if identity != -1:
                            name = self.people[self.identity_ofppl.index(identity)] #+ ", " + str(round_to_1(prob[0]*100)) + "%"
                            print ("[cam{}] ===> [{}] is detected! its identity is {}".format(cam_id, name, identity))

                            #isNewlyDetect is for the bb that is tracked for the first time
                            if not identity in self.tracked_list_of_ppl[cam_id]:
                                isNewlyDetect = True
                                print("[cam{}] A newly detected face! update the result table".format(cam_id))
                                self.tracked_list_of_ppl[cam_id].append(identity)    

                        else:
                            name = "Unknown"

                    # if it is the new unknown which is given a new identity but not yet finish training
                    elif self.unknowntraining:
                        #if collect enough photo for the unknown, start to train face
                        Num_of_img_unknown_need_to_train = 40
                        if identity in self.trainingnumber_foreachide and self.trainingnumber_foreachide[identity] >= Num_of_img_unknown_need_to_train:
                            self.trainFace()
                            self.unknowntraining_list.remove(identity)
                            name = "Finished!"
                            print("{} Learning finished!".format(self.people[self.identity_ofppl.index(identity)]))
                            
                        else:

                            #if self.rgbFrame_mutex.acquire():
                            alignedFace = align[cam_id].align(args.imgDim, rgbFrame, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)  
                                #self.rgbFrame_mutex.release()

                            phash = str(imagehash.phash(Image.fromarray(alignedFace)))
                            if phash in self.images:
                                print("phash in self.image, skip training this")
                                identitylist.append(-99) # -99 means its not going to be tracked
                                replist.append(None)
                                scorelist.append(0.0)                        
                                continue                      

                            #self.trainingPhashs.append(phash)
                            self.trainingAlignFaces.append(alignedFace)
                            self.trainingIdentity.append(identity)
                            self.trainingRep.append(None)
                            self.trainingContent.append(None)
                            self.tolearnimgqueue.put(len(self.trainingContent)-1)    
                            self.trainingnumber += 1

                            if identity in self.trainingnumber_foreachide:
                                self.trainingnumber_foreachide[identity] += 1
                            else:
                                self.trainingnumber_foreachide[identity] = 1

                            percentage = 100.0*(self.trainingnumber_foreachide[identity]/(1.0*Num_of_img_unknown_need_to_train))
                            name = "Learning [" + str(round_to_1(percentage)) + "%]"
                            print("{} -> {}, identity {},in unknown person training mode, {}th record".format(name, self.people[self.identity_ofppl.index(identity)], identity, self.trainingnumber_foreachide[identity]))



                    identitylist.append(identity)
                    replist.append(rep)
                    scorelist.append(BestMatchSimilarityScore)

                    NameAndBB.append((name, bb.left()*inv_scale, bb.top()*inv_scale, bb_width*inv_scale,
                    bb_width*inv_scale, identity, isNewlyDetect, BestMatchSimilarityScore, BB_dataurl))
                    #continue
                
                #when tracking fails or if it is in active learning mode
                else:

                    # One frame at most do one nn forward (hopefully the rest is handled by tracking)
                    # Do this to ensure speed can be maintained
                    if nn_processed_bbs >= 1:
                        print("[cam{}] quota for net.forward() is consumed, treat this bb in next frame!".format(cam_id))
                        identitylist.append(-99) # -99 means its not going to be tracked
                        replist.append(None)
                        scorelist.append(0.0)
                        continue

                    if not self.training and len(self.prev_bbs[cam_id])>0 and matchingresult[0] < 0:
                        print("[cam{}] Tracking fail, tracking dist is {}, matching result is {}".format(cam_id, matchingresult[1], matchingresult[0]))

                    #if self.rgbFrame_mutex.acquire():
                    alignedFace = align[cam_id].align(args.imgDim, rgbFrame, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
                        #self.rgbFrame_mutex.release()
                        
                    if args.verbose and counter % time_log_freq == 0:
                        print("[cam{}] Find landmarks and alignment took {} seconds.".format(cam_id, time.time() - start))
                        start = time.time()

                    if alignedFace is None:
                        identitylist.append(-99) # -99 means its not going to be tracked
                        replist.append(None)
                        scorelist.append(0.0)                        
                        continue

                    # the hash is used as the key for the map
                    phash = str(imagehash.phash(Image.fromarray(alignedFace)))
                    # if args.verbose:
                    #     print("Image hash took {} seconds.".format(time.time() - start))
                    #     start = time.time()

                    #Determine identity by 1. getting representation from nn forward, 2. svm of the representation
                    if phash in self.images and self.training:
                        #identity = self.images[phash].identity
                        print("phash in self.image, skip training this")
                        identitylist.append(-99) # -99 means its not going to be tracked
                        replist.append(None)
                        scorelist.append(0.0)                        
                        continue

                    #active training mode
                    if self.training:

                        #self.trainingPhashs.append(phash)
                        self.trainingAlignFaces.append(alignedFace)
                        self.trainingIdentity.append(identity)
                        self.trainingRep.append(None)
                        self.trainingContent.append(None)
                        self.tolearnimgqueue.put(len(self.trainingContent)-1)    
                        self.trainingnumber += 1

                        if identity in self.trainingnumber_foreachide:
                            self.trainingnumber_foreachide[identity] += 1
                        else:
                            self.trainingnumber_foreachide[identity] = 1

                        name = "Learning [" + str(self.trainingnumber_foreachide[identity]) + "Img]"
                        print("{} -> {}, identity {},in active person training mode".format(name, self.people[self.identity_ofppl.index(identity)], identity)) 

                        #print(name)
                        NameAndBB.append((name, bb.left()*inv_scale, bb.top()*inv_scale,
                        bb_width*inv_scale, bb_width*inv_scale, identity, isNewlyDetect, BestMatchSimilarityScore, BB_dataurl))                     

                    #if not active training and fail for tracking
                    else:

                        rep = None
                        if self.gpu_mutex.acquire():

                            #Sometimes die in this place, dont why
                            try:
                                rep = net.forward(alignedFace)
                            except ValueError:
                                print("[cam{}] net.forward(alignedFace) die, skip".format(cam_id))
                                self.gpu_mutex.release()
                                identitylist.append(-99) # -99 means its not going to be tracked
                                replist.append(None)
                                scorelist.append(0.0)                        
                                continue                                

                            self.gpu_mutex.release()
                        #rep = cpunet.forward(alignedFace)
                        
                        #isNewlyDetect = True
                        #print("A newly detected face!")

                        nn_processed_bbs += 1
                        if args.verbose and counter % time_log_freq == 0:
                            print("[cam{}] Neural network forward pass took {} seconds.".format(cam_id, time.time() - start))
                            start = time.time()

                        #Tracking fails, so need to determine the identity of the bb
                        (identity, BestMatchSimilarityScore) = self.Determine_identity_by_rep(cam_id, rep)

                        if identity not in identities:
                            identities.append(identity)

                        if identity in self.tracked_list_of_ppl[cam_id]:
                            self.tracked_list_of_ppl[cam_id].remove(identity)

                        if identity == -1:
                            print("[cam{}] An unknown is newly detected!".format(cam_id))
                            isNewlyDetect = True
                        
                        identitylist.append(identity)
                        replist.append(rep)
                        scorelist.append(BestMatchSimilarityScore)

                        #Determine the name to display
                        if identity == -1:
                            name = "Unknown"
                        else:
                            name = self.people[self.identity_ofppl.index(identity)] #+ ", " + str(round_to_1(prob[0]*100)) + "%"
                        print ("[cam{}]  ==> [{}] is detected! its identity is {}".format(cam_id, name, identity))

                        NameAndBB.append((name, bb.left()*inv_scale, bb.top()*inv_scale, (bb.right()-bb.left())*inv_scale,
                        (bb.bottom()-bb.top())*inv_scale, identity, isNewlyDetect, BestMatchSimilarityScore, BB_dataurl))
                        #print("[cam{}] A newly detected face! update the result table".format(cam_id))


            # end bbs for loop


            #if self.stop_all_tracking:
                #self.stop_all_tracking = False

            # save this frame bbs and identity rep info for the next frame for tracking
            rescueMaxtime = 5
            if not self.training and len(bbs) > 0 and not StopUpdatePrev:

                #if self.unknowntraining:

                #Make tracking more easy for unknowntraining mode, even one frame miss the target bb
                for idx, val in enumerate(self.prev_identity[cam_id]):

                    if not val in identitylist and val > -1: # and val in self.unknowntraining_list:

                        if not val in self.rescuetimes[cam_id] or val in self.rescuetimes[cam_id] and self.rescuetimes[cam_id][val] < rescueMaxtime:
                            bbs.append(self.prev_bbs[cam_id][idx])
                            identitylist.append(self.prev_identity[cam_id][idx])
                            replist.append(self.prev_rep[cam_id][idx])
                            scorelist.append(self.prev_score[cam_id][idx])

                            if not val in self.rescuetimes[cam_id]:
                                self.rescuetimes[cam_id][val]=1
                            else:
                                self.rescuetimes[cam_id][val]+=1

                            print("[cam{}] rescue identity {} for the {} time".format(cam_id, val, self.rescuetimes[cam_id][val]))

                        else:

                            print("[cam{}] cannot rescue identity {} anymore".format(cam_id, val))
                    
                    elif val in identitylist and val > -1:
                        self.rescuetimes[cam_id][val]=0

                self.prev_bbs[cam_id] = bbs
                self.prev_identity[cam_id] = identitylist
                self.prev_rep[cam_id] = replist
                self.prev_score[cam_id] = scorelist
                #has_prev = True

            # finally, send identities and annotated msg to client
            # if not self.training:
            #     start = time.time()

                #dont send identities msg too often, since no this need
                # if self.counter %10 == 0:
                #     msg = {
                #         "type": "IDENTITIES",
                #         "identities": identities
                #     }
                #     self.sendMessage(json.dumps(msg))
                    # if args.verbose:
                    #     print("Send back the IDENTITIES took {} seconds.".format(
                    #         time.time() - start))
                    #     start = time.time()

            self.lastframetime[cam_id] = time.time() - framestart

            if args.verbose and counter % time_log_freq == 0:
                print("[cam{}] One frame took {} seconds. fps= {}".format(cam_id, self.lastframetime[cam_id], 1/self.lastframetime[cam_id]))       


            if self.processed_content[cam_id].full():
                self.processed_content[cam_id].get()
            self.processed_content[cam_id].put(NameAndBB)



            # if not self.encodedimage[cam_id] is None:
            #     content = self.encodedimage[cam_id]
            #     print("encoded img size in bytes={}".format(utf8len(content)))

            #     msg = {
            #         "type": "ANNOTATED",
            #         "content": NameAndBB,
            #         "fps": round_to_1(1/self.lastframetime[cam_id]),
            #         "id": cam_id,
            #         "image": content,
            #         "dateAndtime": str(datetime.datetime.now())
            #     }        
            #     self.processed_msg[cam_id] = msg
                #self.sendMessage(json.dumps(msg))

            #print("==================================================================")      
        

def main(reactor):
    log.startLogging(sys.stdout)
    factory = WebSocketServerFactory()
    factory.protocol = OpenFaceServerProtocol
    ctx_factory = DefaultOpenSSLContextFactory(tls_key, tls_crt)
    reactor.listenSSL(args.port, factory, ctx_factory)
    return defer.Deferred()

if __name__ == '__main__':
    task.react(main)
