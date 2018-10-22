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

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import openface
import dlib
import time
import cv2gpu
import threading
start = time.time()

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


align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,
                              cuda=args.cuda)
cpunet = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim, cuda=False)

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
        #self.unknowntraining = False
        self.unknowntraininglatestindex = 0
        self.unknowntraining_list = []
        self.people = []
        self.identity_ofppl = [] # w.r.t self.people, i.e. self.identity_ofppl[0] is the identity of self.people[0]
        self.svm = None

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
        self.lastframetime=0.025



        #video stream
        self.rgbFrame = []
        # self.rgbFrame_mutex = threading.Lock()
        # self.zzzjpg_mutex = threading.Lock()
        # self.video_stream_path = 'rtsp://172.18.9.99/axis-media/media.amp'
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
        thread = threading.Thread(target=self.getLearnImgRep, args=())
        thread.daemon = True                            # Daemonize thread
        thread.start()                                  # Start the execution

    def getLearnImgRep(self):
        while True:
            if self.trainingnumber > 0:
                for zz in range(0, self.trainingnumber):
                    ctx = self.trainingContent[zz]
                    if ctx is None:
                        identity = self.trainingIdentity[zz]
                        phash = self.trainingPhashs[zz]
                        alignedFace = self.trainingAlignFaces[zz]
                        self.trainingRep[zz] = cpunet.forward(alignedFace)
                        self.images[phash] = Face(self.trainingRep[zz], identity)
                        # TODO: Transferring as a string is suboptimal.
                        self.trainingContent[zz] = [str(x) for x in cv2.resize(alignedFace, (0,0),
                            fx=0.5, fy=0.5).flatten()]   

                        msg = {
                            "type": "NEW_IMAGE",
                            "hash": phash,
                            "content": self.trainingContent[zz],
                            "identity": identity,
                            "representation": self.trainingRep[zz].tolist()
                        }
                        self.sendMessage(json.dumps(msg))
                        print("One face (index: {}) of identity {} is processed and sent to client.".format(zz, identity))

            time.sleep(0.001)

    def Determine_identity_by_rep(self, rep):
        BestMatchSimilarityScore = 0

        if len(self.people) == 0:
            identity = -1  #unknown

        elif len(self.people) >= 1:

            if self.svm:
                #when added person >1, the identity is the index return by svm
                identity = self.svm.predict(rep)[0]

                #also need to double confirm with the probability of each class
                prob = self.svm.predict_proba(rep)[0]
                print("prob of each class: {}".format(' '.join(str(e) for e in prob)))

                # prob_thd = 0.6
                # if max(prob) < prob_thd:
                #     identity = -1
                #     print("Top prob < {}, not so sure is one of the trained person, treat as unknown".format(prob_thd))
            
            else:
                identity = self.identity_ofppl[0]
            
            #double confirm with class mean and std to confirm
            if not self.mean:
                self.getData()
            if identity >= 0:
                if self.mean and self.std:
                    diff = np.absolute(self.mean[identity]-rep)
                    dist_to_center = np.linalg.norm(diff)
                    print("This bb rep distance to class centre is {}".format(dist_to_center))
                    #print("This class std is : {}".format(self.std[identity]))

                    #Best match: score 1, poorest match: score 0
                    BestMatchSimilarityScore = round_to_1(math.exp(-1*dist_to_center))
                    print("BestMatchSimilarityScore is {}".format(BestMatchSimilarityScore))

                    #check if diff > 6*std in any of the dimension
                    largest_ratio=0
                    for idx, val in enumerate(self.std[identity]):
                        print("idx: {}, Diff: {}, std: {}, ratio: {}".format(idx, diff[idx], val, diff[idx]/val))
                        ratio = diff[idx]/val
                        if ratio > largest_ratio:
                            largest_ratio = ratio

                        ratio_thd = 6.5
                        if ratio > ratio_thd:
                            identity = -1
                            print("Diff > {}*Std, not so sure is one of the trained person, treat as unknown".format(ratio_thd))
                            break

                    print("Largest ratio so far is {}".format(largest_ratio))

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


    def matching(self, bb, bbs_ref, identitylist, bbs_ref_isused):
        min_dist = 9999
        index = -1
        if len(bbs_ref)==0:
            return (index, min_dist)

        bbc = bb_centroid(bb)
        for idx, bbb in enumerate(bbs_ref):
            bbbc = bb_centroid(bbb)
            dist = np.linalg.norm(bbc-bbbc)

            #dist > 0.0 to avoid tracking a wrong detected background
            #In normal use case all the people should be walking, so impossible to have dist == 0.0
            if bbs_ref_isused[idx] == 0 and dist < min_dist and identitylist[idx] != -99:
                min_dist = dist
                index = idx
        
        # calculate the distance threshold, it should be linear to the size of the bb
        ratio = 0.6
        if index > -1 and identitylist[index] in self.unknowntraining_list : # for unknown still under training, make it easier to track, avoid tracking lost
            ratio = 2.0

        dist_thd = max((bb.right()-bb.left())*ratio, 30.0)
        print("dist_thd: {}".format(dist_thd))

        # calculate the bb size ratio for a double check
        if index >= 0:
            bb_ratio = (1.0*((bb.right()-bb.left()))/((bbs_ref[index].right()-bbs_ref[index].left())))
            print("bb_ratio: {}".format(bb_ratio))

            if bb_ratio < 0.6 or bb_ratio > 1.0/0.6:
                index = -1
                print("bb_ratio exceed limit, fail!")

        if min_dist > dist_thd: # or min_dist == 0.0:
            index = -1

        # if IsUnknownTraining:
        #     if (bb.right()-bb.left()) < 50:
        #         index = -1
        #         print("bb width < 50, unknowntraining only accept big enough heads to be learnt")

        if index > -1 and min_dist != 9999:
            bbs_ref_isused[index]=1

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
        print("Frame {},  Received {} message of length {}.".format(
            self.counter, msg['type'], len(raw)))

        if msg['type'] == "ALL_STATE":
            print("loading data!")
            print("Altogether {} images and {} people.".format(len(msg['images']), len(msg['people'])))
            self.loadState(msg['images'], msg['training'], msg['people'], msg['people_ide'])
            print("==================================================================")

        elif msg['type'] == "NULL":
            self.sendMessage('{"type": "NULL"}')

        elif msg['type'] == "FRAME":
            self.processFrame(msg['dataURL'], msg['identity'], msg['id'])
            #self.sendMessage('{"type": "PROCESSED"}')

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
                #self.unknowntraining = False
                self.unknowntraining_list = []

            print("self.training = {}".format(self.training))
            print("==================================================================")

        elif msg['type'] == "ADD_PERSON":
            newpersonname = msg['val'].encode('ascii', 'ignore')
            self.people.append(newpersonname)
            new_identity = msg['ide']
            self.identity_ofppl.append(new_identity)
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

            self.people.remove(idx2_name)
            self.identity_ofppl.remove(idx2)

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

            self.people.remove(personname)
            self.identity_ofppl.remove(ide)

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
            self.images[h] = Face(np.array(jsImage['representation']),
                                  jsImage['identity'])

        for jsPerson in jsPeople:
            self.people.append(jsPerson.encode('ascii', 'ignore'))

        for jside in jsIdentity:
            self.identity_ofppl.append(jside)

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
        for img in self.images.values():
            X.append(img.rep)
            y.append(img.identity)
            
            if img.identity in self.rep_of_each_class:
                self.rep_of_each_class[img.identity].append(img.rep)
            else:
                self.rep_of_each_class[img.identity] = [img.rep]

            print("class {} now has {} representation collected".format(img.identity, len(self.rep_of_each_class[img.identity])))

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
            self.mean[key] = np.mean(value, axis=0)
            self.std[key] = np.std(value, axis=0)
            print("Identity {} has calulcated its class mean and std".format(key))

        print("getData took {} seconds.".format(time.time() - start))
        return (X, y)

    # def sendTSNE(self, people):
    #     d = self.getData()
    #     if d is None:
    #         return
    #     else:
    #         (X, y) = d

    #     X_pca = PCA(n_components=50).fit_transform(X, X)
    #     tsne = TSNE(n_components=2, init='random', random_state=0)
    #     X_r = tsne.fit_transform(X_pca)

    #     yVals = list(np.unique(y))
    #     colors = cm.rainbow(np.linspace(0, 1, len(yVals)))

    #     # print(yVals)

    #     plt.figure()
    #     for c, i in zip(colors, yVals):
    #         name = "Unknown" if i == -1 else people[i]
    #         plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=name)
    #         plt.legend()

    #     imgdata = StringIO.StringIO()
    #     plt.savefig(imgdata, format='png')
    #     imgdata.seek(0)

    #     content = 'data:image/png;base64,' + \
    #               urllib.quote(base64.b64encode(imgdata.buf))
    #     msg = {
    #         "type": "TSNE_DATA",
    #         "content": content
    #     }
    #     self.sendMessage(json.dumps(msg))

    # def stop_all_tracking(self):
    #     self.prev_bbs = []
    #     self.prev_identity = []
    #     self.prev_rep = []
    #     self.prev_score = []
    #     self.tracked_list_of_ppl = []

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
            if self.trainingnumber > 0:
                for zz in range(0, self.trainingnumber):
                    identity = self.trainingIdentity[zz]
                    ctx = self.trainingContent[zz]
                    if ctx is None:
                        print("The {} image of identity {} is not yet computed Rep, please wait...".format(zz, identity))
                        alldataloaded = False
                        time.sleep(0.01)
                        break
                        
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
            (X, y) = d  # d = array(img.rep, img.identity)
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
            self.svm = GridSearchCV(SVC(C=1, probability=True), param_grid, cv=5).fit(X, y)

            print("trainFace (not counting getData) took {} seconds.".format(time.time() - start))

    def processFrame(self, dataURL, identity, cam_id):
        framestart = time.time()
        start = time.time()
        NameAndBB = []
        print("Processing frame from cam {}".format(cam_id))

        #skip frame to achieve more smoothness
        #if self.counter % 2 ==0 or self.training:
        if True:

            head = "data:image/jpeg;base64,"
            assert(dataURL.startswith(head))
            imgdata = base64.b64decode(dataURL[len(head):])

            if args.verbose:
                print("Decode the image took {} seconds.".format(time.time() - start))
                start = time.time()

            imgF = StringIO.StringIO()
            imgF.write(imgdata)
            imgF.seek(0)
            pil_image = Image.open(imgF)
            # img.save("zzz.jpg", "JPEG")
            # if args.verbose:
            #     print("img.save the image took {} seconds.".format(time.time() - start))
            #     start = time.time()

            self.rgbFrame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            ##buf = np.fliplr(np.asarray(img))
            # buf = np.asarray(img)
            if args.verbose:
                print("pil image to opencv mat took {} seconds.".format(time.time() - start))
                start = time.time()  

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

            cv2.imwrite('zzz.jpg', self.rgbFrame)
            #self.rgbFrame = cv2.imread('zzz.jpg')
            if args.verbose:
                print("imwrite the image took {} seconds.".format(time.time() - start))
                start = time.time()

            # if self.training and identity == -1:
            #     if not self.unknowntraining:
            #         print("Now is training unknown people...")
            #     self.unknowntraining = True


            #this_dir = os.path.dirname(os.path.realpath(__file__))

            #if self.zzzjpg_mutex.acquire():
            faces = cv2gpu.find_faces('/root/openface/zzz.jpg')
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

            if args.verbose:
                print("Face detection took {} seconds.".format(time.time() - start))
                start = time.time()

            # if len(bbs) > 1:
            #     print("Number of detected faces: ", len(bbs))

            identitylist = []
            replist = []
            scorelist = []
            BestMatchSimilarityScore = []
            prev_bbs_isused = [0] * (len(self.prev_bbs[cam_id])+1)
            #dist_thd = 20.0 # unit is pixel, the larger the more easy to got matching, save time, but more easy to have mismatch
            prob = [1.0]
            nn_processed_bbs = 0
            StopUpdatePrev = False
            name = None

            if len(bbs) == 0:
                print("No bbs is found in this frame!!")

            #Main loop for each detected bounding boxes
            for idxx, bb in enumerate(bbs):

                isNewlyDetect = False
                BestMatchSimilarityScore = 0
                bb_width = bb.right()-bb.left()
                print("BB:{} ->({}, {}, {}, {})".format(idxx+1, bb.left()*scale_factor, bb.top()*scale_factor,
                 bb_width*scale_factor, bb_width*scale_factor))
                
                if self.training:
                    if bb_width < 50:
                        print("bb width < 50, active training only accept big enough heads to be learnt")
                        identitylist.append(-99) # -99 means its not going to be tracked
                        replist.append(None)
                        scorelist.append(0.0)
                        continue

                # use dlib to confirm again the bb is valid
                start = time.time()
                dlib_margin = 10
                rgbFrame_roi = self.rgbFrame[max(bb.top()-dlib_margin, 0):min(720-1, bb.bottom()+dlib_margin),max(bb.left()-dlib_margin, 0):min(1280-1, bb.right()+dlib_margin)]
                rgbFrame_roi_resize = None

                #resize the ROI otherwise dlib runs very slow
                if bb_width > 100:
                    resize_ratio = 0.6
                    if bb_width > 200:
                        resize_ratio = 0.36
                    elif bb_width > 400:
                        resize_ratio = 0.2
                    rgbFrame_roi_resize = cv2.resize(rgbFrame_roi, (0,0), fx=resize_ratio, fy=resize_ratio)
                else:
                    rgbFrame_roi_resize = rgbFrame_roi.copy()

                dlib_bb = align.getLargestFaceBoundingBox(rgbFrame_roi_resize)
                if args.verbose:
                    print("dlib confirmation took {} seconds.".format(time.time() - start))
                    start = time.time()

                if not dlib_bb:
                    print("dlib confirmation fail!")
                    identitylist.append(-99) # -99 means its not going to be tracked
                    replist.append(None)
                    scorelist.append(0.0)
                    continue

                # landmarks = align.findLandmarks(rgbFrame, bb)
                # if args.verbose:
                #     print("Find landmarks~ took {} seconds.".format(time.time() - start))
                #     start = time.time()

                # Do tracking first, see if we can match with prev bbs, if yes, then 
                # for unknown, dont track more than 4 frames, as it may be wrongly classify into unknown, so give chance to correct
                matchingresult = self.matching(bb, self.prev_bbs[cam_id], self.prev_identity[cam_id], prev_bbs_isused)
                if (not self.training and len(self.prev_bbs[cam_id])>0 and 
                    len(self.prev_identity[cam_id])>0 and len(self.prev_rep[cam_id])>0 and matchingresult[0] >= 0 and 
                    (self.prev_identity[cam_id][matchingresult[0]] >= 0 or (self.prev_identity[cam_id][matchingresult[0]] == -1 and self.counter % 5 !=0))):

                    identity = self.prev_identity[cam_id][matchingresult[0]]
                    print("Tracking successful, matching index is {}, matching identity is {}, matching dist is {}, skip face landmark and nn net forward".format(matchingresult[0], identity, matchingresult[1]))
                    #print("prev_identity: {}".format(' '.join(str(e) for e in self.prev_identity)))
                    #print("prev_rep: {}".format(' '.join(str(e) for e in self.prev_rep)))
                    
                    BestMatchSimilarityScore = self.prev_score[cam_id][matchingresult[0]]
                    rep = self.prev_rep[cam_id][matchingresult[0]]

                    if identity == -1:

                        # double confirm whether it is real unmatch with any person in the database
                        # becoz it is possible that it is a miskake to recongize a real person into unknown
                        alignedFace = align.align(args.imgDim, self.rgbFrame, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE) 
                        phash = str(imagehash.phash(Image.fromarray(alignedFace)))
                        if phash in self.images:
                            print("phash in self.image, skip training this")
                            identitylist.append(-99) # -99 means its not going to be tracked
                            replist.append(None)
                            scorelist.append(0.0)                        
                            continue  
                        rep = net.forward(alignedFace)

                        print("Double check if this rep cannot match with any record in the database...")
                        (ide, sscore) = self.Determine_identity_by_rep(rep)

                        if ide == -1:
                            # add new person automatically since it detects a new bb
                            self.unknowntraininglatestindex += 1

                            identity = 0
                            if self.identity_ofppl:
                                identity = max(self.identity_ofppl)+1
                            newpersonname = "Unknown" + str(self.unknowntraininglatestindex)
                            self.people.append(newpersonname)
                            self.identity_ofppl.append(identity)

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
                        name = self.people[self.identity_ofppl.index(identity)] #+ ", " + str(round_to_1(prob[0]*100)) + "%"
                        print ("[{}] is detected! its identity is {}".format(name, identity))

                        #isNewlyDetect is for the bb that is tracked for the first time
                        if not identity in self.tracked_list_of_ppl[cam_id]:
                            isNewlyDetect = True
                            print("A newly detected face! update the result table")
                            self.tracked_list_of_ppl[cam_id].append(identity)    

                    # if it is the new unknown which is given a new identity but not yet finish training
                    else:
                        #if collect enough photo for the unknown, start to train face
                        Num_of_img_unknown_need_to_train = 40
                        if identity in self.trainingnumber_foreachide and self.trainingnumber_foreachide[identity] >= Num_of_img_unknown_need_to_train:
                            self.trainFace()
                            self.unknowntraining_list.remove(identity)
                            name = "Finished!"
                            print("{} Learning finished!".format(self.people[self.identity_ofppl.index(identity)]))
                            
                        else:

                            #if self.rgbFrame_mutex.acquire():
                            alignedFace = align.align(args.imgDim, self.rgbFrame, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)  
                                #self.rgbFrame_mutex.release()

                            phash = str(imagehash.phash(Image.fromarray(alignedFace)))
                            if phash in self.images:
                                print("phash in self.image, skip training this")
                                identitylist.append(-99) # -99 means its not going to be tracked
                                replist.append(None)
                                scorelist.append(0.0)                        
                                continue                      

                            self.trainingPhashs.append(phash)
                            self.trainingAlignFaces.append(alignedFace)
                            self.trainingIdentity.append(identity)
                            self.trainingRep.append(None)
                            self.trainingContent.append(None)
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
                    bb_width*inv_scale, identity, isNewlyDetect, BestMatchSimilarityScore))
                    #continue
                
                #when tracking fails or if it is in active learning mode
                else:

                    # One frame at most do one nn forward (hopefully the rest is handled by tracking)
                    # Do this to ensure speed can be maintained
                    if nn_processed_bbs >= 1:
                        print("quota for net.forward() is consumed, treat this bb in next frame!")
                        identitylist.append(-99) # -99 means its not going to be tracked
                        replist.append(None)
                        scorelist.append(0.0)
                        continue

                    if not self.training and len(self.prev_bbs[cam_id])>0 and matchingresult[0] < 0:
                        print("Tracking fail, tracking dist is {}, do normal flow".format(matchingresult[1]))

                    #if self.rgbFrame_mutex.acquire():
                    alignedFace = align.align(args.imgDim, self.rgbFrame, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
                        #self.rgbFrame_mutex.release()
                        
                    if args.verbose:
                        print("Find landmarks and alignment took {} seconds.".format(time.time() - start))
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

                        self.trainingPhashs.append(phash)
                        self.trainingAlignFaces.append(alignedFace)
                        self.trainingIdentity.append(identity)
                        self.trainingRep.append(None)
                        self.trainingContent.append(None)
                        self.trainingnumber += 1

                        if identity in self.trainingnumber_foreachide:
                            self.trainingnumber_foreachide[identity] += 1
                        else:
                            self.trainingnumber_foreachide[identity] = 1

                        name = "Learning [" + str(self.trainingnumber_foreachide[identity]) + "Img]"
                        print("{} -> {}, identity {},in active person training mode".format(name, self.people[self.identity_ofppl.index(identity)], identity)) 

                        #print(name)
                        NameAndBB.append((name, bb.left()*inv_scale, bb.top()*inv_scale,
                        bb_width*inv_scale, bb_width*inv_scale, identity, isNewlyDetect, BestMatchSimilarityScore))                               

                    #if not active training and fail for tracking
                    else:
                        rep = net.forward(alignedFace)
                        #isNewlyDetect = True
                        #print("A newly detected face!")

                        nn_processed_bbs += 1
                        if args.verbose:
                            print("Neural network forward pass took {} seconds.".format(
                                time.time() - start))
                            start = time.time()

                        #Tracking fails, so need to determine the identity of the bb
                        (identity, BestMatchSimilarityScore) = self.Determine_identity_by_rep(rep)

                        if identity not in identities:
                            identities.append(identity)

                        if identity in self.tracked_list_of_ppl[cam_id]:
                            self.tracked_list_of_ppl[cam_id].remove(identity)
                        
                        identitylist.append(identity)
                        replist.append(rep)
                        scorelist.append(BestMatchSimilarityScore)

                        #Determine the name to display
                        if identity == -1:
                            name = "Unknown"
                        else:
                            name = self.people[self.identity_ofppl.index(identity)] #+ ", " + str(round_to_1(prob[0]*100)) + "%"
                        print ("[{}] is detected! its identity is {}".format(name, identity))

                        # NameAndBB.append((name, bb.left()*inv_scale, bb.top()*inv_scale, (bb.right()-bb.left())*inv_scale,
                        # (bb.bottom()-bb.top())*inv_scale, identity, isNewlyDetect, BestMatchSimilarityScore))

            # end bbs for loop


            #if self.stop_all_tracking:
                #self.stop_all_tracking = False

            # save this frame bbs and identity rep info for the next frame for tracking
            if not self.training and len(bbs) > 0 and not StopUpdatePrev:

                #Make tracking more easy for unknowntraining mode, even one frame miss the target bb
                for idx, val in enumerate(self.prev_identity[cam_id]):

                    if not val in identitylist and val > -1 and val in self.unknowntraining_list:

                        if not val in self.rescuetimes[cam_id] or val in self.rescuetimes[cam_id] and self.rescuetimes[cam_id][val] < 30:
                            bbs.append(self.prev_bbs[cam_id][idx])
                            identitylist.append(self.prev_identity[cam_id][idx])
                            replist.append(self.prev_rep[cam_id][idx])
                            scorelist.append(self.prev_score[cam_id][idx])

                            if not val in self.rescuetimes:
                                self.rescuetimes[cam_id][val]=1
                            else:
                                self.rescuetimes[cam_id][val]+=1

                            print("in unknowntraining mode, rescue identity {} for the {} time".format(val, self.rescuetimes[cam_id][val]))

                        else:

                            print("in unknowntraining mode, cannot rescue identity {} anymore".format(val))
                    
                    elif val in identitylist and val > -1:
                        self.rescuetimes[cam_id][val]=0

                self.prev_bbs[cam_id] = bbs
                self.prev_identity[cam_id] = identitylist
                self.prev_rep[cam_id] = replist
                self.prev_score[cam_id] = scorelist
                #has_prev = True

            # finally, send identities and annotated msg to client
            if not self.training:
                start = time.time()

                #dont send identities msg too often, since no this need
                if self.counter %10 == 0:
                    msg = {
                        "type": "IDENTITIES",
                        "identities": identities
                    }
                    self.sendMessage(json.dumps(msg))
                    # if args.verbose:
                    #     print("Send back the IDENTITIES took {} seconds.".format(
                    #         time.time() - start))
                    #     start = time.time()

            self.lastframetime = time.time() - framestart

            if args.verbose:
                print("One frame took {} seconds. fps= {}".format(self.lastframetime, 1/self.lastframetime))       

        #else: 
            #print("Skip frame")

        print("==================================================================")   

        msg = {
            "type": "ANNOTATED",
            "content": NameAndBB,
            "fps": round_to_1(1/self.lastframetime),
            "id": cam_id
        }        
        self.sendMessage(json.dumps(msg))

def main(reactor):
    log.startLogging(sys.stdout)
    factory = WebSocketServerFactory()
    factory.protocol = OpenFaceServerProtocol
    ctx_factory = DefaultOpenSSLContextFactory(tls_key, tls_crt)
    reactor.listenSSL(args.port, factory, ctx_factory)
    return defer.Deferred()

if __name__ == '__main__':
    task.react(main)
