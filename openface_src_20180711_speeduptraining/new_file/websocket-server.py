#!/usr/bin/env python2
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, "..", ".."))

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

def round_to_1(x):
    if x == 0:
        return 0
    else:
        return round(x, -int(floor(log10(abs(x)))))

def bb_centroid(bb):
    x = (bb.right()+bb.left())*0.5
    y = (bb.top()+bb.bottom())*0.5
    return np.array((x, y))

def matching(bb, bbs_ref, bbs_ref_isused, dist_thd):
    min_dist = 9999
    index = -1
    if len(bbs_ref)==0:
         return (index, min_dist)

    bbc = bb_centroid(bb)
    for idx, bbb in enumerate(bbs_ref):
        bbbc = bb_centroid(bbb)
        dist = np.linalg.norm(bbc-bbbc)
        if bbs_ref_isused[idx] == 0 and dist < min_dist and dist < dist_thd:
            min_dist = dist
            index = idx
    
    bbs_ref_isused[index]=1
    return (index, min_dist)


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
        self.people = []
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
        #self.has_prev = False

        #To store training alignedFace and process it altogether when training is finished (In order to speed up training)
        self.trainingnumber = 0
        self.trainingIdentity = []
        self.trainingPhashs = []
        self.trainingAlignFaces = []

        self.counter=0

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
            self.loadState(msg['images'], msg['training'], msg['people'])
            print("==================================================================")

        elif msg['type'] == "NULL":
            self.sendMessage('{"type": "NULL"}')

        elif msg['type'] == "FRAME":
            self.processFrame(msg['dataURL'], msg['identity'])
            #self.sendMessage('{"type": "PROCESSED"}')

        elif msg['type'] == "TRAINING":
            self.training = msg['val']
            if not self.training:
                #leave training, so train SVM for the latest image sets
                self.trainFace()
                self.trainingnumber = 0 # reset to zero
                self.trainingIdentity = []
                self.trainingPhashs = []
                self.trainingAlignFaces = []

        elif msg['type'] == "ADD_PERSON":
            self.people.append(msg['val'].encode('ascii', 'ignore'))
            print("A new person is added!")
            print(self.people)
            print("==================================================================")

        elif msg['type'] == "UPDATE_IDENTITY":
            h = msg['hash'].encode('ascii', 'ignore')
            if h in self.images:
                self.images[h].identity = msg['idx']
                print("An image identity is updated to {} !".format(msg['idx']))      
                print("==================================================================")

                if not self.training:
                    self.trainFace()
            else:
                print("Image not found.")

        elif msg['type'] == "REMOVE_IMAGE":
            h = msg['hash'].encode('ascii', 'ignore')
            if h in self.images:
                del self.images[h]
                print("An image is deleted!")      
                print("==================================================================")

                if not self.training:
                    self.trainFace()
            else:
                print("Image not found.")

        elif msg['type'] == 'REQ_TSNE':
            self.sendTSNE(msg['people'])

        else:
            print("Warning: Unknown message type: {}".format(msg['type']))

    def onClose(self, wasClean, code, reason):
        print("WebSocket connection closed: {0}".format(reason))

    def loadState(self, jsImages, training, jsPeople):
        self.training = training

        for jsImage in jsImages:
            h = jsImage['hash'].encode('ascii', 'ignore')
            self.images[h] = Face(np.array(jsImage['representation']),
                                  jsImage['identity'])

        for jsPerson in jsPeople:
            self.people.append(jsPerson.encode('ascii', 'ignore'))

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

        numIdentities = len(set(y + [-1])) - 1
        if numIdentities == 0:
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
            #print("Identity {} has std {}".format(key, self.std[key]))

        print("getData took {} seconds.".format(time.time() - start))
        return (X, y)

    def sendTSNE(self, people):
        d = self.getData()
        if d is None:
            return
        else:
            (X, y) = d

        X_pca = PCA(n_components=50).fit_transform(X, X)
        tsne = TSNE(n_components=2, init='random', random_state=0)
        X_r = tsne.fit_transform(X_pca)

        yVals = list(np.unique(y))
        colors = cm.rainbow(np.linspace(0, 1, len(yVals)))

        # print(yVals)

        plt.figure()
        for c, i in zip(colors, yVals):
            name = "Unknown" if i == -1 else people[i]
            plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=name)
            plt.legend()

        imgdata = StringIO.StringIO()
        plt.savefig(imgdata, format='png')
        imgdata.seek(0)

        content = 'data:image/png;base64,' + \
                  urllib.quote(base64.b64encode(imgdata.buf))
        msg = {
            "type": "TSNE_DATA",
            "content": content
        }
        self.sendMessage(json.dumps(msg))

    def trainFace(self):
        start = time.time()

        # process alignedFaces on the stack
        if self.trainingnumber > 0:
            for zz in range(0, self.trainingnumber):

                phash = self.trainingPhashs[zz]
                identity = self.trainingIdentity[zz]
                alignedFace = self.trainingAlignFaces[zz]
                rep = net.forward(alignedFace)
                self.images[phash] = Face(rep, identity)
                # TODO: Transferring as a string is suboptimal.
                content = [str(x) for x in cv2.resize(alignedFace, (0,0),
                    fx=0.5, fy=0.5).flatten()]
                #content = [str(x) for x in alignedFace.flatten()]
                # if args.verbose:
                #     print("Flatten the alighedFace took {} seconds.".format(
                #         time.time() - start))
                #     start = time.time()

                msg = {
                    "type": "NEW_IMAGE",
                    "hash": phash,
                    "content": content,
                    "identity": identity,
                    "representation": rep.tolist()
                }
                self.sendMessage(json.dumps(msg))

        print("process alignedFaces on the stack took {} seconds.".format(time.time() - start))
        start = time.time()

        print("+ Training Face on {} labeled images.".format(len(self.images)))

        self.rep_of_each_class.clear()
        self.mean.clear()
        self.std.clear()
        d = self.getData()

        if d is None:
            self.svm = None
            return
        else:
            (X, y) = d  # d = array(img.rep, img.identity)
            numIdentities = len(set(y + [-1]))
            if numIdentities <= 1: # no need svm if only one type of sample is present
                print("No need svm since there is only one person added")
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

    def processFrame(self, dataURL, identity):
        framestart = time.time()
        start = time.time()
        head = "data:image/jpeg;base64,"
        assert(dataURL.startswith(head))
        imgdata = base64.b64decode(dataURL[len(head):])
        # if args.verbose:
        #     print("Decode the image took {} seconds.".format(time.time() - start))
        #     start = time.time()

        imgF = StringIO.StringIO()
        imgF.write(imgdata)
        imgF.seek(0)
        img = Image.open(imgF)

        #buf = np.fliplr(np.asarray(img))
        buf = np.asarray(img)
        #rgbFrame_org = np.zeros((720, 1280, 3), dtype=np.uint8)
        #rgbFrame_org = np.zeros((450, 800, 3), dtype=np.uint8)
        #rgbFrame = np.zeros((360, 640, 3), dtype=np.uint8)
        #rgbFrame = np.zeros((216, 384, 3), dtype=np.uint8), frame length 44370, total 55fps, load 4ms, write 3ms, face 8.3ms 
        #rgbFrame = np.zeros((234, 416, 3), dtype=np.uint8), frame length 51450, totoal 50fps, load 4.8ms, write 3.3ms, face 9.5ms
        rgbFrame = np.zeros((252, 448, 3), dtype=np.uint8) # frame length 55282, totoal 48fps, load 5.4ms, write 3.6ms, face 9.6ms
        rgbFrame[:, :, 0] = buf[:, :, 2]
        rgbFrame[:, :, 1] = buf[:, :, 1]
        rgbFrame[:, :, 2] = buf[:, :, 0]

        # if args.verbose:
        #     print("load the image took {} seconds.".format(time.time() - start))
        #     start = time.time()

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
        NameAndBB = []
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

        cv2.imwrite('zzz.jpg',rgbFrame)
        if args.verbose:
            print("load and imwrite the image took {} seconds.".format(time.time() - start))
            start = time.time()

        #this_dir = os.path.dirname(os.path.realpath(__file__))
        faces = cv2gpu.find_faces('/root/openface/zzz.jpg')
        #print(len(faces))
        for (x, y, w, h) in faces:
            #print(x*scale_factor, y*scale_factor, w*scale_factor, h*scale_factor)
            bbs.append(dlib.rectangle(x*scale_factor, y*scale_factor, (x+w)*scale_factor, (y+h)*scale_factor))

        #bbs = [bb] if bb is not None else []

        if args.verbose:
            print("Face detection took {} seconds.".format(time.time() - start))
            start = time.time()

        #if on training, only take the largest boundingbox, since the person going
        #to learn must stand in the front
        if self.training and len(faces) > 1:
            faces = max(faces, key=lambda rect: rect[2] * rect[3])

        # if len(bbs) > 1:
        #     print("Number of detected faces: ", len(bbs))

        identitylist = []
        replist = []
        prev_bbs_isused = [0] * (len(self.prev_bbs)+1)
        dist_thd = 20.0 # unit is pixel, the larger the more easy to got matching, save time, but more easy to have mismatch
        prob = [1.0]
        nn_processed_bbs = 0
        for idxx, bb in enumerate(bbs):
            
            if len(bbs) > 1:
                print("BB:{} ->".format(idxx+1))
            # landmarks = align.findLandmarks(rgbFrame, bb)
            # if args.verbose:
            #     print("Find landmarks~ took {} seconds.".format(time.time() - start))
            #     start = time.time()

            # Do tracking first, see if we can match with prev bbs, if yes, then 
            matchingresult = matching(bb, self.prev_bbs, prev_bbs_isused, dist_thd)
            if not self.training and len(self.prev_bbs)>0 and len(self.prev_identity)>0 and len(self.prev_rep)>0 and matchingresult[0] >= 0:
                print("Tracking successful, matching index is {}, matching dist is {}, skip face landmark and nn net forward".format(matchingresult[0], matchingresult[1]))
                print("prev_identity: {}".format(' '.join(str(e) for e in self.prev_identity)))
                #print("prev_rep: {}".format(' '.join(str(e) for e in self.prev_rep)))
                identity = self.prev_identity[matchingresult[0]]
                rep = self.prev_rep[matchingresult[0]]

                if identity == -1:
                    # if len(self.people) == 1:
                    #     name = self.people[0]
                    # else:
                        name = "Unknown"
                else:
                    prob = [1.0]
                    if self.svm:
                        prob = self.svm.predict_proba(rep)[0]
                    name = self.people[identity]+ ", " + str(round_to_1(prob[0]*100)) + "%"

                print ("[{}] is detected!".format(name))
                identitylist.append(identity)
                replist.append(rep)
                NameAndBB.append((name, bb.left()*inv_scale, bb.top()*inv_scale, (bb.right()-bb.left())*inv_scale, (bb.bottom()-bb.top())*inv_scale))
                continue
            
            else:

                # One frame at most do one nn forward (hopefully the rest is handled by tracking)
                # Do this to ensure speed can be maintained
                if nn_processed_bbs >= 1:
                    print("quota for net.forward() is consumed, treat this bb in next frame!")
                    continue

                if not self.training and len(self.prev_bbs)>0 and matchingresult[0] < 0:
                    print("Tracking fail, do normal flow")
                    
                alignedFace = align.align(args.imgDim, rgbFrame, bb,
                                        #landmarks=landmarks,
                                        landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
                if args.verbose:
                    print("Find landmarks and alignment took {} seconds.".format(time.time() - start))
                    start = time.time()

                if alignedFace is None:
                    continue

                # the hash is used as the key for the map
                phash = str(imagehash.phash(Image.fromarray(alignedFace)))
                # if args.verbose:
                #     print("Image hash took {} seconds.".format(time.time() - start))
                #     start = time.time()

                #Determine identity by 1. getting representation from nn forward, 2. svm of the representation
                if phash in self.images:
                    identity = self.images[phash].identity
                    print("phash in self.image, identity is {}".format(identity))

                else:
                    if self.training:
                        self.trainingnumber += 1
                        self.trainingPhashs.append(phash)
                        self.trainingAlignFaces.append(alignedFace)
                        self.trainingIdentity.append(identity)

                        # self.images[phash] = Face(rep, identity)
                        # # TODO: Transferring as a string is suboptimal.
                        # content = [str(x) for x in cv2.resize(alignedFace, (0,0),
                        #     fx=0.5, fy=0.5).flatten()]
                        # #content = [str(x) for x in alignedFace.flatten()]
                        # # if args.verbose:
                        # #     print("Flatten the alighedFace took {} seconds.".format(
                        # #         time.time() - start))
                        # #     start = time.time()

                        # msg = {
                        #     "type": "NEW_IMAGE",
                        #     "hash": phash,
                        #     "content": content,
                        #     "identity": identity,
                        #     "representation": rep.tolist()
                        # }
                        # self.sendMessage(json.dumps(msg))
                        # if args.verbose:
                        #     print("Send training json took {} seconds.".format(
                        #         time.time() - start))
                        #     start = time.time()
                        
                        #also send the bounding box to indicate the image learnt
                        name = "Learn: OK [" + str(self.trainingnumber) + "]"
                        print(name)
                        NameAndBB.append((name, bb.left()*inv_scale, bb.top()*inv_scale, (bb.right()-bb.left())*inv_scale, (bb.bottom()-bb.top())*inv_scale))


                    else:
                        rep = net.forward(alignedFace)
                        nn_processed_bbs += 1
                        if args.verbose:
                            print("Neural network forward pass took {} seconds.".format(
                                time.time() - start))
                            start = time.time()

                        #Determine the identity of the rep

                        if len(self.people) == 0:
                            identity = -1  #unknown

                        elif len(self.people) >= 1:

                            if len(self.people) == 1:
                                identity = 0

                            elif self.svm:
                                #when added person >1, the identity is the index return by svm
                                identity = self.svm.predict(rep)[0]

                                #also need to double confirm with the probability of each class
                                prob = self.svm.predict_proba(rep)[0]
                                print("prob of each class: {}".format(' '.join(str(e) for e in prob)))

                                if prob[0] < 0.8:
                                    identity = -1
                                    print("Top prob < 0.8, not so sure is one of the trained person, treat as unknown")
                            
                            #double confirm with class mean and std to confirm
                            if not self.mean:
                                self.getData()
                            if identity >= 0:
                                if self.mean and self.std:
                                    diff = np.absolute(self.mean[identity]-rep)
                                    dist_to_center = np.linalg.norm(diff)
                                    print("This bb rep distance to class centre is {}".format(dist_to_center))
                                    #print("This class std is : {}".format(self.std[identity]))

                                    #check if diff > 6*std in any of the dimension
                                    for idx, val in enumerate(self.std[identity]):
                                        print("idx: {}, Diff: {}, std: {}, ratio: {}".format(idx, diff[idx], val, diff[idx]/val))
                                        if diff[idx] > 6*val:
                                            identity = -1
                                            print("Diff > 6*Std, not so sure is one of the trained person, treat as unknown")
                                            break

                                else:
                                    identity = -1

                        else:
                            print("hhh")
                            identity = -1

                        if identity not in identities:
                            identities.append(identity)
                        
                        identitylist.append(identity)
                        replist.append(rep)

            if not self.training:
                start = time.time()

                #Determine the name to display
                if identity == -1:
                    # if len(self.people) == 1:
                    #     name = self.people[0]
                    # else:
                    name = "Unknown"
                else:
                    name = self.people[identity]+ ", " + str(round_to_1(prob[0]*100)) + "%"
                
                print ("[{}] is detected!".format(name))

                NameAndBB.append((name, bb.left()*inv_scale, bb.top()*inv_scale, (bb.right()-bb.left())*inv_scale, (bb.bottom()-bb.top())*inv_scale))

        # end bbs for loop

        # save this frame bbs and identity rep info for the next frame for tracking
        if not self.training and len(bbs) > 0:
            self.prev_bbs = bbs
            self.prev_identity = identitylist
            self.prev_rep = replist
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

        if args.verbose:
            print("One frame took {} seconds. fps= {}".format(
                time.time() - framestart, 1/(time.time() - framestart)))
            print("==================================================================")

        msg = {
            "type": "ANNOTATED",
            "content": NameAndBB,
            "fps": round_to_1(1/(time.time() - framestart))
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
