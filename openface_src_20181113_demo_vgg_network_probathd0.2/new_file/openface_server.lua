#!/usr/bin/env th
--
-- Copyright 2015-2016 Carnegie Mellon University
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.


require 'torch'
require 'nn'
require 'dpnn'
require 'image'

io.stdout:setvbuf 'no'
torch.setdefaulttensortype('torch.FloatTensor')

-- OpenMP-acceleration causes slower performance. Related issues:
-- https://groups.google.com/forum/#!topic/cmu-openface/vqkkDlbfWZw
-- https://github.com/torch/torch7/issues/691
-- https://github.com/torch/image/issues/7
torch.setnumthreads(1)

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Face recognition server.')
cmd:text()
cmd:text('Options:')

cmd:option('-model', '../models/openface/VGG_FACE.t7', 'Path to model.')
cmd:option('-imgDim', 96, 'Image dimension. nn1=224, nn4=96')
cmd:option('-cuda', false)
cmd:text()

opt = cmd:parse(arg or {})
-- print(opt)

net = torch.load(opt.model)
--net = torch.load('./models/openface/VGG_FACE.t7')
net:evaluate()
-- print(net)

--local mean = {129.1863,104.7624,93.5940}
local mean = {93.5940,104.7624,129.1863}

local imgCuda = nil
if opt.cuda then
   require 'cutorch'
   require 'cunn'
   net = net:cuda()
   imgCuda = torch.CudaTensor(1, 3, 224, 224)
   --print("Using torch cuda: torch.CudaTensor")
end

local img = torch.Tensor(1, 3, 224, 224)
local im_bgr = torch.Tensor(1, 3, 224, 224)
while true do
   -- Read a path to an image on stdin and output the representation
   -- as a CSV.
   local imgPath = io.read("*line")
   if imgPath and imgPath:len() ~= 0 then
      --local im = image.load('../models/openface/smile.jpg', 3, 'float')
      local im = image.load(imgPath, 3, 'float')
      im = image.scale(im, 224, 224)
      im = im*255      
      for j=1,3 do im[j]:add(-mean[j]) end

      --local im_bgr[1] = img[1]:index(1,torch.LongTensor{3,2,1})

      local rep
      if opt.cuda then
         imgCuda[1]:copy(im)
         --rep = net:forward(imgCuda):float()
         rep = net:forward(imgCuda):float()
         --print("Using torch cuda: net:forward(imgCuda):float()")
      else
         img[1]:copy(im)
         rep = net:forward(img)
      end

      --print(rep)

      local sz = rep:size(2)
      for i = 1,sz do
         io.write(rep[1][i])
         if i < sz then
            io.write(',')
         end
      end
      io.write('\n')
      io.stdout:flush()
   end
end
