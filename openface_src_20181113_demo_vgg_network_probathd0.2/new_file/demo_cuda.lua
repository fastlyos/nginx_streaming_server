--  Copyright (c) 2015, Omkar M. Parkhi
--  All rights reserved.

require 'image'
require 'nn'
require 'cutorch'
require 'cunn'
require 'torch'
require 'dpnn'

start_time = os.time()
mean = {129.1863,104.7624,93.5940}
img = torch.FloatTensor(1, 3, 224, 224)
imgCuda = torch.CudaTensor(1, 3, 224, 224)
net = torch.load('./VGG_FACE.t7')
net:evaluate()
net = net:cuda()
--im = image.load('./ak.png',3,'float')
im = image.load('./smile.jpg',3,'float')
im = image.scale(im, 224, 224)
im = im*255

for i=1,3 do im[i]:add(-mean[i]) end

im_bgr = im:index(1,torch.LongTensor{3,2,1})
img[1]:copy(im_bgr)
imgCuda[1]:copy(im_bgr)

end_time = os.time()
elapsed_time = os.difftime(end_time-start_time)
print(elapsed_time)

start_time2 = os.time()
print("start to forward")
--prob = net(im_bgr)
--rep = net:forward(img)
rep = net:forward(imgCuda):float()
--maxval,maxid = prob:max(1)
--print(maxval)
print("end forward")
end_time2 = os.time()
elapsed_time2 = os.difftime(end_time2-start_time2)
print(elapsed_time2)
--print(prob)
--print(im_bgr)
print(rep)
