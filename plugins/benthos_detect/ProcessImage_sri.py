#ckwg +28
# Copyright 2015 by Kitware, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither name of Kitware, Inc. nor the names of any contributors may be used
#    to endorse or promote products derived from this software without specific
#    prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from sprokit.pipeline import process
from kwiver.kwiver_process import KwiverProcess
from vital.types import Image
from vital.types import ImageContainer

import numpy as np
import sys
import os
import argparse
import cv2
import caffe
import ImageIO

from selective_search import *
from create_training import *
from FlowSeg import *

class Inference:
    def __init__(self,modelFile,weightsFile,meanFile,imgW,imgH,gpu=False,mode='net',ks=500,opflow_thr=10.0,flowEnable=True):
        """mode=net,sel,flow"""
        self.width = imgW
        self.height = imgH
        self.mode = mode
        self.ks = ks
        self.flowEnable = flowEnable

        if gpu:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        # caffe init
        self.net = caffe.Classifier(modelFile, weightsFile,
                                    mean=np.load(meanFile).mean(1).mean(1),
                                    channel_swap=(2,1,0), image_dims =(256,256),
                                    raw_scale=255)
        # flow segmentation init
        self.fs = FlowSeg()
        self.fs.init(int(self.width),int(self.height), # cols,rows
                     1, #finlev
                     3, #toplev
                     True, #laplacian
                     1, #lev_iterations
                     True) #flow_quantize
        #double opflow_thr, int min_blob_size, int max_blob_size)
        self.fs.params(opflow_thr, 0, int(self.width*self.height/4))

        self.ColorTbl = (
            (  0,  0,  0),# 0: black
            (  0,  0,128),# 1: maroon
            (  0,255,  0),# 2: green
            (  0,128,128),# 3: olive
            (128,  0,  0),# 4: navy
            (128,  0,128),# 5: purple
            (128,128,  0),# 6: teal
            (192,192,192),# 7: silver
            (128,128,128),# 8: gray
            (  0,  0,255),# 9: red
            (  0,255,191),# 10: lime
            (  0,255,255),# 11: yellow
            (255,  0,  0),# 12: blue
            (255,  0,255),# 13: fuchsia
            (255,255,  0),# 14: aqua
            (255,255,255) # 15: white
            )
        print "Inference init done"


    def execute(self,img,fnum):
        # run selective search
        if( self.mode == 'sel' or self.mode == 'net'):
            selRois = self.selSearch(img)
            #print "ssLen=%d"%(len(selRois))
        if( self.mode == 'sel'):
            return (selRois,[])

        if self.flowEnable:
            # run flow segmentation
            if( self.mode == 'flow'):
                (negRois,posRois) = self.flowSegment(img,fnum,debugShow=True)
            if( self.mode == 'net'):
                (negRois,posRois) = self.flowSegment(img,fnum)
                print "flowLen=%d"%(len(posRois))
            if( self.mode == 'flow'):
                return (posRois,[])

            # put two lists together
            rois = selRois + posRois
        else:
            rois = selRois

        # clean up list
        rois = removeDuplicates(rois,0.95)
        rois = removeInvalid(rois,self.width,self.height)
        rois = removeSize(rois,self.width,self.height,0.5,0.0005)
        #print "listLen=%d"%(len(rois))
        # create list of images
        roiImgs = []
        for roi in rois:
            roiImg = ImageIO.crop(img,roi,border=9)
            roiImg = skimage.img_as_float(roiImg).astype(np.float32)
            roiImgs.append(roiImg)
        # call caffe
        scores = self.net.predict(roiImgs, oversample=False)
        # return scores for each roi
        roisWithScores = zip(selRois,scores)
        return (rois,scores)

    def selSearch(self,img):
        """ return list of (x,y,w,h)"""
        imgPyr = cv2.pyrDown(img)

        # selective search
        regions = selective_search(imgPyr,color_spaces = ['rgb'],
                                   ks = [self.ks],
                                   feature_masks=[features.SimilarityMask(size=True,
                                                                          color=True,
                                                                          texture=True,
                                                                          fill=True)])
        selRois = [(x1,y1,x2-x1,y2-y1) for v,(y1,x1,y2,x2) in regions]
        selRois = [(2*x,2*y,2*w,2*h) for (x,y,w,h) in selRois]
        return selRois

    def flowSegment(self,img,fnum,debugShow=False):
        """return (negRois:x,y,w,h,posRois:x,y,w,h) """
        return self.fs.run(img,fnum,debugShow)

    def paint(self,img,rois,scores):
        if(self.mode == 'sel' or self.mode == 'flow'):
            return self.box_paint(img,rois,scores)
        #return self.nms_paint(img,rois,scores)
        #return self.type_paint(img,rois,scores)
        return self.nms_paint3(img,rois,scores)

    def box_paint(self,img,rois,scores):
        for (x,y,w,h) in rois:
            cv2.rectangle(img,(x,y),(x+w-1,y+h-1),(0,128,0),1)
        return img

    def type_paint(self,img,rois,scores):
        """ paint boxed using different color for each type"""
        scored_boxes = []
        lens = len(rois)
        for i in range(lens):
            (x,y,w,h) =  rois[i]
            maxIdx = numpy.unravel_index(scores[i].argmax(),scores[i].shape)
            if(maxIdx[0] != 0):
                print scores[i]
                print "maxIdx=%d"%(maxIdx[0])
            maxScore = scores[i][maxIdx[0]]
            if maxIdx[0]<len(self.ColorTbl):
                thickness = 1
                if maxIdx[0] > 0:
                    thickness = 3
            cv2.rectangle(img,(x,y),(x+w-1,y+h-1),self.ColorTbl[maxIdx[0]],thickness)
        return img

    def nms_paint(self,img,rois,scores):
        scored_boxes = []
        lens = len(rois)
        for i in range(lens):
            (x,y,w,h) =  rois[i]
            pos_score = scores[i][1]
            if pos_score > -0.5:
                scored_box = [x, y, x+w-1, y+h-1, pos_score]
                scored_boxes.append(scored_box)
            cv2.rectangle(img,(x,y),(x+w-1,y+h-1),(0,128,0),1)

        nms_boxes = self.nms(np.array(scored_boxes), 0.1)
        #print nms_boxes

        lens = len(nms_boxes)
        for i in range(lens):
            score = nms_boxes[i][4]
            x0 = int(nms_boxes[i][0])
            y0 = int(nms_boxes[i][1])
            x1 = int(nms_boxes[i][2])
            y1 = int(nms_boxes[i][3])
            if score > 0:
                cv2.rectangle(img,(x0,y0),(x1,y1),(0,0,255),1)
        return img

    def nms_paint3(self,img,rois,scores):
        """
        TypeName = ['0: negative',
                    '1: black_eelpout',
                    '2: crab',
                    '3: longnose_skate',
                    '4: starfish',
                    '5: northpacific_hakefish',
                    '6: rockfish',
                    '7: sea_anemone',
                    '8: seasnail',
                    '9: seaurchin',
                    '10: sunflowerstar']
        """
        TypeName = ['0: Negative',
                    '1: Actinostolidae',
                    '2: Calliostoma_platinum',
                    '3: Chionoecetes_tanneri',
                    '4: Glyptocephalus_zachirus',
                    '5: Lycodes_diapterus',
                    '6: Mediaster_aequalis',
                    '7: Merluccius_productus',
                    '8: Porifera',
                    '9: Psolus_squamatus',
                    '10: Raja_rhina',
                    '11: Rathbunaster_californicus',
                    '12: Sebastes_2species',
                    '13: Sebastolobus_altivelis',
                    '14: Strongylocentrotus_fragilis',
                    '15: Stylasterias_forreri']
        nmsBoxes = self.nmsEx3(rois,scores,0.1)
        rois = []
        roi_id = 0
        for (x0,y0,x1,y1,t,score) in nmsBoxes:
            if x1 - x0 > 0.01 * self.width and y1 - y0 > 0.01 * self.height:
                thickness = 1
                if t > 0:
                    thickness = 3
                cv2.rectangle(img,(int(x0),int(y0)),(int(x1),int(y1)), self.ColorTbl[int(t)],thickness)
                if t > 0:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img,TypeName[int(t)],(int(x0+3), int(y0+3)), font, 0.5,(255,255,255),1)

                label = dict()
                label['label_id'] = int(t)
                label['label_name'] = TypeName[int(t)]
                roi = dict()
                roi['roi_id'] = roi_id
                roi['roi_x'] = int(x0)
                roi['roi_y'] = int(y0)
                roi['roi_w'] = int(x1) - int(x0) + 1
                roi['roi_h'] = int(y1) - int(y0) + 1
                roi['roi_score'] = score
                roi['roi_label'] = label
                rois.append(roi)
                roi_id += 1
        return img, rois

    def nms_paint2(self,img,rois,scores):
        nmsBoxes = self.nmsEx2(rois,scores,0.1)
        typeLens = len(nmsBoxes)
        for t in range(typeLens):
            for (x0,y0,x1,y1,score) in nmsBoxes[t]:
                thickness = 1
                if t > 0:
                    thickness = 3
                cv2.rectangle(img,(int(x0),int(y0)),(int(x1),int(y1)),self.ColorTbl[t],thickness)
        return img

    def nmsEx3(self,rois,scores,overlap):
        """ run nms on whole list
        rois:[(x,y,w,h),...]
        scores:[(...),...]  each roi has score for each type
        return: [(x,y,w,h,type,score),...]
        """
        lens = len(rois) # lens of rois and scores must be same
        # high score index for each roi
        typeIdx = []
        scored_boxes = []
        for i in range(lens):
            maxIdx = numpy.unravel_index(scores[i].argmax(),scores[i].shape)
            typeIdx.append(maxIdx[0])
            (x,y,w,h) =  rois[i]
            score = scores[i][typeIdx[i]]
            scored_box = [x, y, x+w-1, y+h-1,typeIdx[i],score]
            scored_boxes.append(scored_box)
        return self.nms(np.array(scored_boxes), overlap)

    def nmsEx2(self,rois,scores,overlap):
        """
        run nms on each type
        rois:[(x,y,w,h),...]
        scores:[(...),...]  each roi has score for each type
        return:[[(x,y,x1,y1,score),...],...] scored rois for each type
        """
        lens = len(rois) # lens of rois and scores must be same
        # high score index for each roi
        typeIdx = []
        for i in range(lens):
            maxIdx = numpy.unravel_index(scores[i].argmax(),scores[i].shape)
            typeIdx.append(maxIdx[0])
        # nms for each type
        typeLen = len(scores[0])
        retVal = []
        for t in range(typeLen):
            # build score box for each type
            scored_boxes = []
            for i in range(lens):
                if typeIdx[i] != t:
                    continue
                (x,y,w,h) =  rois[i]
                score = scores[i][typeIdx[i]]
            scored_box = [x, y, x+w-1, y+h-1, score]
            scored_boxes.append(scored_box)
            nms_boxes = self.nms(np.array(scored_boxes), overlap)
            retVal.append(nms_boxes)
        return retVal

    def nms(self,boxes, overlap):
        """
        original code: http://github.com/quantombone/exemplarsvm/internal/esvm_nms.m
        boxes: [(x,y,x1,y1,...,score),...]
        return:[(x,y,x1,y1,...,score),...]
        """
        if boxes.size==0:
            return []

        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
        s = boxes[:,-1]

        area = (x2-x1+1) * (y2-y1+1)

        I = np.argsort(s)

        pick = np.zeros(s.size, dtype=np.int)
        counter = 0

        while I.size > 0:
            last = I.size-1
            i = I[-1]
            pick[counter] = i
            counter += 1

            xx1 = np.maximum(x1[i], x1[I[:-1]])
            yy1 = np.maximum(y1[i], y1[I[:-1]])
            xx2 = np.minimum(x2[i], x2[I[:-1]])
            yy2 = np.minimum(y2[i], y2[I[:-1]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            o = w*h / area[I[:-1]]

            I = np.delete(I, np.concatenate([[last], np.nonzero(o > overlap)[0]]))

        pick = pick[:counter]
        top = boxes[pick,:]
        return top

def fishdetect(cvimg):
    f = 1
    model_def = "/media/david/CorticalProcessor/data/MBARI/active_learning/caffe_model/mbari_type_deploy.prototxt"
    pretrained_model = "/media/david/CorticalProcessor/data/MBARI/active_learning/caffe_model/mbari_type_iter_80000.caffemodel"
    mean_file = "/media/david/CorticalProcessor/data/MBARI/active_learning/caffe_model/ilsvrc_2012_mean.npy"
    gpu=True
    mode='net'
    ks=400
    opflow_thr=10.0
    flowEnable=True
    height, width = cvimg.shape[:2]

    # create inferencing object
    inferencing = Inference(model_def,pretrained_model,mean_file,width,height,gpu, mode,ks,opflow_thr,flowEnable)

    cv2.namedWindow("Source",cv2.cv.CV_WINDOW_AUTOSIZE);

    # read image
    (selRois,scores) = inferencing.execute(cvimg,f)

    # Non Maximum Suppression
    #print selRois
    #print scores
    img = inferencing.paint(cvimg,selRois,scores)

    print f
    cv2.imshow("Source",img)
    cv2.waitKey(1000)
    return img

class ProcessImage_sri(KwiverProcess):
    """
    This process gets in an image as input, does some stuff to it and
    sends the modified version to the output port.
    """
    # ----------------------------------------------
    def __init__(self, conf):
        KwiverProcess.__init__(self, conf)

        self.add_config_trait("output", "output", '.',
        'The path of the file to output to.')

        self.declare_config_using_trait( 'output' )

        self.add_port_trait( 'out_image', 'image', 'Processed image' )

        # set up required flags
        optional = process.PortFlags()
        required = process.PortFlags()
        required.add(self.flag_required)

        #  declare our input port ( port-name,flags)
        self.declare_input_port_using_trait('image', required)

        self.declare_output_port_using_trait('out_image', optional )

    # ----------------------------------------------
    def _configure(self):
        print "[DEBUG] ----- configure"
        path = self.config_value('output')

        self._base_configure()

    # ----------------------------------------------
    def _step(self):
        print "[DEBUG] ----- start step"
        # grab image container from port using traits
        in_img_c = self.grab_input_using_trait('image')

        # Get image from container
        in_img = in_img_c.get_image()

        # convert generic image to PIL image
        pil_image = in_img.get_pil_image()
        print 'image size', pil_image.size
    # convert image to cv::Mat
        cvimg = numpy.array(pil_image)
        cvimg = cvimg[:,:,:].copy()
        cvimg = fishdetect(cvimg)
    # convert cvimg back to pil_image
    #pil_image = PIL.Image.fromarray(cvimg)

        """
        # draw on the image to prove we can do it
        num = 37
        import ImageDraw
        draw = ImageDraw.Draw(pil_image)
        draw.line((0, 0) + pil_image.size, fill=128, width=5)
        draw.line((0, pil_image.size[1], pil_image.size[0], 0), fill=32768, width=5)
        #                 x0   y0   x1       y1
        draw.rectangle( [num, num, num+100, num+100], outline=125 )
        del draw
        """
        # get new image handle
        new_image = Image.from_pil( pil_image )
        new_ic = ImageContainer( new_image )

        # push object to output port
        self.push_to_port_using_trait( 'out_image', new_ic )

        self._base_step()

# ==================================================================
def __sprokit_register__():
    from sprokit.pipeline import process_registry

    module_name = 'python:kwiver.ProcessImage_sri'

    reg = process_registry.ProcessRegistry.self()

    if reg.is_module_loaded(module_name):
        return

    reg.register_process('ProcessImage_sri', 'Process image test', ProcessImage_sri)

    reg.mark_module_as_loaded(module_name)
