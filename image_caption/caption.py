## https://github.com/peteanderson80/Up-Down-Captioner/blob/master/scripts/demo.ipynb

import os
import urllib

import numpy as np

import cv2
import caffe
# If you get "No module named _caffe", either you have not built pycaffe or you have not set the pythonpath.

from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect,_get_blobs
from fast_rcnn.nms_wrapper import nms

#=====================================
MIN_BOXES = 10
MAX_BOXES = 40

caffe.set_mode_gpu()
caffe.set_device(0)

cfg['TEST']['RPN_POST_NMS_TOP_N'] = 40  ##number of region proposals, gpu memory exaushted easly, 7G gpu memory only support 40

model_dir=""
rcnn_weights=model_dir+"/resnet101_faster_rcnn_final.caffemodel"
rcnn_proto=model_dir + "/test.prototxt"

caption_weights=model_dir+"/lstm_scst_iter_1000.caffemodel.h5"
caption_proto=model_dir+"/decoder.prototxt"
#======================================

# Load classes
classes = ['__background__']
with open(model_dir + '/objects_vocab.txt') as f:
    for object in f.readlines():
        classes.append(object.split(',')[0].lower().strip())

# Load attributes
attributes = ['__no_attribute__']
with open(model_dir + '/attributes_vocab.txt') as f:
    for att in f.readlines():
        attributes.append(att.split(',')[0].lower().strip())
##
vocab_file = model_dir+'/train_vocab.txt'
vocab = []
with open(vocab_file) as f:
    for word in f:
        vocab.append(word.strip())
print 'Loaded {:,} words into caption vocab'.format(len(vocab))

#=======================================
# Code for getting features from Faster R-CNN

def get_detections_from_im(net, cv2_im, image_id, conf_thresh=0.2):
    im = cv2_im
    scores, boxes, attr_scores = im_detect(net, im)

    # Keep the original boxes, don't worry about the regresssion bbox outputs
    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    blobs, im_scales = _get_blobs(im, None)

    cls_boxes = rois[:, 1:5] / im_scales[0]
    cls_prob = net.blobs['cls_prob'].data
    attr_prob = net.blobs['attr_prob'].data
    pool5 = net.blobs['pool5_flat'].data

    # Keep only the best detections
    max_conf = np.zeros((rois.shape[0]))
    for cls_ind in range(1, cls_prob.shape[1]):
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = np.array(nms(dets, cfg.TEST.NMS))
        max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])

    keep_boxes = np.where(max_conf >= conf_thresh)[0]
    if len(keep_boxes) < MIN_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
    elif len(keep_boxes) > MAX_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]
    objects = np.argmax(cls_prob[keep_boxes][:, 1:], axis=1)
    attrs = np.argmax(attr_prob[keep_boxes][:, 1:], axis=1)

    return {
        'image_id': image_id,
        'image_h': np.size(im, 0),
        'image_w': np.size(im, 1),
        'num_boxes': len(keep_boxes),
        'boxes': cls_boxes[keep_boxes],
        'features': pool5[keep_boxes],
        'objects': objects,
        'attrs': attrs
    }

#=======================================

def lstm_inputs(dets):
    # Inputs to the caption network
    forward_kwargs = {'image_id': np.zeros((1,3),np.float32)}
    forward_kwargs['image_id'][0,1] = dets['image_h']
    forward_kwargs['image_id'][0,2] = dets['image_w']

    forward_kwargs['num_boxes'] = np.ones((1,1), np.float32)*dets['num_boxes']

    forward_kwargs['boxes'] = np.zeros((1,101,4),np.float32)
    forward_kwargs['boxes'][0,1:dets['num_boxes']+1,:] = dets['boxes']

    forward_kwargs['features'] = np.zeros((1,101,2048),np.float32)
    forward_kwargs['features'][0,0,:] = np.mean(dets['features'], axis=0)
    forward_kwargs['features'][0,1:dets['num_boxes']+1,:] = dets['features']
    return forward_kwargs

#========================================
def translate(vocab, blob):
    caption = "";
    w = 0;
    while True:
        next_word = vocab[int(blob[w])]
        if w == 0:
            next_word = next_word.title()
        if w > 0 and next_word != "." and next_word != ",":
            caption += " ";
        if next_word == "\"" or next_word[0] == '"':
            caption += "\\"; # Escape
        caption += next_word;
        w += 1
        if caption[-1] == '.' or w == len(blob):
            break
    return caption

if __name__=="__main__":
    feature_net = caffe.Net(rcnn_proto, rcnn_weights, caffe.TEST)
    caption_net = caffe.Net(caption_proto, caption_weights, caffe.TEST)

    print("\n\n model loaded successfully..... \n")
    im_file = 'COCO_val2014_000000273052.jpg'  # demo image
    im = cv2.imread(im_file)

    ## bottom-up
    dets = get_detections_from_im(feature_net, im, 0)

    ## top-down caption
    forward_kwargs = lstm_inputs(dets)
    caption_net.forward(**forward_kwargs)

    # Decoding the unrolled caption net and print beam search outputs
    image_ids = caption_net.blobs['image_id'].data
    captions = caption_net.blobs['caption'].data
    scores = caption_net.blobs['log_prob'].data
    batch_size = image_ids.shape[0]

    beam_size = captions.shape[0] / batch_size
    print "Beam size: %d" % beam_size
    for n in range(batch_size):
        for b in range(beam_size):
            cap = translate(vocab, captions[n * beam_size + b])
            score = scores[n * beam_size + b]
            print '[%d] %.2f %s' % (b, score, cap)