# -*- coding: utf-8 -*-
"""
Created on Wed May 15 15:51:42 2019

@author: vinhngx
"""


#!/usr/bin/env python
from __future__ import print_function
import argparse
import numpy as np
import time
import pdb

from multiprocessing import Process, cpu_count

import cv2
import os
import tensorflow as tf
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)

from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

parser = argparse.ArgumentParser(description='incetion grpc client flags.')
parser.add_argument('--host', default='localhost', help='inception serving host')
parser.add_argument('--port', default='8500', help='inception serving port')
parser.add_argument('--image', default='/code/data/img.png', help='path to JPEG image file')
FLAGS = parser.parse_args()

def deserialize_image_record(record):
    feature_map = {
        'image/encoded':          tf.FixedLenFeature([ ], tf.string, ''),
        'image/class/label':      tf.FixedLenFeature([1], tf.int64,  -1),
        'image/class/text':       tf.FixedLenFeature([ ], tf.string, ''),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32)
    }
    with tf.name_scope('deserialize_image_record'):
        obj = tf.parse_single_example(record, feature_map)
        imgdata = obj['image/encoded']
        label   = tf.cast(obj['image/class/label'], tf.int32)
        bbox    = tf.stack([obj['image/object/bbox/%s'%x].values
                            for x in ['ymin', 'xmin', 'ymax', 'xmax']])
        bbox = tf.transpose(tf.expand_dims(bbox, 0), [0,2,1])
        text    = obj['image/class/text']
        return imgdata, label, bbox, text

VALIDATION_DATA_DIR = "/data"
BATCH_SIZE = 8

def get_files(data_dir, filename_pattern):
    if data_dir == None:
        return []
    files = tf.gfile.Glob(os.path.join(data_dir, filename_pattern))
    if files == []:
        raise ValueError('Can not find any files in {} with '
                         'pattern "{}"'.format(data_dir, filename_pattern))
    return files

calibration_files = get_files(VALIDATION_DATA_DIR, 'validation*')

print('There are %d calibration files. \n%s\n%s\n...'%(len(calibration_files), calibration_files[0], calibration_files[-1]))
import vgg_preprocessing
def preprocess(record):
    # Parse TFRecord
    imgdata, label, bbox, text = deserialize_image_record(record)
    label -= 1 # Change to 0-based (don't use background class)
    try:    image = tf.image.decode_jpeg(imgdata, channels=3, fancy_upscaling=False, dct_method='INTEGER_FAST')
    except: image = tf.image.decode_png(imgdata, channels=3)

    image = vgg_preprocessing.preprocess_image(image, 224, 224, is_training=False)
    return image, label

dataset = tf.data.TFRecordDataset(calibration_files)    
dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=preprocess, batch_size=BATCH_SIZE, num_parallel_calls=8))


def main():  
  # create prediction service client stubpython
  channel = implementations.insecure_channel(FLAGS.host, int(FLAGS.port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  
  # create request
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'resnet'
  request.model_spec.signature_name = 'serving_default'
  
  start_time = time.time()
  with tf.Session(graph=tf.Graph()) as sess:
        # prepare dataset iterator
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        
        num_hits = 0
        num_predict = 0
        try:
            while True:        
                image_data = sess.run(next_element)    
                img = image_data[0]
                label = image_data[1].squeeze()
                
                # convert to tensor proto and make request
                # shape is in NHWC (num_samples x height x width x channels) format
                tensor = tf.contrib.util.make_tensor_proto(img, shape=list(img.shape))
                request.inputs['input'].CopyFrom(tensor)
                resp = stub.Predict(request, 30.0) #timeout
                #print("Response", resp)                

                prediction = tf.make_ndarray(resp.outputs['classes'])         
                num_hits += np.sum(prediction == label)
                num_predict += len(prediction)
        except tf.errors.OutOfRangeError as e:
            pass
    
        print('Accuracy: %.2f%%'%(100*num_hits/num_predict))
        print('Inference speed: %.2f samples/s'%(num_predict/(time.time()-start_time)))

def run_benchmark(filelist, id, perf_list):
    dataset = tf.data.TFRecordDataset(filelist)    
    dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=preprocess, batch_size=BATCH_SIZE, num_parallel_calls=8))

    # create prediction service client stubpython
    channel = implementations.insecure_channel(FLAGS.host, int(FLAGS.port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  
    # create request
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'resnet'
    request.model_spec.signature_name = 'serving_default'
  
    with tf.Session(graph=tf.Graph()) as sess:
        # prepare dataset iterator
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        
        num_hits = 0
        num_predict = 0
        try:
            while True:        
                image_data = sess.run(next_element)    
                img = image_data[0]
                label = image_data[1].squeeze()
                
                # convert to tensor proto and make request
                # shape is in NHWC (num_samples x height x width x channels) format
                tensor = tf.contrib.util.make_tensor_proto(img, shape=list(img.shape))
                request.inputs['input'].CopyFrom(tensor)
                resp = stub.Predict(request, 30.0) #timeout
                #print("Response", resp)                

                prediction = tf.make_ndarray(resp.outputs['classes'])         
                num_hits += np.sum(prediction == label)
                num_predict += len(prediction)
        except tf.errors.OutOfRangeError as e:
            pass
    print('Thread %d of %d done' %(id, len(perf_list)) )
    perf_list[id] = (num_hits, num_predict)
    print("Thread %d performance: "%id, perf_list)

from multiprocessing.managers import BaseManager, DictProxy         
def main_parallel():  
        
    NUM_JOBS = 8
    print ('Benchmarking with %d threads...'%NUM_JOBS)    
    
    total = len(calibration_files)
    chunk_size = total // NUM_JOBS + 1

    BaseManager.register('dict', dict, DictProxy)
    manager = BaseManager()
    manager.start()
    perf_list = manager.dict()  
    
    processes = []
    start_time = time.time()
    
    id = 0
    for i in range(0, total, chunk_size):
        print('Thread %d of %d start' %(id, len(perf_list)) )
        proc = Process(
            target=run_benchmark,
            args=[
                calibration_files[i:i+chunk_size],
                id,
                perf_list                
            ]
        )
        id += 1
        processes.append(proc)
    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join()

    print("Thread performance: ", perf_list)
    num_hits = 0
    num_predict = 0
    for key, entry in perf_list.items():
        num_hits += entry[0]
        num_predict += entry[1]
    
    print('Total samples: %d'%num_predict)
    print('Accuracy: %.2f%%'%(100*num_hits/num_predict))
    print('Inference speed: %.2f samples/s'%(num_predict/(time.time()-start_time)))       

if __name__ == '__main__':
    #main()
    main_parallel()
