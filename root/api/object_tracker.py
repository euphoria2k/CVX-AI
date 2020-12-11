import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import csv
import random
import math
import operator
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
#color imports
import os
from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier
import os.path
import sys
import colorsys
#prediction = knn_classifier.main('training.data', 'test.data')

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
#flags.DEFINE_string('images', './data/images/kite.jpg', 'path to input image')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')
flags.DEFINE_boolean('color', False, 'maual dtection of images and colour')
flags.DEFINE_boolean('userimage', False, 'maual dtection of images and colour')


flags.DEFINE_boolean('black', False, 'maual dtection of images and colour')
flags.DEFINE_boolean('red', False, 'maual dtection of images and colour')
flags.DEFINE_boolean('blue', False, 'maual dtection of images and colour')
flags.DEFINE_boolean('yellow', False, 'maual dtection of images and colour')
flags.DEFINE_boolean('orange', False, 'maual dtection of images and colour')
flags.DEFINE_boolean('green', False, 'maual dtection of images and colour')
flags.DEFINE_boolean('white', False, 'maual dtection of images and colour')
flags.DEFINE_boolean('violet', False, 'maual dtection of images and colour')

flags.DEFINE_boolean('shirt', False, 'shirt detections from images')
flags.DEFINE_boolean('trouser', False, 'trouser detections from images')
flags.DEFINE_boolean('jeans', False, 'jeans detections from images')
flags.DEFINE_boolean('skirt', False, 'skirt detections from images')
flags.DEFINE_boolean('dress', False, 'dress detections from images')
flags.DEFINE_boolean('footwear', False, 'footwear detections from images')
flags.DEFINE_boolean('suit', False, 'suit detections from images')
flags.DEFINE_boolean('jacket', False, 'jacket detections from images')
flags.DEFINE_boolean('person', False, 'person detections from images')


flags.DEFINE_boolean('Fuzzy_black', False, 'maual dtection of images and colour')
flags.DEFINE_boolean('Fuzzy_red', False, 'maual dtection of images and colour')
flags.DEFINE_boolean('Fuzzy_blue', False, 'maual dtection of images and colour')
flags.DEFINE_boolean('Fuzzy_yellow', False, 'maual dtection of images and colour')
flags.DEFINE_boolean('Fuzzy_orange', False, 'maual dtection of images and colour')
flags.DEFINE_boolean('Fuzzy_green', False, 'maual dtection of images and colour')
flags.DEFINE_boolean('Fuzzy_white', False, 'maual dtection of images and colour')
flags.DEFINE_boolean('Fuzzy_purple', False, 'maual dtection of images and colour')
flags.DEFINE_boolean('Fuzzy_cyan', False, 'maual dtection of images and colour')
flags.DEFINE_boolean('Fuzzy_brown', False, 'maual dtection of images and colour')
flags.DEFINE_boolean('Fuzzy_pink', False, 'maual dtection of images and colour')


PATH = './training.data'
prediction = knn_classifier.main('training.data', 'test.data')

def load_red(filename, test_feature=[]):
    with open(filename) as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(3):
                dataset[x][y] = int(dataset[x][y])
            test_feature.append(dataset[x])
            red = dataset[0][0]
            #blue = dataset[0][1]
            #green = dataset[0][2]
            #print("red" +" "  + str(red))
            #print("test : " +str(test_feature) )
            return red



def load_green(filename, test_feature=[]):
    with open(filename) as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(3):
                dataset[x][y] = int(dataset[x][y])
            test_feature.append(dataset[x])
            green = dataset[0][1]
            #print("green" +" "+ str(green))
            return green


def load_blue(filename, test_feature=[]):
    with open(filename) as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(3):
                dataset[x][y] = int(dataset[x][y])
            test_feature.append(dataset[x])
            blue = dataset[0][2]
            #print("blue" +" " + str(blue))
            return blue
            
def rgb_to_hsv(r, g, b):
    if r and b and g != None:
        r, g, b = r/255.0, g/255.0, b/255.0
        mx = max(r, g, b)
        mn = min(r, g, b)
        df = mx-mn
        if mx == mn:
            h = 0
        elif mx == r:
            h = (60 * ((g-b)/df) + 360) % 360
        elif mx == g:
            h = (60 * ((b-r)/df) + 120) % 360
        elif mx == b:
            h = (60 * ((r-g)/df) + 240) % 360
        if mx == 0:
            s = 0
        else:
            s = (df/mx)*100
        v = mx*100
        return h, s, v



def main(_argv):

    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    #images = FLAGS.images
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0

    if FLAGS.shirt:
        allowed_classes = ['Shirt']
        #ROI = 
    if FLAGS.trouser:
        allowed_classes = ['Trousers']
    if FLAGS.jeans:
        allowed_classes = ['Jeans']
    if FLAGS.dress:
        allowed_classes = ['Dress']
    if FLAGS.footwear:
        allowed_classes = ['Footwear']
    if FLAGS.jacket:
        allowed_classes = ['Jacket']
    if FLAGS.skirt:
        allowed_classes = ['Skirt']
    if FLAGS.suit:
        allowed_classes = ['Suit']

    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        #allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            
        # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            #cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            #cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            #cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

        # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
        
            
            if FLAGS.color:
                PATH = './training.data'
                #(int(bbox[0])):(int(bbox[2])), (int(bbox[1])):(int(bbox[3]))
                #ROI = frame[(int(bbox[0]) +50) :(int(bbox[2]) - 50), (int(bbox[1])+ 50):(int(bbox[3])-50)]
                #ROI = frame[(int(bbox[1])) +15 :(int(bbox[3])-15),(int(bbox[0])+15):(int(bbox[2])-15)]
                ROI = frame[int((int(bbox[1]) + int(bbox[3]))/2):int((int(bbox[1]) + int(bbox[3]))/2)+1,int((int(bbox[0]) + int(bbox[2]))/2):int((int(bbox[0]) + int(bbox[2]))/2)+1]
                #ROI = frame[(int(bbox[1])):(int(bbox[3])),(int(bbox[0])):(int(bbox[2]))]
                #ROI = frame[int(0.5* (int(bbox[1] - 50)+ int(bbox[3] + 50))),int(0.5*(int(bbox[0] - 50) +int(bbox[2] + 50 )))]
                #print(ROI)

                color_histogram_feature_extraction.color_histogram_of_test_image(ROI)
                prediction = knn_classifier.main('training.data','test.data')
                #prediction = 'red'
                red  = load_red('test.data')
                Red = str(red)
                #Red = str(Red_1)

                print('this is the variable of the red:- ' + str(Red))
                green  = load_green('test.data')
                Green  = str(green)
                #Green = str(Green_1)
                print('this is the variable of the green:- ' + str(Green)) 
                blue  = load_blue('test.data')
                #Blue_1 = int(blue)
                Blue = str(blue)
                print('this is the variable of the blue:- ' + str(Blue)) 
                
                #hsv = rgb_to_hsv(red,green,blue)
                #print("HSV: " + str(hsv)) 
                
                if red and blue and green != None:
                    HLS = colorsys.rgb_to_hls(red,green,blue)
                    HUE = int(HLS[0])
                    Light = int(HLS[1])
                    Saturation = int(HLS[2])


                print("HLS is equal to",  HLS)
                print('HUE: ' ,HUE)
                print('LIGHT: ', Light )
                print('Saturation', Saturation)

                if red and blue and green != None:
                    HSV = rgb_to_hsv(red,green,blue)
                    HUE_1 = int(HSV[0])
                    Saturation_1 = int(HSV[1])
                    Value = int(HSV[2])
                
                print("HSV is equal to",  HSV)
                print('Hue: ' , HUE_1)
                print('saturation: ', Saturation_1)
                print('value', Value)
                

                

                print(str(prediction) +" "+ str(class_name))

            if FLAGS.Fuzzy_black:
                #if str(59.7) <= Red < str(200.9)  and  str(74) <= Blue < str(207) and str(70) <= Green < str(203):
                if 0 <= HUE_1 < 210 and  0 <= Saturation_1 < 41 and 0 <= Value < 86:
                    print("THIS IS THE black COLOR yaaaaaaaaaaaaaaaaaaaa")
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                    cv2.putText(frame, class_name + " " + "BLACK" + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            if FLAGS.Fuzzy_red:
                #if  str(139) <= Red < str(255)  and  str(0) <= Green < str(160) and str(0) <= Blue < str(128):
                if 0 <= HUE_1 < 348 and  47 <= Saturation_1 < 100 and 55 <= Value < 100:
                    print("THIS IS THE red COLOR redddddddddddddddddddddddddddddddddddd")
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                    cv2.putText(frame, class_name + " " + "RED" + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            if FLAGS.Fuzzy_orange:
                #if  str(255) <= Red < str(255)  and  str(69) <= Green < str(165) and str(0) <= Blue < str(80):
                if 9 <= HUE_1 < 39 and  69 <= Saturation_1 < 100 and  Value == 100:
                    print("THIS IS THE ORANGE COLOR orangeeeeeeeeeeeeeeeeeeeeeeee")
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                    cv2.putText(frame, class_name + " " + "ORANGE" + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            if FLAGS.Fuzzy_yellow:
                #if  str(189) <= Red < str(255)  and  str(183) <= Green < str(255) and str(0) <= Blue < str(224):
                if 0 <= HUE_1 < 56 and  12 <= Saturation_1 < 100 and 74 <= Value < 100:
                    print("THIS IS THE YELLOW COLOR")
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                    cv2.putText(frame, class_name + " " + "YELLOW" + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            if FLAGS.Fuzzy_blue:
                #if  str(0) <= Red < str(176)  and  str(0) <= Green < str(244) and str(112) <= Blue < str(255):
                if 187 <= HUE_1 < 240 and  21 <= Saturation_1 < 100 and 44 <= Value < 100:
                    print("THIS IS THE BLUE COLOR")
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                    cv2.putText(frame, class_name + " " + "BLUE" + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            if FLAGS.Fuzzy_white:
                #if  str(240) <= Red < str(255)  and  str(228) <= Green < str(255) and str(215) <= Blue < str(255):
                if 0 <= HUE_1 < 340 and  0 <= Saturation_1 < 14 and 96 <= Value < 100:
                    print("THIS IS THE WHITE COLOR")
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                    cv2.putText(frame, class_name + " " + "WHITE" + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            if FLAGS.Fuzzy_purple:
                #if  str(72) <= Red < str(255)  and  str(0) <= Green < str(230) and str(128) <= Blue < str(255):
                if 0 <= HUE_1 < 302 and  8 <= Saturation_1 < 100 and 50 <= Value < 100:
                    print("THIS IS THE PURPLE COLOR")
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                    cv2.putText(frame, class_name + " " + "PURPLE" + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            if FLAGS.Fuzzy_green:
                #if  str(0) <= Red < str(173)  and  str(100) <= Green < str(255) and str(0) <= Blue < str(170):
                if 0 <= HUE_1 < 160 and  24 <= Saturation_1 < 100 and 39 <= Value < 100:
                    print("THIS IS THE green COLOR")
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                    cv2.putText(frame, class_name + " " + "GREEN" + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            if FLAGS.Fuzzy_brown:
                #if  str(128) <= Red < str(255)  and  str(0) <= Green < str(248) and str(0) <= Blue < str(288):
                if 0 <= HUE_1 < 48 and  14 <= Saturation_1 < 100 and 50 <= Value < 100:
                    print("THIS IS THE BROWN COLOR")
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                    cv2.putText(frame, class_name + " " + "BROWN" + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            if FLAGS.Fuzzy_cyan:
                #if  str(0) <= Red < str(244)  and  str(128) <= Green < str(255) and str(128) <= Blue < str(255):
                if 0 <= HUE_1 < 182 and  12 <= Saturation_1 < 100 and 50 <= Value < 100:
                    print("THIS IS THE CYAN COLOR")
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                    cv2.putText(frame, class_name + " " + "CYAN" + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            if FLAGS.Fuzzy_pink:
                #if  str(199) <= Red < str(255)  and  str(20) <= Green < str(192) and str(133) <= Blue < str(203):
                if 322 <= HUE_1 < 351 and  25 <= Saturation_1 < 92 and 78 <= Value < 100:
                    print("THIS IS THE PINK COLOR")
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                    cv2.putText(frame, class_name + " " + "PINK" + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                       

            if FLAGS.black:
                if prediction == 'black':
                    #ROI = frame[int((int(bbox[1]) + int(bbox[3]))/2):int((int(bbox[1]) + int(bbox[3]))/2)+1,int((int(bbox[0]) + int(bbox[2]))/2):int((int(bbox[0]) + int(bbox[2]))/2)+1]
                    #color_histogram_feature_extraction.color_histogram_of_test_image(ROI)
                    #prediction = knn_classifier.main('training.data','test.data')
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                    cv2.putText(frame, class_name + " " + str(prediction) + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            if FLAGS.blue:
                if prediction =='blue':
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                    cv2.putText(frame, class_name + " " + str(prediction) + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            if FLAGS.red:
                if prediction =='red':
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                    cv2.putText(frame, class_name + " " + str(prediction) + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            if FLAGS.yellow:
                if prediction =='yellow':
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1) 
                    cv2.putText(frame, class_name + " " + str(prediction) + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            if FLAGS.orange:
                if prediction =='orange':
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                    cv2.putText(frame, class_name + " " + str(prediction) + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            if FLAGS.violet:
                if prediction =='violet':
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1) 
                    cv2.putText(frame, class_name + " " + str(prediction) + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            if FLAGS.white:
                if prediction =='white':
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1) 
                    cv2.putText(frame, class_name + " " + str(prediction) + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            if FLAGS.green:
                if prediction =='green':
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1) 
                    cv2.putText(frame, class_name + " " + str(prediction) + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                #cv2.putText(frame, class_name + " " + str(prediction) + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                #print('ferture data:' +" " +  feature_data)
                #result_1 = np.asarray(frame)
                #result_1 = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                #cv2.imshow('color classifier', result_1)
                #print(color_histogram_feature_extraction.feature_data)

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
