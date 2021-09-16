import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from .core import utils
from .core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from .core.config import cfg

from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# deep sort imports
from .deep_sort import preprocessing, nn_matching
from .deep_sort.detection import Detection
from .deep_sort.tracker import Tracker
from .tools import generate_detections as gdet

import streamlit as st
import tempfile

# Database insertion related imports
from datetime import datetime
from .dbconfig import connection_string, db_name, video_collection_name
from pymongo import MongoClient

class VideoTracker:
    def __init__(self, framework='tf', tiny=False, model='yolov3'):
        # Model configuration
        self.framework = framework
        self.tiny = tiny
        self.model = model

        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.saved_model_loaded = None
        self.infer = None

        # Tracker config
        self.encoder = None
        self.tracker = None

        # Other constants
        self.nms_max_overlap = 1.0
        self.max_cosine_distance = 0.4


    # Generates interpreter used for box prediction
    def load_model(self):
        # Definition of the parameters
        nn_budget = None

        # Mark status
        status = st.empty()
        
        # initialize deep sort
        print("Initializing deep sort...")
        status.text("Initializing deep sort...")
        model_filename = './pages/model_data/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        # calculate cosine distance metric
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, nn_budget)
        # initialize tracker
        self.tracker = Tracker(metric)

        # load configuration for object detector
        print("Loading config...")
        status.text("Loading config...")
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(self.tiny, self.model)
        session.close()
        
        # load tflite model if flag is set
        print("Loading model...")
        status.text("Loading model...")
        weights_file = './pages/checkpoints/yolov3-kitti'
        if self.framework == 'tflite':
            self.interpreter = tf.lite.Interpreter(model_path=weights_file)
            self.interpreter.allocate_tensors()
            self.input_details = interpreter.get_input_details()
            self.output_details = interpreter.get_output_details()
            print(self.input_details)
            print(self.output_details)
        # otherwise load standard tensorflow saved model
        else:
            self.saved_model_loaded = tf.saved_model.load(weights_file, tags=[tag_constants.SERVING])
            self.infer = self.saved_model_loaded.signatures['serving_default']
        status.text("Loaded model")


    # Returns a video with annotations and well as text listing the results
    def track_video(self, video_path, video_name):
        input_size = 416
        status = st.empty()

        # begin video capture
        print("Capturing video file into cv2 format...")
        status.text("Capturing video...")
        try:
            vid = cv2.VideoCapture(int(video_path))
        except:
            vid = cv2.VideoCapture(video_path)

        # while video is running
        frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_tracked = 0.0
        my_bar = st.progress(0)
        frame_result = st.empty()

        # Configure cv2 writer to output video
        resultTempFile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter(resultTempFile.name, codec, fps, (width, height))
        # Tracks type of each detected object along with its frequency
        results_dict = {}

        print("Starting to track...")
        status.text("Tracking...")
        while True:
            return_value, frame = vid.read()
            if return_value:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
            else:
                print('Video has finished tracking!')
                break
            frame_size = frame.shape[:2]
            image_data = cv2.resize(frame, (input_size, input_size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)

            # run detections on tflite if flag is set
            if self.framework == 'tflite':
                self.interpreter.set_tensor(input_details[0]['index'], image_data)
                self.interpreter.invoke()
                pred = [self.interpreter.get_tensor(self.output_details[i]['index']) for i in range(len(self.output_details))]
                # run detections using yolov3 if flag is set
                if self.model == 'yolov3' and self.tiny == True:
                    boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
                else:
                    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
            else:
                batch_data = tf.constant(image_data)
                pred_bbox = self.infer(batch_data)
                for key, value in pred_bbox.items():
                    boxes = value[:, :, 0:4]
                    pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=0.45,
                score_threshold=0.25
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
            # delete detections that are not in allowed_classes
            bboxes = np.delete(bboxes, deleted_indx, axis=0)
            scores = np.delete(scores, deleted_indx, axis=0)

            # encode yolo detections and feed to tracker
            features = self.encoder(frame, bboxes)
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

            #initialize color map
            cmap = plt.get_cmap('tab20b')
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            # run non-maxima supression
            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
            detections = [detections[i] for i in indices]       

            # Call the tracker
            self.tracker.predict()
            self.tracker.update(detections)
            # update tracks
            for track in self.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()
                class_name = track.get_class()
                
                # draw bbox on screen
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

                # Update results dict
                if not class_name in results_dict.keys():
                    results_dict[class_name] = 1
                elif results_dict[class_name] < track.track_id:
                    results_dict[class_name] = track.track_id

            # calculate frames per second of running detections
            result = np.asarray(frame)
            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # TODO: Find a way to concatenate result (frame) into single video
            frame_result.image(frame)
            # Convert to different format when writing
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out.write(frame)
            # Update Progress bar
            frames_tracked = frames_tracked + 1.0
            my_bar.progress(frames_tracked/frame_count)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        status.text("Finished Tracking!")
        vid.release()
        out.release()
        cv2.destroyAllWindows()

        # Send detection result to DB
        self.update_db(video_name, frame_count, results_dict)

        # Remove tracking elements so that final result can be displayed
        my_bar.empty()
        frame_result.empty()
        return resultTempFile, results_dict

    # Update database with detection
    def update_db(self, filename, framecount, results_dict):
        detections_list = []
        with open('./pages/data/classes/obj.names') as f:
            classes = f.read().splitlines()
        for class_name in classes:
            if class_name in results_dict:
                detections_list.append({class_name : results_dict[class_name]})
            else:
                detections_list.append({class_name: 0})
        date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        video_json = { 'filename': filename, 'date_added': date, 'num_frames': framecount, 'detections_count' : detections_list }

        # Connect to DB
        connection_uri = connection_string
        client = MongoClient(connection_uri)
        db = client[db_name]
        video_collection = db[video_collection_name]
        # Insert record into collection
        try:
            video_collection.insert_one(video_json)
            print("Added Video detection to Database: ", video_json)
        except:
            print("Unable to add video [", filename, "] detection to Database")
        # Close the client
        client.close()
