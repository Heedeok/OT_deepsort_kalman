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
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

# depth for zed
import statistics
# import pyzed.sl as sl
from point.pointer import Pointer
import math

# L515
from utils.new_utils_RS import Realsense

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('target', 'person', 'target for detection')
flags.DEFINE_boolean('l515', True, 'camera for L515')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')
flags.DEFINE_string('save', None, 'save point data to npy')
flags.DEFINE_string('load', None, 'load point data from npy')


def main(_argv):

    ### display target
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d') 
    # ax = fig.add_subplot(111) 

    ### display velocity
    # fig_vel = plt.figure(figsize=(13, 8))
    # ax_vel = fig_vel.add_subplot(111) 

    ### Flags define
    input_size = FLAGS.size
    video_path = FLAGS.video
    l515_path = FLAGS.l515
    save_path = FLAGS.save
    load_path = FLAGS.load
    target = FLAGS.target

    ### Display or searching mode

    if load_path is None:
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


        if l515_path:
            ss = Realsense(1)
            ss.get_Intrinsics()
        else:

            print('Realsense cannot detect!, loading video file...')
            # begin video capture
            try:
                vid = cv2.VideoCapture(int(video_path))
            except:
                vid = cv2.VideoCapture(video_path)

        out = None

        # get video ready to save locally if flag is set
        if FLAGS.output:
            # by default VideoCapture returns float instead of int
            width = 1280
            height = 720
            fps = 30
            codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
            out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

        info = []
        frame_num = 0
        
        # while video is running
        while True:

            if l515_path is not None:
                cviz, dviz = ss.output_image()
                frame = cv2.cvtColor(cviz, cv2.COLOR_BGR2RGB)

            else:
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
            for i, track in enumerate(tracker.tracks):
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

            # if enable info flag then print details about each track
                if FLAGS.info:
                    print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

                ### extract target coordi ---- L515 ----
                if l515_path is not None: 
                    if track.get_class() == target:
                        x, y, z = get_target_coordi(track, ss, ax)
                        info.append([track.track_id, frame_num, x,y,z])



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
    
        ### result processing
        target_pointer = point_split(info)

        ### using kalman 3d filter
        for point in target_pointer.points:

            point.initiate_kalman(60)
            point.calculate_velocity(point.kf.fps)
            
            ### target point
            ax.scatter(point.x, point.y, point.z, marker='o', s=4, label='id : {}'.format(point.idx))
            ax.scatter(point.kf.x, point.kf.y, point.kf.z, marker='o', s=8,label='kalman : {}'.format(point.idx))
            
            ### target velocity
            ax_vel.plot(range(point.frame_number[-1]- point.frame_number[0] + 1), point.v, label='id : {}'.format(point.idx))
            ax_vel.plot(range(point.frame_number[-1]- point.frame_number[0] + 1), point.kf.v, label='kalman : {}'.format(point.idx))


    else :
        target_pointer = np.load(load_path, allow_pickle = True)

        ### display target center
        for point in target_pointer:

            ### target point
            ax.scatter(point.x, point.y, point.z, marker='o', s=4, label='person : {}'.format(point.idx))
            ax.scatter(point.kf.x, point.kf.y, point.kf.z, marker='o', s=8,label='kalman : {}'.format(point.idx))
            
            ### target velocity
            # ax_vel.plot(range(point.frame_number[-1]- point.frame_number[0] + 1), point.v, label='person : {}'.format(point.idx))
            # ax_vel.plot(range(point.frame_number[-1]- point.frame_number[0] + 1), point.kf.v, label='kalman : {}'.format(point.idx))
           
    if save_path is not None:
        np.save(save_path,target_pointer.points)

    ax.set_xlabel('X [m]', fontsize=15)
    ax.set_ylabel('Y [m]', fontsize=15) 
    ax.set_zlabel('Z [m]', fontsize=15)
    ax.set_xlim(-3,3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(4.5,3)

    # ax_vel.set_xlabel('Frame', fontsize=20)
    # ax_vel.set_ylabel('Velocity [m/s]', fontsize=20)
    # ax_vel.set_ylim(-1,30)

    plt.legend(fontsize=18, loc='upper right')
    plt.show()


def get_object_depth_l515(depth, bounds):
    '''
    Calculates the median x, y, z position of top slice(area_div) of point cloud
    in camera frame.
    Arguments:
        depth: Point cloud data of whole frame.
        bounds: Bounding box for object in pixels.
            bounds[0]: x-center
            bounds[1]: y-center
            bounds[2]: width of bounding box.
            bounds[3]: height of bounding box.

    Return:
        x, y, z: Location of object in meters.
    '''
    area_div = 2

    x_vect = []
    y_vect = []
    z_vect = []
    # print('depth shape : {}'.format(depth.shape))

    for j in range(int(bounds[0] - area_div), int(bounds[0] + area_div)):
        for i in range(int(bounds[1] - area_div), int(bounds[1] + area_div)):
            x, y, z = depth.get_Depth_Point(j, i)
            
            if not np.isnan(z) and not np.isinf(z):
                x_vect.append(x)
                y_vect.append(y)
                z_vect.append(z)
    try:
        x_median = statistics.median(x_vect)
        y_median = statistics.median(y_vect)
        z_median = statistics.median(z_vect)
    except Exception:
        x_median = -1
        y_median = -1
        z_median = -1
        pass

    return x_median, y_median, z_median

def get_target_coordi(track, depth, ax):   
        
    bbox = track.to_tlwh()

    center_x = bbox[0] + bbox[2]//2
    center_y = bbox[1] + bbox[3]//2

    bounds = [center_x, center_y, bbox[2], bbox[3]]       
    x, y, z = get_object_depth_l515(depth, bounds)

    return x,y,z

def point_split(info):

    pointer = Pointer()

    idx = 0
    items = 0

    info.sort()

    for pt in info:
        if pt[0] != idx:
            idx = pt[0]
            items += 1
            pointer.new_point(idx)
        
        pointer.points[items-1].frame_number.append(pt[1])
        pointer.points[items-1].x.append(pt[2])
        pointer.points[items-1].y.append(pt[3])
        pointer.points[items-1].z.append(pt[4])

    return pointer


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
