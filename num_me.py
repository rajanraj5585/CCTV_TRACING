from Detector import *
from faceRecogination import *
import cv2

modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
#modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz"
#modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v1_fpn_512x512_kpts_coco17_tpu-8.tar.gz"


classFile="coco.names"
imagePath="test/1.jpg"
videoPath=cv2.VideoCapture(0)
threshold=0.5

detector = Detector()
facereco = faceRecogination()
# detector.readClasses(classFile)
# detector.downloadModel(modelURL)
# detector.loadModel()
# #detector.predictImage(imagePath,threshold)
# detector.predictVideo(videoPath,threshold)
facereco.FaceReco(videoPath)
