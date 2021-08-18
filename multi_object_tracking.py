import face_alignment
import collections
import numpy as np
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os, sys
import subprocess
from skimage import io
from os.path import isfile, join
from os import listdir
from utils import find_q2k, find_outliers
import pdb
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = str(1)
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video_path", type=str,
	help="path to input video file")
ap.add_argument("-sv", "--save_video_path", type=str,
	help="path to output video file")
ap.add_argument("-m", "--mode", type=str, default="both",
        help="select detect, tracking or both", )
ap.add_argument("-t", "--tracker", type=str, default="csrt",
	help="OpenCV object tracker type")
args = vars(ap.parse_args())

# initialize a dictionary that maps strings to their corresponding
# OpenCV object tracker implementations
OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.TrackerCSRT_create,
	"kcf": cv2.TrackerKCF_create,
	"boosting": cv2.TrackerBoosting_create,
	"mil": cv2.TrackerMIL_create,
	"tld": cv2.TrackerTLD_create,
	"medianflow": cv2.TrackerMedianFlow_create,
	"mosse": cv2.TrackerMOSSE_create,
    "goturn":cv2.TrackerGOTURN_create
}
                        

fa=face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu',flip_input=False, face_detector='sfd')

fa_probs_threshold  = 0.95

# initialize OpenCV's special multi-object tracker
#trackers            = cv2.MultiTracker_create()


fps=30
size=(224,224)

video_list=[x for x in listdir(args["video_path"]) if ".avi" in x]
video_list=sorted(video_list)
#video_list=video_list[::-1]
video_count=0

for video_name in video_list:

    try:
        
        
        start_time=time.time()
        video=args["video_path"]+str(video_name)
        out_path = args["save_video_path"] + str(video_name)   
    
        if os.path.exists(out_path):
            print("already existed: ", video_name)
            continue
    
        vs = cv2.VideoCapture(video)
        
        # loop over frames from the video stream
        n_frame=0
        files=[]
        count=0
        video_count+=1
        overlapped_list=[]
        
        
        while True:
            # grab the current frame, then handle if we are using a
            # VideoStream or VideoCapture object
            
            hasFrame, frame = vs.read()
            if not hasFrame:
                break
        #    frame = frame[1] if args.get("video", False) else frame
            #frame = frame[:,int(240):int(1680)]
            #frame = cv2.resize(frame, dsize=(640,480),interpolation=cv2.INTER_LINEAR)
           
            count+=1
            #print("count", count)
            # check to see if we have reached the end of the stream
            #if frame is None:
            #    break
            
            if args["mode"] == "tracking":
                ############################ face detect at first frame ############################
                if n_frame == 0:
                    bbox = []
                    pred, probs = fa.get_landmarks(frame)
                    pdb.set_trace()
                    if len(probs) > 1:
                        for prob in probs:
                            overlapped_list.append(prob)
                        min_index=overlapped_list.index(max(overlapped_list))
                        pred=[pred[min_index]]
                        overlapped_list=[]
        
                    pred = np.squeeze(pred)
                    x = pred[:,0]
                    y = pred[:,1]
                    min_x = min(x)
                    min_y = min(y)
                    max_x = max(x)
                    max_y = max(y)
                    height = int((max_y-min_y)/2)
                    width = int((max_x-min_x)/2)
                    standard=max(height,width)
                    box = [int(min_x), int(min_y), int(max_x), int(max_y)]
                #####################################################################################
#                    print("first box", box)   
                # create a new object tracker for the bounding box and add it
                # to our multi-object tracker
        #            for idx_b in range(len(bbox)): 
                    box = tuple(box)
                    tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
                    tracker.init(frame, box)
                    #pdb.set_trace()
                else:
                    #pdb.set_trace() 
                    (success, boxes) = tracker.update(frame)
                    #pdb.set_trace()
                    box=[]
                    #pdb.set_trace()
                    for i in range(len(boxes)):
                        box.append(int(boxes[i]))
                    box=tuple(box)
#                    print("after box",box)
        
            else:
                print("NotImplementMode Error")
                sys.exit()
        
            # loop over the bounding boxes and draw then on the frame
            # If you want to add crop function, you can this for loop #
            #pdb.set_trace() 
            (x, y, w, h) = [int(v) for v in box]
                # here, i index means person_id. crop i axis and resize
            #left_boundary=max(int(0),int((h+y)/2)-standard)
            #right_boundary=min(int(640),int((h+y)/2)+standard)
            #top_boundary=max(int(0),int((w+x)/2)-standard)
            #bottom_boundary=min(int(480),int((w+x)/2)+standard)
            
            left_boundary=int((h+y)/2)-standard
            right_boundary=int((h+y)/2)+standard
            top_boundary=int((w+x)/2)-standard
            bottom_boundary=int((w+x)/2)+standard


            crop_img = frame[left_boundary:right_boundary,top_boundary:bottom_boundary]
            resized_crop_img=cv2.resize(crop_img, dsize=(224,224),interpolation=cv2.INTER_LINEAR)
            files.append(resized_crop_img) 
            n_frame += 1
        
        #########save cropped video##########
        #frame_array=[]
        #for j in range(len(files)):
            #height, width, layers = files[j].shape
            #size = (width,height)
            #frame_array.append(files[j])
        
        
        #out_path = args["save_video_path"] + video_name
           
        print("mpg vs mpg_crop: {} vs {}".format(count,len(files)))
        #pdb.set_trace()

        out = cv2.VideoWriter(
                out_path,
                cv2.VideoWriter_fourcc(*'DIVX'),
                fps,
                size,
            ) 
        print("now starting to save cropped video")
        #pdb.set_trace()

        for k in range(len(files)):
            out.write(files[k])
        out.release()
        
        vs.release()
        print(video_name, " saved",video_count)
        print("time: ",time.time()-start_time) 
    except Exception as ex:
        print("ERROR:",ex,video_name)


        
