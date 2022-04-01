# Import
import numpy as np
from PIL import Image
from torchvision import transforms as T
import cv2 as cv
import torch
import torch.utils.data
import os
import time

# Set Variable
os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Variables
BOX_COLOR = (0, 0, 225) # Red
BOX_THICKESS = 5
TEXT_COLOR = (100, 255, 0)
TEXT_FONT = 0
IMG_SIZE = (200, 200)

# Damage categories
# category_id_to_name = { 1: "dent", 2: "broken_glass", 3: "deflated_wheel", 4: "scratch", 5: "broken_headlight"}
category_id_to_name = { 1 : "damage"}

transform_to_tensor = T.ToTensor()

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


def draw_box(_image, _left, _top, _right, _bottom, _name):
    TEXT_SIZE = round(get_optimal_font_scale(_name, _image.shape[1])/2)
    cv.rectangle(_image, (_left, _top), (_right, _bottom), BOX_COLOR, BOX_THICKESS)
    cv.putText(_image, _name, (_right+10, _bottom), TEXT_FONT, TEXT_SIZE, TEXT_COLOR)
    return _image


def get_optimal_font_scale(text, width):
    for scale in range(60, 0, -1):
        textSize = cv.getTextSize(text, fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
        if (textSize[0][0] <= width):
            return scale/10
    return 1


def get_image_with_boxes(img=None, prediction=None, resize=False, fx=0, fy=0, tolerance=0):
    h = 0
    array_bbox = []
    for left, top, right, bottom in prediction[0]['boxes']:
        if float(prediction[0]["scores"][h]) > tolerance:
            left, top, right, bottom = int(left), int(top), int(right), int(bottom)
            name = category_id_to_name.get(int(prediction[0]['labels'][h]))
            if resize:
                assert(fx != 0)
                assert(fy != 0)
                left = int(round(left * fx))
                top  = int(round(top * fy))
                right = round(right * fx)
                bottom = int(round(bottom * fy))
            array_bbox.append( (left, top, right, bottom, name) )                  # Save boxes
            img = draw_box(img, left, top, right, bottom, name)
        h += 1
    return img, array_bbox


def show_webcam(MIRROR=False, VIDEO_PATH=0, SHOW_FPS=False, STREAMING=False, RESIZE=False, CONSOLE=False):
    model = torch.load(PATH, map_location=device).eval()
    cam = cv.VideoCapture(VIDEO_PATH)
    frame_exist = cam.isOpened()
    frame_id = 0
    last_bboxes = []
    fps_history = []
    FX = 1
    FY = 1

    prev_frame_time = 0
    new_frame_time = 0

    while frame_exist:
        new_frame_time = time.time()
        frame_exist, img = cam.read()

        if not frame_exist: break
        if MIRROR: img = cv.flip(img, 1)

        '''
        Optimization trick:
        Nearest frames in camera or video streaming have the same objects which are approximately placed closely
        We can skip some next frames from predictor and move the boxes from previous frame to increase FPS in STREAMING mode
        Skipped frames output without latency
        '''
        FLAG = False
        if frame_id % FRAME_PRED == 0:
            if RESIZE:
                FX = IMG_SIZE[0] / img.shape[1]
                FY = IMG_SIZE[1] / img.shape[0]
                debug_img = cv.resize(img, (0,0), fx=FX, fy=FY)
            else:
                debug_img = img

            FLAG = True
            with torch.no_grad(): prediction = model([transform_to_tensor(debug_img).to(device)])

            last_bboxes.clear()
            if len(prediction[0]["labels"]) > 0:
                img, last_bboxes = get_image_with_boxes(img=img, prediction=prediction, resize=RESIZE, fx=1/FX, fy=1/FY, tolerance=TOLERANCE)

        elif STREAMING and len(last_bboxes):
            for left, top, right, bottom, name in last_bboxes:
                img = draw_box(img, left, top, right, bottom, name)
        
        if (FLAG or STREAMING) and SHOW_FPS:
            fps = 1/(new_frame_time - prev_frame_time)
            TEXT_SIZE = round(get_optimal_font_scale(f"FPS: 000000", img.shape[1])/2)
            prev_frame_time = new_frame_time
            fps_history.append(int(fps))
            if not CONSOLE:
                cv.putText(img, f"FPS: { sum(fps_history)/len(fps_history):0.2f}", (7, 70), TEXT_FONT, TEXT_SIZE, TEXT_COLOR, 3, cv.LINE_AA)
        
        if not CONSOLE and (FLAG or STREAMING):
            cv.imshow('my webcam', img)

        frame_id += 1
        if cv.waitKey(1) == 27: 
            break  # esc to quit
    if CONSOLE and len(fps_history):
        print(f"Average FPS: { sum(fps_history)/len(fps_history):0.2f}")
    cv.destroyAllWindows()
    return 


FRAME_PRED = 10                     # FPS of Prediction pipeline
TOLERANCE = 0.4                     # Predictors confidence of class
PATH = "include/model_4.pt"        # Path to predictor model


show_webcam(VIDEO_PATH=0, MIRROR=False, SHOW_FPS=True, STREAMING=True)
