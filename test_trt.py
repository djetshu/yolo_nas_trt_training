# Packages 
import cv2
from PIL import Image 
import time
import tensorrt as trt
import argparse

#Custom packages
from inference import InferenceSession
from dataset import COCO_DETECTION_CLASSES_LIST, CUSTOM_DETECTION # Add your custom dataset list

def main(model_path):
    #Start
    #Creating windows to visualize images
    cv2.namedWindow("YOLO_NAS")
    cv2.namedWindow("preview")

    #Capture video
    vc = cv2.VideoCapture(0)
    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    # Calculate the FPS
    start_time = time.time()
    x = 1 # displays the frame rate every 1 second
    counter = 0

    # Create the YOLO tensorRT
    # Remember path of trt file needs to be according the current terminal' path
    with InferenceSession(model_path, (640, 640), trt_logger = trt.Logger(trt.Logger.VERBOSE)) as session: 
        # Main loop
        while rval:
            # Show the preprocess image
            cv2.imshow("preview", frame)

            # Get the current image
            rval, frame = vc.read()

            # Conversion CV to PIL image
            im_pil = Image.fromarray(frame)

            # Process the image with YOLO tensorRT
            result = session(im_pil)

            # Show the Image and results of YOLO TRT
            image = session.show_predictions_from_batch_format(im_pil, result, CUSTOM_DETECTION)
            cv2.imshow("YOLO_NAS", image)

            # Get the FPS after processing
            counter+=1
            if (time.time() - start_time) > x :
                print("FPS: ", counter / (time.time() - start_time))
                counter = 0
                start_time = time.time()

            # ESC key to break the loop
            key = cv2.waitKey(20)
            if key == 27: # exit on ESC
                break
        
        # Release webcam and break visualization windows
        vc.release()
        cv2.destroyWindow("preview")
        cv2.destroyWindow("YOLO_NAS")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLO NAS TensorRT Inference')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the TensorRT model file')

    args = parser.parse_args()
    main(args.model_path)