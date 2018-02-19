# Real-Time-Object-Detection
Implementation of YOLO algorithm for real-time object detection and classification 

## Application: Lane Changer

### Dependencies
* Python 3
* Tensorflow CPU/GPU
* Numpy
* OpenCV

### Dataset
This application has been trained on the COCO test-dev dataset.

### About the Algorithm
You only look once (YOLO) is a state-of-the-art, real-time object detection system. Prior work on object detection repurposes classifiers to perform detection. Instead, YOLO frames object detection as a regression problem to spatially separated bounding boxes and associated class probabilities. 
<br/>
A single neural network predicts bounding boxes and class probabilities directly from full images in one evaluation. Since the whole detection pipeline is a single network, it can be optimized end-to-end directly on detection performance. This network divides the image into regions and predicts bounding boxes and probabilities for each region. These bounding boxes are weighted by the predicted probabilities.
<br/>
YOLO looks at the whole image at test time so its predictions are informed by global context in the image, instead of the sliding window approach.

### Test the model
1. Clone the repository.
1. Install the dependencies.
1. #### Object Detection in Image
   1. Save test image in **darkflow-master** as 'test.jpg'.
   1. Run 'object_detect_image.py'.
1. #### Object Detection in Video
   1. Save the test video in **darkflow-master** as 'test-video.mp4'.
   1. For faster implementation on CPU (unavilability of GPU), run 'reduce_frame_speed.py'.
   1. Execute 
         > python flow --model cfg/yolo.cfg --load bin/yolo.weights --demo videofile.mp4 --gpu 1.0 --saveVideo
      <br/>
      Omit the '--gpu 1.0' for Tensorflow CPU version. 
1. #### Object Detection in Real-Time
   1. Save the test video in **darkflow-master** as 'test-video.mp4'.
   1. Run 'object_detect_video_realtime.py'. The quality and frame speed is an attribute of hardware compatibility available. 
   
### Tested Samples
<img src = "https://user-images.githubusercontent.com/31643223/36366210-1ad74128-1573-11e8-9c55-84ab5188718d.jpg" width="500" height="400">
<img src = "https://user-images.githubusercontent.com/31643223/36366256-4c3b01f0-1573-11e8-9d3f-2ede6970d5e1.jpg" width="500" height="400">
<img src = "https://user-images.githubusercontent.com/31643223/36366306-808a66b2-1573-11e8-9fea-dc595dc6e581.jpg" width="500" height="400">
<img src = "https://user-images.githubusercontent.com/31643223/36366294-6c9a3024-1573-11e8-87dc-e823b06c9fa2.jpg" width="500" height="400">
