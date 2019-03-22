# Kalman Filtered and Stabilized OpenCV Face Tracking
OpenCV face tracking with Karman filter applied. It also averages the position to the last 10 frames for dat extra smoothness.

It calculates the position and the distance of the head and eyes. Note: It uses a single camera, so you need to mess around with the KNOWN_DISTANCE and FOCAL_LENGTH parameters found in face_tracking.py file. More info here: https://www.pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/

It also sends the data over a UDP socket. It sends the offset from the center and the distance in the following format:

>"x:{}|y:{}|w:{}|h:{}|d:{}"

>x: x position offset from the middle of the screen

>y: y position offset from the middle of the screen

>w: width of the bounding box

>h: height of the bounding box

>d: distance from camera.
## To run
pip install -r requirements.txt

python face_tracking.py
