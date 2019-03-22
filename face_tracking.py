import numpy as np
import cv2
import socket

def center(points):
    x = np.float32((points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4.0)
    y = np.float32((points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4.0)
    return np.array([np.float32(x), np.float32(y)], np.float32)

UDP_IP = "127.0.0.1"
UDP_PORT = 5065

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
kalman = cv2.KalmanFilter(4,2)
kalmanWH = cv2.KalmanFilter(4,2)
kalman.measurementMatrix = np.array([[1,0,0,0],
                                     [0,1,0,0]],np.float32)

kalman.transitionMatrix = np.array([[1,0,1,0],
                                    [0,1,0,1],
                                    [0,0,1,0],
                                    [0,0,0,1]],np.float32)

kalman.processNoiseCov = np.array([[1,0,0,0],
                                   [0,1,0,0],
                                   [0,0,1,0],
                                   [0,0,0,1]],np.float32) * 0.03

kalmanWH.measurementMatrix = np.array([[1,0,0,0],
                                     [0,1,0,0]],np.float32)

kalmanWH.transitionMatrix = np.array([[1,0,1,0],
                                    [0,1,0,1],
                                    [0,0,1,0],
                                    [0,0,0,1]],np.float32)

kalmanWH.processNoiseCov = np.array([[1,0,0,0],
                                   [0,1,0,0],
                                   [0,0,1,0],
                                   [0,0,0,1]],np.float32) * 0.03


# initialize the known distance from the camera to the object, which
# in this case is 24 inches
KNOWN_DISTANCE = 24.0

KNOWN_WIDTH_EYE = 1.0
KNOWN_WIDTH_HEAD = 6.0

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

FOCAL_LENGTH = 1000

cap.set(3, CAMERA_WIDTH)
cap.set(4, CAMERA_HEIGHT)
print(CAMERA_WIDTH)
print(CAMERA_HEIGHT)
frameNum = 0
avgDist = 0
avgDistArray = []
avgX = 0
avgXArray = []
avgY = 0
avgYArray = []
avgW = 0
avgWArray = []
avgH = 0
avgHArray = []
while True:
    frameNum = frameNum + 1
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    distance = 0
    for face in faces:
        x, y, w, h = face
        rect = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        points = np.array([np.float32(x+(w/2)), np.float32(y+(h/2))], np.float32)
        pointsWH = np.array([np.float32(w), np.float32(h)], np.float32)
        kalman.correct(points)
        kalmanWH.correct(pointsWH)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y: y+h, x: x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            break
        break

    prediction = kalman.predict()
    predictionWH = kalmanWH.predict()

    avgXArray.append(prediction[0])
    avgYArray.append(prediction[1])
    avgWArray.append(predictionWH[0])
    avgHArray.append(predictionWH[1])

    if len(avgXArray) > 10:
        avgXArray = avgXArray[1:]
        avgYArray = avgYArray[1:]
    if len(avgWArray) > 10:
        avgWArray = avgWArray[1:]
        avgHArray = avgHArray[1:]

    avgX = sum(avgXArray) / len(avgXArray)
    avgY = sum(avgYArray) / len(avgYArray)
    avgW = sum(avgWArray) / len(avgWArray)
    avgH = sum(avgHArray) / len(avgHArray)

    frame = cv2.rectangle(frame, (avgX - (0.5 * avgW), avgY - (0.5 * avgH)),
                          (avgX + (0.5 * avgW), avgY + (0.5 * avgH)),
                          (0, 255, 0), 2)
    distance = (KNOWN_WIDTH_HEAD * FOCAL_LENGTH) / avgW
    avgDistArray.append(distance)
    if len(avgDistArray) > 10:
        avgDistArray = avgDistArray[1:]

    avgDist = sum(avgDistArray) / len(avgDistArray)

    cv2.putText(frame, "%.2fcm" % avgDist[0],
                    (frame.shape[1] - 200, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    2.0, (0, 255, 0), 3)

    cv2.putText(frame, "x:{} y:{} w:{} h:{} ".format(avgX[0], avgY[0], avgW[0], avgH[0]), (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 1)
    offX = (avgX + (avgW / 2)) - (CAMERA_WIDTH / 2)
    offY = (avgY + (avgH / 2)) - (CAMERA_HEIGHT / 2)
    cv2.putText(frame, "offX:{} offY:{}".format(offX[0], offY[0]), (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 1)

    # Display the resulting frame

    cv2.imshow('frame', frame)

    try:
        sock.sendto(("x:{}|y:{}|w:{}|h:{}|d:{}".format(offX[0], offY[0], avgW[0], avgH[0], avgDist[0])).encode(), (UDP_IP, UDP_PORT))
    except Exception:
        pass
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
