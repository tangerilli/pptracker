import sys

import cv2
import numpy as np
import json


def findBall(frame, params):
    h, w, _ = frame.shape
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # threshold the image to isolate the target colour
    lower_bound = np.array([params.lowH, params.lowS, params.lowV])
    upper_bound = np.array([params.highH, params.highS, params.highV])
    thresholded = cv2.inRange(frame_hsv, lower_bound, upper_bound)

    kernel = np.ones((5, 5), np.uint8)
    thresholded = cv2.erode(thresholded, kernel)
    thresholded = cv2.dilate(thresholded, kernel)

    thresholded = cv2.dilate(thresholded, kernel)
    thresholded = cv2.erode(thresholded, kernel)

    moments = cv2.moments(thresholded)
    if moments['m00'] > 10000:
        x = int(moments['m10'] / moments['m00'])
        y = int(moments['m01'] / moments['m00'])
        return True, x, y, thresholded
    else:
        return False, None, None, thresholded


class Params:
    def __init__(self, data={}):
        self.lowH = data.get('lowH', 0)
        self.lowS = data.get('lowS', 0)
        self.lowV = data.get('lowV', 0)
        self.highH = data.get('highH', 179)
        self.highS = data.get('highS', 255)
        self.highV = data.get('highV', 255)


def updateParams(params, name, val):
    setattr(params, name, val)


def loadConfig(path):
    try:
        f = open(path, 'r')
        config = json.load(f)
        f.close()
        return Params(config)
    except:
        return Params()


def writeConfig(path, params):
    f = open(path, 'w')
    config = {
        'lowH': params.lowH,
        'lowS': params.lowS,
        'lowV': params.lowV,
        'highH': params.highH,
        'highS': params.highS,
        'highV': params.highV
    }
    json.dump(config, f)
    f.close()


def main(args):
    calibrate = False
    if len(args) > 1 and args[1] == 'calibrate':
        calibrate = True
    cv2.namedWindow('controls')

    params = loadConfig('config.json')
    cv2.createTrackbar('LowH', 'controls', params.lowH, 179, lambda x: updateParams(params, 'lowH', x))
    cv2.createTrackbar('LowS', 'controls', params.lowS, 255, lambda x: updateParams(params, 'lowS', x))
    cv2.createTrackbar('LowV', 'controls', params.lowV, 255, lambda x: updateParams(params, 'lowV', x))
    cv2.createTrackbar('HighH', 'controls', params.highH, 179, lambda x: updateParams(params, 'highH', x))
    cv2.createTrackbar('HighS', 'controls', params.highS, 255, lambda x: updateParams(params, 'highS', x))
    cv2.createTrackbar('HighV', 'controls', params.highV, 255, lambda x: updateParams(params, 'highV', x))

    cap = cv2.VideoCapture(0)
    while(True):
        ret, raw_frame = cap.read()
        found, x, y, thresholded = findBall(raw_frame, params)

        if calibrate:
            frame = thresholded
        else:
            frame = raw_frame
        if found:
            cv2.rectangle(frame, (x + 20, y), (x, y + 20), (255, 255, 255), 1)

        # Display the resulting frame
        cv2.imshow('pptracker', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            writeConfig('config.json', params)
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv)
