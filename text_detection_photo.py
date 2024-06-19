# Code adapted from:
# https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/

# import the necessary packages
from imutils.object_detection import non_max_suppression
import tensorflow as tf
import numpy as np
import argparse
import imutils
import time
import cv2
import json
import keras_ocr
import socket

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimum probability required to inspect a region")
ap.add_argument("-ip", "--ip", type=str, default='localhost',
	help="the ip address of the server")
ap.add_argument("-cores", "--cores", type=int, default=4,
	help="the number of cores to allocate to the tflite interpreter")
ap.add_argument("-w", "--width", type=int, default=320,
	help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=320,
	help="resized image height (should be multiple of 32)")
args = vars(ap.parse_args())


def decode_predictions(scores, geometry):
	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the
		# geometrical data used to derive potential bounding box
		# coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability,
			# ignore it
			if scoresData[x] < args["min_confidence"]:
				continue

			# compute the offset factor as our resulting feature
			# maps will be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction and
			# then compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# use the geometry volume to derive the width and height
			# of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# add the bounding box coordinates and probability score
			# to our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	# return a tuple of the bounding boxes and associated confidences
	return (rects, confidences)

def preprocess_image(image, mean):
	image = image.astype("float32")
	image -= mean
	image = np.expand_dims(image, 0)
	return image

recognizer = keras_ocr.recognition.Recognizer()
# reader = easyocr.Reader(['en'], gpu=False, download_enabled=True, detector=False)
def OCR(image):
    return recognizer.recognize(image)

# initialize the TFLite interpreter
interpreter = tf.lite.Interpreter(model_path="./text_detector.tflite", num_threads=args["cores"])
input_details = interpreter.get_input_details()
interpreter.allocate_tensors()

def run_inference(image):
	# perform inference and parse the outputs
	interpreter.set_tensor(input_details[0]['index'], image)
	interpreter.invoke()
	scores = interpreter.tensor(
		interpreter.get_output_details()[0]['index'])()
	geometry = interpreter.tensor(
		interpreter.get_output_details()[1]['index'])()
	scores = np.transpose(scores, (0, 3, 1, 2)) 
	geometry = np.transpose(geometry, (0, 3, 1, 2))

	return (scores, geometry)

def doAI(image):
    print("Started AI")
    # define the channel-wise mean array to perform mean subtraction
    mean = np.array([123.68, 116.779, 103.939][::-1], dtype="float32")

    # initialize the original frame dimensions, new frame dimensions,
    # and ratio between the dimensions
    (W, H) = (None, None)
    (newW, newH) = (args["width"], args["height"])
    (rW, rH) = (None, None)

    # Save the image to disk
    # cv2.imwrite(".\output.jpg", image)

    # # Open the data as an image using PIL
    frame = image

    # resize the frame, maintaining the aspect ratio
    frame = imutils.resize(frame, width=1000)
    orig = frame.copy()

    # if our frame dimensions are None, we still need to compute the
    # ratio of old frame dimensions to new frame dimensions
    if W is None or H is None:
        (H, W) = frame.shape[:2]
        rW = W / float(newW)
        rH = H / float(newH)

    # resize the frame, this time ignoring aspect ratio and preprocess
    # it
    frame = cv2.resize(frame, (newW, newH))
    frame = preprocess_image(frame, mean)

    # perform inference and parse the outputs
    (scores, geometry) = run_inference(frame)

    # decode the predictions, then  apply non-maxima suppression to
    # suppress weak, overlapping bounding boxes
    (rects, confidences) = decode_predictions(scores, geometry)
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    cropped_image = None
    output = []
    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        if startX < 0:
            startX = 0
        if startY < 0:
            startY = 0
        if endX > orig.shape[1]:
            endX = orig.shape[1]
        if endY > orig.shape[0]:
            endY = orig.shape[0]

        # draw the bounding box on the frame
        # cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
        # cv2.imwrite(".\output_boxes.jpg", orig)
        cropped_image = orig[startY:endY, startX:endX]
        # if cropped_image.size != 0:
        #     cv2.imwrite(".\cropped_image" + str(startX) + ".jpg", cropped_image)
        output.append(OCR(cropped_image))

    return(output)

cv2.namedWindow("Display Window")

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to a specific address and port
server_address = (args["ip"], 10000)
print('starting up on {} port {}'.format(*server_address))
sock.bind(server_address)

# Listen for incoming connections
sock.listen(1)

while True:
    print('waiting for a connection')
    connection, client_address = sock.accept()

    try:
        print('connection from', client_address)

        while True:
            # Receive the input data as a byte array
            data = connection.recv(100000)
            if data and len(data) > 1:
                # Convert the data to a numpy array
                img_array = np.frombuffer(data, dtype=np.uint8)

                # Load the numpy array as an image using cv2
                img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

                # cv2.imwrite(".\output.jpg", img)
                # Show the image on screen
                cv2.imshow("Display Window", img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Process the image and get the output
                output = doAI(img)
                output = [str(x) for x in output]  # Convert each element to a string
                output = ",".join(output)

                if output == "":
                    output = "unrecognized"
                print("OCR output: " + output)

                # Send the output back to the client
                connection.sendall(output.encode('utf-8'))

            else:
                # No more data, so break out of the loop
                break
    # except:
    #     print("oups")
    finally:
        # Clean up the connection
        print("Closed connection")
        connection.close()

cv2.destroyAllWindows()