import numpy as np
import cv2

min_confidence=0.5
min_threshold=0.3

# Given an image, detect objects and return: a copy of the image with boxes drawn, and the detection results
def get_boxes(image):
	# Load class labels
	labelsPath = './pages/data/classes/obj.names'
	labels = open(labelsPath).read().strip().split('\n')

	# Load model architecture
	net = cv2.dnn.readNetFromDarknet('./pages/single-image/yolov3-kitti.cfg', './pages/data/yolov3-kitti_best.weights')
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	# Set input and get output
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(ln)

	boxes, confidences, classIDs = [], [], []
	(height, width) = image.shape[:2]

	get_boxes_and_confidences(layerOutputs, boxes, confidences, classIDs, width, height)
	# Perform non maximal suppression
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, min_threshold)

    # Create a copy of the image to draw bounding boxes on
	img_clone = image.copy()
	detections = draw_boxes(idxs, img_clone, boxes, classIDs, confidences, labels)

	return img_clone, detections

# From detections, get bounding boxes and their respective confidence and classID
def get_boxes_and_confidences(layerOutputs, boxes, confidences, classIDs, width, height):
    for output in layerOutputs:
        for detection in output:
            # Get classID and confidence from object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # Filter out weak predictions
            if confidence > min_confidence:
                box = detection[0:4] * np.array([width, height, width, height])
                (centerX, centerY, box_width, box_height) = box.astype('int')

                # Get top left corner of the box
                x = int(centerX - (box_width/2))
                y = int(centerY - (box_height/2))

                # Update boxes, confidences, and classIDs
                boxes.append([x, y, int(box_width), int(box_height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

# Draw bounding boxes on the image
def draw_boxes(idxs, image, boxes, classIDs, confidences, labels):
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
    detections = []

    if len(idxs) > 0: 
        # If object was detected only
        index = 1
        for i in idxs.flatten():
            # Get bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # Add detection information to the image
            try:
                color = [int(c) for c in colors[classIDs[i]]]
                # Draw the bounding box on the image
                cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
                text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
                detections.append(text)
                # Label the boxes with index of the object
                cv2.rectangle(image, (x, y-30), (x+15, y), color, -1)
                cv2.putText(image, str(index), (x, y-10), 0, 0.75, (255,255,255), 2)
                index = index + 1
            except IndexError:
                print("Invalid ID " + str(classIDs[i]))
    return detections