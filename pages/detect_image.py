import streamlit as st
import numpy as np
import cv2
from .image_detection import get_boxes

def app():
	st.header('Detect Objects in Image')
	image_file = st.file_uploader('Upload your Image', type=['png', 'jpg', 'jpeg'])
	if image_file is not None:
		st.markdown('**_Original Image_**')
		st.image(image_file)

	if image_file and st.button('Draw Boxes on Image'):
		img = read_image(image_file)

		# Detect boxes in the image
		img_with_boxes, detections = get_boxes(img)

		st.markdown("**_Here is the image with detected boxes_**")
		st.image(img_with_boxes)
		if (len(detections) == 0):
			st.text("No objects were detected in this image")
		else:
			# Print each item found by YOLO followed by its confidence
			st.text("The following objects were detected:")
			results_text = ""
			for count, detection in enumerate(detections):
				results_text += str(count+1) + ". " + detection + "\n"
			st.text(results_text)

def read_image(image_file):
	# Convert the file to an opencv image
	file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
	image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	return image