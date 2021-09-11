import streamlit as st
import os
import cv2
import tempfile
#from .video_tracker import track_video # TODO Create Tracker class like what was there before
from .video_tracker import VideoTracker
import traceback

# Initialize model and stuff outside of app()
tracker = VideoTracker()
tracker.load_model()

def app():
	st.header('Detect Objects in Video')

	# Temp file to hold result
	resultTempFile = None

	video_file = st.file_uploader('Upload your Video', type=['mp4'])
	if video_file is not None:
		st.markdown('**_Original Video_**')
		st.video(video_file)

	if video_file and st.button('Draw Boxes on Video'):
		# If an existing result file already exists, delete it
		close_file(resultTempFile)

		# Read input video file
		inputTempFile = tempfile.NamedTemporaryFile(delete=False)
		inputTempFile.write(video_file.read())

		# Detect images on the video
		resultTempFile, tracked_objects = tracker.track_video(inputTempFile.name)
		close_file(inputTempFile)

		# Display resulting video
		try:
			st.video(resultTempFile.name)
		except Exception as ex:
			print("Error reading video file", resultTempFile.name)
			traceback.print_exc()
			st.text("Video cannot be displayed :(")

		# Print detection results
		result_string = ''
		st.markdown('**_The tracker found the following objects: _**')
		for key, val in tracked_objects.items():
			result_string = result_string + key + ': ' + str(val) + '\n'
		st.text(result_string)
	close_file(resultTempFile)

# Closes tempfile to save resources
def close_file(tempFile):
	if tempFile is not None:
		tempFile.close()
		os.unlink(tempFile.name)