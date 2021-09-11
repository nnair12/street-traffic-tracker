import streamlit as st

# Custom imports 
from multipage import MultiPage
from pages import detect_video, detect_image

# Create an instance of the app 
app = MultiPage()

# Title of the main page
st.title('Webapp Street Detection')

# Add applications here
app.add_page("Detect from Video", detect_video.app)
app.add_page("Detect from Image", detect_image.app)

# The main app
app.run()