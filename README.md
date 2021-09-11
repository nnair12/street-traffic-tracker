# Street Traffic Predictor
Detects vehicles and pedestrians on images and videos

Using this webapp, a user can upload a photo or video and detect vehicles in them. Bounding boxes are drawn around the detected objects.

For videos, these objects are tracked frame by frame. The total number of each type of object is printed.

For images, boxes are simply drawn over each object. Each detected object and their confidence is printed.

## How to Run

After downloading this project, first run the following to download all the needed packages

```pip install requirements.txt```

You will then need to download the weights file. You can put in any weights file in ```pages/data```, but the file name should be ```yolov3-kitti_best.weights```.

Finally, you can run the app with the following command:

```streamlit run app.py```

## Relevant Sources

DeepSort Implementation (Used to Track Objects On Video): https://github.com/nnair12/yolov4-deepsort
