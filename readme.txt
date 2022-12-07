Nikhil Reddy Midterm

Prompt 1:

ColorDetection class used to check if there was more red or blue in area. Then converted frame to hue, saturation and value and applied red and blue masks. findcontours() was then used for the masks and calculate the sum of the area of each contour and checked if the red area was greater than the blue area to determine if the frame contained red or blue armor plates.

Capture class with a ColorDetection object and is initialized with video source. Capture class contains process_frame() which uses object detection pipeline on each frame. Confidence score is displayed using after bounding boxes are used to detect blue armor plates.

These masks are beneficial as they get rid of additional information that can cause the program to slow down.

Used thresholding to detect the blue armor plates.

Prompt 2:

The DepthCamera class used bounding box coordinates to get the angle offset values. get_coordinates function gets the coordinates of the frame and returns (x_min, y_min, x_max, y_max) coordinates are then used to find median distance from heatmap.

Prompt 3:

Followed the instructions on the tutorial about SystemD. The service files runs object detection pipeline and the service is started. It is turned on when booted. I selected my service type by using the information on the tutorial and making it a simple type.
