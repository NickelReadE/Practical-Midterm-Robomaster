import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2
import statistics
import torch

# Configure depth and color streams


class DepthCamera:

    # Constructor
    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        self.load_model()
        # self.cap = cv2.VideoCapture("./CV_Detection-code-refactor-arjun/utils/vid.mp4")

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        depth_sensor = device.query_sensors()[0]
        depth_sensor.set_option(rs.option.laser_power, 0)

        # Start streaming
        self.pipeline.start(config)

    def load_model(self): 
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='./Algorithm/pt_files/best.pt')

	
  
  

    # Get Depth and Color Frame
    def get_frame(self):
        try:
            frames = self.pipeline.wait_for_frames()
        except:
            return False, None, None
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        if not depth_frame or not color_frame:
            return False, None, None

        return True, depth_image, color_image

    def release(self):
        self.pipeline.stop()

    def get_coordinates(self, frame, model):
        
        results = self.model(frame)
       
        rows = results.pandas().xyxy[0].to_numpy()
        if len(rows) != 0:
            x_min, y_min, x_max, y_max = rows['x_min'][0], rows['y_min'][0], rows['x_max'][0], rows['y_max'][0]
            return (x_min, y_min, x_max, y_max)
        return None

    def process_frame(self,depth_frame, x_min, y_min, x_max, y_max):
        values = []
        for x in range(x_min - 1, x_max):
            for y in range(y_min - 1, y_max):
                values.append(depth_frame[y, x])

        med = statistics.median(values)
        return med


# Display BBOX around detection with Estimated median depth.


    def show_frame(self,color_frame, depth_frame, depth, coordinates):
        # Display Text for distance
        if coordinates != None:
            cv2.putText(color_frame, "Median: {}mm".format(
                depth), (coordinates[0], coordinates[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0))

            # Display Rectangle overlay
            cv2.rectangle(color_frame, (coordinates[0], coordinates[1]), (
                coordinates[2], coordinates[3]), (0, 0, 255), 10)
            cv2.rectangle(depth_frame, (coordinates[0], coordinates[1]), (
                coordinates[2], coordinates[3]), (0, 0, 255), 10)

        # Show Both
        cv2.imshow("Video", color_frame)
        cv2.imshow("Video_Depth", depth_frame)

    def det_move_(self, obj_x_coord, obj_y_coord, xres, yres):
        obj_y_coord = yres-obj_y_coord
        centerx, centery = xres/2.0, yres/2.0

        
        move_x = obj_x_coord-centerx
        move_y = obj_y_coord-centery
        if(move_x != 0):
            move_x /= centerx
        if(move_y != 0):
            move_y /= centery

        return(move_x, move_y)




cam = DepthCamera()

oldCords = None
depth = None


while True:
    # Start Video Capture
    try:
        ret, depth_frame, color_frame = cam.get_frame()
    except:
        print("Error getting frame")

    # If frame is not empty
    if ret:

        key = cv2.waitKey(1)
        if key == 27:
            break

        # Get coordinates from color frame
        try:
            coordinates = cam.get_coordinates(color_frame, cam.model)

        except:
            print("Error getting cordinates\n")

        if coordinates != None:
            # Get Median Depth from depth frame
            try:
                depth = cam.process_frame(
                    depth_frame, coordinates[0], coordinates[1], coordinates[2], coordinates[3])
            except:
                print("Error processing_frame")

            # Debug mode
            if Debug_flag == 1:
                print("In: ", coordinates)
                print(coordinates)
                print(depth)
                cam.show_frame(color_frame, depth_frame, depth, coordinates)

                
                
           # try:
            final_cords = cam.det_move(
                (coordinates[0]+coordinates[2])/2,
                (coordinates[1]+coordinates[3])/2,
                640,
                480)
            print("This is run", final_cords)


            
if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
