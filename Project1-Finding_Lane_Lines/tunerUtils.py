"""Parts of this code and the idea for this approach is taken from a git hub repository
https://github.com/maunesh/opencv-gui-parameter-tuner.git and then modified on it to suit my application."""

import cv2
import numpy as np


class Tuner:
    def __init__(self, filename, threshold=50, minllen=150, maxlgap=186, filter_size=5, threshold1=150, threshold2=200):
        # Initialisation of Argument Variables
        self.filename = filename
        self._threshold = threshold
        self._minlinlen = minllen
        self._maxlingap = maxlgap
        self._filter_size = filter_size
        self._threshold1 = threshold1
        self._threshold2 = threshold2

        # Functions for Tuner Panel Trackbar
        def onchangethreshold(pos):
            self._threshold = pos

        def onchangeminlinlen(pos):
            self._minlinlen = pos

        def onchangemaxlingap(pos):
            self._maxlingap = pos

        def onchangethreshold1(pos):
            self._threshold1 = pos

        def onchangethreshold2(pos):
            self._threshold2 = pos

        def onchangefiltersize(pos):
            self._filter_size = pos
            self._filter_size += (self._filter_size + 1) % 2

        # Create Trackbars for all parameters(Canny & Hough) in a window named 'Panel'
        cv2.namedWindow('Panel')
        cv2.createTrackbar('Low Threshold', 'Panel', self._threshold1, 255, onchangethreshold1)
        cv2.createTrackbar('High Threshold', 'Panel', self._threshold2, 255, onchangethreshold2)
        cv2.createTrackbar('Kernel Size', 'Panel', self._filter_size, 20, onchangefiltersize)
        cv2.createTrackbar('Threshold', 'Panel', self._threshold, 1000, onchangethreshold)
        cv2.createTrackbar('Min Line Length', 'Panel', self._minlinlen, 500, onchangeminlinlen)
        cv2.createTrackbar('Max Line Gap', 'Panel', self._maxlingap, 500, onchangemaxlingap)

        self._render()

        print("Adjust the parameters as desired.  Hit q key to close.")

        cv2.waitKey(100)

        cv2.destroyAllWindows()

    # Functions that returns parameter values
    def threshold(self):
        return self._threshold

    def minlinlen(self):
        return self._minlinlen

    def maxlingap(self):
        return self._maxlingap

    def threshold1(self):
        return self._threshold1

    def threshold2(self):
        return self._threshold2

    def filtersize(self):
        return self._filter_size

    # render function is responsible for processing the frames and displaying and saving output
    def _render(self):
        cap = cv2.VideoCapture(self.filename)  # Create object to read video

        # Create object to write video output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for saving video output
        out = cv2.VideoWriter('test_videos_output/solidWhiteRight.mp4', fourcc, 20.0, (960, 540))

        # Starting and Ending Y coordinates of lane annotation
        start_y = 320
        end_y = 539

        while cap.isOpened():
            ret, image = cap.read()  # Read Image Frame

            # Comment End of Video if condn and uncomment Playback loop if condn for ease of calibrating parameters
            # or vice-versa to save video output to file
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):  # End of Video if condn
                break
            """if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):  # Playback Loop if condn
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, img = cap.read()"""

            # Apply Gaussian Blur, convert to greyscale and apply canny edge detection
            blur_img = cv2.GaussianBlur(image, (self._filter_size, self._filter_size), 0)
            gray = cv2.cvtColor(blur_img, cv2.COLOR_RGB2GRAY)
            edge_img = cv2.Canny(gray, self._threshold1, self._threshold2)

            # Create Mask from region of interest, vertices is obtained by running the function
            # draw_roi(uses mouse callbacks) seperately
            mask = np.zeros_like(edge_img)
            vertices = np.array([[(35, 539), (445, 320), (525, 320), (935, 539)]], dtype=np.int32)
            cv2.fillPoly(mask, vertices, 255)
            masked_edges = cv2.bitwise_and(edge_img, mask)
            cv2.imshow('edges', masked_edges)

            line_img = np.copy(image) * 0  # Blank Image upon which the annotated lane will be drawn

            # Apply Hough Transform on the masked image
            lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, self._threshold, np.array([]),
                                    self._minlinlen, self._maxlingap)

            # Empty List to store Left and Right Lane points
            L_lanes = []
            R_lanes = []

            # Left and Right lanes are classified based on their slopes, Left lane if m>0 or Right Lane if m<0
            for line in lines:
                for x1, y1, x2, y2 in line:
                    m = (y2 - y1) / (x2 - x1)
                    if m < 0:
                        R_lanes.append([x1, y1])
                        R_lanes.append([x2, y2])
                    elif m > 0:
                        L_lanes.append([x1, y1])
                        L_lanes.append([x2, y2])

            # Using fitLine method to determine left lane and draw using line() in a blank image defined as line_img
            if L_lanes:
                vx, vy, x, y = cv2.fitLine(np.array(L_lanes, dtype=np.int32), cv2.DIST_L2, 0, 0.01, 0.01)
                m = vy / vx
                b = y - m * x
                start_x = (start_y - b) / m
                end_x = (end_y - b) / m
                cv2.line(line_img, (start_x[0], start_y), (end_x[0], end_y), (0, 0, 255), 10)

            # Using fitLine method to determine right lane and draw using line() in a blank image defined as line_img
            if R_lanes:
                vx, vy, x, y = cv2.fitLine(np.array(R_lanes, dtype=np.int32), cv2.DIST_L2, 0, 0.01, 0.01)
                m = vy / vx
                b = y - m * x
                start_x = (start_y - b) / m
                end_x = (end_y - b) / m
                cv2.line(line_img, (start_x[0], start_y), (end_x[0], end_y), (0, 0, 255), 10)

            # Draw the lines on the image and display on a window named Result
            line_edges = cv2.addWeighted(line_img, 0.8, image, 1, 0)
            cv2.imshow('Result', line_edges)

            # Write the resulting image to object out for saving to file
            out.write(line_edges)

            # Press 'q' when running the program to quit execution
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

        cap.release()
