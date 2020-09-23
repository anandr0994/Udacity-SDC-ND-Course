""" This code is obtained from https://programmersought.com/article/3449903953/  and I made a few changes
to it for suiting my application"""

import cv2

pts = []


def draw_roi(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:  # Left click, select point
        if len(pts) < 4:
            pts.append((x, y))
        else:
            print('Only 4 points can be selected. You have exceeded limit')
            print('If you are satisfied with selection press s key or esc key')
            print('Else, Right click to delete last selected point and reselect again')

    if event == cv2.EVENT_RBUTTONDOWN:  # Right click to cancel the last selected point
        pts.pop()

    if len(pts) > 0:
        # Draw the last point in pts
        cv2.circle(img, pts[-1], 3, (0, 0, 255), -1)

    if len(pts) > 1:
        # Connect the points with lines
        for i in range(len(pts) - 1):
            cv2.circle(img, pts[i], 5, (0, 0, 255), -1)  # x ,y is the coordinates of the mouse click place
            cv2.line(img=img, pt1=pts[i], pt2=pts[i + 1], color=(255, 0, 0), thickness=2)

    cv2.imshow('Select ROI', img)


cap = cv2.VideoCapture('test_videos/solidWhiteRight.mp4')
cv2.namedWindow('Select ROI')
cv2.setMouseCallback('Select ROI', draw_roi)
while cap.isOpened():
    ret, img = cap.read()
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, img = cap.read()

    if cv2.waitKey(100) & 0xFF == 27:
        saved_data = [pts]
        print(saved_data)
        break
    if cv2.waitKey(100) & 0xFF == ord("s"):
        saved_data = [pts]
        print(saved_data)
        break

cap.release()
cv2.destroyAllWindows()
