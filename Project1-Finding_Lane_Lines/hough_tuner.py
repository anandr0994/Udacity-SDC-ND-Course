"""Parts of this code and the idea for this approach is taken from a git hub repository
https://github.com/maunesh/opencv-gui-parameter-tuner.git and then modified on it to suit my application."""

import cv2
from tunerUtils import Tuner


def main():
    filename = 'test_videos/solidWhiteRight.mp4'
    params = Tuner(filename)

    print("Edge Parameters:")
    print(f"GaussianBlur Filter Size: {params.filtersize()}")
    print(f"Low Threshold: {params.threshold1()}")
    print(f"High Threshold: {params.threshold2()}")
    print("Line Parameters:")
    print(f"Threshold: {params.threshold()}")
    print(f"Min Line Length: {params.minlinlen()}")
    print(f"Max Line Gap: {params.maxlingap()}")

    cv2.destroyAllWindows()


if __name__ == main():
    main()
