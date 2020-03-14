import cv2
from random import randint 

def select_bounding_box(frame):
    ## Select boxes
    bboxes = []
    colors = [] 

    # OpenCV's selectROI function doesn't work for selecting multiple objects in Python
    # So we will call this function in a loop till we are done selecting all objects
    while True:
      # draw bounding boxes over objects
      # selectROI's default behaviour is to draw box starting from the center
      # when fromCenter is set to false, you can draw box starting from top left corner
        bbox = cv2.selectROI('MultiTracker', frame)
        bboxes.append(bbox)
        print("Box selected")
        colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
        print("Press q to quit selecting boxes and start tracking")
        print("Press any other key to select next object")
        k = cv2.waitKey(0) & 0xFF
        if (k == ord('q')):  # q is pressed
            break
        break
    print('Selected bounding boxes {}'.format(bboxes))
    cv2.destroyAllWindows()
    return bboxes,colors