import cv2

class Tracker:
    def __init__(self,bboxes,frame,trackerType = "CSRT"):
        self.trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
        self.multiTracker = self.create_multiTracker(bboxes,frame,trackerType)
    
    def createTrackerByName(self,trackerType):
  # Create a tracker based on tracker name
        if trackerType == self.trackerTypes[0]:
            tracker = cv2.TrackerBoosting_create()
        elif trackerType == self.trackerTypes[1]: 
            tracker = cv2.TrackerMIL_create()
        elif trackerType == self.trackerTypes[2]:
            tracker = cv2.TrackerKCF_create()
        elif trackerType == self.trackerTypes[3]:
            tracker = cv2.TrackerTLD_create()
        elif trackerType == self.trackerTypes[4]:
            tracker = cv2.TrackerMedianFlow_create()
        elif trackerType == self.trackerTypes[5]:
            tracker = cv2.TrackerGOTURN_create()
        elif trackerType == self.trackerTypes[6]:
            tracker = cv2.TrackerMOSSE_create()
        elif trackerType == self.trackerTypes[7]:
            tracker = cv2.TrackerCSRT_create()
        else:
            tracker = None
            print('Incorrect tracker name')
            print('Available trackers are:')
            for t in trackerTypes:
                print(t)
        return tracker
        
    def create_multiTracker(self,bboxes,frame,trackerType):
    # Specify the tracker type
        trackerType = trackerType
        # Create MultiTracker object
        multiTracker = cv2.MultiTracker_create()
        # Initialize MultiTracker 
        for bbox in bboxes:
            multiTracker.add(self.createTrackerByName(trackerType), frame, bbox)
        print("Tracker Created ...")
        return multiTracker



