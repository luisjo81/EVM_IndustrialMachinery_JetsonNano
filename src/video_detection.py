import cv2
from tracker import *

#Area for machine1
area_size = 15
#Area for machine2
#area_size = 150
#Area for machine3
#area_size = 50
#Area for machine4
#area_size = 500

#Create tracker
tracker = EuclideanDistTracker()

#Get video source
cap = cv2.VideoCapture("./results/machineEVM.avi")

#Create object detection
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape

    #Extract area of interest (change for every video used)
    #For machine1
    aoi = frame[300: 470, 980: 1900]
    #For machine2
    #aoi = frame[300: 550, 650: 850]
    #For machine3
    #aoi = frame[600: 1050, 250: 1000]
    #For machine4
    #aoi = frame[50: 800, 400: 1900]

    #Object detection
    mask = object_detector.apply(aoi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    objects_list = []
    for cnt in contours:
        #Object maximum size for detection
        area = cv2.contourArea(cnt)
        if area > area_size:
            x, y, w, h = cv2.boundingRect(cnt)
            objects_list.append([x, y, w, h])

    #Object tracking
    movement_id = tracker.update(objects_list)
    for box_id in movement_id:
        x, y, w, h, id = box_id
        cv2.putText(aoi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(aoi, (x, y), (x + w, y + h), (0, 255, 0), 3)

    #Show the three windows
    cv2.imshow("Area of insterest", aoi)
    cv2.imshow("Full image", frame)
    cv2.imshow("Mask", mask)

    #Keeps the program running
    key = cv2.waitKey(30)
    if key == 27:
        break

#Close the program
cap.release()
cv2.destroyAllWindows()