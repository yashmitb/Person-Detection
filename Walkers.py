import cv2


# Create our body classifier
path = cv2.CascadeClassifier(
    "Face_Recognition/PRO-106-ProjectTemplate-main/PRO-106-ProjectTemplate-main/haarcascade_fullbody.xml")

# Initiate video capture for video file
vid = cv2.VideoCapture('walking.avi')
path = cv2.CascadeClassifier(
    "C:\Python310\Lib\site-packages\cv2\data\haarcascade_fullbody.xml")


while True:
    ret, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = path.detectMultiScale(gray, 1.1, 3)
    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    cv2.imshow("vid", frame)
    if (cv2.waitKey(25) == 32):
        break
vid.release()
cv2.destroyAllWindows()
