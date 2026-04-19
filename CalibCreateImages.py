import cv2

cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4096)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
cap.set(cv2.CAP_PROP_FPS, 10)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # prevents lag

num = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite("img" + str(num) + '.png', frame)
        print("image saved!")
        num += 1

    cv2.imshow("Img",frame)

# Release and destroy all windows before termination
cap.release()

cv2.destroyAllWindows()