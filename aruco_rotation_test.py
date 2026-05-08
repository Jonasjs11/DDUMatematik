import cv2
import cv2.aruco as aruco
import pickle
import math
import numpy as np

# Open webcam
cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4096)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
cap.set(cv2.CAP_PROP_FPS, 5)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # prevents lag

# Load kamera-barrel-distortion-kalibrations-filer
fs = cv2.FileStorage("calibration.yaml", cv2.FILE_STORAGE_READ)
cameraMatrix = fs.getNode("cameraMatrix").mat()
dist = fs.getNode("distCoeffs").mat()
fs.release()

# Load ArUco dictionary
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

detector_params = aruco.DetectorParameters()

marker_length = 0.02   # 2 cm

def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0

    return (
        math.degrees(roll),
        math.degrees(pitch),
        math.degrees(yaw)
    )

while True:
    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, rejected = aruco.detectMarkers(
        frame, dictionary, parameters=detector_params
    )

    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)

        for i in range(len(ids)):
            marker_id = int(ids[i][0])

            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                corners[i],
                marker_length,
                cameraMatrix,
                dist
            )

            cv2.drawFrameAxes(
                frame,
                cameraMatrix,
                dist,
                rvec,
                tvec,
                0.02
            )

            # Convert rotation vector to matrix
            R, _ = cv2.Rodrigues(rvec[0][0])

            # Convert to roll pitch yaw
            roll, pitch, yaw = rotationMatrixToEulerAngles(R)

            # Marker center for text placement
            pts = corners[i][0]
            center_x = int(np.mean(pts[:, 0]))
            center_y = int(np.mean(pts[:, 1]))

            # Show marker ID
            cv2.putText(
                frame,
                f"ID:{marker_id}",
                (center_x, center_y - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Yaw:{yaw:.1f}",
                (center_x, center_y + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )



        

    cv2.imshow("ArUco Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()