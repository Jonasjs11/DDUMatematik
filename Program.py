# Biblioteker
import sympy as sp
from sympy.parsing.latex import parse_latex
import numpy as np
import cv2
import time

# Egne filer
import HelperFunc
import StartCalib
import HMER
import OpenCVPlot


# Start kamera
cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2) # cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4096)#1920 eller 4096
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)#1080 eller 2160
cap.set(cv2.CAP_PROP_FPS, 5)

# Load kamera filer
fs = cv2.FileStorage("calibration.yaml", cv2.FILE_STORAGE_READ)
cameraMatrix = fs.getNode("cameraMatrix").mat()
dist = fs.getNode("distCoeffs").mat()
fs.release()

calibration_succes_grid_animation_t = 0
calibration_succes_grid_animation_speed = 5

run_main_program = False

if __name__ == "__main__":
    StartCalib.setup_windows()
    #HMER.setup_processor()
    
    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break



        ret, frame = cap.read()
        if not ret:
            continue

        undistorted = cv2.undistort(frame, cameraMatrix, dist)

        cv2.imshow("Camera", undistorted)



        if StartCalib.do_calibration:
            projector_image = StartCalib.generate_projector_screen().copy()

            cv2.imshow("Projector", projector_image)

            all_camproj_cam_points, all_camproj_proj_points, all_camboard_cam_points, all_camboard_board_points = StartCalib.detect_markers(undistorted)
                
            success, homography_camproj, homography_camboard, homography_camsnippet = StartCalib.try_generate_homographies(all_camproj_cam_points, all_camproj_proj_points, all_camboard_cam_points, all_camboard_board_points)
            if success:
                StartCalib.do_calibration = False
                projector_image = np.zeros((StartCalib.screen_h, StartCalib.screen_w), dtype=np.uint8)
        if(StartCalib.do_calibration == False):
            StartCalib.debug_board_grid(homography_camproj, homography_camboard, projector_image, np.clip(calibration_succes_grid_animation_t, 0, 1))
            cv2.imshow("Projector", projector_image)

            calibration_succes_grid_animation_t += 0.02 * calibration_succes_grid_animation_speed

            if(calibration_succes_grid_animation_t >= 1.5):
                run_main_program = True
        if(run_main_program):
            projector_image = np.zeros((StartCalib.screen_h, StartCalib.screen_w), dtype=np.uint8)
            cv2.imshow("Projector", projector_image)

            width = StartCalib.measured_whiteboard_height # Swapped for some weird reason
            height = StartCalib.measured_whiteboard_width # Swapped for some weird reason
            warped = cv2.warpPerspective(frame, homography_camsnippet, (width, height))
            warped_rotated = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)

            gray = cv2.cvtColor(warped_rotated, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            gray = cv2.GaussianBlur(src=gray, ksize=(3,3), sigmaX=0, sigmaY=0)
            thresh = cv2.adaptiveThreshold(src=gray, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY_INV, blockSize=41, C=5) # Blocksize er område for threshold (kun ulige tal), C er en konstant som trækkes fra gennemsnittet

            # remove tiny noise but keep strokes
            kernel = np.ones((2,2), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

            # connect text lightly
            kernel = np.ones((3,3), np.uint8)
            thresh = cv2.dilate(thresh, kernel, iterations=1)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_contours = HelperFunc.filter_contours_by_size(contours, minimum_area=75, minimum_wh=5)
            cropped = HelperFunc.find_grouped_contours_and_crop(warped_rotated, filtered_contours, horizontal_thresh=100, vertical_thresh=20, pad=20)

            gray = cv2.cvtColor(warped_rotated, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            corners, ids, rejected = StartCalib.detector.detectMarkers(gray)

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(warped_rotated, corners, ids)

                if 6 in ids.flatten(): # flatten() laver en liste med elementer, i stedet for et array med lister med størrelser 1
                    marker_box = HelperFunc.get_marker_box(corners, ids, 6)
                    for contour in filtered_contours:
                        if HelperFunc.contour_contains_marker(contour, marker_box):
                            x, y, w, h = cv2.boundingRect(contour)
                            crop = warped_rotated[y:y+h, x:x+w]
                            cv2.imshow("matched_contour 6", crop)
                            #print(HMER.doHMER(crop))

                if 7 in ids.flatten(): # flatten() laver en liste med elementer, i stedet for et array med lister med størrelser 1
                    marker_box = HelperFunc.get_marker_box(corners, ids, 7)
                    for contour in filtered_contours:
                        if HelperFunc.contour_contains_marker(contour, marker_box):
                            x, y, w, h = cv2.boundingRect(contour)
                            crop = warped_rotated[y:y+h, x:x+w]
                            cv2.imshow("matched_contour 7", crop)
                            #print(HMER.doHMER(crop))



            cv2.imshow("threshold", thresh)
            cv2.imshow("warped", warped_rotated)

            if key == ord('d'):
                calibration_succes_grid_animation_t = 0
                StartCalib.do_calibration = True
                run_main_program = False
                all_camproj_cam_points = []
                all_camproj_proj_points = []
                all_camboard_cam_points = []
                all_camboard_board_points = []
                homography_camproj = None
                homography_camboard = None
                frame = None
                undistorted = None


    
    cap.release()
    cv2.destroyAllWindows()