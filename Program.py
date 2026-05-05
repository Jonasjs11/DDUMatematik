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

marker_7_was_here = False
drawing_plot = False

if __name__ == "__main__":
    StartCalib.setup_windows()
    HMER.setup_processor()
    
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
            cv2.imshow("Camera", undistorted)

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
            projector_image = np.zeros((StartCalib.screen_h, StartCalib.screen_w, 3), dtype=np.uint8) # 3 is 3 color channel
            if drawing_plot == False:
                cv2.imshow("Projector", projector_image)
            drawing_plot = False

            height = StartCalib.measured_whiteboard_width # Swapped for some weird reason
            width = StartCalib.measured_whiteboard_height # Swapped for some weird reason
            warped = cv2.warpPerspective(frame, homography_camsnippet, (width, height))
            warped_rotated = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)

            #Swap back
            temp_height = height
            height = width
            width = temp_height

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
            grouped_boxes = HelperFunc.group_contours_to_boxes(filtered_contours, horizontal_thresh=100, vertical_thresh=20, pad=10)
            HelperFunc.draw_boxes(warped_rotated, grouped_boxes, color=(255, 255, 0), thickness=1, show_index=True)

            gray = cv2.cvtColor(warped_rotated, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            corners, ids, rejected = StartCalib.detector.detectMarkers(gray)

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(warped_rotated, corners, ids)

                if 6 in ids.flatten(): # flatten() laver en liste med elementer, i stedet for et array med lister med størrelser 1
                    marker_box = HelperFunc.get_marker_box(corners, ids, 6)
                    HelperFunc.draw_boxes(warped_rotated, [marker_box], color=(255, 255, 255), thickness=1, show_index=True)
                    for box in grouped_boxes:
                       if HelperFunc.box_contains_marker(box, marker_box):
                            x1, y1, x2, y2 = box
                            mx1, my1, mx2, my2 = marker_box
                            marker_extra_padding = 15
                            crop = warped_rotated[y1:y2, mx2+marker_extra_padding:x2]
                            cv2.imshow("matched_contour 6", crop)
                            #print(HMER.doHMER(crop))

                if 7 in ids.flatten(): # flatten() laver en liste med elementer, i stedet for et array med lister med størrelser 1
                    marker_box = HelperFunc.get_marker_box(corners, ids, 7)
                    HelperFunc.draw_boxes(warped_rotated, [marker_box], color=(255, 255, 255), thickness=1, show_index=True)
                    for box in grouped_boxes:
                        if HelperFunc.box_contains_marker(box, marker_box):
                            x1, y1, x2, y2 = box
                            mx1, my1, mx2, my2 = marker_box
                            marker_extra_padding = 15
                            crop = warped_rotated[y1:y2, mx2+marker_extra_padding:x2]
                            if crop is not None and crop.size > 0:
                                cv2.imshow("matched_contour 7", crop)
                            else:
                                print("Marker 7 area is too small")
                                break
                            
                            dominant_color = HelperFunc.get_dominant_non_white_color(crop,
                                    k=3,
                                    sat_thresh=30,   # lower = keeps lighter reds
                                    val_thresh=250,   # high so bright marker isn't removed
                                    sat_boost=4.0,
                                    debug=True
                                )
                            print(dominant_color)

                            LaTeX = HMER.doHMER(crop)
                            if not marker_7_was_here:
                                OpenCVPlot.setup_plot(100, 100, 500, 500, LaTeX)
                            else:
                                OpenCVPlot.set_function(LaTeX)
                            
                            if 4 in ids.flatten():
                                center = HelperFunc.get_box_center(HelperFunc.get_marker_box(corners, ids, 4))
                                x, y = center

                                #normalize to (0,0) to (1,1)
                                x = x/width
                                y = 1 - y/height

                                OpenCVPlot.bottom_corner_moved_do_recenter(x, y)

                            if 5 in ids.flatten():
                                center = HelperFunc.get_box_center(HelperFunc.get_marker_box(corners, ids, 5))
                                x, y = center

                                #normalize to (0,0) to (1,1)
                                x = x/width
                                y = 1 - y/height

                                OpenCVPlot.top_corner_moved_do_recenter(x, y)


                            H = StartCalib.get_board2proj_H(homography_camproj, homography_camboard)
                            #H = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]], dtype=np.float32)
                            OpenCVPlot.draw_plot(projector_image, H, 1)
                            cv2.imshow("Projector", projector_image)
                            drawing_plot = True
                    marker_7_was_here = True
                else:
                    marker_7_was_here = False





            #cv2.imshow("threshold", thresh)
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