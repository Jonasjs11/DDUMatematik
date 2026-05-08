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
cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2) # Adressen/identifikationnen af kameraet. Alternativ er cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")) # Tving den til at bruge MJPG som virker bedst
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4096) # Horizontal resolution. Relevante muligheder er 1920 eller 4096
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160) # Vertikal resolution. Relevante muligheder er 1080 eller 2160
cap.set(cv2.CAP_PROP_FPS, 5) # Max framerate
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Den gemmer kun på det nyeste billede

# Load kamera-barrel-distortion-kalibrations-filer
fs = cv2.FileStorage("calibration.yaml", cv2.FILE_STORAGE_READ)
cameraMatrix = fs.getNode("cameraMatrix").mat()
dist = fs.getNode("distCoeffs").mat()
fs.release()

calibration_succes_grid_animation_t = 0
calibration_succes_grid_animation_speed = 5

run_main_program = False

function_to_draw = False

debug = False


def setup_windows():

    cv2.namedWindow('Projector', cv2.WINDOW_NORMAL)
    cv2.moveWindow('Projector', 2500, 0)
    cv2.resizeWindow('Projector', 1920, 1080)

    if debug:
        cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Camera', 1920, 1080)
        cv2.moveWindow('Camera', -2000, 2000)

if __name__ == "__main__":
    setup_windows()
    HMER.setup_processor()
    OpenCVPlot.setup_plot(100, 100, 500, 500)
    
    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break

        ret, frame = cap.read() # Forsøg at læse billede fra kamera
        if not ret: # Hvis der ikke kunne læses et billede, prøv igen i ny iteration
            continue

        undistorted = cv2.undistort(frame, cameraMatrix, dist) # Fjern barrel distortion
        if debug:
            cv2.imshow("Camera", undistorted)

        if StartCalib.do_calibration: # Homografierne skal findes
            projector_image = StartCalib.generate_projector_screen().copy()

            cv2.imshow("Projector", projector_image)

            all_camproj_cam_points, all_camproj_proj_points, all_camboard_cam_points = StartCalib.detect_markers(undistorted)
            if debug:
                cv2.imshow("Camera", undistorted)

            success, homography_camproj, homography_camboard = StartCalib.try_generate_homographies(all_camproj_cam_points, all_camproj_proj_points, all_camboard_cam_points)
            if success:
                StartCalib.do_calibration = False
                projector_image = np.zeros((StartCalib.screen_h, StartCalib.screen_w), dtype=np.uint8)
                board2proj_H = StartCalib.get_board2proj_H(homography_camproj, homography_camboard)
        if(StartCalib.do_calibration == False and run_main_program == False):# Vis test grid på tavlen
            StartCalib.debug_board_grid(homography_camproj, homography_camboard, projector_image, [StartCalib.measured_whiteboard_width, StartCalib.measured_whiteboard_height], np.clip(calibration_succes_grid_animation_t, 0, 1))
            cv2.imshow("Projector", projector_image)

            calibration_succes_grid_animation_t += 0.02 * calibration_succes_grid_animation_speed

            if(calibration_succes_grid_animation_t >= 1.5):
                projector_image = np.zeros((StartCalib.screen_h, StartCalib.screen_w), dtype=np.uint8)
                cv2.imshow("Projector", projector_image)
            if(calibration_succes_grid_animation_t >= 2.5):
                run_main_program = True
        if run_main_program:
            projector_image = np.zeros((StartCalib.screen_h, StartCalib.screen_w, 3), dtype=np.uint8) # Reset med projektor-billedet til sort, hvor 3 er for 3 color channels
            function_to_draw = False

            # Nu skal whiteboardet klippes ud af kamera billedet
            width = StartCalib.measured_whiteboard_width
            height = StartCalib.measured_whiteboard_height
            whiteboard_image = cv2.warpPerspective(frame, homography_camboard, (width, height))

            # Nu skal alt det der er på tavlen findes, først laver vi et "binært" threshold billede, som viser hvad der er tavle, og hvad der ikke er. Derefter skal øerne identificeres og grupperes.
            gray = cv2.cvtColor(whiteboard_image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(src=gray, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY_INV, blockSize=41, C=5) # Blocksize er område for threshold (kun ulige tal), C er en konstant som trækkes fra gennemsnittet
            thresh = cv2.dilate(thresh, kernel=np.ones((3,3), np.uint8), iterations=1) # connect text lightly

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Start med at finde alle øer i thresholden
            filtered_contours = HelperFunc.filter_contours_by_size(contours, minimum_area=75, minimum_wh=5) # Filtrer alle fra, der er for lille, og bare er noise/støj
            grouped_boxes = HelperFunc.group_contours_to_boxes(filtered_contours, horizontal_thresh=50, vertical_thresh=20, pad=10) # Grupper contours hvis de er inden for thresholdværdierne til boxes
            if debug:
                HelperFunc.draw_boxes(whiteboard_image, grouped_boxes, color=(255, 255, 0), thickness=1, show_index=True) # Debug, tegn de fundne kasser

            # Nu skal de markørerne til brugerinteraktion findes
            gray = cv2.cvtColor(whiteboard_image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            corners, ids, rejected = StartCalib.detector.detectMarkers(gray)

            if ids is not None:
                if debug:
                    cv2.aruco.drawDetectedMarkers(whiteboard_image, corners, ids)

                if 4 in ids.flatten():
                    center = HelperFunc.get_box_center(HelperFunc.get_marker_box(corners, ids, 4))
                    x, y = center
                    OpenCVPlot.bottom_corner_moved_do_recenter(x, y)

                if 5 in ids.flatten():
                    center = HelperFunc.get_box_center(HelperFunc.get_marker_box(corners, ids, 5))
                    x, y = center
                    OpenCVPlot.top_corner_moved_do_recenter(x, y)

                extra_padding_right_of_marker = 15
                if 7 in ids.flatten(): # flatten() laver en liste med elementer, i stedet for et array med lister med størrelser 1
                    marker_box = HelperFunc.get_marker_box(corners, ids, 7)
                    if debug:
                        HelperFunc.draw_boxes(whiteboard_image, [marker_box], color=(255, 255, 255), thickness=1, show_index=True)
                    for box in grouped_boxes:
                        if HelperFunc.box_contains_marker(box, marker_box):
                            x1, y1, x2, y2 = box
                            mx1, my1, mx2, my2 = marker_box
                            crop = whiteboard_image[y1:y2, mx2+extra_padding_right_of_marker:x2]
                            if crop is not None and crop.size > 0:
                                if debug:
                                    cv2.imshow("matched_contour 7", crop)
                            else:
                                print("Marker 7 area is too small")
                                break

                            LaTeX = HMER.doHMER(crop)
                            OpenCVPlot.set_function(LaTeX, (255, 127, 0), func_is_f1=True, debug=debug)
                            function_to_draw = True
                
                if 6 in ids.flatten(): # flatten() laver en liste med elementer, i stedet for et array med lister med størrelser 1
                    marker_box = HelperFunc.get_marker_box(corners, ids, 6)
                    if debug:
                        HelperFunc.draw_boxes(whiteboard_image, [marker_box], color=(255, 255, 255), thickness=1, show_index=True)
                    for box in grouped_boxes:
                        if HelperFunc.box_contains_marker(box, marker_box):
                            x1, y1, x2, y2 = box
                            mx1, my1, mx2, my2 = marker_box
                            crop = whiteboard_image[y1:y2, mx2+extra_padding_right_of_marker:x2]
                            if crop is not None and crop.size > 0:
                                if debug:
                                    cv2.imshow("matched_contour 6", crop)
                            else:
                                print("Marker 6 area is too small")
                                break

                            LaTeX = HMER.doHMER(crop)
                            OpenCVPlot.set_function(LaTeX, (0, 0, 255), func_is_f1=False, debug=debug)
                            function_to_draw = True
                
                if function_to_draw:
                    OpenCVPlot.draw_plot(projector_image, board2proj_H, debug=debug)
                    cv2.imshow("Projector", projector_image)

            if debug:
                cv2.imshow("Whiteboard", whiteboard_image)

            if key == ord('d'): # Hvis der trykkes "d" skal homografierne findes igen, hvor alt først skal resettes
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