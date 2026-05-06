import cv2
import cv2.aruco as aruco
import numpy as np

def nothing(x):
    pass

def board2proj(camproj_H, camboard_H, point): # https://chatgpt.com/share/69f6075f-8a4c-83eb-8ab9-b34097ed70ac
    H = get_board2proj_H(camproj_H, camboard_H)

    # Convert point to correct shape for cv2
    pt = np.array([[point]], dtype=np.float32)  # shape (1,1,2)

    # Apply perspective transform
    projected_pt = cv2.perspectiveTransform(pt, H)

    return projected_pt[0][0]

def get_board2proj_H(camproj_H, camboard_H):
    # Invert camboard homography (board <- camera → camera <- board)
    camboard_inv = np.linalg.inv(camboard_H)

    # Compose transformations: board -> camera -> projector
    H = camproj_H @ camboard_inv

    return H

def debug_board_grid(camproj_H, camboard_H, image, size, animation_t):
    w, h = size
    line_spacing = 50
    for x in range(w):
        if x % line_spacing == 0:
            # line() skal bruge int værdier i en tuple. map(int, tuple) bruger int() på alle værdierne og returnerer en iterator. tuple() laver det om til en tuple igen
            vertical_point_top =        tuple(map(int, board2proj(camproj_H, camboard_H, (x, 0))))
            vertical_point_bottom =     tuple(map(int, board2proj(camproj_H, camboard_H, (x, animation_t * h))))
            cv2.line(image, vertical_point_top,    vertical_point_bottom,   (255, 255, 255), 1, cv2.LINE_AA)
    
    for y in range(h):
        if y % line_spacing == 0:
            # line() skal bruge int værdier i en tuple. map(int, tuple) bruger int() på alle værdierne og returnerer en iterator. tuple() laver det om til en tuple igen
            horizontal_point_left =     tuple(map(int, board2proj(camproj_H, camboard_H, (0, y))))
            horizontal_point_right =    tuple(map(int, board2proj(camproj_H, camboard_H, (animation_t * w, y))))
            cv2.line(image, horizontal_point_left,  horizontal_point_right, (255, 255, 255), 1, cv2.LINE_AA)

def generate_projector_screen():
    global proj_marker_image, proj_marker_positions

    proj_marker_positions = { # Top Left corner of the markers
        10: (np.clip(marker_spacing_x + marker_offset_x, 0, screen_w)                           , np.clip(screen_h - marker_spacing_y - marker_size + marker_offset_y, 0, screen_h)), # LB
        11: (np.clip(marker_spacing_x + marker_offset_x, 0, screen_w)                           , np.clip(marker_spacing_y + marker_offset_y, 0, screen_h)),                          # LT
        12: (np.clip(screen_w - marker_spacing_x - marker_size + marker_offset_x, 0, screen_w)  , np.clip(marker_spacing_y + marker_offset_y, 0, screen_h)),                          # RT
        13: (np.clip(screen_w - marker_spacing_x - marker_size + marker_offset_x, 0, screen_w)  , np.clip(screen_h - marker_spacing_y - marker_size + marker_offset_y, 0, screen_h)), # RB
    }

    proj_marker_image = np.ones((screen_h, screen_w), dtype=np.uint8) * 255

    for i, marker in enumerate(proj_markers): # enumerate skaber indicies sammen med værdierne
        proj_marker_image[proj_marker_positions[i+10][1]:proj_marker_positions[i+10][1]+marker_size, proj_marker_positions[i+10][0]:proj_marker_positions[i+10][0]+marker_size] = marker

    return proj_marker_image

def dist(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def try_generate_homographies(all_camproj_cam_points, all_camproj_proj_points, all_camboard_cam_points):
    global measured_whiteboard_width, measured_whiteboard_height
    if len(all_camproj_cam_points) == 16 and len(all_camboard_cam_points) == 4:
        all_camproj_cam_points = np.array(all_camproj_cam_points, dtype=np.float32)#Laver til numpy matrix med form (N, 2)
        all_camproj_proj_points = np.array(all_camproj_proj_points, dtype=np.float32)

        all_camboard_cam_points = np.array(all_camboard_cam_points, dtype=np.float32)

        measured_whiteboard_height = int((dist(all_camboard_cam_points[0], all_camboard_cam_points[1])+
                                          dist(all_camboard_cam_points[2], all_camboard_cam_points[3])) / 2)
        measured_whiteboard_width  = int((dist(all_camboard_cam_points[1], all_camboard_cam_points[2])+
                                          dist(all_camboard_cam_points[3], all_camboard_cam_points[0])) / 2)
        #print(str(dist(all_camboard_cam_points[0], all_camboard_cam_points[1])) + " " + str(dist(all_camboard_cam_points[2], all_camboard_cam_points[3])) + " " + str(dist(all_camboard_cam_points[1], all_camboard_cam_points[2])) + " " + str(dist(all_camboard_cam_points[3], all_camboard_cam_points[0])))
        #print(str(measured_whiteboard_height) + " " + str(measured_whiteboard_width))

        board_pts = np.array([
            [0, measured_whiteboard_height],
            [0, 0],
            [measured_whiteboard_width, 0],
            [measured_whiteboard_width, measured_whiteboard_height]
        ], dtype=np.float32)

        homography_camproj, _ = cv2.findHomography(all_camproj_cam_points, all_camproj_proj_points)
        homography_camboard, _ = cv2.findHomography(all_camboard_cam_points, board_pts)

        return True, homography_camproj, homography_camboard
    
    return False, None, None
    
def detect_markers(undistorted):
    gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
    #clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    #gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(undistorted, corners, ids)

        all_camproj_cam_points = []
        all_camproj_proj_points = []

        all_camboard_cam_points = []

        # Pair ids with corners and sort by id
        markers = sorted(
            zip(ids.flatten(), corners),
            key=lambda x: x[0]
        )

        for marker_id, corner in markers:
            if marker_id in proj_marker_positions:
                cam_pts = corner[0].astype(np.float32)

                proj_x, proj_y = proj_marker_positions[marker_id]

                proj_pts = np.array([
                    [proj_x, proj_y],                               # LT
                    [proj_x + marker_size, proj_y],                 # RT
                    [proj_x + marker_size, proj_y + marker_size],   # RB
                    [proj_x, proj_y + marker_size]                  # LB
                ], dtype=np.float32)

                all_camproj_cam_points.extend(cam_pts)
                all_camproj_proj_points.extend(proj_pts)
                
            if marker_id in board_marker_positions:
                cam_pts = corner[0].astype(np.float32)

                marker_center = np.mean(cam_pts, axis=0)

                all_camboard_cam_points.append(marker_center)
    
    return all_camproj_cam_points, all_camproj_proj_points, all_camboard_cam_points




screen_w = 1920
screen_h = 1080
marker_spacing_x = 750
marker_spacing_y = 400
marker_offset_x = 0
marker_offset_y = 50
marker_size = 100


measured_whiteboard_width = -1
measured_whiteboard_height = -1


dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

markerLB = aruco.generateImageMarker(dictionary, 10, marker_size, 1)
markerLT = aruco.generateImageMarker(dictionary, 11, marker_size, 1)
markerRT = aruco.generateImageMarker(dictionary, 12, marker_size, 1)
markerRB = aruco.generateImageMarker(dictionary, 13, marker_size, 1)
proj_markers = [markerLB, markerLT, markerRT, markerRB]

board_marker_positions = {
    0: (-1, -1), # LB
    1: (-1, -1), # LT
    2: (-1, -1), # RT
    3: (-1, -1)  # RB
}

parameters = aruco.DetectorParameters()
parameters.adaptiveThreshWinSizeMin = 3
parameters.adaptiveThreshWinSizeMax = 35
parameters.adaptiveThreshWinSizeStep = 4
parameters.adaptiveThreshConstant = 5
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
parameters.minMarkerPerimeterRate = 0.005  # default ~0.03
detector = aruco.ArucoDetector(dictionary, parameters)

do_calibration = True