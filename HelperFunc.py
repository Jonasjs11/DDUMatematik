import cv2
import numpy as np

def find_sorrounding_contour_and_crop(frame, contours):
    xs = []
    ys = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        xs.append(x)
        xs.append(x+w)
        ys.append(y)
        ys.append(y+h)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        #cv2.rectangle(thresh, (x,y), (x+w,y+h), (255,0,0), 2)

    if contours:
        x1 = min(xs)
        x2 = max(xs)
        y1 = min(ys)
        y2 = max(ys)

        pad = 20
        x1 = max(x1 - pad, 0)
        y1 = max(y1 - pad, 0)
        x2 = min(x2 + pad, frame.shape[1])
        y2 = min(y2 + pad, frame.shape[0])

        cropped = frame[y1:y2, x1:x2]
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        #cv2.rectangle(thresh, (x1,y1), (x2,y2), (0,255,0), 2)
    else:
        cropped = frame

    return cropped

def group_contours_to_boxes(contours, horizontal_thresh=40, vertical_thresh=80, pad=20):
    boxes = []

    # Convert contours to bounding boxes
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        boxes.append([x, y, x + w, y + h])

    merged = True
    while merged:
        merged = False
        new_boxes = []

        while boxes:
            x1, y1, x2, y2 = boxes.pop(0)
            merged_box = [x1, y1, x2, y2]

            i = 0
            while i < len(boxes):
                bx1, by1, bx2, by2 = boxes[i]

                dx, dy = rect_distance(merged_box, boxes[i])

                if dx < horizontal_thresh and dy < vertical_thresh:
                    merged_box = [
                        min(merged_box[0], bx1),
                        min(merged_box[1], by1),
                        max(merged_box[2], bx2),
                        max(merged_box[3], by2),
                    ]

                    boxes.pop(i)
                    merged = True
                else:
                    i += 1

            new_boxes.append(merged_box)

        boxes = new_boxes

    # Apply padding AFTER merging
    padded_boxes = []
    for x1, y1, x2, y2 in boxes:
        padded_boxes.append([
            x1 - pad,
            y1 - pad,
            x2 + pad,
            y2 + pad
        ])

    return padded_boxes

def boxes_to_contours(boxes):
    contours = []

    for x1, y1, x2, y2 in boxes:
        contour = np.array([
            [[x1, y1]],
            [[x2, y1]],
            [[x2, y2]],
            [[x1, y2]]
        ], dtype=np.int32)

        contours.append(contour)

    return contours

def find_grouped_contours_and_crop(frame, contours, horizontal_thresh=40, vertical_thresh=80, pad=20):
    boxes = []

    # Get bounding boxes
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        boxes.append([x, y, x + w, y + h])

    merged = True
    while merged:
        merged = False
        new_boxes = []

        while boxes:
            box = boxes.pop(0)
            x1, y1, x2, y2 = box

            merged_box = box

            i = 0
            while i < len(boxes):
                bx1, by1, bx2, by2 = boxes[i]

                # compute gap between boxes
                horizontal_gap = min(abs(bx1 - x2), abs(x1 - bx2))
                vertical_gap = min(abs(by1 - y2), abs(y1 - by2))

                # merge if close enough
                dx, dy = rect_distance(merged_box, boxes[i])

                if dx < horizontal_thresh and dy < vertical_thresh:
                    merged_box = [
                        min(x1, bx1),
                        min(y1, by1),
                        max(x2, bx2),
                        max(y2, by2),
                    ]

                    boxes.pop(i)
                    merged = True
                    x1, y1, x2, y2 = merged_box
                else:
                    i += 1

            new_boxes.append(merged_box)

        boxes = new_boxes

    crops = []

    for (x1, y1, x2, y2) in boxes:
        x1 = max(x1 - pad, 0)
        y1 = max(y1 - pad, 0)
        x2 = min(x2 + pad, frame.shape[1])
        y2 = min(y2 + pad, frame.shape[0])

        crop = frame[y1:y2, x1:x2]
        crops.append(crop)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return crops

def filter_contours_by_size(contours, minimum_area=75, minimum_wh=5):
    filtered_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        # --- HARD FILTERS ---
        if area < minimum_area:          # remove tiny noise (tune: 30–200)
            continue
        if w < minimum_wh or h < minimum_wh:     # remove tiny boxes
            continue

        filtered_contours.append(cnt)
    return filtered_contours

def rect_distance(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB

    # horizontal gap
    if ax2 < bx1:
        dx = bx1 - ax2
    elif bx2 < ax1:
        dx = ax1 - bx2
    else:
        dx = 0  # overlap

    # vertical gap
    if ay2 < by1:
        dy = by1 - ay2
    elif by2 < ay1:
        dy = ay1 - by2
    else:
        dy = 0  # overlap

    return dx, dy

def get_marker_box(corners, ids, target_id):
    for i, marker_id in enumerate(ids.flatten()):
        if marker_id == target_id:
            pts = corners[i][0].astype(int)

            x, y, w, h = cv2.boundingRect(pts)

            # convert to x1, y1, x2, y2
            return [x, y, x + w, y + h]

    print(f"Unable to find marker box {target_id}")
    return None

def get_box_center(box):
    x1, y1, x2, y2 = box
    return [(x1+x2)/2, (y1+y2)/2]

def box_contains_marker(box, marker_box):
    if marker_box is None:
        return False

    x1, y1, x2, y2 = box
    mx1, my1, mx2, my2 = marker_box

    return (mx1 >= x1    and my1 >= y1 and
            mx2 <= x2    and my2 <= y2)

def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2, show_index=False):
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # Optional index label
        if show_index:
            cv2.putText(
                image,
                str(i),
                (x1, max(y1 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA
            )

import cv2
import numpy as np

def get_dominant_non_white_color(
    crop,
    k=3,
    sat_thresh=50,
    val_thresh=245,
    sat_boost=255.0,   # <-- key parameter
    debug=False
):
    if crop is None or crop.size == 0:
        return None

    # --- Step 1: Convert to HSV ---
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)

    # --- Step 2: Boost saturation ---
    hsv[:, :, 1] *= sat_boost
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

    hsv = hsv.astype(np.uint8)

    # --- Step 3: Mask non-white pixels ---
    mask = (hsv[:, :, 1] > sat_thresh) & (hsv[:, :, 2] < val_thresh)

    pixels = crop[mask]

    if len(pixels) == 0:
        return None

    # --- Step 4: K-means clustering ---
    pixels = pixels.reshape(-1, 3).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        10,
        1.0
    )

    _, labels, centers = cv2.kmeans(
        pixels,
        k,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    counts = np.bincount(labels.flatten())
    dominant = centers[np.argmax(counts)]

    dominant_color = tuple(int(c) for c in dominant)

    # --- Debug ---
    if debug:
        cv2.imshow("crop", crop)

        sat_vis = hsv[:, :, 1]
        cv2.imshow("boosted_saturation", sat_vis)

        mask_vis = (mask.astype(np.uint8) * 255)
        cv2.imshow("mask", mask_vis)

        filtered_vis = np.zeros_like(crop)
        filtered_vis[mask] = crop[mask]
        cv2.imshow("filtered_pixels", filtered_vis)

    return dominant_color