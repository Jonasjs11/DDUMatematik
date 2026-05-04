import cv2

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

def find_grouped_contours_and_crop(
    frame,
    contours,
    horizontal_thresh=40,
    vertical_thresh=80,
    pad=20
):
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
            return cv2.boundingRect(pts)

    print("Unable to find marker box " + str(target_id))
    return None

def contour_contains_marker(contour, marker_box):
    if marker_box is None:
        return False

    x, y, w, h = cv2.boundingRect(contour)
    mx, my, mw, mh = marker_box

    return (mx >= x             and my >= y and
            mx + mw <= x + w    and my + mh <= y + h)
