import cv2
import numpy as np
import sympy as sp
from sympy.parsing.latex import parse_latex
import math

sample_points = 500

plot_width, plot_height = 300, 300
plot_origin_x, plot_origin_y = 500, 500

plot_x_range_min, plot_x_range_max = -10, 10
plot_y_range_min, plot_y_range_max = -10, 10

x_values = np.linspace(plot_x_range_min, plot_y_range_max, sample_points)
y_values_f1 = [0] * sample_points
y_values_f2 = [0] * sample_points

x = sp.symbols('x')
last_accepted_input_f1 = r"f(x)=x"
last_accepted_input_f2 = r"g(x)=x"

color_f1 = (255, 255, 255)
color_f2 = (255, 255, 255)

corner_t_x, corner_t_y, corner_b_x, corner_b_y = 0, 0, 0, 0



def setup_plot(_corner_1_x, _corner_1_y, _corner_2_x, _corner_2_y):
    global plot_width, plot_height, plot_origin_x, plot_origin_y, corner_t_x, corner_t_y, corner_b_x, corner_b_y, last_accepted_input_f1, last_accepted_input_f2

    corner_t_x = _corner_1_x
    corner_t_y = _corner_1_y
    corner_b_x = _corner_2_x
    corner_b_y = _corner_2_y

    plot_width = corner_b_x - corner_t_x
    plot_height = corner_b_y - corner_t_y # Bottom marker has highest y

    plot_origin_x = corner_t_x + plot_width/2
    plot_origin_y = corner_t_y + plot_height/2

    last_accepted_input_f1 = ""
    last_accepted_input_f2 = ""
    



def set_function(LaTeX, color, func_is_f1, debug):
    global change_animation_t, change_animation_done, last_accepted_input_f1, last_accepted_input_f2, y_values_f1, y_values_f2, color_f1, color_f2

    correctedLaTeX = str(LaTeX).replace(r"\chi", r"x").replace(r"\times", r"x").replace(r"X", r"x")

    expected_start = "f ( x ) = " if func_is_f1 else "g ( x ) = "

    if func_is_f1 and last_accepted_input_f1 == correctedLaTeX:
        if debug:
            print("Input as last accepted f1: " + str(correctedLaTeX) + " from: " + str(LaTeX))
        return
    if (not func_is_f1) and last_accepted_input_f2 == correctedLaTeX:
        if debug:
            print("Input as last accepted f2: " + str(correctedLaTeX) + " from: " + str(LaTeX))
        return
    
    if len(correctedLaTeX) <= 10:
        if debug:
            print("Input not long enough: " + str(correctedLaTeX))
        return

    if correctedLaTeX[:10] == expected_start:
        correctedLaTeX = correctedLaTeX[10:]
    else:
        if debug:
            print("Is not a the correct function of x: " + str(expected_start) + " Instead: " + str(correctedLaTeX))
        return
    
    try:
        expr = parse_latex(correctedLaTeX, backend="lark")
        current_function = sp.lambdify(x, expr, "numpy")
        for i in range(sample_points):
            if func_is_f1:
                y_values_f1[i] = current_function(x_values[i])
                if(math.isnan(y_values_f1[i])):
                    y_values_f1[i] = 0
                    if debug:
                        print("Value is nan")
            else:
                y_values_f2[i] = current_function(x_values[i])
                if(math.isnan(y_values_f2[i])):
                    y_values_f2[i] = 0
                    if debug:
                        print("Value is nan")

        if func_is_f1:
            last_accepted_input_f1 = LaTeX
            color_f1 = color
            if debug:
                print("Set function f1: " + str(LaTeX))
        else:
            last_accepted_input_f2 = LaTeX
            color_f2 = color
            if debug:
                print("Set function f2: " + str(LaTeX))
    except Exception as e:
        if debug:
            print("Error with input: " + str(LaTeX) + " And error: " + str(e))


def top_corner_moved_do_recenter(corner_new_x, corner_new_y):
    global plot_origin_x, plot_origin_y, corner_t_x, corner_t_y, plot_width, plot_height

    corner_t_x = corner_new_x
    corner_t_y = corner_new_y

    plot_width = corner_b_x - corner_t_x
    plot_height = corner_b_y - corner_t_y # Bottom marker has highest y

    plot_origin_x = corner_t_x + plot_width/2
    plot_origin_y = corner_t_y + plot_height/2

    #print("pos: " + str(corner_t_x) + " " + str(corner_t_y) + " " + str(corner_b_x) + " " + str(corner_b_y))
    #print("wh: " + str(plot_width) + " " + str(plot_height))


def bottom_corner_moved_do_recenter(corner_new_x, corner_new_y):
    global plot_origin_x, plot_origin_y, corner_b_x, corner_b_y, plot_width, plot_height
    
    corner_b_x = corner_new_x
    corner_b_y = corner_new_y

    plot_width = corner_b_x - corner_t_x
    plot_height = corner_b_y - corner_t_y # Bottom marker has highest y

    plot_origin_x = corner_t_x + plot_width/2
    plot_origin_y = corner_t_y + plot_height/2


def w2p(point, H): # WHITEBOARD TO PROJECTOR
    # Convert point to correct shape for cv2
    pt = np.array([[point]], dtype=np.float32)  # shape (1,1,2)

    # Apply perspective transform
    projected_pt = cv2.perspectiveTransform(pt, H)

    x, y = projected_pt[0, 0]

    return (int(x), int(y))

def normalize_point(point, debug):
    try:
        x, y = point

        if (y > plot_y_range_max or y < plot_y_range_min):
            return None

        x_range = plot_x_range_max - plot_x_range_min
        y_range = plot_y_range_max - plot_y_range_min

        nx = plot_origin_x + (x / x_range) * plot_width
        ny = plot_origin_y - (y / y_range) * plot_height

        return (nx, ny)
    except Exception as e:
        if debug:
            print("Unable to normalize: " + str(point) + " With error: " + str(e))
        return (0, 0)

def draw_ticks(image, H, tick_spacing=1, tick_size=0.02, debug=False):
    # X-axis ticks (along y = 0)
    x = math.ceil(plot_x_range_min)
    while x <= plot_x_range_max:
        p1 = normalize_point((x, -tick_size), debug)
        p2 = normalize_point((x, tick_size), debug)

        if p1 and p2:
            cv2.line(image, w2p(p1, H), w2p(p2, H), (255, 255, 255), 1, cv2.LINE_AA)
            text = str(x)
            text_point = normalize_point((x, -4*tick_size), debug)
            cv2.putText(image, text, w2p(text_point, H), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1, cv2.LINE_AA)

        x += tick_spacing

    # Y-axis ticks (along x = 0)
    y = math.ceil(plot_y_range_min)
    while y <= plot_y_range_max:
        p1 = normalize_point((-tick_size, y), debug)
        p2 = normalize_point((tick_size, y), debug)

        if p1 and p2:
            cv2.line(image, w2p(p1, H), w2p(p2, H), (255, 255, 255), 1, cv2.LINE_AA)
            text = str(y)
            text_point = normalize_point((2*tick_size, y), debug)
            cv2.putText(image, text, w2p(text_point, H), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1, cv2.LINE_AA)

        y += tick_spacing

def draw_grid(image, H, color, line_spacing=1, debug=False):
    x = math.ceil(plot_x_range_min)
    while x <= plot_x_range_max:
        p1 = normalize_point((x, plot_y_range_min), debug)
        p2 = normalize_point((x, plot_y_range_max), debug)

        if p1 and p2:
            cv2.line(image, w2p(p1, H), w2p(p2, H), color, 1, cv2.LINE_AA)

        x += line_spacing

    y = math.ceil(plot_y_range_min)
    while y <= plot_y_range_max:
        p1 = normalize_point((plot_x_range_min, y), debug)
        p2 = normalize_point((plot_x_range_max, y), debug)

        if p1 and p2:
            cv2.line(image, w2p(p1, H), w2p(p2, H), color, 1, cv2.LINE_AA)

        y += line_spacing

def draw_plot(image, H, debug):
    global initial_animation_done, initial_animation_t, change_animation_done, change_animation_t
    # Draw Axis
    axis_h_left = normalize_point((plot_x_range_min, 0), debug)
    axis_h_right = normalize_point((plot_x_range_max, 0), debug)
    axis_v_bottom = normalize_point((0, plot_y_range_min), debug)
    axis_v_top = normalize_point((0, plot_y_range_max), debug)
    cv2.line(image, w2p(axis_h_left, H), w2p(axis_h_right, H), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(image, w2p(axis_v_bottom, H), w2p(axis_v_top, H), (255, 255, 255), 1, cv2.LINE_AA)

    draw_grid(image, H, (100, 100, 100), line_spacing=1, debug=debug)
    draw_ticks(image, H, tick_spacing=1, tick_size=0.2, debug=debug)

    points_to_draw_f1 = []
    points_to_draw_f2 = []

    for i in range(sample_points):
        point_f1 = normalize_point((x_values[i],y_values_f1[i]), debug)
        if point_f1 is not None:
            points_to_draw_f1.append(w2p(point_f1, H))

        point_f2 = normalize_point((x_values[i],y_values_f2[i]), debug)
        if point_f2 is not None:
            points_to_draw_f2.append(w2p(point_f2, H))
        
    if len(points_to_draw_f1) > 1:
        points = np.array(points_to_draw_f1, dtype=np.int32)
        points = [points]
        color = color_f1
        cv2.polylines(image, points, False, color, 1, cv2.LINE_AA)

    if len(points_to_draw_f2) > 1:
        points = np.array(points_to_draw_f2, dtype=np.int32)
        points = [points]
        color = color_f2
        cv2.polylines(image, points, False, color, 1, cv2.LINE_AA)