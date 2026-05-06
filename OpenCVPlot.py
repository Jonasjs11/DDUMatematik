import cv2
import numpy as np
import sympy as sp
from sympy.parsing.latex import parse_latex
import math

sample_points = 100

plot_width, plot_height = 300, 300
plot_origin_x, plot_origin_y = 500, 500

plot_x_range_min, plot_x_range_max = -10, 10
plot_y_range_min, plot_y_range_max = -10, 10

x_values = np.linspace(plot_x_range_min, plot_y_range_max, sample_points)
y_values = [0] * sample_points

initial_animation_t = 0
initial_animation_duration = 1
initial_animation_done = False

change_animation_t = 0
change_animation_duration = 1
change_animation_done = False
change_animation_old_y_points = [0] * sample_points
change_animation_new_y_points = [0] * sample_points

x = sp.symbols('x')
current_function = sp.lambdify(x, parse_latex(r"x", backend="lark"), "numpy")
last_accepted_input = r"f(x)=x"

corner_t_x, corner_t_y, corner_b_x, corner_b_y = 0, 0, 0, 0



def setup_plot(_corner_1_x, _corner_1_y, _corner_2_x, _corner_2_y, LaTeX):
    global plot_width, plot_height, plot_origin_x, plot_origin_y, corner_t_x, corner_t_y, corner_b_x, corner_b_y, initial_animation_t, initial_animation_done, last_accepted_input

    corner_t_x = _corner_1_x
    corner_t_y = _corner_1_y
    corner_b_x = _corner_2_x
    corner_b_y = _corner_2_y

    #plot_width = np.abs(corner_2_x - corner_1_x)
    #plot_height = np.abs(corner_2_y - corner_1_y)
    #plot_origin_x = np.mean([corner_1_x, corner_2_x])
    #plot_origin_y = np.mean([corner_1_y, corner_2_y])

    initial_animation_t = 0
    initial_animation_done = False

    last_accepted_input = ""

    set_function(LaTeX)
    



def set_function(LaTeX):
    global current_function, change_animation_t, change_animation_done, change_animation_old_y_points, change_animation_new_y_points, last_accepted_input

    correctedLaTeX = str(LaTeX).replace(r"\chi", r"x").replace(r"\times", r"x").replace(r"X", r"x")

    if correctedLaTeX[:10] == "f ( x ) = ":
        correctedLaTeX = correctedLaTeX[10:]
    else:
        print("Is not a f function of x: " + str(LaTeX))
        return
    if last_accepted_input == correctedLaTeX:
        print("Input as last accepted: " + str(correctedLaTeX) + " from: " + str(LaTeX))
        return
    if "x" not in correctedLaTeX:
        print("No x in input: " + str(correctedLaTeX) + " from: " + str(LaTeX))
        return
    
    try:
        change_animation_old_y_points = y_values.copy()

        expr = parse_latex(correctedLaTeX, backend="lark")
        current_function = sp.lambdify(x, expr, "numpy")
        for i in range(sample_points):
            change_animation_new_y_points[i] = current_function(x_values[i])
            if(math.isnan(change_animation_new_y_points[i])):
                change_animation_new_y_points[i] = 0
                print("Value is nan")
        
        change_animation_t = 0
        change_animation_done = False

        last_accepted_input = correctedLaTeX
        print("Set function: " + str(correctedLaTeX))
    except Exception as e:
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

def normalize_point(point):
    x, y = point

    if (y > plot_y_range_max or y < plot_y_range_min):
        return None

    x_range = plot_x_range_max - plot_x_range_min
    y_range = plot_y_range_max - plot_y_range_min

    nx = plot_origin_x + (x / x_range) * plot_width
    ny = plot_origin_y - (y / y_range) * plot_height

    return (nx, ny)

def draw_ticks(image, H, tick_spacing=1, tick_size=0.02):
    # X-axis ticks (along y = 0)
    x = math.ceil(plot_x_range_min)
    while x <= plot_x_range_max:
        p1 = normalize_point((x, -tick_size))
        p2 = normalize_point((x, tick_size))

        if p1 and p2:
            cv2.line(image, w2p(p1, H), w2p(p2, H), (255, 255, 255), 1, cv2.LINE_AA)

        x += tick_spacing

    # Y-axis ticks (along x = 0)
    y = math.ceil(plot_y_range_min)
    while y <= plot_y_range_max:
        p1 = normalize_point((-tick_size, y))
        p2 = normalize_point((tick_size, y))

        if p1 and p2:
            cv2.line(image, w2p(p1, H), w2p(p2, H), (255, 255, 255), 1, cv2.LINE_AA)

        y += tick_spacing

def draw_grid(image, H, color, line_spacing=1):
    x = math.ceil(plot_x_range_min)
    while x <= plot_x_range_max:
        p1 = normalize_point((x, plot_y_range_min))
        p2 = normalize_point((x, plot_y_range_max))

        if p1 and p2:
            cv2.line(image, w2p(p1, H), w2p(p2, H), color, 1, cv2.LINE_AA)

        x += line_spacing

    y = math.ceil(plot_y_range_min)
    while y <= plot_y_range_max:
        p1 = normalize_point((plot_x_range_min, y))
        p2 = normalize_point((plot_x_range_max, y))

        if p1 and p2:
            cv2.line(image, w2p(p1, H), w2p(p2, H), color, 1, cv2.LINE_AA)

        y += line_spacing

def draw_func_name(image, H, color):
    box, _ = cv2.getTextSize(last_accepted_input, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    print(box)
    textUpperLeft = normalize_point((plot_x_range_max, y_values[len(y_values)-1]))
    cv2.putText(image, last_accepted_input, w2p(textUpperLeft, H), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def draw_plot(image, H, dt):
    global initial_animation_done, initial_animation_t, change_animation_done, change_animation_t, y_values
    # Draw Axis
    axis_h_left = normalize_point((plot_x_range_min, 0))
    axis_h_right = normalize_point((plot_x_range_max, 0))
    axis_v_bottom = normalize_point((0, plot_y_range_min))
    axis_v_top = normalize_point((0, plot_y_range_max))
    cv2.line(image, w2p(axis_h_left, H), w2p(axis_h_right, H), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(image, w2p(axis_v_bottom, H), w2p(axis_v_top, H), (255, 255, 255), 1, cv2.LINE_AA)

    draw_ticks(image, H, tick_spacing=1, tick_size=0.2)
    draw_grid(image, H, (100, 100, 100), line_spacing=1)

    points_to_draw = []

    if not initial_animation_done:
        initial_animation_t += dt
        if initial_animation_t >= initial_animation_duration:
            initial_animation_done = True
        for i in range(sample_points):
            point_location = i / sample_points
            time_location = initial_animation_t / initial_animation_duration
            if time_location >= point_location:
                point = normalize_point((x_values[i],y_values[i]))
                if point is not None:
                    points_to_draw.append(w2p(point, H))
    elif not change_animation_done:
        change_animation_t += dt
        if change_animation_t >= change_animation_duration:
            change_animation_done = True
            for i in range(sample_points):
                y_values[i] = change_animation_new_y_points[i]
        for i in range(sample_points):
            percent_done = change_animation_t / change_animation_duration
            y_values[i] = change_animation_old_y_points[i] + (change_animation_new_y_points[i]-change_animation_old_y_points[i]) * percent_done
            point = normalize_point((x_values[i],y_values[i]))
            if point is not None:
                points_to_draw.append(w2p(point, H))
    else:
        for i in range(sample_points):
            point = normalize_point((x_values[i],y_values[i]))
            if point is not None:
                points_to_draw.append(w2p(point, H))

    if len(points_to_draw) > 1:
        points = np.array(points_to_draw, dtype=np.int32)
        points = [points]
        func_color = (0, 0, 255)
        cv2.polylines(image, points, False, func_color, 1, cv2.LINE_AA)
        draw_func_name(image, H, func_color)
