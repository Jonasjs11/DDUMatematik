import cv2
import numpy as np
import sympy as sp
from sympy.parsing.latex import parse_latex
import math

sample_points = 100

plot_width, plot_height = 100, 100
plot_origin_x, plot_origin_y = 100, 100

plot_x_range_min, plot_x_range_max = -10, 10
plot_y_range_min, plot_y_range_max = -10, 10

x_values = np.linspace(plot_x_range_min, plot_y_range_min, sample_points)
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

corner_1_x, corner_1_y, corner_2_x, corner_2_y = 0, 0, 0, 0



def setup_plot(_corner_1_x, _corner_1_y, _corner_2_x, _corner_2_y, LaTeX):
    global plot_width, plot_height, plot_origin_x, plot_origin_y, corner_1_x, corner_1_y, corner_2_x, corner_2_y, initial_animation_t, initial_animation_done

    corner_1_x = _corner_1_x
    corner_1_y = _corner_1_y
    corner_2_x = _corner_2_x
    corner_2_y = _corner_2_y

    plot_width = np.abs(corner_2_x - corner_1_x)
    plot_height = np.abs(corner_2_y - corner_1_y)
    plot_origin_x = np.mean([corner_1_x, corner_2_x])
    plot_origin_y = np.mean([corner_1_y, corner_2_y])

    initial_animation_t = 0
    initial_animation_done = False

    set_function(LaTeX)
    



def set_function(LaTeX):
    global current_function, change_animation_t, change_animation_done, change_animation_old_y_points, change_animation_new_y_points

    correctedLaTeX = str(LaTeX).replace(r"\chi", r"x").replace(r"\times", r"x").replace(r"X", r"x")

    if last_accepted_input == correctedLaTeX:
        print("Input as last accepted: " + str(last_accepted_input))
        return
    if "x" not in correctedLaTeX:
        print("No x in input: " + str(correctedLaTeX))
        return
    
    try:
        change_animation_old_y_points = y_values.copy()

        expr = parse_latex(correctedLaTeX, backend="lark")
        current_function = sp.lambdify(x, expr, "numpy")
        for i in range(sample_points):
            change_animation_new_y_points[i] = current_function(x_values[i])
            if(math.isnan(change_animation_new_y_points[i])):
                change_animation_new_y_points = 0
                print("Value is nan")
        
        change_animation_t = 0
        change_animation_done = False

        last_accepted_input = correctedLaTeX
        print("Set function: " + str(correctedLaTeX))
    except Exception as e:
        print("Error with input: " + str(LaTeX) + " And error: " + str(e))


def top_corner_moved(corner_new_x, corner_new_y):
    global plot_width, plot_height, plot_x_range_min, plot_x_range_max, plot_y_range_min, plot_y_range_max, corner_2_x, corner_2_y
    range_x_per_width = (plot_x_range_max - plot_x_range_min) / plot_width
    range_y_per_height = (plot_y_range_max - plot_y_range_min) / plot_height

    corner_1_x = corner_new_x
    corner_1_y = corner_new_y

    plot_width = np.abs(corner_2_x - corner_1_x)
    plot_height = np.abs(corner_2_y - corner_1_y)

    plot_x_range_min = range_x_per_width * plot_width
    plot_y_range_min = range_y_per_height * plot_height

def bottom_corner_moved(corner_new_x, corner_new_y):
    global plot_width, plot_height, plot_x_range_min, plot_x_range_max, plot_y_range_min, plot_y_range_max, corner_2_x, corner_2_y
    range_x_per_width = (plot_x_range_max - plot_x_range_min) / plot_width
    range_y_per_height = (plot_y_range_max - plot_y_range_min) / plot_height

    corner_2_x = corner_new_x
    corner_2_y = corner_new_y

    plot_width = np.abs(corner_2_x - corner_1_x)
    plot_height = np.abs(corner_2_y - corner_1_y)

    plot_x_range_max = range_x_per_width * plot_width
    plot_y_range_max = range_y_per_height * plot_height

def pan_x_axis(amount):
    global plot_x_range_min, plot_x_range_max, x_values, newPointsY
    plot_x_range_min += amount
    plot_x_range_max += amount

    x_values = np.linspace(plot_x_range_min, plot_x_range_max, sample_points)
    y_values = current_function(x_values)

def pan_y_axis(amount):
    global plot_y_min, plot_y_max
    plot_x_range_min += amount
    plot_x_range_max += amount

def w2p(point, H): # whiteboard 2 projector
    return (0,0)

def draw_plot(image, H, dt):
    # Draw Axis
    cv2.line(image, w2p((plot_origin_x - plot_width/2, plot_origin_y), H),      w2p((plot_origin_x + plot_width/2, plot_origin_y), H),  (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(image, w2p((plot_origin_x, plot_origin_y - plot_height/2), H),     w2p((plot_origin_x, plot_origin_y + plot_height/2), H), (255, 255, 255), 1, cv2.LINE_AA)

    points_to_draw = []

    if not initial_animation_done:
        initial_animation_t += dt
        if initial_animation_t >= initial_animation_duration:
            initial_animation_done = True
        for i in range(sample_points):
            point_location = i / sample_points
            time_location = initial_animation_t / initial_animation_duration
            if time_location >= point_location:
                points_to_draw.append(w2p((x_values[i], y_values[i]), H))
    elif not change_animation_done:
        change_animation_t += dt
        if change_animation_t >= change_animation_duration:
            change_animation_done = True
            for i in range(sample_points):
                y_values[i] = change_animation_new_y_points[i]
        for i in range(sample_points):
            percent_done = change_animation_t / change_animation_duration
            y_values[i] = change_animation_old_y_points[i] + (change_animation_new_y_points[i]-change_animation_old_y_points[i]) * percent_done
            points_to_draw.append(w2p((x_values[i], y_values[i]), H))
    else:
        for i in range(sample_points):
            points_to_draw.append(w2p((x_values[i], y_values[i]), H))

    if len(points_to_draw) > 1:
        cv2.polylines(image, points_to_draw, False, (255, 255, 255), 1, cv2.LINE_AA)
