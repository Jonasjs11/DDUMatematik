import sympy as sp
from sympy.parsing.latex import parse_latex
import numpy as np
import pygame
import math

import demo

amountOfPoints = 100

plot_width_screenspace, plot_height_screenspace = 500, 500
plot_origin_x_screenspace, plot_origin_y_screenspace = 1500, 1000
plot_x_min, plot_x_max = -25, 25
plot_y_min, plot_y_max = -25, 25

x_values = np.linspace(plot_x_min, plot_x_max, amountOfPoints) # min, max, amount
y_values = [0] * amountOfPoints

oldPointsY = [0] * amountOfPoints
newPointsY = [0] * amountOfPoints

dt = 0.05 # animation time step
animationInitialDuration = 1.0 # seconds
animationInitialT = 0
animationInitialComplete = False
animationChangePlotDuration = 0.2 # seconds
animationChangePlotT = 0
animationChangePlotComplete = False

x = sp.symbols('x')
currentFunction = sp.lambdify(x, parse_latex(r"x", backend="lark"), "numpy")
lastAcceptedInput = r"f(x)=x"

def pan_x_axis(amount):
    global plot_x_min, plot_x_max, x_values, newPointsY
    plot_x_min += amount
    plot_x_max += amount

    x_values = np.linspace(plot_x_min, plot_x_max, amountOfPoints)
    newPointsY = currentFunction(x_values)

def pan_y_axis(amount):
    global plot_y_min, plot_y_max
    plot_x_min += amount
    plot_x_max += amount

def set_new_function(newInput):
    global y_values, newPointsY, newPointsY, currentFunction, lastAcceptedInput, animationChangePlotT, animationChangePlotComplete

    newInput = str(newInput).replace(r"\chi", r"x").replace(r"\times", r"x").replace(r"X", r"x")

    if lastAcceptedInput == newInput:
        print("Last input as last accepted: " + str(lastAcceptedInput))
        return
    if "x" not in newInput:
        print("No x in input: " + str(newInput))
        return
    try:
        oldPointsY = y_values.copy()

        expr = parse_latex(newInput, backend="lark")
        f = sp.lambdify(x, expr, "numpy")
        for i in range(amountOfPoints):
            newPointsY[i] = f(x_values[i])
            if(math.isnan(newPointsY[i])):
                newPointsY[i] = 0

        animationChangePlotT = 0
        animationChangePlotComplete = False

        currentFunction = f
        lastAcceptedInput = newInput
        print("Set new function: " + str(newInput))
    except Exception as e:
        print("Error with input: " + str(newInput))
        #print("Input: " + str(newInput) + " Error: " + str(e))

def map_plot_coords_2_screen_coords(plot_x, plot_y):
    screen_x = plot_origin_x_screenspace + (plot_x / (plot_x_max - plot_x_min)) * plot_width_screenspace
    screen_y = plot_origin_y_screenspace + (plot_y / (plot_y_max - plot_y_min)) * plot_height_screenspace
    return flip_coords(int(screen_x), int (screen_y))

def flip_coords(x, y):
    w, h = pygame.display.get_surface().get_size()
    return x, h - y

demo.setup_camera_and_processor()

running = True

pygame.init()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
pygame.display.set_caption("Plot")
clock = pygame.time.Clock()

set_new_function(r"x^{2}")

while running:
    screen.fill((0, 0, 0))

    pygame.draw.line(screen, (255, 255, 255), map_plot_coords_2_screen_coords(plot_x_min, 0), map_plot_coords_2_screen_coords(plot_x_max, 0), 2)
    pygame.draw.line(screen, (255, 255, 255), map_plot_coords_2_screen_coords(0, plot_y_min), map_plot_coords_2_screen_coords(0, plot_y_max), 2)

    points_to_draw = []
    if animationInitialComplete:
        for i in range(amountOfPoints):
            if type(x_values[i]) is not np.float64 or type(y_values[i]) is not np.float64:
                break

            points_to_draw.append(map_plot_coords_2_screen_coords(x_values[i], y_values[i]))
    else:
        for i in range(amountOfPoints):
            if type(x_values[i]) is not np.float64 or type(y_values[i]) is not np.float64:
                break

            pointLocation = i / amountOfPoints
            timeLocation = animationInitialT / animationInitialDuration
            if timeLocation >= pointLocation:
                points_to_draw.append(map_plot_coords_2_screen_coords(x_values[i], y_values[i]))
    
    if len(points_to_draw) > 1:
        pygame.draw.lines(screen, (255, 0, 0), False, points_to_draw, 4)


    detectedLatex, cropped, full = demo.get_latex_from_image()
    set_new_function(detectedLatex)
    if cropped is not None:
        height, width = cropped.shape[:2]
        pygame_image_cropped = pygame.image.frombuffer(cropped.tobytes(), (width, height), 'RGB')
        height, width = full.shape[:2]
        pygame_image_full = pygame.image.frombuffer(full.tobytes(), (width, height), 'RGB')
        screen.blit(pygame_image_cropped, (0, 1000))
        screen.blit(pygame_image_full, (0, 0))



    pygame.display.flip() # update screen
    clock.tick(60) # limit to 60 fps

    if animationInitialComplete == False:
        animationInitialT += dt
        if  animationInitialT > animationInitialDuration:
            animationInitialComplete = True

    if animationChangePlotComplete == False:
        animationChangePlotT += dt
        if animationChangePlotT > animationChangePlotDuration:
            animationChangePlotComplete = True
            for i in range(amountOfPoints):
                y_values[i] = newPointsY[i]
        else:
            for i in range(amountOfPoints):
                y_values[i] = oldPointsY[i] + (newPointsY[i] - oldPointsY[i]) * (animationChangePlotT / animationChangePlotDuration)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            demo.release_camera()