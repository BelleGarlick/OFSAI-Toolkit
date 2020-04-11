# Image Annotation Visualisation
This library has a method `draw_annotations` in `from fsai.visualisation.image_annotations` which is used to render image annotations using OpenCV. The main use of this tool is to visualise the outputs that an object detector provides, however you can also use it to check the annotations within a dataset directly from python.

#### Parameters:  
`labels_path` - If given labels will load from a file at this path.  
`annotation_load_function` - This mutable function dictates how the labels should be loaded into the darknet format ([See here](/fsai/utils/README.md)).   
`label_annotations` - Pre-loaded annotations can be passed here.  
`image_path` - The path to an image to load and render labels upon.   
`image` - A pre-loaded OpenCV image to render upon.   
`colors` - A list of colours for the labels to render as.   
`class_names` - Labels for the classes, passed to the annotation loader.   
`show_names` - If True then the class labels will be render along side the drawn annotations.   
`line_width` - The line width to render the boxes with.   
`return` - Returns the labeled image.   

#### Basic Usage:
```python
import cv2
from fsai.visualisation.image_annotations import draw_annotations

cv2.imshow(
    "Annotated Image",
    draw_annotations(
        labels_path="1.txt",
        image_path="1.jpg",
        colors=[(255, 100, 0), (0, 200, 200), (0, 150, 255), (0, 0, 255)],
        class_names=["Blue", "Yellow", "Orange", "Big"],
        show_names=True,
        line_width=3
    )
)
cv2.waitKey(0)
cv2.destroyAllWindows()
```  

# 2D Renderer
OFSAI provides multiple ways to view the track. The different methods require similar parameters allowing you to interchange the different methods without requiring too much code to be altered. The different methods share the same parameters `lines`, `points`, `cones`, `cars`, `background`, `padding`. The given items will be rendered and scaled to fit within the screen_size given with a given padding. The lines, points and cones parameter take a list of tuples `[color, size, list[objects]]` which enables you to customise the rendering quite flexibly.

### OpenCV Renderer
This function allows you to render the scene as an OpenCV image.

#### Parameters:  
`image_size` - The size of the image that will be produced from this function.  
`lines` - A List of tuples containing a colour, line-width and list of points to be rendered.  
`cones` - A List of tuples containing a colour, radius and list of points to be rendered.  
`points` - A List of tuples containing a colour, radius and list of points to be rendered.    
`cars` - A List of cars to be rendered within the scene.  
`background` - The background colour of the scene (as an integer).  
`padding` - How much padding (pixels) to surround the image with (for aesthetic reasons).  
`returns` - Return the rendered image.  

#### Usage:  
```python
import cv2
import random

from fsai.objects.point import Point
from fsai.objects.track import Track
from fsai.visualisation.draw_opencv import render

track = Track("examples/data/tracks/imola.json")
blue_lines, yellow_lines, orange_lines = track.get_boundary()

random_points = []
for i in range(20):
    random_points.append(Point(
        (random.random() - 0.5) * 70,
        (random.random() - 0.5) * 35)
    )

image = render(
    image_size=(800, 600),
    lines=[
        ((255, 0, 0), 2, blue_lines),
        ((0, 255, 255), 2, yellow_lines),
        ((0, 100, 255), 2, orange_lines),
    ],
    cones=[
        ((255, 0, 0), 4, track.blue_cones),
        ((0, 255, 255), 4, track.yellow_cones),
        ((0, 100, 255), 4, track.big_cones),
    ],
    points=[
        ((0, 255, 0), 3, random_points)
    ],
    cars=track.cars,
    background=100,
    padding=50
)
cv2.imshow("", image)
cv2.waitKey(0)
```

### Py Game Renderer
This function allows you to render your scene as in a pygame screen. 

#### Parameters:  
`pygame_screen` - The pygame screen to render on to.  
`screen_size` - The size of the image that will be produced from this function.  
`lines` - A List of tuples containing a colour, line-width and list of points to be rendered.  
`cones` - A List of tuples containing a colour, radius and list of points to be rendered.  
`points` - A List of tuples containing a colour, radius and list of points to be rendered.    
`cars` - A List of cars to be rendered within the scene.  
`background` - The background colour of the scene (as an integer).  
`padding` - How much padding (pixels) to surround the image with (for aesthetic reasons).  

#### Usage:  
Using pygame as a rendering system allows us to view simulations. In this example usage we can use a joystick to control the vehicle.
```python
import time
import pygame as pygame
from fsai.objects.track import Track
from fsai.visualisation.draw_pygame import render

# Set up pygame and a pygame controller
pygame.init()
screen_size = [1000, 800]
screen = pygame.display.set_mode(screen_size)
joystick = None
if pygame.joystick.get_count() > 0:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

# Load track and boundary
track = Track("examples/data/tracks/azure_circuit.json")
blue_lines, yellow_lines, orange_lines = track.get_boundary()

running = True
last_update = time.time()
while running:
    now = time.time()
    dt = now - last_update
    pygame.event.get()

    # Set car driving based upon the joystick inputs
    if joystick is not None:
        track.cars[0].throttle = max(0, -joystick.get_axis(2))
        track.cars[0].steer = joystick.get_axis(0)
        track.cars[0].brake = max(0, joystick.get_axis(2))
        
    # update the physics
    track.cars[0].physics.update(dt)

    # draw and show the track
    render(
        screen,
        screen_size,
        cones=[
            ((255, 255, 0), 5, track.yellow_cones),
            ((0, 0, 255), 5, track.blue_cones)
        ],
        lines=[
            ((0, 0, 255), 2, blue_lines),
            ((255, 255, 0), 2, yellow_lines),
            ((255, 100, 0), 2, orange_lines)
        ],
        cars=track.cars
    )

    pygame.display.flip()
    last_update = now
pygame.quit()
```
