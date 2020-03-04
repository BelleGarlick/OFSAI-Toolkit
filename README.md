# OFSAI-Toolkit
An easy-to-use, open-source toolkit for self driving racing car development. Please try and improve on the tools offered and contribute to help everyone. 

Features:
 - Cone detection 
 - Track Boundary Estimation
 - Cone Mapping
 - Visualise image annotations
 
TODO:
 - Visualise 2D track
 - Midline calculation
 - Racing line calculation
 - On-the-line image annotator (with active learning)
 - On-the-line custom track maker
 - Yolo Trains Via Data generator where each imagine is loaded differently with random alterations
 - Visualise 3D track
 - Driver model
 - SLAM


# Objects
This repository comes with, and utilises, a few objects that represent objects found in the FS-AI competition.

**Point** - `from fsai.objects.point import Point`

Used to represent the position of an object in 2D.

`point = Point(x, y)`


**Cone** - `from fsai.objects.cone import Cone`

Used to represent a cone in the track. Each cone contains a point object representing the position of a cone, as well as the colour of the cone.

`fsai.objects.cone.CONE_COLOR_BLUE = 0`

`fsai.objects.cone.CONE_COLOR_YELLOW = 1`

`fsai.objects.cone.CONE_COLOR_BIG_ORANGE = 2`

`fsai.objects.cone.CONE_COLOR_ORANGE = 3`

The cone object can be constructed as follows:

`cone = Cone()  # blue cone at 0, 0`

`cone = Cone(point=Point(9, 2), color=CONE_COLOR_YELLOW)  # yellow cone at 9, 2`

`cone = Cone(x=4, y=6, color=CONE_COLOR_ORANGE)  # orange cone at 4, 6`


**Line** - `from fsai.objects.line import Line`

Object containing two points. Used to represent lines such as track boundaries.

# Visualisations
doc coming soon...

# Cone Detections
doc coming soon...

# Mapping
doc coming soon...

# Track Boundary Estimation
doc coming soon...
