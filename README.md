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
 - SLAM (EKF)
 - Compile to Cython

# Contents
1. Objects
2. Visualisations
3. Perception
4. Mapping
5. Track Boundary Estimation


# 1. Objects
This repository comes with, and utilises, a few objects that represent objects found in the FS-AI competition.

### Point 
Used to represent the position of an object in 2D.
```python
from fsai.objects.point import Point

point = Point(x, y)
```

### Cone 
Used to represent a cone in the track. Each cone contains a point object representing the position of a cone, as well as the colour of the cone.  

```python
from fsai.objects.cone import Cone

fsai.objects.cone.CONE_COLOR_BLUE = 0  
fsai.objects.cone.CONE_COLOR_YELLOW = 1  
fsai.objects.cone.CONE_COLOR_BIG_ORANGE = 2  
fsai.objects.cone.CONE_COLOR_ORANGE = 3
```
  
The cone object can be constructed as follows:  
```python
cone = Cone()  # blue cone at 0, 0
cone = Cone(point=Point(9, 2), color=CONE_COLOR_YELLOW)  # yellow cone at 9, 2
cone = Cone(x=4, y=6, color=CONE_COLOR_ORANGE)  # orange cone at 4, 6
```


### Line  
Object containing two points. Used to represent lines such as track boundaries.
```python
from fsai.objects.line import Line

line = Line(a=Point(0, 0), b=Point(1, 0))
```


### Track  
An object used to encapsulate cones into one handy class. 
```python
from fsai.objects.track import Track

track = Track()
```
Additionally this class allows you to save and load tracks in the json format:
```json
{
    "blue_cones": [
        {"x": 1.25, "y": -14.75},
        {"x": 5, "y": -15.5}
    ],
    "yellow_cones": [
        {"x": 2.875, "y": -9.875},
        {"x": 6, "y": -10.375}
    ],
    "big_orange_cones": [
        {"x": -2, "y": -13.625},
        {"x": -3.75, "y": -12.875}
    ]
    "orange_cones": []
}
```

```python
from fsai.objects.track import Track

# Loading
track = Track("examples/data/tracks/laguna_seca.json")
# -or-
track.load_track("examples/data/tracks/brands_hatch.json")

# Saving
track.save_track("output_track.json")
# -or-
with open("file", "w+") as file:
    file.write(track.to_json())
```

# 2. Visualisations
### Image Annotations

Annotates a given image with given annotations, returning an openCV formatted image.  
`<class> <x> <y> <w> <h>`
(Relative to the size of the image)

```python
annotated_image = annotate(labels_path, label_annotations, image_path, image, colors, class_names, line_width)
```  

#### Parameters:  
`labels_path` - If provided, the labels will be loaded from a file destination  
`label_annotations` - If provided, the annotations will be draw from the given information  
`image_path` - If provided the image will load from the provided path  
`image` - If provided the labels will be draw onto this opencv image  
`colors` - If provided the annotations will use the colours provided. By default colours are procedural generated  
`class_names` - If provided the annotations will draw class names as well  
`line_width` - Annotations will be draw with the width provided  
`returns` - OpenCV Image
    
#### Example Usages:
 ```python
from fsai.visulisation.image_annotations import annotate

image = annotate(label_path="labels/0.txt", image_path="images/0.jpg", colors=[(255, 0, 0), (0, 255, 0)], line_width=10)
cv2.imshow("Cones", image)
cv2.waitKey(0)
```
```python
from fsai.visulisation.image_annotations import annotate

image = cv2.imread("images/1.png")
annotations = [
    [0, 0.4, 0.6, 0.25, 0.25],
    [1, 0.8, 0.8, 0.1, 0.1],
]
image = annotate(image=image, label_annotations=annotations, class_names=["dog", "plane"], line_width=2)
cv2.imshow("Cones", image)
cv2.waitKey(0)
```


### 2D Track Renderer
doc coming soon...

# 3. Perception
doc coming soon...

# 4. Mapping
doc coming soon...

# 5. Track Boundary Estimation
doc coming soon...
