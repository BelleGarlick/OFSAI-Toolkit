# Image Annotator
This repository comes with an image annotator blueprint for the Flask API (https://flask.palletsprojects.com/en/1.1.x/). This allows this tool to run as a web page on a server (AWS or RaspPi) to allow for distributed annotation with a centralised data source. This means all data can be kept together on the server but allow for people to request the webpage and annotate it themselves. This tool is designed to be quite open allowing you to customise the annotation reading and saving methods which in turn allow you to create a custom active learning function. This means that you can create a custom function which loads the image and provides annotation estimations to help the user annotate the image. 

# TODO PASSWORD
# TODO DOWNLOAD
# TODO HELP
# FIXC Mouse pos and corner radius

# TODO README - Parameters list an explanation here
# TODO README - add links and reference to other code and stuff

### Basic Usage
In `fsai.tools.image_annotator` there is a method `image_annotator_blueprint` which, given relevant information, will build the blueprint to add to flask (shown below). 

Setting up this blueprint has a few parameter requirements. Four directories will be required: pre-annotated images, post-annotated images, pre-annotated labels, post-annotated labels. The pre-annotated labels may seem a little fatuous since the whole point of this tool is to annotate the images, but this tool allows you to pre-generate annotations for all images in the data set or it can be used when you want to correct badly annotated images.

In addition to these image paths you'll need to provide the class labels and (optionaly) provide the class colours. These class colours are the colours used to render labels in the webpage.

```python
from flask import Flask
from fsai.tools.image_annotator import image_annotator_blueprint

input_images_path = "../images/"  # input unannotated images
input_annotations_path = "../annotations/"  # dir for annotations pre-human annotation
output_images_path = "../annotated-images/"  # dir for images once they're annotated
output_annotations_path = "../annotated-annotations/"  # dir for human annotated images

classes = ["blue_cone", "yellow_cone", "orange_cone", "big_cone"]
class_colours = ["#0000ff", "#dddd00", "#ff9900", "#ff3300"]

app = Flask(__name__)
app.register_blueprint(image_annotator_blueprint(
    input_images_path,
    input_annotations_path,
    output_images_path,
    output_annotations_path,
    classes,
    colors=class_colours
))

if __name__ == "__main__":
    app.run("0.0.0.0", port=5000, debug=True)
```

### Custom Loading / Saving functions
By default this tool is able to read darknet formatted annotation files and VOC xml formatted files. However it is relatively simple to create and use your own methods and functions. 

It's important to understand how the pipeline works before creating your own. When a user loads the page, the server will look for an image to provide the user, once's it's found an images, it will then try and search for a corresponding label for the image. This is done by iterating through a list of provided file formats (defaults to `.txt` and `.xml`). If a match is found the server will attempt to load the annotations from the file. By default this is done using 'fsai.utils.annotations.auto_detect_annotation_loader' method which will try and detect which method is should use. All annotations should be stores as a list of tuples of class_num, x, y, w, h (relative to the size of the image) -> `List[Tuple[int, float, float, float, float]]`:. 

Different annotation formats store the data differently, for example, VOC stores classes by the given name whereas darknet stores it by class_index. Because of this, the import/export functions will require parameters to allow you to load/save into the correct format. These requires parameters are as follows:

custom_image_loader:  
`label_path` - Path to the label file to load (if available)  
`image_path` - Path to the image being annotated.  
`class_names` - List of all the class names  

export_as_json:  
`output_label_path` - Path to the label to save to.  
`image_width` - Width of the image to save.  
`image_height` - Height of the image to save.  
`annotations` - Annotations in the format: `List[Tuple[int, float, float, float, float]]`

With this knowledge we can easily customise the loading and saving functions of this tool.

### Active learning example