# Image Annotator
This repository comes with an image annotator blueprint for the Flask API (https://flask.palletsprojects.com/en/1.1.x/). This allows this tool to run as a web page on a server (AWS or RaspPi) to allow for distributed annotation with a centralised data source. This means all data can be kept together on the server but allow for people to request the webpage and annotate it themselves. This tool is designed to be quite open allowing you to customise the annotation reading and saving methods which in turn allow you to create a custom active learning function. This means that you can create a custom function which loads the image and provides annotation estimations to help the user annotate the image. 

`from fsai.tools.image_annotator import image_annotator_blueprint`

`images_path` - Path to a directory of images to annotate.  
`images_labels_path` - Path to a directory of pre-generated annotations.  
`output_image_path` - Path to a directory to store all images that have been annotated.  
`output_labels_path` - Path to a director where all annotations are stored.  
`classes` - The list of classes contained within the dataset.  
`colors` - If given these colours will be used to represent the different classes in the UI.  
`file_formats` - The file formats the labels are stored in. This is to help find the label file, it will not be used to determine how the labels are loaded. By default this param is set to ["txt", "xml"].  
`output_file_format` - The output file format the labels are saved as.  
`session_key` - If set the users session must have a variable with this key set to 'valid' to allow the user to view the page.   
`redirect_url` - If the user is not validated with a session_id, they will be sent to this url. Otherwise they'll return a HTTP 401 Error.  
`annotation_loader` - This function is used to load the text file and return the valid annotation format. List[Tuple[int, float, float, float, float]] - List[class_num, x/i_width, y/i_height, box_width/i_width, box_height/i_height]. By default labels can be loaded from the Darknet Yolo format or the Pascal VOC format.   
`export_function` - This function determines how the annotations will be saved. By default all annotations will be saved in the Darknet Yolo Format.
`download` - If set, this function will be called when the user wishes to download a file. By default this is set to None which hides the download button in the UI. [See download usage.](#download)

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
By default this tool is able to read darknet formatted annotation files and Pascal VOC xml formatted files. However it is relatively simple to create and use your own methods and functions. 

It's important to understand how the pipeline works before creating your own. When a user loads the page, the server will look for an image to provide the user, once's it's found an images, it will then try and search for a corresponding label for the image. This is done by iterating through a list of provided file formats (defaults to `.txt` and `.xml`). If a match is found the server will attempt to load the annotations from the file. By default this is done using 'fsai.utils.annotations.auto_detect_annotation_loader' method which will try and detect which method is should use. All annotations should be stores as a list of tuples of class_num, x, y, w, h (relative to the size of the image) -> `List[Tuple[int, float, float, float, float]]`:. 

Different annotation formats store the data differently, for example, Pascal VOC stores classes by the given name whereas darknet stores it by class_index. Because of this, the import/export functions will require parameters to allow you to load/save into the correct format. These requires parameters are as follows:

custom_image_loader:  
`label_path` - Path to the label file to load (if available)  
`image_path` - Path to the image being annotated.  
`class_names` - List of all the class names  

export_as_json:  
`output_label_path` - Path to the label to save to.  
`image_width` - Width of the image to save.  
`image_height` - Height of the image to save.  
`annotations` - Annotations in the format: `List[Tuple[int, float, float, float, float]]`

With this knowledge we can easily customise the loading and saving functions of this tool to load and save annotations using a custom format.

```python
import json
from flask import Flask
from fsai.tools.image_annotator import image_annotator_blueprint


input_images_path = "../images/"  # input unannotated images
input_annotations_path = "../annotations/"  # dir for annotations pre-human annotation
output_images_path = "../annotated-images/"  # dir for images once they're annotated
output_annotations_path = "../annotated-annotations/"  # dir for human annotated images

classes = ["blue_cone", "yellow_cone", "orange_cone", "big_cone"]
class_colours = ["#0000ff", "#dddd00", "#ff9900", "#ff3300"]
files_extensions = ["json"]
output_file_format = "json"


def custom_image_loader(label_path, image_path, class_names):
    # Customise load function
    with open(label_path, "r") as label_file:
        label_string = label_file.read()

        annotation_json = json.loads(label_string)
    iw, ih = annotation_json["width"], annotation_json["height"]

    annotations = []
    # loop through all boxes in the json to formatted the data into the correct c, x, y, w, h format
    for box in annotation_json["boxes"]:
        annotations.append((
            class_names.index(box["class"]),
            ((box["x1"] + box["x2"]) / 2) / iw,  # calc x from x1, x2
            ((box["y1"] + box["y2"]) / 2) / ih,  # calc y from y1, y2
            (box["x2"] - box["x1"]) / iw,  # calc w from x1, x2
            (box["y2"] - box["y1"]) / ih  # calc h from y1, y2
        ))

    return annotations


def export_as_json(output_label_path, image_width, image_height, annotations):
    output = {"width": image_width, "height": image_height, "boxes": []}

    for box in annotations:
        class_num, x, y, w, h = box
        # translate x, y, w, h -> x1, y1, x2, y2
        output["boxes"].append({
            "class": classes[class_num],
            "x1": (x - w / 2) * image_width,
            "y1": (y - h / 2) * image_height,
            "x2": (x + w / 2) * image_width,
            "y2": (y + h / 2) * image_height
        })

    # save the json
    with open(output_label_path, "w+") as label_file:
        label_file.write(json.dumps(output, indent=4))


# set up the flask server with the custom functions
app = Flask(__name__)
app.register_blueprint(image_annotator_blueprint(
    input_images_path,
    input_annotations_path,
    output_images_path,
    output_annotations_path,
    classes,
    colors=class_colours,
    file_formats=files_extensions,
    output_file_format=output_file_format,
    annotation_loader=custom_image_loader,
    export_function=export_as_json
))


if __name__ == "__main__":
    app.run(port=5000, debug=True)
```


### Active learning example
We can alter the loading function to create active learning. Rather than loading the labels from a given path we can load the image from the given image path and then use some form of object detection to outline where cones might be which can then be passed on to the user currently annotating an image. This means you can take unannotated images and allow the server to pre-generate annotations to help aid the user.
```python
import cv2
from flask import Flask
from fsai.tools.image_annotator import image_annotator_blueprint


input_images_path = "../images/"  # input unannotated images
input_annotations_path = "../annotations/"  # dir for annotations pre-human annotation
output_images_path = "../annotated-images/"  # dir for images once they're annotated
output_annotations_path = "../annotated-annotations/"  # dir for human annotated images

classes = ["blue_cone", "yellow_cone", "orange_cone", "big_cone"]
class_colours = ["#0000ff", "#dddd00", "#ff9900", "#ff3300"]


def active_image_loader(label_path, image_path, class_names):
    # Customise load function
    image = cv2.imread(image_path)
    
    # create your own function here to generate annotations for the loaded image
    return generate_annotations(image)


# set up the flask server with the custom functions
app = Flask(__name__)
app.register_blueprint(image_annotator_blueprint(
    input_images_path,
    input_annotations_path,
    output_images_path,
    output_annotations_path,
    classes,
    colors=class_colours,
    annotation_loader=active_image_loader,
))


if __name__ == "__main__":
    app.run(port=5000, debug=True)
```

### Password Only
If you're feeling quite secretive about your dataset you can apply a password only filter to the blueprint. The way the blueprint authenticates someone is by checking the user's session contains a given id equals "generate_annotations". By default the parameter `generate_annotations` is set to None which means no password will be applied. However, if this value is set to a given string then the blueprint will check to see if the user's session contains that key equal to 'valid'. If the user does have a valid session id then they'll be allowed through, else they'll be redirected to a given url using the parameter `redirect_url`, by default redirect_url is None, meaning the user will be given a 401 error (Permission Denied).
```python
from flask import Flask, request, session, redirect
from fsai.tools.image_annotator import image_annotator_blueprint

session_key_id = "annotation_allowed"

app = Flask(__name__)
app.secret_key = b'i_love_ofsai'  # used to encode the session
app.register_blueprint(image_annotator_blueprint(
    "../images/",  # input unannotated images
    "../annotations/",  # dir for annotations pre-human annotation
    "../annotated-images/",  # dir for images once they're annotated
    "../annotated-annotations/",  # dir for human annotated images
    ["blue_cone", "yellow_cone", "orange_cone", "big_cone"],
    ["#0000ff", "#dddd00", "#ff9900", "#ff3300"],
    session_key=session_key_id,  # requires session key id
    redirect_url="login/",  # if not valid, take user to login page
))


@app.route("/login/")
def login_page():
    # login page
    title = "<h2>OFSAI Image Annotator Login</h2>"
    form = "<form method='post' action='/login-submit/'><input name=password type='password'/></form>"
    table_style = "style='position: absolute; height: 100%; width: 100%; top: 0px; left: 0px;'"
    return "<table {}><tr><td align=center>{}{}</td></tr></table>".format(table_style, title, form)


@app.route("/login-submit/", methods=['POST'])
def submit_login():
    # on submit check password is valid, if so set the session key to 'valid'
    if request.form.get("password", None) == "ofsai":
        session[session_key_id] = "valid"
        return redirect("/")
    return redirect("/login/")


if __name__ == "__main__":
    app.run(port=5000, debug=True)
```

### Download
In the event you wish to allow users to download the dataset straight from the website then you can set the `download` 
parameter to a function that gets called when the user hits the download button. By default the `download` param is set
to None meaning no download option appears. The download function should be returned like any other flask function since
the download button simply opens a webpage which calls the download function. There are multiple ways you could create
the download function for example you could cache a copy of the downloads onto the server which is sent to the user
upon download. Or you archive the whole of the annotated dataset and download that. If you're running this of a low power computer then it might be ideal to cache the downloads before hand and manually update it every week.  
```python
from flask import Flask, send_file
from fsai.tools.image_annotator import image_annotator_blueprint


def on_download():
    """
    This function is passed in to the server which is called upon request by the user
    """
    return send_file(
        "dataset.zip",
        as_attachment=True
    )


app = Flask(__name__)
app.register_blueprint(image_annotator_blueprint(
    "../images/",  # input unannotated images
    "../annotations/",  # dir for annotations pre-human annotation
    "../annotated-images/",  # dir for images once they're annotated
    "../annotated-annotations/",  # dir for human annotated images
    ["blue_cone", "yellow_cone", "orange_cone", "big_cone"],
    ["#0000ff", "#dddd00", "#ff9900", "#ff3300"],
    download=on_download   # pass in the download function
))
```