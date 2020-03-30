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
