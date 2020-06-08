import random
import json
import os
import time
import uuid

import waitress
from flask import Flask, escape, request, Response, abort

app = Flask(__name__)

MAX_ITERATIONS = 50

completed_folder = "C:\\Users\\SamGa\\Documents\\GitHub\\OFSAI-Toolkit\\server_testing\\annotated\\"
tracks_folder = "C:\\Users\\SamGa\\Documents\\GitHub\\OFSAI-Toolkit\\server_testing\\tracks\\"
processing_folder = "C:\\Users\\SamGa\\Documents\\GitHub\\OFSAI-Toolkit\\server_testing\\processing\\"

overview_code = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css">
  <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.16.0/umd/popper.min.js"></script>
  <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.0/js/bootstrap.min.js"></script>
</head>
<body>


<div class="container">
</br>
  <table class="table">
    <thead>
      <tr>
        <th></th>
        <th>Track</th>
        <th>Time</th>
      </tr>
    </thead>
    <tbody>{}</tbody>
  </table>
</div>

</body>
</html>"""

def list_tracks():
    tracks = {}
    for track in os.listdir(tracks_folder):
        if track[0] != ".":
            track_name = track.replace(".json", "")
            load_track_variants(tracks, track_name)
    return tracks


def load_track_variants(data, track_name):
    data[track_name] = load_overview_data(track_name)
    data[track_name + ".reversed"] = load_overview_data(track_name + ".reversed")
    # data[track_name + ".tight"] = load_overview_data(track_name + ".tight")
    # data[track_name + ".reversed.tight"] = load_overview_data(track_name + ".reversed.tight")


def load_overview_data(track_name):
    data = {
        "time": -1,
        "distance": 0,
        "processing": False,
        "processed": False,
        "process_time": -1,
        "iterations": 0
    }

    completed_path = completed_folder + track_name + ".json"
    processing_path = processing_folder + track_name + ".json"

    if os.path.exists(completed_path):
        data["processed"] = True
        processed_data = load_json(completed_path)
        data["time"] = processed_data["time"]
        data["distance"] = processed_data["distance"]
        data["iterations"] = processed_data["iterations"]

    elif os.path.exists(processing_path):
        data["processing"] = True
        processed_data = load_json(processing_path)
        data["process_time"] = time.time() - processed_data["start"]

    return data


def load_json(path):
    with open(path) as file:
        return json.loads(file.read())


def get_random_unprocessed():
    all_tracks = list_tracks()

    unprocessed_tracks = [track_name for track_name in all_tracks.keys() if not all_tracks[track_name]["processing"] and not all_tracks[track_name]["processed"]]

    if len(unprocessed_tracks) > 0:
        return random.choice(unprocessed_tracks)
    return None


@app.route('/new/')
def get_track():
    current_track = get_random_unprocessed()
    print(current_track)
    if current_track is not None:
        id = uuid.uuid1()
        with open(processing_folder + current_track + ".json", "w+") as file:
            file.write(json.dumps({
                "start": time.time(),
                "uuid": str(id)
            }))

        return {
            "name": current_track,
            "data": load_json(tracks_folder + current_track.split(".")[0] + ".json"),
            "uuid": str(id),
            "intervals": MAX_ITERATIONS
        }
    return abort(444)


@app.route('/save/', methods=["post"])
def save_track():
    print(request.form)
    name = request.form.get("name", None)
    data = request.form.get("data", None)
    print(data)
    uuid = request.form.get("uuid", None)

    if name and data and uuid:
        processing_path = processing_folder + name + ".json"
        if os.path.exists(processing_path):
            processing_data = load_json(processing_path)
            if processing_data["uuid"] == uuid:
                os.remove(processing_path)
                with open(completed_folder + name + ".json", "w+") as file:
                    file.write(data)
                return "cheers"
    return "norty"


@app.route('/')
def hello_world():
    table_code = ""

    tracks = list_tracks()
    for track in tracks.keys():
        table_code += """
            <tr>
                <td>{}</td>
                <td>{}</td>
                <td>{}</td>
            </tr>
        """.format(
            get_track_name(tracks[track]),
            track,
            tracks[track]["time"]
        )
    return overview_code.format(table_code)


def get_track_name(track_data):
    color = "red"
    if track_data["processing"]:
        color = "gold"
    if track_data["processed"]:
        color = "green"
    return """
        <div style='width: 10px; height: 10px; border-radius: 10px; background: {};'></div>

    """.format(
        color
    )


if __name__ == "__main__":
    app.run(port=80, debug=True)
