import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_dangerously_set_inner_html
import mediapipe as mp
import SquatPosture as sp
from flask import Flask, Response
import cv2
import numpy as np
from utils import landmarks_list_to_array, label_params
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

def gen(camera):
    cap = camera.video
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                # continue
                break

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            image_height, image_width, _ = image.shape

            params = sp.get_params(results)

            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            coords = landmarks_list_to_array(results.pose_landmarks, image.shape)
            label_params(image, params, coords)

            ret, jpeg = cv2.imencode('.jpg', image)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


server = Flask(__name__)
# external_stylesheets = ['./app.css']
app = dash.Dash(__name__, server=server)
app.title = "Posture"


@server.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()) ,mimetype='multipart/x-mixed-replace; boundary=frame')


# app.layout = html.Div(className="main", children=[
#     html.Link(
#         rel="stylesheet",
#         href="/static/stylesheet.css"
#     ),
#     html.Div(className="container", children=[
#         html.H2(
#         children= "Posture",
#         className = "head"
#     ),
#     html.Br(),
#     html.Img(
#         src="/video_feed",
#         className = "feed"
#     )
#     ]),
    
# ])

app.layout = html.Div(className="main", children=[
    html.Link(
        rel="stylesheet",
        href="/static/stylesheet.css"
    ),
    dash_dangerously_set_inner_html.DangerouslySetInnerHTML("""
        <div class="container">
            <table cellspacing="20px" class="table">
                <tr class="row">
                    <td> <h2 class="head"> Posture </h2> </td>
                </tr>
                <tr class="row">
                    <td> <img src="/video_feed" class="feed"/> </td>
                </tr>
            </table>
        </div>
    """),
])

if __name__ == '__main__':
    app.run_server(debug=True)