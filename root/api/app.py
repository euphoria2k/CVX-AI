from PIL import ImageTk, Image as PILImage
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS

import os
import subprocess
import json
import imageio
from pathlib import Path
from tkinter import filedialog
from tkinter import *

app = Flask(__name__)
CORS(app)

global setColor
setColor = '--color'

# Referencing functions into Vue go here:
@app.route("/setPythonDir", methods=['POST'])
def SetDirectories():
    root = Tk()
    root.pyLoad = filedialog.askopenfilename(initialdir="C:/Users/HP/Documents/CVX_AI_WebUI/root/api",
                                               title="Locate object_tracker.py file...", filetypes=(("Python files", "*.py"),))
    pyLoad_JSON = '{"dir":"' + root.pyLoad + '"}'
    global pyLoad_parse
    pyLoad_parse = json.loads(pyLoad_JSON)
    print("Python file was set!")
    print(pyLoad_JSON)
    print(pyLoad_parse["dir"])
    print("*********************************************************************************\n")

    root.weightDir = filedialog.askdirectory(initialdir="C:/Users/HP/Documents/CVX_AI_WebUI/root/api/checkpoints", title="Select weights directory...")
    weightDir_JSON = '{"dir":"' + root.weightDir + '"}'
    global weightDir_parse
    weightDir_parse = json.loads(weightDir_JSON)
    print("Weights directory set!")
    print(weightDir_JSON)
    print(weightDir_parse["dir"])
    print("*********************************************************************************\n")

    root.videoLoad = filedialog.askopenfilename(initialdir="C:/Users/HP/Documents/CVX_AI_WebUI/root/api/data/video",
                                               title="Select video for analysis...", filetypes=(("Video files", "*.mp4 *.avi *.mov *.3gp"),))
    videoLoad_JSON = '{"dir":"' + root.videoLoad + '"}'
    global videoLoad_parse
    videoLoad_parse = json.loads(videoLoad_JSON)
    print("Video file selected!")
    print(videoLoad_JSON)
    print(videoLoad_parse["dir"])
    print("*********************************************************************************\n")
    
    root.saveAsFileName = filedialog.asksaveasfilename(initialdir="C:/Users/HP/Documents/CVX_AI_WebUI/root/api/data/output",
                                               title="Select video for analysis...", filetypes=(("mp4 files", "*.mp4"),))
    saveAsFileName_JSON = '{"dir":"' + root.saveAsFileName + '.mp4"}'
    global saveAsFileName_parse
    saveAsFileName_parse = json.loads(saveAsFileName_JSON)
    print("Output file name and directory set!")
    print(saveAsFileName_JSON)
    print(saveAsFileName_parse["dir"])
    print("*********************************************************************************\n")

    res = make_response(jsonify({"message": "Output directory set"}))
    return res

# Full command format syntax:
# {python} {python file} {weights} {weights location} {video} {video location} {output} {output location} {--color} {--colorname} {--class}
# Ex: python object_tracker.py -weights ./checkpoints/lastweights -video ./data/video/test.mp4 --black --trouser

# {python} - acquired from JS backend
@app.route("/execute", methods=['POST'])
def receive_exec():
    json_req = request.get_json()
    global execPy
    execPy = json_req["exec"]
    print(execPy)
    print("\n")

    res = make_response(jsonify({"message": "Exec received"}))
    return res

# {python file} - Flask
# @app.route("/arg1", methods=['POST'])
# def receive_arg1():
#     json_req1 = request.get_json()
#     global execPy_dir
#     execPy_dir= json_req1["arg1"]
#     print(execPy_dir)
#     print("\n")

    # res = make_response(jsonify({"message": "Arg1 received"}))
    # return res

# {weights} - acquired from JS backend
@app.route("/arg2", methods=['POST'])
def receive_arg2():
    json_req2 = request.get_json()
    global flagWeight
    flagWeight= json_req2["arg2"]
    print(flagWeight)
    print("\n")

    res = make_response(jsonify({"message": "Arg2 received"}))
    return res

# {weights file} - Flask
# @app.route("/arg3", methods=['POST'])
# def receive_arg3():
#     json_req3 = request.get_json()
#     global flagWeight_dir
#     flagWeight_dir = json_req3["arg3"]
#     print(flagWeight_dir)
#     print("\n")

#     res = make_response(jsonify({"message": "Arg3 received"}))
#     return res

# {video} - acquired from JS backend
@app.route("/arg4", methods=['POST'])
def receive_arg4():
    json_req4 = request.get_json()
    global flagVideo
    flagVideo= json_req4["arg4"]
    print(flagVideo)
    print("\n")

    res = make_response(jsonify({"message": "Arg4 received"}))
    return res

# {video location} - Flask
# @app.route("/arg5", methods=['POST'])
# def receive_arg5():
#     json_req5 = request.get_json()
#     global flagVideo_dir
#     flagVideo_dir= json_req5["arg5"]
#     print(flagVideo_dir)
#     print("\n")

#     res = make_response(jsonify({"message": "Arg1 received"}))
#     return res

# {output} - acquired from JS backend
@app.route("/arg6", methods=['POST'])
def receive_arg6():
    json_req6 = request.get_json()
    global flagOutput
    flagOutput = json_req6["arg6"]
    print(flagOutput)
    print("\n")

    res = make_response(jsonify({"message": "Arg1 received"}))
    return res

# {output location} - Flask
# @app.route("/arg7", methods=['POST'])
# def receive_arg7():
#     json_req7 = request.get_json()
#     global flagOutput_dir
#     flagOutput_dir = json_req7["arg7"]
#     print(flagOutput_dir)
#     print("\n")

#     res = make_response(jsonify({"message": "Arg1 received"}))
#     return res

# {color} - acquired from JS backend
@app.route("/arg8", methods=['POST'])
def receive_arg8():
    json_req8 = request.get_json()
    global flagColor
    flagColor = json_req8["value"]
    print(flagColor)
    print("\n")

    res = make_response(jsonify({"message": "Arg1 received"}))
    return res

# {class} - acquired from JS backend
@app.route("/arg9", methods=['POST'])
def receive_arg9():
    json_req9 = request.get_json()
    global flagClass
    flagClass= json_req9["value"]
    print(flagClass)
    print("\n")

    res = make_response(jsonify({"message": "Arg1 received"}))
    return res

@app.route("/cvx_start", methods=['POST'])
def call_CVX_AI():
    subprocess.call([execPy, pyLoad_parse["dir"], flagWeight, weightDir_parse["dir"], flagVideo, videoLoad_parse["dir"], flagOutput, saveAsFileName_parse["dir"], setColor, flagColor, flagClass], shell=True)

    res = make_response(jsonify({"message": "PROCESS_STARTED"}))
    return res

