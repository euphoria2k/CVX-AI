# CVX-AI - The Computer Vision AI
This project was submitted to Mahidol University as a research project to fulfill our graduation requirements in B.Eng Computer Engineering. The original project paper can be found at 

## Overview
![Imgur](/cvx-ai.gif)  
Safety and maintaining order in society are two factors that make the world safe. This is made possible because of laws. Despite this, crime exists due to lawbreakers. Law enforcement agencies are interested in identifying and locating these persons of interest (POI). Descriptive details of potential suspects help find POI. Useful descriptions come from eyewitnesses present at the scene. Remembering details can be problematic with short-term memory since crucial details may be forgotten. Eyewitness accounts are not always the same, because every individual remembers events differently. Sometimes, extra details are added unintentionally. The purpose of this project is to make the identification of POIs more reliable and efficient with an emphasis on privacy.

## Features
CVX-AI utilizes artificial intelligence and machine learning algorithms supplemented by YOLOv4 (with DarkNet) and TensorFlow. The Python backend is controlled by a web UI created with Vue.js and and Flask. The classes that are used for finding POI are selectable from the dropdown boxes provided on the web UI.  
The classes are split into 2 categories:
### Color Classes:
The colors utilize two different models for classification. The first model uses YOLO's k-Nearest Neighbor classifier and is capable of detecting 8 colors:
* Black
* Red
* Blue
* Yellow
* Orange
* Green
* White
* Violet

The second model attempts to implement a new way of classifying colors by utilizing a fuzzy membership classification model. This feature is one of the key highlights of this project. This model can identify 11 colors based on the X11 color groups.
* Pink
* Red
* Orange
* Yellow
* Brown
* Green
* Blue
* Cyan
* Purple/Violet/Magenta
* White
* Gray/Black

### Clothing Classes:
The clothing classification used uses YOLO's DarkNet object detection. It detects:
* Person
* Man
* Woman
* Shirt
* Jacket
* Suit
* Trousers
* Jeans
* Dress
* Skirt
* Footwear

## Setup, Installation, and User Guide

## Changelog

## Future Plans

## Credits
This project wouldn't have been possible without the use of the following modules, frameworks and algorithms:
* YOLOv4
* DarkNet
* TensorFlow
* Python in tandem with Anaconda
* Tkinter
* Vue.JS framework
* ArchitectUI by DashboardPack
* Flask framework
* SyncFusion for Vue, notably their NodeJS FileManager
