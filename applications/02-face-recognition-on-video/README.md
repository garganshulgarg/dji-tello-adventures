# Face Recognition Using Webcam

## What You Need
- Python 3.x installed.
- A terminal (for Mac users).
- Install OpenCV:
```
brew install opencv
```

## Setting Up Your Local Environment

Go to the folder `dji-tello-adventures/applications/02-face-recognition-on-video`

### Steps to Set Up the Virtual Environment

#### 1️⃣ **Check Python**
Make sure Python is installed:
```bash
python3 --version
```

#### 2️⃣ Create a Virtual Environment
Go to your project folder and run:
```
python3 -m venv venv
```

#### 3️⃣ Activate the Virtual Environment
Run:
```
source venv/bin/activate
```

#### 4️⃣ Install Required Libraries
Run:
```
pip3 install -r requirements.txt
```

#### 5️⃣ Deactivate the Virtual Environment
When done, run:
```
deactivate
```

## Concepts

We will use an OpenCV tutorial for detecting objects with Haar Cascades. You can find more details at the link below:
[Face Detection using Haar Cascade](./docs/face_detection_using_haar_cascade.md)



## Run

To run the code, execute the following command:
```
python video_face_detection.py
```

### Output

After running the code, you should see a window displaying the video feed from your webcam with detected faces highlighted by rectangles. Below is a sample output video demonstrating the face recognition in action:

![Video preview](output/sample_output.gif)