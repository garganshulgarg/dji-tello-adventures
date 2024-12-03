import cv2 as cv

# Initialize the video capture object to read from the default camera
video_capture = cv.VideoCapture(1)

# Load the pre-trained Haar Cascade classifiers for face and eye detection
face_cascade = cv.CascadeClassifier('data/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('data/haarcascade_eye.xml')

while True:
    # Capture frame-by-frame from the video stream
    ret, frame = video_capture.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to grayscale for face detection
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.5, minNeighbors=5)
    print(f'Number of faces detected: {len(faces)}')

    # Loop through each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Get the region of interest (ROI) for the face in both grayscale and color frames
        face_gray = gray_frame[y:y+h, x:x+w]
        face_color = frame[y:y+h, x:x+w]
        
        # Detect eyes within the face ROI
        eyes = eye_cascade.detectMultiScale(face_gray)
        
        # Loop through each detected eye
        for (ex, ey, ew, eh) in eyes:
            # Draw a rectangle around the eye
            cv.rectangle(face_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    # Display the resulting frame with detected faces and eyes
    cv.imshow('Video Face Detection', frame)
    
    # Break the loop if 'q' key is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
video_capture.release()
cv.destroyAllWindows()