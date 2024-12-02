import cv2 as cv

# Load the pre-trained Haar Cascade classifiers for face and eye detection
face_cascade = cv.CascadeClassifier('data/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('data/haarcascade_eye.xml')

# Read the input image
img = cv.imread('data/test_image.jpg')

# Convert the image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
print(len(faces))  # Print the number of faces detected

# Loop through each detected face
for (x, y, w, h) in faces:
    # Draw a rectangle around the face
    cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Get the region of interest (ROI) for the face in both grayscale and color images
    eye_gray = gray[y:y+h, x:x+w]
    eye_color = img[y:y+h, x:x+w]
    
    # Detect eyes within the face ROI
    eyes = eye_cascade.detectMultiScale(eye_gray)
    
    # Loop through each detected eye
    for (ex, ey, ew, eh) in eyes:
        # Draw a rectangle around the eye
        cv.rectangle(eye_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

# Display the output image with detected faces and eyes
cv.imshow('img', img)
cv.waitKey(0)  # Wait for a key press to close the window
cv.destroyAllWindows()  # Close all OpenCV windows
