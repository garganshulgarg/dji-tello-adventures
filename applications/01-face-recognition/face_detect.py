import cv2 as cv
import os

# Load the pre-trained Haar Cascade classifiers for face and eye detection
face_cascade = cv.CascadeClassifier('data/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('data/haarcascade_eye.xml')

# Define the input directory and output directory
input_dir = 'data/images'
output_dir = 'output/images'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get the list of image files in the input directory
image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Process each image file in the input directory
for image_file in image_files:
    # Read the input image
    img = cv.imread(os.path.join(input_dir, image_file))

    if img is None:
        continue  # Skip if the image is not read correctly
    
    # Convert the image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print(f'Processing {image_file}: {len(faces)} faces detected.')

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

    # Save the processed image in the output directory
    output_path = os.path.join(output_dir, f'processed_{image_file}')
    cv.imwrite(output_path, img)

    # Optionally, display the output image
    # cv.imshow('Processed Image', img)
    # cv.waitKey(0)

cv.destroyAllWindows()  # Close all OpenCV windows
