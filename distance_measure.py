import cv2
import math

# Load the Haar Cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_face.xml')

cap = cv2.VideoCapture(0)

# Constants for distance measurement
KNOWN_DISTANCE = 100  # Define a known distance (in cm) from the camera to the face
KNOWN_FACE_WIDTH = 15  # Define the width of the face (in cm) at the known distance

while True:
    # Read the frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Calculate the distance to the face using the known distance and face width
        face_width_pixels = w
        distance = (KNOWN_FACE_WIDTH * cap.get(3)) / (2 * face_width_pixels * math.tan(cap.get(4) * math.pi / 360))
        
        # Draw the distance on the frame
        cv2.putText(frame, f"Distance: {distance:.2f} cm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the frame with distance information
    cv2.imshow('Distance Measurement', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
