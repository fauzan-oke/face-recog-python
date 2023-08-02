import face_recognition
import cv2

# Load an image with known faces
known_image = face_recognition.load_image_file("Zanu.jpg")

# Encode the known face(s)
known_face_encoding = face_recognition.face_encodings(known_image)[0]
known_face_names = ["Zanu"]  # Provide names for the known faces (can be a list of names)

# Read a video stream from the webcam (change 0 to the path of a video file if using pre-recorded video)
video_capture = cv2.VideoCapture(0)

while True:
    # Read a single frame from the video stream
    ret, frame = video_capture.read()

    # Find all face locations and face encodings in the frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_encoding in face_encodings:
        # Compare the face with known faces
        matches = face_recognition.compare_faces([known_face_encoding], face_encoding)
        name = "Unknown"

        if True in matches:
            # Find the index of the first match and use the corresponding name
            match_index = matches.index(True)
            name = known_face_names[match_index]

        # Draw a rectangle around the face and label it with the name
        top, right, bottom, left = face_locations[0]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the OpenCV window
video_capture.release()
cv2.destroyAllWindows()
