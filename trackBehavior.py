import face_recognition
import cv2
import mediapipe as mp

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Load an image with known faces
known_image = face_recognition.load_image_file("Fauzan.jpg")
known_face_encoding = face_recognition.face_encodings(known_image)[0]
known_face_names = ["Fauzan"]

# Read a video stream from the webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform pose detection
    results = pose.process(rgb_frame)
    behavior = "Normal"

    if results.pose_landmarks:
        # Extract landmarks for nose and right hand
        landmarks = results.pose_landmarks.landmark
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX]

        # Convert normalized coordinates to pixel values
        frame_height, frame_width, _ = frame.shape
        nose_coords = (int(nose.x * frame_width), int(nose.y * frame_height))
        right_hand_coords = (int(right_hand.x * frame_width), int(right_hand.y * frame_height))

        # Calculate distance between right hand and nose
        distance = ((nose_coords[0] - right_hand_coords[0]) ** 2 + (nose_coords[1] - right_hand_coords[1]) ** 2) ** 0.5

        # Define behavior based on distance
        if distance < 50:  # Threshold for "scratching nose"
            behavior = "Scratching Nose"

        # Draw landmarks and behavior text
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.putText(frame, behavior, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Recognize faces
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces([known_face_encoding], face_encoding)
        name = "Unknown"
        if True in matches:
            name = known_face_names[0]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow("Video", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
