import face_recognition
import cv2

# Load the image of the known face
known_face_image = face_recognition.load_image_file("fauzan.jpg")
known_face_encoding = face_recognition.face_encodings(known_face_image)[0]

# Load the photo containing the face to be verified
photo_image = face_recognition.load_image_file("fauzan_verify.jpg")
face_locations = face_recognition.face_locations(photo_image)
if len(face_locations) == 0:
    print("No face found in the photo.")
else:
    photo_face_encoding = face_recognition.face_encodings(photo_image, face_locations)[0]

    # Compare the face with the known face
    face_distance = face_recognition.face_distance([known_face_encoding], photo_face_encoding)
    threshold = 0.6  # Set a threshold for face verification (you can adjust this value)
    if face_distance[0] <= threshold:
        print("Face verified. It's the known person.")
    else:
        print("Face not verified. It's not the known person.")
