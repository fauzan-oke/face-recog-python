import face_recognition
import cv2
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the image of the known face
known_face_image = face_recognition.load_image_file("fauzan.jpg")
known_face_encoding = face_recognition.face_encodings(known_face_image)[0]
threshold = 0.6  # Set a threshold for face verification (you can adjust this value)

@app.route("/verify_face", methods=["POST"])
def verify_face():
    try:
        # Get the uploaded image from the POST request
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        photo_image = face_recognition.load_image_file(request.files["image"])
        face_locations = face_recognition.face_locations(photo_image)

        if len(face_locations) == 0:
            return jsonify({"result": "No face found in the photo"}), 200
        else:
            photo_face_encoding = face_recognition.face_encodings(photo_image, face_locations)[0]

            # Compare the face with the known face
            face_distance = face_recognition.face_distance([known_face_encoding], photo_face_encoding)

            if face_distance[0] <= threshold:
                return jsonify({"result": "Face verified. It's the known person"}), 200
            else:
                return jsonify({"result": "Face not verified. It's not the known person"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
