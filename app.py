from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)

# Cheating detection thresholds
WARNING_HAND_MOVEMENT = 0.08  # Warning level for hand movement
MAX_HAND_MOVEMENT = 0.15  # Cheating level for hand movement
MAX_HEAD_TILT = 0.03  # Cheating level for head tilt

# YOLOv8 model initialization
model = YOLO('yolov8n-pose.pt')


def detect_cheating(keypoints, image_width):
    # YOLOv8-pose keypoint indices
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    NOSE = 0
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6

    # Get x-coordinates normalized by image width
    left_hand_x = keypoints[LEFT_WRIST][0] / image_width
    right_hand_x = keypoints[RIGHT_WRIST][0] / image_width
    nose_x = keypoints[NOSE][0] / image_width
    left_shoulder_x = keypoints[LEFT_SHOULDER][0] / image_width
    right_shoulder_x = keypoints[RIGHT_SHOULDER][0] / image_width

    # Calculate shoulder midpoint x
    shoulder_midpoint_x = (left_shoulder_x + right_shoulder_x) / 2

    # Calculate hand movement
    left_hand_movement = abs(left_hand_x - left_shoulder_x)
    right_hand_movement = abs(right_hand_x - right_shoulder_x)

    # Calculate head tilt
    head_tilt = abs(nose_x - shoulder_midpoint_x)

    # Cheating detection logic
    if head_tilt > MAX_HEAD_TILT:
        return "cheating"

    if left_hand_movement > MAX_HAND_MOVEMENT or right_hand_movement > MAX_HAND_MOVEMENT:
        return "cheating"
    elif left_hand_movement > WARNING_HAND_MOVEMENT or right_hand_movement > WARNING_HAND_MOVEMENT:
        return "warning"

    return "not_cheating"


@app.route('/detect', methods=['POST'])
def detect():
    # Check for file in POST request
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    image_file = request.files['image']

    # Convert image to OpenCV format
    np_img = np.frombuffer(image_file.read(), np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    image_height, image_width, _ = frame.shape

    # Run YOLOv8 pose estimation
    results = model(frame, stream=True)

    response = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        keypoints = result.keypoints.cpu().numpy()

        for person_idx, (box, kpts) in enumerate(zip(boxes, keypoints)):
            # Detect cheating for this person
            cheating_status = detect_cheating(kpts.data[0], image_width)

            # Prepare response
            response.append({
                "person_id": person_idx,
                "cheating_status": cheating_status
            })
    
    return jsonify(response)


# Run Flask app
if __name__ == '_main_':
    app.run(host='0.0.0.0', port=5000)