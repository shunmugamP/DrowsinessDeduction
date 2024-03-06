from scipy.spatial import distance as dist #to find dist of coordinates
from imutils import face_utils #get the coord of different parts
import imutils #resize
import dlib #face detection & Landmarks on face
import cv2 #frame acquit
import numpy as np
import os

frequency = 2500
duration = 1000
#For Audio path
# import soundfile as sf
# import sounddevice as sd

# # Specify the path to the audio file
# audio_path = 'beep-01a.wav'
# # Load the audio file
# audio_data, fs = sf.read(audio_path)
#Audio path end

nose_image = cv2.imread('nose.png', cv2.IMREAD_UNCHANGED)
def eyeAspectRatio(eye):
    #vertical
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    #horizontal
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

count = 0
earThresh = 0.3 #distance between vertical eye coordinate Threshold
earFrames = 48 #consecutive frames for eye closure
shapePredictor = "shape_predictor_68_face_landmarks.dat"

# Get the camera index from the environment variable
camera_index = int(os.getenv('CAMERA_INDEX', '0'))
print(f"Using camera index: {camera_index}")

# def find_camera_index():
#     # Try camera indices from 0 to 10
#     for i in range(10):
#         cap = cv2.VideoCapture(i)
#         if cap.isOpened():
#             print(f"Found camera at index {i}")
#             cap.release()
#             return i
#     print("Failed to find camera")
#     return None

# # Find the camera index
# camera_index = find_camera_index()
# print(f"Using camera index---: {camera_index}")


cam = cv2.VideoCapture(camera_index) #camera init
if not cam.isOpened():
    print("Error: Could not open camera.")
    exit()
else:
    print("Camera opened successfully.")

detector = dlib.get_frontal_face_detector() #Face detection alg.
predictor = dlib.shape_predictor(shapePredictor) #68 Landmarks load

#get the coord of left & right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]

# Nose draw
while True:
    _, frame = cam.read() #read frame from camera
    if frame is None:
        print("Error: Could not read frame from camera.")
        break

    frame = imutils.resize(frame, width=450) #resizing 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #color 2 Gray

    rects = detector(gray, 0) #detect the face

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        nose_points = shape[nStart:nEnd]

        # Calculate the width and height of the nose image based on the distance between the landmark points
        nose_width = int(np.linalg.norm(nose_points[6] - nose_points[0]) * 1)
        nose_height = int(nose_width * nose_image.shape[0] / nose_image.shape[1])

        # Resize the nose image to fit the nose region
        nose_image_resized = cv2.resize(nose_image, (nose_width, nose_height))

        # Calculate the position to overlay the nose image
        nose_x = nose_points[0][0] - nose_width // 2
        nose_y = nose_points[0][1] - nose_height // 2 + nose_height // 4

        # Ensure the nose image fits within the frame
        if nose_x < 0:
            nose_x = 0
        if nose_y < 0:
            nose_y = 0
        if nose_x + nose_width > frame.shape[1]:
            nose_x = frame.shape[1] - nose_width
        if nose_y + nose_height > frame.shape[0]:
            nose_y = frame.shape[0] - nose_height

        # Calculate the angle between the eyes
        eye_slope = (shape[45][1] - shape[36][1]) / (shape[45][0] - shape[36][0])
        eye_angle = np.degrees(np.arctan(eye_slope))

        # Rotate the nose image
        rotation_matrix = cv2.getRotationMatrix2D((nose_width // 2, nose_height // 2), -eye_angle, 1.0)
        nose_image_rotated = cv2.warpAffine(nose_image_resized, rotation_matrix, (nose_width, nose_height))


        # For draw 68 points
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Overlay the rotated nose image onto the frame
        for c in range(0, 3):
            if nose_image_rotated is not None:
                frame[nose_y:nose_y + nose_height, nose_x:nose_x + nose_width, c] = \
        nose_image_rotated[:, :, c] * (nose_image_rotated[:, :, 3] / 255.0) + \
        frame[nose_y:nose_y + nose_height, nose_x:nose_x + nose_width, c] * \
        (1.0 - nose_image_rotated[:, :, 3] / 255.0)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eyeAspectRatio(leftEye)
        rightEAR = eyeAspectRatio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

        # print("ear", ear, " count ", count, " earThresh ", earThresh, " earFrames ", earFrames)
        if ear < earThresh:
            count += 1

            if count >= earFrames:
                cv2.putText(frame, "DROWSINESS DETECTED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
   
        # Play the audio
                # sd.play(audio_data, fs)
                # sd.wait()
        else:
            count = 0

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
