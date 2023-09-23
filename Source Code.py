import cv2
import pyautogui
import mediapipe as mp
from math import hypot

# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose

# Setup the Pose function for videos.
pose_video = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.7,
                          min_tracking_confidence=0.7)

# Initialize mediapipe drawing class.
mp_drawing = mp.solutions.drawing_utils

# Function to detect pose in an image and draw landmarks.
def detectPose(image, pose, draw=False):
    output_image = image.copy()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)

    if results.pose_landmarks and draw:
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255),
                                                                               thickness=3, circle_radius=3),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(49, 125, 237),
                                                                                 thickness=2, circle_radius=2))
    return output_image, results

# Function to check if hands are joined.
def checkHandsJoined(image, results, draw=False):
    height, width, _ = image.shape
    output_image = image.copy()
    left_wrist_landmark = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * width,
                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * height)
    right_wrist_landmark = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * width,
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * height)
    euclidean_distance = int(hypot(left_wrist_landmark[0] - right_wrist_landmark[0],
                                   left_wrist_landmark[1] - right_wrist_landmark[1]))

    hand_status = 'Hands Not Joined'
    color = (0, 0, 255)

    if euclidean_distance < 130:
        hand_status = 'Hands Joined'
        color = (0, 255, 0)

    if draw:
        cv2.putText(output_image, hand_status, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
        cv2.putText(output_image, f'Distance: {euclidean_distance}', (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 2, color, 3)

    return output_image, hand_status

# Function to check horizontal position.
def checkLeftRight(image, results, draw=False):
    height, width, _ = image.shape
    output_image = image.copy()
    left_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width)
    right_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width)

    horizontal_position = None

    if (right_x <= width // 2 and left_x <= width // 2):
        horizontal_position = 'Left'
    elif (right_x >= width // 2 and left_x >= width // 2):
        horizontal_position = 'Right'
    elif (right_x >= width // 2 and left_x <= width // 2):
        horizontal_position = 'Center'

    if draw:
        cv2.putText(output_image, horizontal_position, (5, height - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
        cv2.line(output_image, (width // 2, 0), (width // 2, height), (255, 255, 255), 2)

    return output_image, horizontal_position

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to check jumping and crouching
def checkJumpCrouch(image, results, MID_Y, draw=False):
    height, width, _ = image.shape
    output_image = image.copy()
    MID_Y = 300

    # Calculate the positions of the shoulders
    left_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height)
    right_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height)

    # Calculate the actual MID_Y
    actual_mid_y = abs(right_y + left_y) // 2

    # Define the bounds for jump and crouch detection
    lower_bound = MID_Y - 15
    upper_bound = MID_Y + 100

    # Determine the posture
    posture = None

    if actual_mid_y < lower_bound:
        posture = 'Jumping'
    elif actual_mid_y > upper_bound:
        posture = 'Crouching'
    else:
        posture = 'Standing'

    if draw:
        # Draw the horizontal line
        cv2.line(output_image, (0, MID_Y), (width, MID_Y), (255, 255, 255), 2)
        # Draw the posture text
        cv2.putText(output_image, posture, (5, height - 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)

    return output_image, posture

# Initialize the VideoCapture object to read from the webcam.
camera_video = cv2.VideoCapture(0)
camera_video.set(3, 1280)
camera_video.set(4, 960)

# Create named window for resizing purposes.
cv2.namedWindow('Subway Surfers with Pose Detection', cv2.WINDOW_NORMAL)

# Initialize variables and parameters.
time1 = 0
game_started = False
x_pos_index = 1
y_pos_index = 1
MID_Y = None
counter = 0
num_of_frames = 10

# Main loop for processing frames.
while camera_video.isOpened():
    ok, frame = camera_video.read()

    if not ok:
        continue

    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape

    frame, results = detectPose(frame, pose_video, draw=game_started)

    if results.pose_landmarks:
        if game_started:
            frame, _ = checkLeftRight(frame, results, draw=True)
        else:
            cv2.putText(frame, 'JOIN BOTH HANDS TO START THE GAME', (5, frame_height - 10), cv2.FONT_HERSHEY_PLAIN,
                        2, (0, 255, 0), 3)

        if checkHandsJoined(frame, results)[1] == 'Hands Joined' and not game_started:
            MID_Y = int((results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y +
                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2)
            game_started = True

        if game_started:
            if x_pos_index < 0:
                x_pos_index = 0
            elif x_pos_index > 2:
                x_pos_index = 2

            if y_pos_index < 0:
                y_pos_index = 0
            elif y_pos_index > 2:
                y_pos_index = 2

            frame, posture = checkJumpCrouch(frame, results, MID_Y, draw=True)

            if posture == 'Jumping' and counter % num_of_frames == 0:
                if x_pos_index == 0:
                    pyautogui.press('left')
                elif x_pos_index == 2:
                    pyautogui.press('right')
                else:
                    pyautogui.press('up')

            elif posture == 'Crouching' and counter % num_of_frames == 0:
                pyautogui.press('down')

            counter += 1

    cv2.imshow('Subway Surfers with Pose Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and destroy any OpenCV windows.
camera_video.release()
cv2.destroyAllWindows()
