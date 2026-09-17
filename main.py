import cv2
import mediapipe as mp
import time
import math


GREEN = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FINGERTIPS = (4, 8, 12, 16, 20)
PALM_POINTS = (5, 9, 13, 17)
REFERENCE_PALM_SIZE = 100
MIN_HAND_SCALE = 0.35
MAX_HAND_SCALE = 2.5


def draw_hand(frame, hand, category, connections):
    height, width = frame.shape[:2]
    points = [(int(p.x * width), int(p.y * height)) for p in hand]

    palm_width = math.hypot(
        (hand[5].x - hand[17].x) * width,
        (hand[5].y - hand[17].y) * height
    )
    palm_length = math.hypot(
        (hand[0].x - hand[9].x) * width,
        (hand[0].y - hand[9].y) * height
    )
    scale = max(MIN_HAND_SCALE, min(
        max(palm_width, palm_length) / REFERENCE_PALM_SIZE, MAX_HAND_SCALE
    ))
    point_radius = max(1, round(5 * scale))
    fingertip_radius = max(2, round(10 * scale))
    palm_radius = max(5, round(16 * scale))
    line_thickness = max(1, round(scale))
    text_scale = 0.6 * scale
    text_thickness = max(1, round(2 * scale))

    for connection in connections:
        cv2.line(
            frame, points[connection.start], points[connection.end],
            GREEN, line_thickness, cv2.LINE_AA
        )

    for index, point in enumerate(points):
        cv2.circle(frame, point, point_radius, GREEN, line_thickness, cv2.LINE_AA)
        if index in FINGERTIPS:
            cv2.circle(frame, point, fingertip_radius, GREEN, line_thickness, cv2.LINE_AA)

    knuckles_x = sum(hand[i].x for i in PALM_POINTS) / len(PALM_POINTS)
    knuckles_y = sum(hand[i].y for i in PALM_POINTS) / len(PALM_POINTS)
    center_x = int((hand[0].x + knuckles_x) * 0.5 * width)
    center_y = int((hand[0].y + knuckles_y) * 0.5 * height)
    label = "L" if category == "Left" else "R"
    (text_width, text_height), _ = cv2.getTextSize(label, FONT, text_scale, text_thickness)

    cv2.circle(frame, (center_x, center_y), palm_radius, GREEN, line_thickness, cv2.LINE_AA)
    cv2.putText(
        frame, label,
        (center_x - text_width // 2, center_y + text_height // 2),
        FONT, text_scale, GREEN, text_thickness, cv2.LINE_AA
    )


options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7
)

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    raise IOError("Cannot open webcam")

try:
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
        connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
        previous_time = time.perf_counter()

        while True:
            success, frame = camera.read()

            if not success:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp = int(time.monotonic() * 1000)
            result = detector.detect_for_video(image, timestamp)

            for hand_index, hand in enumerate(result.hand_landmarks):
                category = result.handedness[hand_index][0].category_name
                draw_hand(frame, hand, category, connections)

            current_time = time.perf_counter()
            fps = 1 / (current_time - previous_time)
            previous_time = current_time

            cv2.putText(
                frame,
                f"FPS: {int(fps)}",
                (20, 40),
                FONT,
                0.8,
                GREEN,
                2,
                cv2.LINE_AA
            )

            cv2.imshow("Hand Tracker", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()