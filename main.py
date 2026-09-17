import cv2
import mediapipe as mp
import time


GREEN = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FINGERTIPS = (4, 8, 12, 16, 20)
PALM_POINTS = (5, 9, 13, 17)


def draw_hand(frame, hand, category, connections):
    height, width = frame.shape[:2]
    points = [(int(p.x * width), int(p.y * height)) for p in hand]

    for connection in connections:
        cv2.line(
            frame, points[connection.start], points[connection.end],
            GREEN, 1, cv2.LINE_AA
        )

    for index, point in enumerate(points):
        cv2.circle(frame, point, 5, GREEN, 0, cv2.LINE_AA)
        if index in FINGERTIPS:
            cv2.circle(frame, point, 10, GREEN, 1, cv2.LINE_AA)

    knuckles_x = sum(hand[i].x for i in PALM_POINTS) / len(PALM_POINTS)
    knuckles_y = sum(hand[i].y for i in PALM_POINTS) / len(PALM_POINTS)
    center_x = int((hand[0].x + knuckles_x) * 0.5 * width)
    center_y = int((hand[0].y + knuckles_y) * 0.5 * height)
    label = "L" if category == "Left" else "R"
    (text_width, text_height), _ = cv2.getTextSize(label, FONT, 0.6, 2)

    cv2.circle(frame, (center_x, center_y), 16, GREEN, 1, cv2.LINE_AA)
    cv2.putText(
        frame, label,
        (center_x - text_width // 2, center_y + text_height // 2),
        FONT, 0.6, GREEN, 2, cv2.LINE_AA
    )


options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1
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