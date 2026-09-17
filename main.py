import cv2
import mediapipe as mp
import time
import math


def raised_fingertips(hand, width, height):
    """Retorna os índices das pontas dos dedos que parecem estendidos."""
    points = [(point.x * width, point.y * height) for point in hand]

    def distance(a, b):
        return math.dist(points[a], points[b])

    def is_straight(a, joint, b):
        ux = points[a][0] - points[joint][0]
        uy = points[a][1] - points[joint][1]
        vx = points[b][0] - points[joint][0]
        vy = points[b][1] - points[joint][1]
        length = math.hypot(ux, uy) * math.hypot(vx, vy)
        return length > 0 and (ux * vx + uy * vy) / length < -0.85

    tips = []
    for base, joint, end_joint, tip in (
        (5, 6, 7, 8),
        (9, 10, 11, 12),
        (13, 14, 15, 16),
        (17, 18, 19, 20),
    ):
        if (is_straight(base, joint, end_joint)
                and is_straight(joint, end_joint, tip)
                and distance(0, tip) > distance(0, joint) * 1.1):
            tips.append(tip)

    if (is_straight(1, 2, 3) and is_straight(2, 3, 4)
            and distance(4, 5) > distance(3, 5) * 1.2):
        tips.append(4)
    return tips


def label_position(hand, width, height, text):
    tips = raised_fingertips(hand, width, height)
    if len(tips) == 1:
        anchor = hand[tips[0]]
        center_x = anchor.x * width
        top_y = anchor.y * height
    else:
        center_x = (min(p.x for p in hand) + max(p.x for p in hand)) * width / 2
        top_y = min(p.y for p in hand) * height

    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    )
    x = int(center_x - text_width / 2)
    y = int(top_y - 20)
    x = max(5, min(x, width - text_width - 5))
    y = max(text_height + 5, min(y, height - baseline - 5))
    return x, y


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

            height, width, _ = frame.shape

            connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

            for hand_index, hand in enumerate(result.hand_landmarks):

                handedness = result.handedness[hand_index][0]
                hand_label = handedness.category_name

                if hand_label == "Left":
                    hand_label = "Esquerda"
                else:
                    hand_label = "Direita"

                x, y = label_position(hand, width, height, hand_label)

                cv2.putText(
                    frame,
                    hand_label,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )
                for connection in connections:
                    start = hand[connection.start]
                    end = hand[connection.end]

                    x1 = int(start.x * width)
                    y1 = int(start.y * height)

                    x2 = int(end.x * width)
                    y2 = int(end.y * height)

                    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

                for landmark in hand:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            current_time = time.perf_counter()
            fps = 1 / (current_time - previous_time)
            previous_time = current_time

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.imshow("Hand Tracker", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()