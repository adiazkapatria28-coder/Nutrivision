import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "pose_landmarker_lite.task"
IMAGE_PATH = "test.jpeg"

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    num_poses=1
)

detector = vision.PoseLandmarker.create_from_options(
    options
)

image = mp.Image.create_from_file(IMAGE_PATH)

cv_image = cv2.imread(IMAGE_PATH)

result = detector.detect(image)

print(
    "Image size:",
    image.width,
    image.height
)

print("Number of poses detected:", len(result.pose_landmarks))

if len(result.pose_landmarks) > 0:
    print("Success!")
    print("Number of landmarks:", len(result.pose_landmarks[0]))
    landmarks = result.pose_landmarks[0]

    print("Left shoulder:", landmarks[11].x, landmarks[11].y)
    print("Right shoulder:", landmarks[12].x, landmarks[12].y)
    print("Left hip:", landmarks[23].x, landmarks[23].y)
    print("Right hip:", landmarks[24].x, landmarks[24].y)

    height = image.height
    width = image.width

    landmarks = result.pose_landmarks[0]

    xs = [
        landmarks[11].x,
        landmarks[12].x,
        landmarks[23].x,
        landmarks[24].x
    ]

    ys = [
        landmarks[11].y,
        landmarks[12].y,
        landmarks[23].y,
        landmarks[24].y
    ]

    x_min = int(min(xs) * width) - 80
    x_max = int(max(xs) * width) + 80

    y_min = int(min(ys) * height) - 80
    y_max = int(max(ys) * height) + 80

    print(x_min, x_max)
    print(y_min, y_max)

    top_y = min(
        landmark.y
        for landmark in landmarks
    )

    bottom_y = max(
        landmark.y
        for landmark in landmarks
    )

    pixel_height = (
        bottom_y - top_y
    ) * image.height

    print("Top Y:", top_y)
    print("Bottom Y:", bottom_y)

    print(
        "Estimated body height:",
        round(pixel_height, 2),
        "pixels"
    )

    torso = cv_image[
        y_min:y_max,
        x_min:x_max
    ]

    gray = cv2.cvtColor(
        torso,
        cv2.COLOR_BGR2GRAY
    )

    for value in [180, 190, 200, 210]:
        _, thresh = cv2.threshold(
        gray,
        220,
        255,
        cv2.THRESH_BINARY
    )
    cv2.imshow(
        "White Areas",
        thresh
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No pose detected.")