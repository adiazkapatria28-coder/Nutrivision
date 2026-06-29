import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def estimate_height(image_path):
    MODEL_PATH = "pose_landmarker_lite.task"    

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

    image = mp.Image.create_from_file(image_path)
    cv_image = cv2.imread(image_path)

    if cv_image is None:
        return None

    result = detector.detect(image)

    if len(result.pose_landmarks) > 0:
        landmarks = result.pose_landmarks[0]

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

        padding = 80

        x_min = max(0, int(min(xs) * width) - padding)
        x_max = min(width, int(max(xs) * width) + padding)

        y_min = max(0, int(min(ys) * height) - padding)
        y_max = min(height, int(max(ys) * height) + padding)

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

        torso = cv_image[
            y_min:y_max,
            x_min:x_max
        ]

        h, w = torso.shape[:2]

        crop_x = int(w * 0.15)
        crop_y = int(h * 0.05)

        crop_w = int(w * 0.70)
        crop_h = int(h * 0.90)

        torso = torso[
            crop_y:crop_y + crop_h,
            crop_x:crop_x + crop_w
        ]

        # Convert torso to HSV
        hsv = cv2.cvtColor(
            torso,
            cv2.COLOR_BGR2HSV
        )

        # Detect white colors
        lower_white = (0, 0, 170)
        upper_white = (180, 60, 255)

        mask = cv2.inRange(
            hsv,
            lower_white,
            upper_white
        )

        # Remove small gaps/noise
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (9, 9)
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )
        # ============================================
        # Detect paper by expanding from image center
        # ============================================

        height, width = mask.shape

        center_x = width // 2
        center_y = height // 2

        # Check that the paper is actually near the center
        if mask[center_y, center_x] != 255:
            print("Paper is not centered.")
            return None
        else:

            # Expand left
            left = center_x
            while left > 0 and mask[center_y, left] == 255:
                left -= 1

            # Expand right
            right = center_x
            while right < width - 1 and mask[center_y, right] == 255:
                right += 1

            # Expand upward
            top = center_y
            while top > 0 and mask[top, center_x] == 255:
                top -= 1

            # Expand downward
            bottom = center_y
            while bottom < height - 1 and mask[bottom, center_x] == 255:
                bottom += 1

            paper_width = right - left
            paper_height = bottom - top

            cv2.rectangle(
                torso,
                (left, top),
                (right, bottom),
                (0, 255, 0),
                2
            )

            cm_per_pixel = 29.7 / paper_height
            estimated_height = pixel_height * cm_per_pixel

            print(
                "Estimated Height:",
                round(estimated_height, 1),
                "cm"
            )

            return round(estimated_height, 1)
    else:
        print("No pose detected.")
        return None