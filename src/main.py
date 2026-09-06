import cv2
import mediapipe as mp
import numpy as np
import joblib
import pandas as pd
import os
from typing import Tuple, Optional, Dict, Any

# Physical and configuration constants
GATHER_DATA: bool = False
VIDEO_PATH: str = 'video.mp4' # Change it to ur video path
MODEL_PATH: str = 'model.pkl' # Change it to ur model path
DATASET_PATH: str = 'dataset.csv' # Change it to ur dataset path
IS_GOOD_SHOT: int = 1


def calculate_angle(a: Any, b: Any, c: Any) -> Tuple[Optional[float], Optional[str]]:
    """Calculates angles between joints. Returns angle in degrees and its string representation."""
    if a.visibility > 0.55 and b.visibility > 0.55 and c.visibility > 0.55:
        a_pos = np.array([a.x, a.y, a.z])
        b_pos = np.array([b.x, b.y, b.z])
        c_pos = np.array([c.x, c.y, c.z])

        vector1 = a_pos - b_pos
        vector2 = c_pos - b_pos

        cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        rad = np.arccos(cos_angle)
        deg = float(np.rad2deg(rad))
        deg_text = str(int(deg))

        return deg, deg_text
    return None, None


def draw_skeleton(a: Tuple[int, int], b: Tuple[int, int], c: Tuple[int, int], deg: Optional[float], deg_text: Optional[str], frame: np.ndarray) -> None:
    """Draws lines representing connections and circles in the middle of a body part."""
    if deg is not None and deg_text is not None:
        cv2.line(frame, a, b, (0, 255, 0), 4)
        cv2.line(frame, b, c, (0, 255, 0), 4)
        cv2.circle(frame, a, 6, (0, 0, 255), -1)
        cv2.circle(frame, b, 6, (0, 0, 255), -1)
        cv2.circle(frame, c, 6, (0, 0, 255), -1)
        cv2.putText(frame, deg_text, (b[0] + 20, b[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


def data_or_model(switch: bool, dip: float, release_angle: float, is_good_shot: int, model: Optional[Any] = None) -> Tuple[str, Tuple[int, int, int]]:
    """Switches between data gathering mode and model inference mode."""
    if switch:
        file_exist: bool = os.path.isfile(DATASET_PATH)
        with open(DATASET_PATH, 'a', encoding='utf-8') as file:
            if not file_exist:
                file.write("dip,release,is_good\n")
            file.write(f"{int(dip)},{int(release_angle)},{is_good_shot}\n")

        # Return default values when writing to CSV
        return "Waiting...", (255, 255, 255)

    else:
        if model is None:
            raise ValueError("Model must be provided when GATHER_DATA is False")

        features = pd.DataFrame([[dip, release_angle]], columns=['dip', 'release'])
        model_score = model.predict(features)[0]

        if model_score == 1:
            return "Good angles", (0, 255, 0)
        else:
            return "Bad angles", (0, 0, 255)


def main() -> None:
    """Main execution loop for the basketball shot analysis."""
    print("Initializing Basketball Vision System...")

    # Model setup
    ai_model: Optional[Any] = None
    if not GATHER_DATA:
        ai_model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")

    video = cv2.VideoCapture(VIDEO_PATH)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Tracking variables
    dip1: float = 360.0
    release_angle1: float = 0.0
    frame_counter: int = 0
    dip_frame: int = 0
    release_frame: int = 0

    # Initial statuses
    model_status: str = "Waiting..."
    status_color2: Tuple[int, int, int] = (0, 255, 255)

    try:
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break

            fps: float = video.get(cv2.CAP_PROP_FPS)
            frame_counter += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks is not None:
                landmarks = results.pose_landmarks.landmark
                height, width, _ = frame.shape

                # Extracting body parts
                body_parts: Dict[str, Any] = {
                    'l_hip': landmarks[23], 'r_hip': landmarks[24],
                    'l_knee': landmarks[25], 'r_knee': landmarks[26],
                    'l_ankle': landmarks[27], 'r_ankle': landmarks[28],
                    'l_shoulder': landmarks[11], 'r_shoulder': landmarks[12],
                    'l_elbow': landmarks[13], 'r_elbow': landmarks[14],
                    'l_wrist': landmarks[15], 'r_wrist': landmarks[16]
                }

                # Converting to screen pixels
                pixels: Dict[str, Tuple[int, int]] = {}
                for name, landmark in body_parts.items():
                    pixels[name] = (int(landmark.x * width), int(landmark.y * height))

                # Calculating angles
                deg1, deg1_text = calculate_angle(body_parts['l_hip'], body_parts['l_knee'], body_parts['l_ankle'])
                deg3, deg3_text = calculate_angle(body_parts['r_hip'], body_parts['r_knee'], body_parts['r_ankle'])
                deg2, deg2_text = calculate_angle(body_parts['r_shoulder'], body_parts['r_elbow'], body_parts['r_wrist'])
                deg4, deg4_text = calculate_angle(body_parts['l_shoulder'], body_parts['l_elbow'], body_parts['l_wrist'])

                # Tracking dip
                current_dip: Optional[float] = None
                if deg1 is not None and deg3 is not None:
                    current_dip = min(deg1, deg3)
                elif deg1 is not None:
                    current_dip = deg1
                elif deg3 is not None:
                    current_dip = deg3

                if current_dip is not None and current_dip < dip1 and (
                        body_parts['r_wrist'].y < body_parts['r_hip'].y or body_parts['l_wrist'].y < body_parts['l_hip'].y):
                    dip_frame = frame_counter
                    dip1 = current_dip

                # Reset status when hands drop
                if body_parts['r_wrist'].y > body_parts['r_hip'].y or body_parts['l_wrist'].y > body_parts['l_hip'].y:
                    model_status = "Waiting..."
                    status_color2 = (255, 255, 255)

                # Tracking release
                deg_temp: Optional[float] = deg2 if body_parts['r_wrist'].y < body_parts['l_wrist'].y else deg4

                if deg_temp is not None and deg_temp > release_angle1:
                    release_angle1 = deg_temp
                    release_frame = frame_counter

                # Model evaluation or data logging
                if release_frame != 0 and (release_frame + fps <= frame_counter) and (deg_temp is not None and deg_temp < 90):
                    model_status, status_color2 = data_or_model(GATHER_DATA, dip1, release_angle1, IS_GOOD_SHOT, ai_model)

                    # Reset mechanics tracking
                    dip1 = 360.0
                    release_angle1 = 0.0
                    dip_frame = 0
                    release_frame = 0

                # Kinetic chain evaluation
                if dip_frame > 0 and release_frame > 0:
                    if dip_frame < release_frame and (release_frame - dip_frame < fps):
                        timing_status = "Good"
                        status_color = (0, 255, 0)
                    else:
                        timing_status = "Poor"
                        status_color = (0, 0, 255)
                else:
                    timing_status = "Waiting..."
                    status_color = (0, 255, 255)

                # Rendering
                draw_skeleton(pixels['r_hip'], pixels['r_knee'], pixels['r_ankle'], deg3, deg3_text, frame)
                draw_skeleton(pixels['l_hip'], pixels['l_knee'], pixels['l_ankle'], deg1, deg1_text, frame)
                draw_skeleton(pixels['r_shoulder'], pixels['r_elbow'], pixels['r_wrist'], deg2, deg2_text, frame)
                draw_skeleton(pixels['l_shoulder'], pixels['l_elbow'], pixels['l_wrist'], deg4, deg4_text, frame)

                cv2.putText(frame, f"Max dip: {int(dip1)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Max release angle: {int(release_angle1)}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Kinetic Chain: {timing_status}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

            if not GATHER_DATA:
                cv2.putText(frame, f"Model Verdict: {model_status}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color2, 2)

            cv2.imshow("Video", frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    finally:
        video.release()
        pose.close()
        cv2.destroyAllWindows()
        print("Resources released. Shutting down.")


if __name__ == "__main__":
    main()
