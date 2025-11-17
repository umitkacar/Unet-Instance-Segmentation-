"""
Facial Landmark Detection using MediaPipe and Dlib
Comprehensive example covering both libraries
"""

import cv2
import numpy as np
import mediapipe as mp
import dlib
from typing import List, Tuple, Optional
import time


class FacialLandmarkDetector:
    """Unified interface for facial landmark detection"""

    def __init__(self, method='mediapipe'):
        """
        Initialize landmark detector

        Args:
            method: 'mediapipe' or 'dlib'
        """
        self.method = method

        if method == 'mediapipe':
            self._init_mediapipe()
        elif method == 'dlib':
            self._init_dlib()
        else:
            raise ValueError("Method must be 'mediapipe' or 'dlib'")

    def _init_mediapipe(self):
        """Initialize MediaPipe Face Mesh"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        print("MediaPipe Face Mesh initialized (478 landmarks)")

    def _init_dlib(self):
        """Initialize Dlib face detector and shape predictor"""
        # Download shape predictor model from:
        # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

        self.detector = dlib.get_frontal_face_detector()

        try:
            self.predictor = dlib.shape_predictor(
                "shape_predictor_68_face_landmarks.dat"
            )
            print("Dlib initialized (68 landmarks)")
        except RuntimeError:
            print("Error: shape_predictor_68_face_landmarks.dat not found!")
            print("Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            raise

    def detect_landmarks(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect facial landmarks

        Args:
            image: Input image (BGR format)

        Returns:
            List of (x, y) landmark coordinates
        """
        if self.method == 'mediapipe':
            return self._detect_mediapipe(image)
        else:
            return self._detect_dlib(image)

    def _detect_mediapipe(self, image: np.ndarray) -> List[Tuple[float, float, float]]:
        """Detect landmarks using MediaPipe"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process image
        results = self.face_mesh.process(image_rgb)

        landmarks = []
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            h, w = image.shape[:2]
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                z = landmark.z  # Depth information
                landmarks.append((x, y, z))

        return landmarks

    def _detect_dlib(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """Detect landmarks using Dlib"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.detector(gray)

        landmarks = []
        if len(faces) > 0:
            # Get landmarks for first face
            face = faces[0]
            shape = self.predictor(gray, face)

            for i in range(68):
                x = shape.part(i).x
                y = shape.part(i).y
                landmarks.append((x, y))

        return landmarks

    def visualize_landmarks(
        self,
        image: np.ndarray,
        landmarks: List[Tuple],
        show_numbers: bool = False,
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """
        Visualize landmarks on image

        Args:
            image: Input image
            landmarks: List of landmark coordinates
            show_numbers: Whether to show landmark numbers
            color: Color for landmarks (B, G, R)

        Returns:
            Image with landmarks drawn
        """
        result = image.copy()

        for idx, landmark in enumerate(landmarks):
            x, y = int(landmark[0]), int(landmark[1])

            # Draw point
            cv2.circle(result, (x, y), 2, color, -1)

            # Optionally draw number
            if show_numbers:
                cv2.putText(
                    result,
                    str(idx),
                    (x + 3, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),
                    1
                )

        # Draw connections for Dlib 68-point model
        if self.method == 'dlib' and len(landmarks) == 68:
            self._draw_facial_features(result, landmarks, color)

        return result

    def _draw_facial_features(
        self,
        image: np.ndarray,
        landmarks: List[Tuple],
        color: Tuple[int, int, int]
    ):
        """Draw facial feature connections for 68-point model"""
        # Jawline
        for i in range(16):
            cv2.line(image, landmarks[i], landmarks[i+1], color, 1)

        # Right eyebrow
        for i in range(17, 21):
            cv2.line(image, landmarks[i], landmarks[i+1], color, 1)

        # Left eyebrow
        for i in range(22, 26):
            cv2.line(image, landmarks[i], landmarks[i+1], color, 1)

        # Nose bridge
        for i in range(27, 30):
            cv2.line(image, landmarks[i], landmarks[i+1], color, 1)

        # Nose bottom
        for i in range(31, 35):
            cv2.line(image, landmarks[i], landmarks[i+1], color, 1)

        # Right eye
        for i in range(36, 41):
            cv2.line(image, landmarks[i], landmarks[i+1], color, 1)
        cv2.line(image, landmarks[41], landmarks[36], color, 1)

        # Left eye
        for i in range(42, 47):
            cv2.line(image, landmarks[i], landmarks[i+1], color, 1)
        cv2.line(image, landmarks[47], landmarks[42], color, 1)

        # Outer mouth
        for i in range(48, 59):
            cv2.line(image, landmarks[i], landmarks[i+1], color, 1)
        cv2.line(image, landmarks[59], landmarks[48], color, 1)

        # Inner mouth
        for i in range(60, 67):
            cv2.line(image, landmarks[i], landmarks[i+1], color, 1)
        cv2.line(image, landmarks[67], landmarks[60], color, 1)

    def benchmark(self, image: np.ndarray, iterations: int = 100) -> dict:
        """
        Benchmark detection speed

        Args:
            image: Test image
            iterations: Number of iterations

        Returns:
            Dictionary with timing statistics
        """
        times = []

        for _ in range(iterations):
            start_time = time.time()
            _ = self.detect_landmarks(image)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms

        return {
            'method': self.method,
            'avg_time_ms': np.mean(times),
            'std_time_ms': np.std(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times),
            'fps': 1000 / np.mean(times)
        }


def extract_facial_features(landmarks: List[Tuple]) -> dict:
    """
    Extract facial features from 68-point landmarks (Dlib format)

    Args:
        landmarks: List of 68 landmark coordinates

    Returns:
        Dictionary with facial features
    """
    if len(landmarks) != 68:
        raise ValueError("This function requires 68 landmarks (Dlib format)")

    features = {}

    # Eye aspect ratio (EAR) - for blink detection
    def eye_aspect_ratio(eye_points):
        # Vertical distances
        A = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
        B = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
        # Horizontal distance
        C = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
        return (A + B) / (2.0 * C)

    # Left eye (points 42-47)
    left_eye = [landmarks[i] for i in range(42, 48)]
    features['left_eye_ear'] = eye_aspect_ratio(left_eye)

    # Right eye (points 36-41)
    right_eye = [landmarks[i] for i in range(36, 42)]
    features['right_eye_ear'] = eye_aspect_ratio(right_eye)

    # Mouth aspect ratio (MAR) - for yawn/mouth opening detection
    mouth_top = landmarks[51]
    mouth_bottom = landmarks[57]
    mouth_left = landmarks[48]
    mouth_right = landmarks[54]

    vertical_dist = np.linalg.norm(np.array(mouth_top) - np.array(mouth_bottom))
    horizontal_dist = np.linalg.norm(np.array(mouth_left) - np.array(mouth_right))
    features['mouth_aspect_ratio'] = vertical_dist / horizontal_dist

    # Inter-pupil distance (approximate using eye centers)
    left_eye_center = np.mean([landmarks[i] for i in range(42, 48)], axis=0)
    right_eye_center = np.mean([landmarks[i] for i in range(36, 42)], axis=0)
    features['inter_pupil_distance'] = np.linalg.norm(left_eye_center - right_eye_center)

    # Face width (jawline)
    features['face_width'] = np.linalg.norm(
        np.array(landmarks[0]) - np.array(landmarks[16])
    )

    # Nose length
    features['nose_length'] = np.linalg.norm(
        np.array(landmarks[27]) - np.array(landmarks[33])
    )

    return features


def compare_methods(image_path: str):
    """Compare MediaPipe and Dlib on the same image"""
    print("="*60)
    print("Facial Landmark Detection Comparison")
    print("="*60)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        # Create demo image if file not found
        print("Creating demo image...")
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    print(f"\nImage size: {image.shape}")

    # MediaPipe
    print("\n1. MediaPipe Face Mesh:")
    try:
        detector_mp = FacialLandmarkDetector(method='mediapipe')
        landmarks_mp = detector_mp.detect_landmarks(image)
        print(f"   Landmarks detected: {len(landmarks_mp)}")

        # Benchmark
        benchmark_mp = detector_mp.benchmark(image, iterations=50)
        print(f"   Average time: {benchmark_mp['avg_time_ms']:.2f} ms")
        print(f"   FPS: {benchmark_mp['fps']:.1f}")

        # Visualize
        result_mp = detector_mp.visualize_landmarks(image, landmarks_mp)
        cv2.imwrite('facial_landmarks_mediapipe.jpg', result_mp)
        print("   Saved: facial_landmarks_mediapipe.jpg")
    except Exception as e:
        print(f"   Error: {e}")

    # Dlib
    print("\n2. Dlib Shape Predictor:")
    try:
        detector_dlib = FacialLandmarkDetector(method='dlib')
        landmarks_dlib = detector_dlib.detect_landmarks(image)
        print(f"   Landmarks detected: {len(landmarks_dlib)}")

        # Benchmark
        benchmark_dlib = detector_dlib.benchmark(image, iterations=50)
        print(f"   Average time: {benchmark_dlib['avg_time_ms']:.2f} ms")
        print(f"   FPS: {benchmark_dlib['fps']:.1f}")

        # Visualize
        result_dlib = detector_dlib.visualize_landmarks(
            image,
            landmarks_dlib,
            show_numbers=False
        )
        cv2.imwrite('facial_landmarks_dlib.jpg', result_dlib)
        print("   Saved: facial_landmarks_dlib.jpg")

        # Extract features
        if len(landmarks_dlib) == 68:
            features = extract_facial_features(landmarks_dlib)
            print("\n3. Extracted Facial Features:")
            for key, value in features.items():
                print(f"   {key}: {value:.2f}")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n" + "="*60)


def video_demo(method='mediapipe', camera_id=0):
    """
    Real-time facial landmark detection from webcam

    Args:
        method: 'mediapipe' or 'dlib'
        camera_id: Camera device ID
    """
    print(f"Starting video demo with {method}...")
    print("Press 'q' to quit")

    # Initialize detector
    detector = FacialLandmarkDetector(method=method)

    # Open webcam
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    fps_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # Detect landmarks
        landmarks = detector.detect_landmarks(frame)

        # Visualize
        if len(landmarks) > 0:
            frame = detector.visualize_landmarks(frame, landmarks)

        # Calculate FPS
        fps = 1.0 / (time.time() - start_time)
        fps_list.append(fps)

        # Display FPS
        cv2.putText(
            frame,
            f"FPS: {fps:.1f} | Landmarks: {len(landmarks)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Show frame
        cv2.imshow('Facial Landmark Detection', frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nAverage FPS: {np.mean(fps_list):.1f}")


def main():
    """Main demo function"""
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "test_face.jpg"

    # Compare methods
    compare_methods(image_path)

    # Uncomment to run video demo
    # video_demo(method='mediapipe')


if __name__ == "__main__":
    main()
