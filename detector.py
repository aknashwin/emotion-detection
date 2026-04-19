from deepface import DeepFace
from utils import draw_emotion

class EmotionDetector:
    def __init__(self, confidence_threshold=10):
        """
        Initialise the emotion detector.
        confidence_threshold: minimum confidence % to display a detection
        """
        self.confidence_threshold = confidence_threshold
        self.backend = "opencv"

    def detect(self, image):
        """
        Run emotion detection on an image.
        Returns annotated image and list of detections.
        """
        annotated = image.copy()
        detections = []

        try:
            results = DeepFace.analyze(
                image,
                actions=["emotion"],
                detector_backend=self.backend,
                enforce_detection=False
            )

            for face in results:
                emotion = face["dominant_emotion"]
                emotions = face["emotion"]
                confidence = emotions[emotion]
                region = face["region"]

                x = region["x"]
                y = region["y"]
                w = region["w"]
                h = region["h"]

                if confidence >= self.confidence_threshold:
                    annotated = draw_emotion(
                        annotated, x, y, w, h, emotion, confidence
                    )
                    detections.append({
                        "emotion": emotion,
                        "confidence": round(confidence, 2),
                        "all_emotions": {
                            k: round(v, 2) for k, v in emotions.items()
                        },
                        "bbox": [x, y, w, h]
                    })

        except Exception as e:
            print(f"Detection error: {e}")

        return annotated, detections