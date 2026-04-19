import cv2
import os
import argparse
from detector import EmotionDetector
from utils import load_image, save_image

def run_on_image(image_path, output_path, confidence=10):
    """Run emotion detection on a single image."""
    print(f"Loading image: {image_path}")
    image = load_image(image_path)

    print("Running emotion detection...")
    detector = EmotionDetector(confidence_threshold=confidence)
    annotated_image, detections = detector.detect(image)

    os.makedirs("results", exist_ok=True)
    save_image(annotated_image, output_path)

    print(f"\nFaces detected: {len(detections)}")
    for i, d in enumerate(detections):
        print(f"\n  Face {i + 1}:")
        print(f"    Dominant emotion : {d['emotion']} ({d['confidence']:.1f}%)")
        print(f"    All emotions:")
        for emotion, score in sorted(
            d['all_emotions'].items(), key=lambda x: x[1], reverse=True
        ):
            bar = "█" * int(score / 5)
            print(f"      {emotion:<10} {score:>6.1f}%  {bar}")

def run_on_webcam(confidence=10):
    """Run real-time emotion detection using webcam."""
    print("Starting webcam detection... Press 'q' to quit.")
    detector = EmotionDetector(confidence_threshold=confidence)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection every 3 frames for performance
        if frame_count % 3 == 0:
            annotated_frame, detections = detector.detect(frame)
        
        cv2.imshow("Emotion Detection", annotated_frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam detection stopped.")

def main():
    parser = argparse.ArgumentParser(description="Real-Time Emotion Detection System")
    parser.add_argument("--mode", type=str, default="image",
                        choices=["image", "webcam"],
                        help="Run on image or webcam")
    parser.add_argument("--input", type=str, default="sample_images/test.jpg",
                        help="Path to input image (image mode only)")
    parser.add_argument("--output", type=str, default="results/output.jpg",
                        help="Path to save output image (image mode only)")
    parser.add_argument("--confidence", type=float, default=10,
                        help="Minimum confidence threshold (0-100)")
    args = parser.parse_args()

    if args.mode == "image":
        run_on_image(args.input, args.output, args.confidence)
    elif args.mode == "webcam":
        run_on_webcam(args.confidence)

if __name__ == "__main__":
    main()