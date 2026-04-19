import cv2

# Emotion to colour mapping for bounding boxes
EMOTION_COLORS = {
    "happy":     (0, 255, 0),      # Green
    "sad":       (255, 0, 0),      # Blue
    "angry":     (0, 0, 255),      # Red
    "surprise":  (0, 255, 255),    # Yellow
    "fear":      (128, 0, 128),    # Purple
    "disgust":   (0, 128, 0),      # Dark Green
    "neutral":   (255, 255, 255),  # White
}

def draw_emotion(image, x, y, w, h, emotion, confidence):
    """
    Draw a bounding box and emotion label on the image.
    Colour changes based on the detected emotion.
    """
    color = EMOTION_COLORS.get(emotion.lower(), (255, 255, 255))

    # Draw bounding box
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    # Draw emotion label and confidence
    label = f"{emotion}: {confidence:.0f}%"
    cv2.putText(image, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return image

def load_image(image_path):
    """Load an image from file."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return image

def save_image(image, output_path):
    """Save an image to file."""
    cv2.imwrite(output_path, image)
    print(f"Result saved to {output_path}")