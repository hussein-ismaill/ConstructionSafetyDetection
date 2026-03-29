import cv2
import math
import os
from ultralytics import YOLO

# --- CONFIGURATION ---
# Path where your customized model weights get saved after training
MODEL_PATH = "runs/detect/train/weights/best.pt"  
# Fallback model to test the code without training (only detects generic things like 'person')
FALLBACK_MODEL = "yolov8n.pt"  

def load_model():
    """Load the custom trained model or fallback to a pretrained generic one."""
    if os.path.exists(MODEL_PATH):
        print(f"[*] Loading custom model from '{MODEL_PATH}'...")
        model = YOLO(MODEL_PATH)
        is_custom = True
    else:
        print(f"[!] Custom model not found at '{MODEL_PATH}'.")
        print(f"[*] Loading fallback pretrained model '{FALLBACK_MODEL}'...")
        print("[!] Note: Pretrained model only detects generic objects (e.g., 'person').")
        print("    To detect helmets and vests, you MUST train the model first.")
        model = YOLO(FALLBACK_MODEL)
        is_custom = False
    return model, is_custom

def main():
    model, is_custom = load_model()
    
    # Custom classes matching the data.yaml order
    custom_classes = ["helmet", "vest", "person"]

    # Initialize video capture. '0' opens your laptop's default webcam.
    # To use a video, change it to: cap = cv2.VideoCapture("construction_video.mp4")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return

    print("[*] Starting real-time detection. Press 'q' in the window to quit.")

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read frame.")
            break

        # Run YOLOv8 inference on the current frame
        results = model(img, stream=True, verbose=False)

        # Let YOLO natively draw its beautiful bounding boxes & labels!
        # Because stream=True returns a generator, we process the first result
        for r in results:
            annotated_img = r.plot()
            boxes = r.boxes
            
            # Store coordinates for rule violation logic later
            persons = []
            helmets = []

            for box in boxes:
                # Bounding Box Coordinates: (x1, y1) -> Top-Left | (x2, y2) -> Bottom-Right
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get the detected Class Index
                cls = int(box.cls[0])
                
                # Resolve the correct Class Name
                if is_custom:
                    current_class = custom_classes[cls] if cls < len(custom_classes) else "Unknown"
                else:
                    # Fallback for the basic pretrained model
                    current_class = model.names.get(cls, "Unknown")
                
                # Categorize detections for our safety rules
                if current_class == "person":
                    persons.append((x1, y1, x2, y2))
                elif current_class == "helmet":
                    helmets.append((x1, y1, x2, y2))

            # --- SAFETY VIOLATION LOGIC ---
            # If we have the custom model loaded, let's enforce PPE rules!
            if is_custom:
                # Check every person bounding box
                for px1, py1, px2, py2 in persons:
                    has_helmet = False
                    # Try to map a detected helmet to this person
                    for hx1, hy1, hx2, hy2 in helmets:
                        # Calculate center point of the helmet box
                        hcx = (hx1 + hx2) / 2
                        hcy = (hy1 + hy2) / 2
                        
                        # Rule check: Is the helmet center physically located inside the person's bounding box?
                        if px1 <= hcx <= px2 and py1 <= hcy <= py2:
                            has_helmet = True
                            break
                    
                    # If the loop finished and no helmet matched this person
                    if not has_helmet:
                        # Draw a bright RED box around the violator
                        cv2.rectangle(annotated_img, (px1, py1), (px2, py2), (0, 0, 255), 4)
                        # Add a red warning message right above them
                        cv2.putText(annotated_img, "WARNING: NO HELMET", (px1, max(25, py1 - 25)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Display the result to the screen
            cv2.imshow("Smart City - Construction Safety PPE Monitor", annotated_img)

        # Wait 1ms, and check if the user pressed 'q' to close the app
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up and release the camera
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
