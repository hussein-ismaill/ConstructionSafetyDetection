from ultralytics import YOLO
import sys

def main():
    print("🚀 Initializing YOLOv8 base model...")
    # Load the base pretrained Nano model
    model = YOLO('yolov8n.pt') 

    print("⏳ Starting training on your custom PPE dataset config (data.yaml)...")
    print("This may take some time depending on your laptop's speed.")
    
    try:
        # Run the YOLOv8 Training loop using Python code directly
        results = model.train(
            data='Construction-Site-Safety-30/data.yaml',    # Path to your dataset configuration
            epochs=25,           # Number of training rounds (epochs)
            imgsz=640,           # Image pixel resolution size
            batch=8,             # RESTORED to 8 to avoid memory indexing bugs
            device='cpu'         # RESTORED to 'cpu' (MPS currently has a bug with tensor indices)
        )
        
        print("\n✅ Training Complete!")
        print(f"Your new custom PPE model is automatically saved at:")
        print(f"    --> {results.save_dir}/weights/best.pt")
        print("You can now run 'streamlit run app.py' and it will instantly detect Helmets and Vests! 👷‍♂️")
        
    except Exception as e:
        print("\n❌ Error starting training!")
        print("Please make sure you have put your custom images inside the 'train/images' folder!")
        print(f"Details: {e}")

if __name__ == '__main__':
    # Run the main function
    main()
