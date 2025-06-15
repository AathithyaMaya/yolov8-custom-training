from ultralytics import YOLO
import os

def train_yolov8():
    # Load a pre-trained YOLOv8 model
    model = YOLO("yolov8n.pt")  # Use 'yolov8s.pt', 'yolov8m.pt', etc., for larger models

    # Define the path to data.yaml
    data_yaml_path = os.path.join("")  # Update this path with yaml file

    # Train the model with optimized settings
    results = model.train(
        data=data_yaml_path,  # Path to your dataset configuration file
        epochs=100,           # Number of epochs
        batch=16,             # Batch size
        imgsz=640,            # Imizage se
        device="0",           # Use GPU (e.g., "0" for GPU 0)
        name="yolov8_optimized_train",  # Name of the training run
        patience=10,          # Early stopping if no improvement for 10 epochs
        lr0=0.01,             # Initial learning rate
        lrf=0.01,             # Final learning rate (lr0 * lrf)
        momentum=0.937,        # Momentum
        weight_decay=0.0005,   # Weight decay
        warmup_epochs=3,      # Warmup epochs
        warmup_momentum=0.8,  # Warmup momentum
        warmup_bias_lr=0.1,   # Warmup bias learning rate
        box=7.5,              # Box loss gain
        cls=0.5,              # Clasoss gain
        dfl=1.5,              # Distribution Focs lal Loss gain
        hsv_h=0.015,          # Image HSV-Hue augmentation
        hsv_s=0.7,            # Image augmentation
        flipud=0.5,           # Image flip up-down proe HSV-Saturation augmentation
        hsv_v=0.4,            # Image HSV-Valubability
        fliplr=0.5,           # Image flip left-right probability
        mosaic=1.0,           # Mosaic augmentation probability
        mixup=0.1,            # Mixup augmentation probability
        copy_paste=0.1,       # Copy-paste augmentation probability
        degrees=0.0,          # Image rotation (+/- degrees)
        translate=0.1,        # Image translation (+/- fraction)
        scale=0.5,            # Image scale (+/- gain)
        shear=0.0,            # Image shear (+/- degrees)
        perspective=0.0,      # Image perspective (+/- fraction)
        half=True,            # Use mixed precision training
    )

    # Validate the model
    metrics = model.val(data=data_yaml_path, imgsz=640, device="0")

    # Export the model to ONNX format
    model.export(format="onnx")

    print("Training and validation completed successfully!")

if __name__ == "__main__":
    train_yolov8()