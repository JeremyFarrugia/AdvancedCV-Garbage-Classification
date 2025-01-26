import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

def prepare_yolov11_dataset():
    """
    Validate and return the paths for training, validation, and test datasets.

    The function checks if the dataset structure is complete and outputs
    the required paths for training and evaluation.
    """
    # Define dataset structure
    dataset_dir = "C:/Users/User/Desktop/Garbage-Classification/Garbage Detection.v3i.yolov5pytorch"
    train_images = os.path.join(dataset_dir, "train/images")
    valid_images = os.path.join(dataset_dir, "valid/images")
    train_labels = os.path.join(dataset_dir, "train/labels")
    valid_labels = os.path.join(dataset_dir, "valid/labels")
    test_images = os.path.join(dataset_dir, "test/images")
    test_labels = os.path.join(dataset_dir, "test/labels")

    # Validate the dataset structure
    for path in [train_images, valid_images, train_labels, valid_labels, test_images, test_labels]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required folder: {path}")

    print("Dataset structure is valid.")

    return {
        "train_images": train_images,
        "valid_images": valid_images,
        "train_labels": train_labels,
        "valid_labels": valid_labels,
        "test_images": test_images,
        "test_labels": test_labels,
        "data_yaml_path": os.path.join(dataset_dir, "data.yaml")
    }


def evaluate_model(model, dataset):
    """
    Evaluate the model using mAP and TensorBoard.

    Args:
        model: Trained YOLOv11 model.
        dataset (dict): Dictionary containing dataset paths.

    Returns:
        dict: Evaluation metrics including mAP scores.
    """
    print("Evaluating model performance...")

    # Run validation
    results = model.val(
        data=dataset["data_yaml_path"],
        project="runs",
        name="evaluation YOLOv11 Model",
        imgsz=416,
        split='test',
        save_json=True,
        plots=True,
        exist_ok=True  # This will overwrite the existing evaluation folder
    )

    # Extract metrics safely
    try:
        metrics = {
            "mAP@0.5": results.box.map50,      # mAP@0.5
            "mAP@0.5:0.95": results.box.map    # mAP@0.5:0.95
        }
        print(f"Evaluation metrics: {metrics}")
        return metrics
    except AttributeError as e:
        raise ValueError("The results object does not contain expected metrics.") from e

def train_yolov11(dataset, epochs=20, batch_size=16, pretrained_model=None):
    """
    Train YOLOv11 on the prepared dataset, optionally starting from a pretrained model.
    
    Args:
        dataset (dict): A dictionary containing dataset paths.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        pretrained_model (str): Path to a pretrained model to continue training. Default is None.
    """
    # Ensure data.yaml exists
    if not os.path.exists(dataset["data_yaml_path"]):
        raise FileNotFoundError(f"data.yaml file not found: {dataset['data_yaml_path']}")

    # Debug: Check if the dataset paths are correct
    print(f"Training with dataset located at: {dataset['data_yaml_path']}")
    
    # Load YOLOv11 model (optionally using a pretrained model)
    model = YOLO(pretrained_model if pretrained_model else "yolo11m.pt")

    # Debug: Verify the model is loaded
    print(f"Model {pretrained_model if pretrained_model else 'yolo11m.pt'} loaded successfully.")
    
    # Train the model
    model.train(
        data=dataset["data_yaml_path"],
        epochs=50,                   # Increase epochs if accuracy is low
        batch=6,                     # Adjust based on GPU capacity
        imgsz=416,                   # Recommended image size
        lr0=0.0008,                  # Lower learning rate for stability
        lrf=0.000001,                # Learning rate decay factor
        optimizer="Adam",            # Switch optimizer if needed
        patience=7,                  # Allow more epochs before early stopping
        val=True,
        project="runs",
        name="yolov11_garbage_optimized",
        plots=True,
        device=0
    )

    print("YOLOv11 training complete.")
    
    # Save the model after training
    model.save("yolov11_garbage_detection_model.pt")
    print("Model saved as 'yolov11_garbage_detection_model.pt'.")


def visualize_predictions_and_analytics(model, test_images_path, confidence_threshold=0.5):
    """
    Visualize predictions and provide analytics for detected objects.

    Args:
        model: The trained YOLOv11 model.
        test_images_path (str): Path to the test images directory.
        confidence_threshold (float): Minimum confidence level for predictions.
    """
    # Get list of test images
    test_images = [f for f in os.listdir(test_images_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    total_bags_detected = 0  # Counter for detected bags

    for image_name in test_images:
        image_path = os.path.join(test_images_path, image_name)

        # Predict using the model
        results = model.predict(image_path)

        if not results or len(results[0].boxes) == 0:
            print(f"No objects detected in {image_name}.")
            continue

        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bags_detected = 0

        for box in results[0].boxes:
            conf = box.conf[0]
            if conf < confidence_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = results[0].names[int(box.cls[0])]

            if label.lower() != "other waste":  # Assuming "bag" is a class name
                bags_detected += 1

            # Draw bounding boxes and labels
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label} ({conf:.2f})"
            cv2.putText(img_rgb, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Update total bag count
        total_bags_detected += bags_detected

        # Display analytics for the image
        print(f"{image_name}: {bags_detected} bags detected.")

        # Visualize the predictions
        plt.imshow(img_rgb)
        plt.title(f"Predictions for {image_name}")
        plt.axis("off")
        plt.show()

    print(f"Total bags detected across test images: {total_bags_detected}")


if __name__ == "__main__":
    try:
        # Prepare the dataset
        dataset = prepare_yolov11_dataset()

        # Train the model
        train_yolov11(dataset, epochs=30, batch_size=16)

        # Load the trained model
        model = YOLO("yolov11_garbage_detection_model.pt")

        # Evaluate model performance
        metrics = evaluate_model(model, dataset)

        # Visualize predictions and provide analytics
        visualize_predictions_and_analytics(model, dataset["test_images"], confidence_threshold=0.6)

    except Exception as e:
        print(f"Error: {e}")
