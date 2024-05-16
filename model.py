import os
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import keras_cv
import keras_cv.visualization as visualization
from keras_cv.models import YOLOV8Detector
import matplotlib.patches as patches

# Constants
IMAGE_SIZE = (640, 640)  # Input size for YOLOv8
BATCH_SIZE = 3  # Number of samples per batch
NUM_CLASSES = 1  # Example number of classes, adjust as needed
BOUNDING_BOX_FORMAT = "xywh"  # YOLO bounding box format
PAD_TO_ASPECT_RATIO = True  # To maintain aspect ratio when resizing

# Paths to the datasets
TRAIN_IMAGES_DIR = Path("dataset/train/images/")
TRAIN_LABELS_DIR = Path("dataset/train/labels/")
VAL_IMAGES_DIR = Path("dataset/val/images/")
VAL_LABELS_DIR = Path("dataset/val/labels/")

# Function to load YOLO annotations
def load_yolo_annotations(label_path, image_size):
    annotations = []
    with open(label_path, 'r') as file:
        for line in file:
            parts = line.strip().split(" ")
            if len(parts) != 5:
                continue  # Skip lines that don't match expected format

            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # Convert normalized "xywh" to pixel-based "xyxy" format
            x_min = (x_center - width / 2) * image_size[0]
            y_min = (y_center - height / 2) * image_size[1]
            x_max = (x_center + width / 2) * image_size[0]
            y_max = (y_center + height / 2) * image_size[1]

            annotations.append([x_min, y_min, x_max, y_max, class_id])

    return np.array(annotations, dtype=np.float32)

# Function to load image and corresponding annotations
def load_sample(image_path, labels_dir):
    image_path_str = tf.keras.backend.get_value(image_path).decode("utf-8")  # Convert tensor to string
    image = Image.open(image_path_str).resize(IMAGE_SIZE)  # Resize to 640x640
    image = np.array(image) / 255.0  # Normalize
    
    # Construct the label path and validate its existence
    image_stem = Path(image_path_str).stem
    label_path = os.path.join(labels_dir, image_stem + ".txt")

    if not os.path.isfile(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")

    # Load YOLO annotations
    annotations = load_yolo_annotations(label_path, IMAGE_SIZE)  # Load annotations
    return image, annotations

# Function to resize image for inference
def inference_resizing(image, annotations):
    print(f"Image shape: {image.shape}, Annotations: {annotations}")
    resized_image = tf.image.resize(image, [224, 224])
    return resized_image, annotations

# Function to normalize image data
def normalize_image_data(image):
    # Convert TensorFlow tensor to NumPy array
    image = image.numpy()  # Explicit conversion
    # If data is in float format, scale to [0, 255]
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)  # Scale to [0, 255]
    return image

# Function to convert from BGR to RGB if needed
def ensure_rgb_format(image):
    # If the image appears incorrect, try converting BGR to RGB
    if image.shape[-1] == 3:  # Assuming three channels (RGB or BGR)
        return image[..., ::-1]  # Reverse the color channels to convert BGR to RGB
    return image

# Function to filter out empty annotations
def filter_empty_annotations(image, annotations):
    return tf.size(annotations) > 0

# Function to pad annotations to ensure consistent shape
def pad_annotations(image, annotations, max_annotations=5):
    padding = [[0, max_annotations - tf.shape(annotations)[0]], [0, 0]]
    annotations = tf.pad(annotations, padding, constant_values=-1)  # Using -1 to indicate padding
    return image, annotations

# Data loader with flexible batch size and error handling
def data_loader(images_dir, labels_dir, batch_size):
    image_paths = list(Path(images_dir).rglob("*.jpg")) + list(Path(images_dir).rglob("*.png"))
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {images_dir}. Check your dataset path.")

    # Create TensorFlow dataset object from list of image paths
    dataset = tf.data.Dataset.from_tensor_slices([str(p) for p in image_paths])

    # Map function to load images and annotations with error handling
    dataset = dataset.map(
        lambda x: tf.py_function(
            lambda y: load_sample(y, labels_dir),
            [x],
            [tf.float32, tf.float32]
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Apply the resizing function
    dataset = dataset.map(
        lambda image, annotations: tf.py_function(
            func=lambda img, ann: inference_resizing(img, ann),
            inp=[image, annotations],
            Tout=[tf.float32, tf.float32]
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Normalize image data
    dataset = dataset.map(
        lambda image, annotations: tf.py_function(
            func=lambda img, ann: (normalize_image_data(img), ann),
            inp=[image, annotations],
            Tout=[tf.float32, tf.float32]
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Ensure RGB format
    dataset = dataset.map(
        lambda image, annotations: tf.py_function(
            func=lambda img, ann: (ensure_rgb_format(img), ann),
            inp=[image, annotations],
            Tout=[tf.float32, tf.float32]
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Filter and pad annotations
    dataset = dataset.filter(filter_empty_annotations)
    dataset = dataset.map(lambda image, annotations: pad_annotations(image, annotations))

    # Apply batching, allowing for partial batches
    dataset = dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    
    return dataset

# Function to visualize the dataset
def visualize_dataset(dataset, value_range, default_rows, default_cols, bounding_box_format):
    # Get the first batch from the dataset
    batch = next(iter(dataset.take(1)))  # Get the first batch

    # Extract images and raw bounding boxes
    images, bounding_boxes_raw = batch

    # Adjust the number of rows and columns for visualization
    rows = default_rows
    cols = default_cols

    fig, axs = plt.subplots(rows, cols, figsize=(15, 15))
    axs = axs.flatten() if rows * cols > 1 else [axs]

    for ax, image, bboxes in zip(axs, images, bounding_boxes_raw):
        ax.imshow(image, vmin=value_range[0], vmax=value_range[1])
        for bbox in bboxes:
            if tf.reduce_all(bbox != -1):  # Ensure bounding box is not just padding
                x_min, y_min, x_max, y_max, _ = bbox
                rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

    plt.tight_layout()
    plt.show()

# Create datasets for training, validation, and testing
train_dataset = data_loader(TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, BATCH_SIZE)
val_dataset = data_loader(VAL_IMAGES_DIR, VAL_LABELS_DIR, BATCH_SIZE // 2)

# Visualize the dataset
visualize_dataset(train_dataset, value_range=(0, 1), default_rows=1, default_cols=1, bounding_box_format="xyxy")
