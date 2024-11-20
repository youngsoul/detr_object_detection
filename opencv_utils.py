import numpy as np
import cv2
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch

def calculate_label_size(box_width, box_height, frame_width, frame_height, min_scale=0.4, max_scale=1.2):
    """
    Calculate the size of a label based on the dimensions of a box and a frame,
    scaling it within specified minimum and maximum scale limits.

    :param box_width: Width of the box in pixels
    :param box_height: Height of the box in pixels
    :param frame_width: Width of the frame in pixels
    :param frame_height: Height of the frame in pixels
    :param min_scale: Minimum scale factor (default is 0.4)
    :param max_scale: Maximum scale factor (default is 1.2)
    :return: A tuple containing the calculated width and height of the label in pixels
    """
    box_size_ratio = (box_width * box_height)/ (frame_width * frame_height)

    font_scale = min_scale + (np.log(1+box_size_ratio*100)/5) * (max_scale - min_scale)

    return np.clip(font_scale, min_scale, max_scale)

def generate_color_palette(num_classes):
    """
    Generates a deterministic color palette for a given number of classes.

    Each class is assigned a unique RGB color tuple.

    :param num_classes: The number of classes for which the color palette is to be generated.
    :type num_classes: int
    :return: A dictionary with class labels as keys and RGB color tuples as values.
    :rtype: dict
    """
    np.random.seed(42)
    return {label: tuple(np.random.randint(0, 255, size=3)) for label in range(num_classes)}

_processor = None
_model = None

def get_processor_and_model():
    global _processor, _model
    if _processor is not None and _model is not None:
        return _processor, _model

    _processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    _model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    return _processor, _model


def predict_and_plot_boxes(image_path, threshold=0.9):
    # Step 1: Load and preprocess the image
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        # I assume its an actual image
        image_rgb = image_path

    processor, model = get_processor_and_model()

    inputs = processor(images=image_rgb, return_tensors="pt")

    # Step 2: Run the inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Step 3: Post-process the results
    # Convert the outputs (logits) into bounding boxes and labels
    target_sizes = torch.tensor([image_rgb.shape[:2]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

    scores = results["scores"].numpy()
    keep = scores > threshold

    boxes = results["boxes"].numpy()[keep]
    labels = results["labels"].numpy()[keep]
    scores = scores[keep]

    for box, label, score in zip(boxes, labels, scores):
        xmin, ymin, xmax, ymax = map(int, box)
        box_width = xmax - xmin
        box_height = ymax - ymin
        img_width, img_height = image_rgb.shape[:2]
        font_scale = calculate_label_size(box_width, box_height, img_width, img_height)
        color_palette_map = generate_color_palette(model.config.num_labels)

        label_text = f"{model.config.id2label[label]}: {score:.2f}"
        color1 = color_palette_map[label]
        color = tuple(int(i) for i in color1)

        (label_width, label_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

        cv2.rectangle(image_rgb, (xmin, ymin), (xmax, ymax), color, max(2, int(font_scale*3)))
        cv2.rectangle(image_rgb, (xmin, ymin-label_height-baseline-5),
                      (xmin+label_width, ymin), color, -1)

        cv2.putText(image_rgb, f"{model.config.id2label[label]}: {score:.2f}", (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), max(1, int(font_scale*1.5)), cv2.LINE_AA)

    return image_rgb



def process_video(video_path, window_name="Video Predict", skip_first_frames:int=0, max_frames:int=5, threshold=0.9):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    while cap.isOpened():
        frame_count += 1
        print(frame_count)
        if frame_count > max_frames+skip_first_frames:
            break
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If the frame is read correctly, ret is True
        if not ret:
            print("Finished processing the video.")
            break

        # Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame_count < skip_first_frames:
            continue

        # Pass the RGB frame to the predict_and_plot_boxes function
        predict_image = predict_and_plot_boxes(frame_rgb, threshold=threshold)
        cv2.imshow(window_name, predict_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the video capture object
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_path = "./images/my_videos/coyote_backyard.mp4"
    process_video(video_path, max_frames=100, threshold=0.7)
