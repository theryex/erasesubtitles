from text_detection.predict import get_text_boxes
import cv2
import numpy as np

def resize_img(image, max_dim=1000):
    """Resizes an image so its largest dimension is `max_dim`."""
    h, w = image.shape[:2]
    rescale_fac = max(h, w) / max_dim
    if rescale_fac > 1.0:
        h = int(h / rescale_fac)
        w = int(w / rescale_fac)
    return h, w, rescale_fac

def get_coords(num_of_frames, masks, gpu_id=0):
    """
    Detects the subtitle region by aggregating text boxes from sample frames.
    Includes robust validation and coordinate scaling.
    """
    if not masks:
        return None

    max_height, max_width = masks[0].shape[:2]
    rh, rw, rescale_fac = resize_img(masks[0])

    print('Original Dimensions: ', max_height, 'x', max_width)
    print('Rescaled Dimensions: ', rh, 'x', rw)

    # Initialize with values that will be overridden by the first valid coordinates
    overall_xmin, overall_ymin = max_width, max_height
    overall_xmax, overall_ymax = 0, 0

    text_detected = False

    for i in range(num_of_frames):
        input_img = cv2.resize(masks[i], (rw, rh))
        text_boxes = get_text_boxes(input_img, gpu_id=gpu_id)

        # Check if the text_boxes are empty (can be a list or numpy array)
        if isinstance(text_boxes, np.ndarray) and text_boxes.size == 0:
            continue
        if isinstance(text_boxes, list) and not text_boxes:
            continue

        for box in text_boxes:
            # Scale coordinates back to original image dimensions
            box = np.array(box) * rescale_fac

            # Validate that the coordinate array has an even number of elements
            if box.size % 2 != 0:
                print(f"Skipping malformed bounding box (odd number of coordinates): {box}")
                continue

            # Reshape the flat array into a 2D array of (x, y) pairs
            box = box.reshape(-1, 2)

            # Get bounding box for the current text detection
            xmin, ymin = np.min(box, axis=0)
            xmax, ymax = np.max(box, axis=0)

            # Sanity check: ensure coordinates are within the frame boundaries
            if xmax > max_width or ymax > max_height:
                continue

            text_detected = True

            # Update the overall bounding box
            overall_xmin = min(overall_xmin, xmin)
            overall_ymin = min(overall_ymin, ymin)
            overall_xmax = max(overall_xmax, xmax)
            overall_ymax = max(overall_ymax, ymax)

    if not text_detected:
        print("No valid text boxes found across sampled frames.")
        return None

    # Add padding to the final coordinates, ensuring they stay within bounds
    final_xmin = max(0, int(overall_xmin - 10))
    final_ymin = max(0, int(overall_ymin - 10))
    final_xmax = min(max_width, int(overall_xmax + 10))
    final_ymax = min(max_height, int(overall_ymax + 20))

    # Final validation to ensure the coordinates are logical
    if final_xmin >= final_xmax or final_ymin >= final_ymax:
        print(f"Invalid final coordinates generated: {[final_xmin, final_ymin, final_xmax, final_ymax]}")
        return None

    return [final_xmin, final_ymin, final_xmax, final_ymax]
