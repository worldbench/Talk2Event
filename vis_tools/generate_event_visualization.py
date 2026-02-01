import os
import cv2
import json
import numpy as np
import h5py
import hdf5plugin
from utils import StackedHistogram
from scipy.ndimage import zoom
from einops import rearrange, reduce
import shutil 
from PIL import Image, ImageDraw, ImageFont

def ev_repr_to_img(img, x):
    ch, ht, wd = x.shape[-3:]
    assert ch > 1 and ch % 2 == 0
    ev_repr_reshaped = rearrange(x, '(posneg C) H W -> posneg C H W', posneg=2)
    img_neg = np.asarray(reduce(ev_repr_reshaped[0], 'C H W -> H W', 'sum'), dtype='int32')
    img_pos = np.asarray(reduce(ev_repr_reshaped[1], 'C H W -> H W', 'sum'), dtype='int32')
    img_diff = img_pos - img_neg

    # comment to draw event on rgb img, otherwise only event
    img = np.zeros((ht, wd, 3), dtype=np.uint8)

    img[img_diff > 0] = np.array([255, 0, 0])
    img[img_diff < 0] = np.array([0, 0, 255])
    return img

def get_text_size(text, font):
    """Return (width, height) of the given text using font.getbbox()."""
    bbox = font.getbbox(text)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width, height

def draw_box_with_caption(draw, bbox, caption, box_color, font, text_color, thickness=3, padding=4):
    """
    Draws a rectangle (bounding box) and a caption on the given PIL.ImageDraw object.
    
    Parameters:
      - draw: PIL.ImageDraw.Draw object.
      - bbox: dict with keys 'x', 'y', 'w', 'h'
      - caption: text string to display.
      - box_color: tuple (R, G, B) for the rectangle color.
      - font: a PIL ImageFont instance.
      - text_color: tuple (R, G, B) for the text.
      - thickness: rectangle border thickness.
      - padding: padding around the text.
    """
    # Extract and convert coordinates to int
    x, y, w, h = int(bbox['x']), int(bbox['y']), int(bbox['w']), int(bbox['h'])
    
    # Draw the bounding box (multiple rectangles to simulate thickness)
    for i in range(thickness):
        draw.rectangle([x - i, y - i, x + w + i, y + h + i], outline=box_color)
    
    # Prepare caption text: wrap if too long.
    max_text_width = max(w, 100)
    words = caption.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = f"{current_line} {word}".strip()
        if get_text_size(test_line, font)[0] <= max_text_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    
    # Determine text dimensions using get_text_size helper.
    line_height = get_text_size("Ag", font)[1]
    text_height = line_height * len(lines) + 2 * padding
    text_width = max([get_text_size(line, font)[0] for line in lines]) + 2 * padding
    
    # Decide text background position: try above bbox if possible.
    text_x = x
    text_y = y - text_height if y - text_height >= 0 else y
    
    # Draw a solid background rectangle for text
    draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], fill=(0, 0, 0))
    
    # Draw each line of text
    for i, line in enumerate(lines):
        line_y = text_y + padding + i * line_height
        draw.text((text_x + padding, line_y), line, font=font, fill=text_color)

def main():
    # Define paths
    split = 'test'
    sequence = 'thun_02_a'
    root = f'/ssd/dylu/data/talk2event_v4/'
    meta_root = os.path.join(root, 'meta_data', split)
    output_root = os.path.join(root, 'visualzation', split, sequence)
    os.makedirs(output_root, exist_ok=True)
    
    # Read meta data
    meta_file_path = os.path.join(meta_root, f"{sequence}.json")
    with open(meta_file_path, 'r') as f:
        data = json.load(f)
    
    for idx in range(len(data)):
        if data[idx]['id'] == 'thun_02_a_1_53':
            print(f"Found item with ID: {data[idx]['id']}")
            item = data[idx]
            break
    
    # valid_items_list = [item for item in data if item.get("answer") == "match"]
    
    # # Build a dictionary for quick lookup by ID
    # id_to_item = {item.get("id"): item for item in valid_items_list}
    
    # # Try to load a custom TrueType font; fallback to default if not found.
    # try:
    #     # Adjust the font path and size as needed.
    #     font = ImageFont.truetype("arial.ttf", 16)
    # except IOError:
    #     font = ImageFont.load_default()

    # # Define colors (PIL expects (R,G,B))
    # main_box_color = (0, 255, 255)   # Cyan for main bbox
    # other_box_color = (0, 255, 0)     # Green for others
    # text_color = (255, 255, 255)      # White text

    # Process each valid item
    # for item in valid_items_list:
    image_path = os.path.join(root, item.get("image_path").split(".")[0] + ".png")

    img = cv2.imread(image_path)
    # Optionally load event data if needed:
    event_path = os.path.join(root, item.get("event_path"))
    event = np.load(event_path)['events']

    #event visualization
    img = ev_repr_to_img(img, event)

    # Convert the image from BGR (OpenCV) to RGB (PIL)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # pil_img = Image.fromarray(img_rgb)
    # draw = ImageDraw.Draw(pil_img)

    # # Draw bounding boxes and captions for "others"
    # for other in item.get("others", []):
    #     other_id = other.get("id")
    #     if other_id in id_to_item:
    #         other_item = id_to_item[other_id]
    #         other_bbox = other_item.get("bbox")
    #         other_caption = other_item.get("caption", "")
    #         draw_box_with_caption(draw, other_bbox, other_caption, other_box_color, font, text_color)

    # # Draw the main object's bounding box and caption
    # main_bbox = item.get("bbox")
    # main_caption = item.get("caption", "")
    # draw_box_with_caption(draw, main_bbox, main_caption, main_box_color, font, text_color)

    # # Convert back to OpenCV BGR format and save
    # pil_img_rgb = np.array(pil_img)
    # final_img = cv2.cvtColor(pil_img_rgb, cv2.COLOR_RGB2BGR)
    # filename = os.path.basename(item.get("image_path"))
    # output_path = os.path.join(output_root, filename)
    cv2.imwrite('sample.png', img)

if __name__ == "__main__":
    main()

