import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.ops import box_convert
from groundingdino.util.inference import load_model, load_image, predict, annotate
import groundingdino.datasets.transforms as T

def post_process(boxes, logits, phrases):
    logit_idx = logits.argmax()
    return boxes[logit_idx][None, :], torch.Tensor([logits[logit_idx]]), [phrases[logit_idx]]

def post_process_talk2event(boxes, logits, phrases, image_shape):
    if logits.shape[0] == 0:
        return np.zeros([4])
    logit_idx = logits.argmax()
    box = boxes[logit_idx]
    box = decode_box(box, image_shape)
    return box

def transform_image(image_source):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_transformed, _ = transform(image_source, None)
    return np.asarray(image_source), image_transformed

def decode_box(box, image_shape):
    h, w, _ = image_shape
    box = box * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=box, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    return xyxy

class GroundingDino:
    # def __init__(self, pretrained_weight_path, m3ed_name = ("person, bicycle, car, motorcycle, bus, train, truck, horse")):
    def __init__(self, 
                 config_path="/data/yyang/workspace/magiclidar/submodules/gdino/configs/GroundingDINO_SwinT_OGC.py",
                 pretrained_weight_path="submodules/pretrained/groundingdino_swint_ogc.pth", 
                 ):
        self.pretrained_weight_path = pretrained_weight_path
        self.config_path = config_path
        self.build_model()

    def build_model(self):
        model = load_model(self.config_path, self.pretrained_weight_path)
        self.model = model
        self.model.eval()

    def run_image(self, image, text):
        image_source, image = transform_image(Image.fromarray(image.astype(np.uint8)))

        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=text,
            box_threshold=0.0,
            text_threshold=0.0
        )
        # box, logit, phrase = post_process(boxes, logits, phrases)
        # annotated_frame = annotate(image_source=image_source, boxes=box, logits=logit, phrases=phrase)
        # cv2.imwrite("temp_test/annotated_image.jpg", annotated_frame)
        box = post_process_talk2event(boxes, logits, phrases, image_source.shape)
        return box

if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES']='2'

    model = load_model("/data/yyang/workspace/magiclidar/submodules/gdino/configs/GroundingDINO_SwinT_OGC.py", "submodules/pretrained/groundingdino_swint_ogc.pth")
    IMAGE_PATH = "temp_test/000157.png"
    TEXT_PROMPT = "a white car facing the viewer."
    BOX_TRESHOLD = 0.0
    TEXT_TRESHOLD = 0.0

    image_source, image = load_image(IMAGE_PATH)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )
    box, logit, phrase = post_process(boxes, logits, phrases)


    annotated_frame = annotate(image_source=image_source, boxes=box, logits=logit, phrases=phrase)
    cv2.imwrite("temp_test/annotated_image.jpg", annotated_frame)