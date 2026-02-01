import requests
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection

from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

class OWL_VOTV2:
    # def __init__(self, pretrained_weight_path, m3ed_name = ("person, bicycle, car, motorcycle, bus, train, truck, horse")):
    def __init__(self, 
                 model_path=None,
                 processer_path=None, 
                 ):
        self.model_path = model_path
        self.processer_path = processer_path
        self.build_model()

    def build_model(self):
        self.processor = Owlv2Processor.from_pretrained("google/owlv2-large-patch14", cache_dir='submodules/pretrained')
        self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-large-patch14", cache_dir='submodules/pretrained')
        self.model = self.model.to('cuda')

    def run_image(self, image, text):
        texts = [[text]]
        inputs = self.processor(text=texts, images=image, return_tensors="pt")
        if inputs.data['input_ids'].shape[1] > 16:
            inputs.data['input_ids'] = inputs.data['input_ids'][:,:16]
            inputs.data['attention_mask'] = inputs.data['attention_mask'][:,:16]

        for key, value in inputs.data.items():
            inputs.data[key] = value.to('cuda')
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Convert outputs (bounding boxes and class logits) to COCO API
        results = self.processor.post_process_object_detection(outputs=outputs, threshold=0.0, target_sizes= torch.Tensor([image.shape[:2]]))
        boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
        if scores.shape[0] == 0:
            return np.zeros([4])
        logit_index = scores.argmax()
        box = boxes[logit_index].detach().cpu().numpy()
        return box

if __name__ == "__main__":

    from groundingdino.util.inference import load_model, load_image, predict, annotate

    processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16", cache_dir='submodules/pretrained')
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16", cache_dir='submodules/pretrained')

    image = Image.open("temp_test/000157.png")
    texts = [["a white car, facing the viewer"]]
    inputs = processor(text=texts, images=image, return_tensors="pt")
    for key, value in inputs.data.items():
        inputs.data[key] = value.to('cuda')
    outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]])
    # Convert outputs (bounding boxes and class logits) to COCO API
    results = processor.post_process_object_detection(outputs=outputs, threshold=0.0, target_sizes=target_sizes)

    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
    logit_index = scores.argmax()
    box = boxes[logit_index].detach()

    annotated_frame = annotate(image_source=np.asarray(image), boxes=box[None,:], logits=torch.Tensor([scores[logit_index]]), phrases=[text[0]], rescale_box=False)
    cv2.imwrite("temp_test/annotated_image.jpg", annotated_frame)