import numpy as np
from ultralytics import YOLOE

class YOLO_E:
    # def __init__(self, pretrained_weight_path, m3ed_name = ("person, bicycle, car, motorcycle, bus, train, truck, horse")):
    def __init__(self, pretrained_weight_path):

        self.pretrained_weight_path = pretrained_weight_path
        self.build_model()

    def build_model(self):
        model = YOLOE(self.pretrained_weight_path).cuda()
        self.model = model
        self.model.eval()

    def run_image(self, image, text):
        self.model.set_classes(text, self.model.get_text_pe(text))
        results = self.model.predict(image)
        # results[0].show()

        scores = results[0].boxes.conf
        if scores.shape[0] == 0:
            return np.zeros([4])
        
        top_1_idx = int(scores.argmax())

        return results[0].boxes.data[top_1_idx][:4].detach().cpu().numpy()

if __name__ == "__main__":
    import cv2
    model = YOLO_E('/data/yyang/workspace/magiclidar/submodules/pretrained/yoloe-11l-seg.pt')
    model.run_image(
        cv2.imread('/data/yyang/workspace/magiclidar/temp_test/bus.jpg'),
        ("a bus parked on left side of road"))