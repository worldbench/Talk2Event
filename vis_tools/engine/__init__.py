from .detector import Detector
from .yoloworld_detector import YoloWorld_Detector
from .yoloe_detector import YoloE_Detector
from .groundingdino_detector import GroundingDino_Detector
from .owl_vit_detector import OWL_VIT_Detector, OWL_VIT_V2_Detector

__all__ = {
    'our': Detector,
    'yolo_world': YoloWorld_Detector,
    'yoloe': YoloE_Detector,
    'groundingdino': GroundingDino_Detector,
    'owl_vit': OWL_VIT_Detector,
    'owl_vitv2': OWL_VIT_V2_Detector

}