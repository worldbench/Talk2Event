import numpy as np
from pathlib import Path
import cv2
from einops import rearrange, reduce
# import open3d as o3d
import supervision as sv
box_annotator = sv.BoxAnnotator()

def load_points(lidar_path, item_info, only_fov=False):
    if not isinstance(lidar_path, Path):
        lidar_path = Path(lidar_path)

    if lidar_path.suffix == '.npy':
        points = np.load(lidar_path).reshape(-1, 5)
    elif lidar_path.suffix == '.bin':
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    else:
        raise NotImplementedError('No support such data format.')
    
    if not only_fov:
        return points
    else:
        extristric = np.array(item_info['extristric'])
        K = np.array(item_info['image_intrinsic'])
        D = np.array(item_info['camera_distortion'])
        keep = get_fov_flag(points, extristric, K, D)
        return points[keep]

def cart_to_hom(pts):
    """
    :param pts: (N, 3 or 2)
    :return pts_hom: (N, 4 or 3)
    """
    pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
    return pts_hom

def get_fov_flag(points, extristric, K, D):
    extend_points = cart_to_hom(points[:,:3])
    points_cam = extend_points @ extristric.T
    rvecs = np.zeros((3,1))
    tvecs = np.zeros((3,1))

    pts_img, _ = cv2.projectPoints(points_cam[:,:3].astype(np.float32), rvecs, tvecs,
            K, D)
    imgpts = pts_img[:,0,:]
    depth = points_cam[:,2]

    imgpts = np.round(imgpts)
    kept1 = (imgpts[:, 1] >= 0) & (imgpts[:, 1] < 800) & \
                (imgpts[:, 0] >= 0) & (imgpts[:, 0] < 1280) & \
                (depth > 0.2)
    return kept1
        
def label_img(img, bbox_2d=None, class_2d=None):
    labels = [class_2d] * bbox_2d.shape[0]
    gt_detections = sv.Detections(
        xyxy=bbox_2d[0][np.newaxis, :],
        class_id=np.zeros([1])
    )
    pred_detections = sv.Detections(
        xyxy=bbox_2d[1][np.newaxis, :],
        class_id=np.ones([1])

    )
    # annotated_image = box_annotator.annotate(img.copy(), detections, labels)

    gt_bbox_annotator = sv.BoxAnnotator(sv.Color(r=0, g=253, b=255), thickness=4)
    pred_bbox_annotator = sv.BoxAnnotator(sv.Color(r=163, g=8, b=232), thickness=4)

    # label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
    # annotated_frame = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
    annotated_frame = gt_bbox_annotator.annotate(scene=img, detections=gt_detections)
    annotated_frame = pred_bbox_annotator.annotate(scene=annotated_frame, detections=pred_detections)
    # annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    # 0 253 255
    # 0 250 0
    return annotated_frame

def load_img(img_path, bbox_2d=None, class_2d=None):
    img = cv2.imread(img_path)
    annotated_frame = label_img(img, bbox_2d, class_2d)
    return annotated_frame

def ev_repr_to_img(x):
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

def load_event(event_path, bbox_2d=None, class_2d=None):
    event = np.load(event_path)['events']
    img = ev_repr_to_img(event)
    annotated_frame = label_img(img, bbox_2d, class_2d)
    return annotated_frame