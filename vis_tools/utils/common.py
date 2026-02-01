import torch
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import find_contours
from matplotlib.patches import Polygon

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.375, 0.66, 0.089], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

COLORS = [[0.375, 0.66, 0.089], [1, 0.38, 0]]

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    if out_bbox.dim() == 2:
      img_h, img_w = size
      b = box_cxcywh_to_xyxy(out_bbox)
      b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).view(-1, 4)
      b[:,0].clamp_(0, img_w)
      b[:,2].clamp_(2, img_w)
      b[:,1].clamp_(1, img_h)
      b[:,3].clamp_(3, img_h)
      return b
    else:
        for b_idx in out_bbox.shape[0]:
          out_bbox[b_idx] = rescale_bboxes(out_bbox[b_idx])
        return out_bbox



def plot_results(pil_img, scores, boxes, labels, masks=None):
    plt.figure(figsize=(16,10))
    np_image = np.array(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    if masks is None:
      masks = [None for _ in range(len(scores))]
    assert len(scores) == len(boxes) == len(labels) == len(masks)
    for s, (xmin, ymin, xmax, ymax), l, mask, c in zip(scores, boxes.tolist(), labels, masks, colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=6))
        text = f'{l}'
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='white', alpha=0.8))

        if mask is None:
          continue
        np_image = apply_mask(np_image, mask, c)

        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
          # Subtract the padding and flip (y, x) to (x, y)
          verts = np.fliplr(verts) - 1
          p = Polygon(verts, facecolor="none", edgecolor=c)
          ax.add_patch(p)

    plt.imshow(np_image)
    plt.axis('off')
    plt.savefig('box.png')