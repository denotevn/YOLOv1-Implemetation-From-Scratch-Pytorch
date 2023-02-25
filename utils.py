import torch
import torch.nn as nn


def intersection_over_union(preds, labels, format='midpoint'):
    '''
    Computes the Intersection over Union (IoU) between two bounding boxes.
        preds, labels(tensor): Predictions and Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    '''
    # midpoint format represents 
    # the bounding box by its center coordinates (x,y) and its width and height (w,h)
    if format == 'midpoint':
        box1_x1 = preds[..., 0:1] - preds[..., 2:3] / 2
        box1_y1 = preds[..., 1:2] - preds[..., 3:4] / 2
        box1_x2 = preds[..., 0:1] + preds[..., 2:3] / 2
        box1_y2 = preds[..., 1:2] + preds[..., 3:4] / 2
        box2_x1 = labels[..., 0:1] - labels[..., 2:3] / 2
        box2_y1 = labels[..., 1:2] - labels[..., 3:4] / 2
        box2_x2 = labels[..., 0:1] + labels[..., 2:3] / 2
        box2_y2 = labels[..., 1:2] + labels[..., 3:4] / 2
    
    # Corners format represents the bounding 
    # box by its top-left and bottom-right coordinates (x1,y1) and (x2,y2) respectively. 
    if format == "corners":
        box1_x1 = preds[..., 0:1]
        box1_y1 = preds[..., 1:2]
        box1_x2 = preds[..., 2:3]
        box1_y2 = preds[..., 3:4]  # (N, 1)
        box2_x1 = labels[..., 0:1]
        box2_y1 = labels[..., 1:2]
        box2_x2 = labels[..., 2:3]
        box2_y2 = labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

# Loại bỏ các bounding box trùng lặp và giữ lại các bounding box có độ tin cậy cao nhất.
# In YOlov1 IOU threshold = 0.5
def NMS(bboxes, iou_threshold=0.5, threshold=0.5, box_format="corners"):
    """
    Does Non Max Suppression given bboxes
    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU) 
        box_format (str): "midpoint" or "corners" used to specify bboxes
    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list
    # filtered boxes with IOU > 0.5 and sorted
    # first element is box with higest confident of boxes
    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []
    # selects the box with the highest confidence score 
    # and removes all other boxes that have a high overlap with the selected box.
    while bboxes:
        # selects the first box in the list and removes it from the list. 
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(torch.tensor(chosen_box[2:]),
                    torch.tensor(box[2:]),box_format=box_format,
            )
            < iou_threshold
            # It's mean IOU < 0.5 when those two boxes are different 
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


