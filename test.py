# reference: https://zhuanlan.zhihu.com/p/270663039
import yolo_object_detection as yolo_obj
import torch
import numpy as np
import math
import cv2
'''
def bbox_overlaps_ciou(bboxes1, bboxes2):
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    cious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return cious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        cious = torch.zeros((cols, rows))
        exchange = True

    #w1 = bboxes1[:, 2] - bboxes1[:, 0]
    #h1 = bboxes1[:, 3] - bboxes1[:, 1]
    #w2 = bboxes2[:, 2] - bboxes2[:, 0]
    #h2 = bboxes2[:, 3] - bboxes2[:, 1]

    w1 = bboxes1[2] - bboxes1[0]
    h1 = bboxes1[3] - bboxes1[1]
    w2 = bboxes2[2] - bboxes2[0]
    h2 = bboxes2[3] - bboxes2[1]
    area1 = w1 * h1
    area2 = w2 * h2
    print("area1:%d" % area1)
    print("area2:%d" % area2)


    #center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    #center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    #center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    #center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2
    
    center_x1 = (bboxes1[2] + bboxes1[0]) / 2
    center_y1 = (bboxes1[3] + bboxes1[1]) / 2
    center_x2 = (bboxes2[2] + bboxes2[0]) / 2
    center_y2 = (bboxes2[3] + bboxes2[1]) / 2

    print("center_x1:%d" % center_x1)
    print("center_y1:%d" % center_y1)
    print("center_x2:%d" % center_x2)
    print("center_y2:%d" % center_y2)

    #inter_max_xy = torch.min(bboxes1[:, 2:],bboxes2[:, 2:])
    #inter_min_xy = torch.max(bboxes1[:, :2],bboxes2[:, :2])
    #out_max_xy = torch.max(bboxes1[:, 2:],bboxes2[:, 2:])
    #out_min_xy = torch.min(bboxes1[:, :2],bboxes2[:, :2])


    inter_max_xy = min(bboxes1[2],bboxes2[2])
    inter_min_xy = max(bboxes1[2],bboxes2[2])
    out_max_xy = max(bboxes1[2],bboxes2[2])
    out_min_xy = min(bboxes1[2],bboxes2[2])

    print("inter_max_xy:%d" % inter_max_xy)
    print("inter_min_xy:%d" % inter_min_xy)
    print("out_max_xy:%d" % out_max_xy)
    print("out_min_xy:%d" % out_min_xy)


    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1+area2-inter_area
    u = (inter_diag) / outer_diag
    iou = inter_area / union
    with torch.no_grad():
        arctan = torch.atan(w2 / h2) - torch.atan(w1 / h1)
        v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
        S = 1 - iou
        alpha = v / (S + v)
        w_temp = 2 * w1
    ar = (8 / (math.pi ** 2)) * arctan * ((w1 - w_temp) * h1)
    cious = iou - (u + alpha * ar)
    cious = torch.clamp(cious,min=-1.0,max = 1.0)
    if exchange:
        cious = cious.T
    return cious
'''

def calculate_diou(box_1, box_2):
    """
    calculate diou
    :param box_1: (x0, y0, x1, y1)
    :param box_2: (x0, y0, x1, y1)
    :return: value of diou
    """
    # calculate area of each box
    area_1 = (box_1[2] - box_1[0]) * (box_1[3] - box_1[1])
    area_2 = (box_2[2] - box_2[0]) * (box_1[3] - box_1[1])

    # calculate center point of each box
    center_x1 = (box_1[2] - box_1[0]) / 2
    center_y1 = (box_1[3] - box_1[1]) / 2
    center_x2 = (box_2[2] - box_2[0]) / 2
    center_y2 = (box_2[3] - box_2[1]) / 2

    # calculate square of center point distance
    p2 = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2

    # calculate square of the diagonal length
    width_c = max(box_1[2], box_2[2]) - min(box_1[0], box_2[0])
    height_c = max(box_1[3], box_2[3]) - min(box_1[1], box_2[1])
    c2 = width_c ** 2 + height_c ** 2

    # find the edge of intersect box
    top = max(box_1[0], box_2[0])
    left = max(box_1[1], box_2[1])
    bottom = min(box_1[3], box_2[3])
    right = min(box_1[2], box_2[2])

    # calculate the intersect area
    area_intersection = (right - left) * (bottom - top)

    # calculate the union area
    area_union = area_1 + area_2 - area_intersection

    # calculate iou
    iou = float(area_intersection) / area_union
    print("iou:%f" % iou)

    # calculate diou(iou - p2/c2)
    diou = iou - float(p2) / c2
    print("diou:%f" % diou)

    return diou

def calculate_ciou(box_1, box_2):
    """
    calculate ciou
    :param box_1: (x0, y0, x1, y1)
    :param box_2: (x0, y0, x1, y1)
    :return: value of ciou
    """
    # calculate area of each box
    width_1 = box_1[2] - box_1[0]
    height_1 = box_1[3] - box_1[1]
    area_1 = width_1 * height_1

    width_2 = box_2[2] - box_2[0]
    height_2 = box_2[3] - box_2[1]
    area_2 = width_2 * height_2

    # calculate center point of each box
    center_x1 = (box_1[2] - box_1[0]) / 2
    center_y1 = (box_1[3] - box_1[1]) / 2
    center_x2 = (box_2[2] - box_2[0]) / 2
    center_y2 = (box_2[3] - box_2[1]) / 2

    # calculate square of center point distance
    p2 = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2

    # calculate square of the diagonal length
    width_c = max(box_1[2], box_2[2]) - min(box_1[0], box_2[0])
    height_c = max(box_1[3], box_2[3]) - min(box_1[1], box_2[1])
    c2 = width_c ** 2 + height_c ** 2

    # find the edge of intersect box
    left = max(box_1[0], box_2[0])
    top = max(box_1[1], box_2[1])
    bottom = min(box_1[3], box_2[3])
    right = min(box_1[2], box_2[2])

    # calculate the intersect area
    area_intersection = (right - left) * (bottom - top)

    # calculate the union area
    area_union = area_1 + area_2 - area_intersection

    # calculate iou
    iou = float(area_intersection) / area_union
    print("iou:%f" % iou)

    # calculate v
    arctan = math.atan(float(width_2) / height_2) - math.atan(float(width_1) / height_1)
    v = (4.0 / math.pi ** 2) * (arctan ** 2)

    # calculate alpha
    alpha = float(v) / (1 - iou + v)

    # calculate ciou(iou - p2 / c2 - alpha * v)
    ciou = iou - float(p2) / c2 - alpha * v
    print("ciou:%f" % ciou)
    return ciou

def use_ROI_select(frame):
    name = "ROI"
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 1920, 1280)
    ROI = cv2.selectROI(name, frame, False)
    cv2.destroyWindow(name)
    return ROI


if __name__ == '__main__':
    img = cv2.imread('./image/1.jpg') 

    # bbox1 from yolo
    yolo = yolo_obj.yolo_object_detection()
    bbox_temp = yolo.run_detection(img, 'person')
    print(bbox_temp)
    bbox_yolo = []

    bbox_yolo.append(bbox_temp[0][0])
    bbox_yolo.append(bbox_temp[0][1])
    bbox_yolo.append(bbox_temp[0][0] + bbox_temp[0][2])
    bbox_yolo.append(bbox_temp[0][1] + bbox_temp[0][3])
    print(bbox_yolo)

    # bbox2 from user

    bbox_roi = []
    bbox_roi = use_ROI_select(img)
    bbox_user = []
    bbox_user.append(bbox_roi[0])
    bbox_user.append(bbox_roi[1])
    bbox_user.append(bbox_roi[0] + bbox_roi[2])
    bbox_user.append(bbox_roi[1] + bbox_roi[3])
    print(bbox_user)

    ciou = calculate_ciou(bbox_yolo, bbox_user)
