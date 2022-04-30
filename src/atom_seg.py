import json
import numpy as np
from pycocotools import mask
from skimage import measure
import os
import cv2

annotation_id = 0
black = [0,0,0]

category_list = [{
            "supercategory": "textile",
            "isthing": 0,
            "id": 92,
            "name": "banner"
        },
        {
            "supercategory": "textile",
            "isthing": 0,
            "id": 93,
            "name": "blanket"
        },
        {
            "supercategory": "building",
            "isthing": 0,
            "id": 95,
            "name": "bridge"
        },
        {
            "supercategory": "raw-material",
            "isthing": 0,
            "id": 100,
            "name": "cardboard"
        },
        {
            "supercategory": "furniture-stuff",
            "isthing": 0,
            "id": 107,
            "name": "counter"
        },
        {
            "supercategory": "textile",
            "isthing": 0,
            "id": 109,
            "name": "curtain"
        },
        {
            "supercategory": "furniture-stuff",
            "isthing": 0,
            "id": 112,
            "name": "door-stuff"
        },
        {
            "supercategory": "floor",
            "isthing": 0,
            "id": 118,
            "name": "floor-wood"
        },
        {
            "supercategory": "plant",
            "isthing": 0,
            "id": 119,
            "name": "flower"
        },
        {
            "supercategory": "food-stuff",
            "isthing": 0,
            "id": 122,
            "name": "fruit"
        },
        {
            "supercategory": "ground",
            "isthing": 0,
            "id": 125,
            "name": "gravel"
        },
        {
            "supercategory": "building",
            "isthing": 0,
            "id": 128,
            "name": "house"
        },
        {
            "supercategory": "furniture-stuff",
            "isthing": 0,
            "id": 130,
            "name": "light"
        },
        {
            "supercategory": "furniture-stuff",
            "isthing": 0,
            "id": 133,
            "name": "mirror-stuff"
        },
        {
            "supercategory": "structural",
            "isthing": 0,
            "id": 138,
            "name": "net"
        },
        {
            "supercategory": "textile",
            "isthing": 0,
            "id": 141,
            "name": "pillow"
        },
        {
            "supercategory": "ground",
            "isthing": 0,
            "id": 144,
            "name": "platform"
        },
        {
            "supercategory": "ground",
            "isthing": 0,
            "id": 145,
            "name": "playingfield"
        },
        {
            "supercategory": "ground",
            "isthing": 0,
            "id": 147,
            "name": "railroad"
        },
        {
            "supercategory": "water",
            "isthing": 0,
            "id": 148,
            "name": "river"
        },
        {
            "supercategory": "ground",
            "isthing": 0,
            "id": 149,
            "name": "road"
        },
        {
            "supercategory": "building",
            "isthing": 0,
            "id": 151,
            "name": "roof"
        },
        {
            "supercategory": "ground",
            "isthing": 0,
            "id": 154,
            "name": "sand"
        },
        {
            "supercategory": "water",
            "isthing": 0,
            "id": 155,
            "name": "sea"
        },
        {
            "supercategory": "furniture-stuff",
            "isthing": 0,
            "id": 156,
            "name": "shelf"
        },
        {
            "supercategory": "ground",
            "isthing": 0,
            "id": 159,
            "name": "snow"
        },
        {
            "supercategory": "furniture-stuff",
            "isthing": 0,
            "id": 161,
            "name": "stairs"
        },
        {
            "supercategory": "building",
            "isthing": 0,
            "id": 166,
            "name": "tent"
        },
        {
            "supercategory": "textile",
            "isthing": 0,
            "id": 168,
            "name": "towel"
        },
        {
            "supercategory": "wall",
            "isthing": 0,
            "id": 171,
            "name": "wall-brick"
        },
        {
            "supercategory": "wall",
            "isthing": 0,
            "id": 175,
            "name": "wall-stone"
        },
        {
            "supercategory": "wall",
            "isthing": 0,
            "id": 176,
            "name": "wall-tile"
        },
        {
            "supercategory": "wall",
            "isthing": 0,
            "id": 177,
            "name": "wall-wood"
        },
        {
            "supercategory": "water",
            "isthing": 0,
            "id": 178,
            "name": "water-other"
        },
        {
            "supercategory": "window",
            "isthing": 0,
            "id": 180,
            "name": "window-blind"
        },
        {
            "supercategory": "window",
            "isthing": 0,
            "id": 181,
            "name": "window-other"
        },
        {
            "supercategory": "plant",
            "isthing": 0,
            "id": 184,
            "name": "tree-merged"
        },
        {
            "supercategory": "structural",
            "isthing": 0,
            "id": 185,
            "name": "fence-merged"
        },
        {
            "supercategory": "ceiling",
            "isthing": 0,
            "id": 186,
            "name": "ceiling-merged"
        },
        {
            "supercategory": "sky",
            "isthing": 0,
            "id": 187,
            "name": "sky-other-merged"
        },
        {
            "supercategory": "furniture-stuff",
            "isthing": 0,
            "id": 188,
            "name": "cabinet-merged"
        },
        {
            "supercategory": "furniture-stuff",
            "isthing": 0,
            "id": 189,
            "name": "table-merged"
        },
        {
            "supercategory": "floor",
            "isthing": 0,
            "id": 190,
            "name": "floor-other-merged"
        },
        {
            "supercategory": "ground",
            "isthing": 0,
            "id": 191,
            "name": "pavement-merged"
        },
        {
            "supercategory": "solid",
            "isthing": 0,
            "id": 192,
            "name": "mountain-merged"
        },
        {
            "supercategory": "plant",
            "isthing": 0,
            "id": 193,
            "name": "grass-merged"
        },
        {
            "supercategory": "ground",
            "isthing": 0,
            "id": 194,
            "name": "dirt-merged"
        },
        {
            "supercategory": "raw-material",
            "isthing": 0,
            "id": 195,
            "name": "paper-merged"
        },
        {
            "supercategory": "food-stuff",
            "isthing": 0,
            "id": 196,
            "name": "food-other-merged"
        },
        {
            "supercategory": "building",
            "isthing": 0,
            "id": 197,
            "name": "building-other-merged"
        },
        {
            "supercategory": "solid",
            "isthing": 0,
            "id": 198,
            "name": "rock-merged"
        },
        {
            "supercategory": "wall",
            "isthing": 0,
            "id": 199,
            "name": "wall-other-merged"
        },
        {
            "supercategory": "textile",
            "isthing": 0,
            "id": 200,
            "name": "rug-merged"
        }]

def create_annotation_format(masks, category_id, image_id):
    global annotation_id
    annotation = {
            "segmentation": [],
            "area": [],
            "iscrowd": int(0),
            "image_id": int(image_id),
            "bbox": [],
            "category_id": int(category_id),
            "id": int(annotation_id)
        }
    ground_truth_binary_mask= cv2.copyMakeBorder(masks,1,1,1,1,cv2.BORDER_CONSTANT,value=black)
    fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
    encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
    ground_truth_area = mask.area(encoded_ground_truth)
    ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
    contours = measure.find_contours(ground_truth_binary_mask, 0.5)
    annotation["area"] = int(ground_truth_area)
    annotation["category_id"] = int(category_id)
    annotation["bbox"] = ground_truth_bounding_box.tolist()
    for contour in contours:
        contour = np.flip(contour, axis=1).astype(int)
        segmentation = contour.ravel().tolist()
        annotation["segmentation"].append(segmentation)
    annotation_id += 1
    return annotation


def create_category_annotation(category_dict):
    # category_list = []
    global category_list
    # category_list = category

    for key, value in category_dict.items():
        category = {
            "supercategory": key,
            "id": int(value),
            "name": key
        }
        category_list.append(category)

    return category_list

def create_image_annotation(file_name, width, height, image_id):
    images = {
        "file_name": file_name + '.jpg',
        "height": int(height),
        "width": int(width),
        "id": int(image_id)
    }
    return images

def create_image_panpotic_annotation(file_name, width, height, image_id):
    images = {
        "file_name": file_name,
        "height": int(height),
        "width": int(width),
        "id": int(image_id)
    }
    return images

def get_coco_json_format():
    # Standard COCO format 
    coco_format = {
        "info": {},
        "licenses": [],
        "images": [{}],
        "categories": [{}],
        "annotations": [{}]
    }

    return coco_format

def get_coco_json_panoptic_format():
    # Standard COCO format 
    coco_format = {
        "info": {},
        "licenses": [],
        "images": [{}],
        "categories": [{}],
        "annotations": [{}]
    }

    return coco_format

seg_id = 0

def create_seg_info(result):
    # global seg_id
    # for i, info in enumerate(result["segments_info"]):
    #     result["segments_info"][i]["id"] = seg_id
    #     seg_id +=1
    return result["segments_info"]

annotation_id_panoptic = 0

def create_panoptic_annotation_format(image_id, file_name, result):
    segments_info =create_seg_info(result)
    annotation = {
        "segments_info": segments_info,
        "file_name": file_name,
        "image_id": image_id,
    }
    return annotation