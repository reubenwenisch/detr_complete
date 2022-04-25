from PIL import Image                                      # (pip install Pillow)
import numpy as np                                         # (pip install numpy)
from skimage import measure                                # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon         # (pip install Shapely)
import os
import json

def create_sub_masks(mask_image, width, height):
    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x,y))[:3]

            # Check to see if we have created a sub-mask...
            pixel_str = str(pixel)
            sub_mask = sub_masks.get(pixel_str)
            if sub_mask is None:
               # Create a sub-mask (one bit per pixel) and add to the dictionary
                # Note: we add 1 pixel of padding in each direction
                # because the contours module doesn"t handle cases
                # where pixels bleed to the edge of the image
                sub_masks[pixel_str] = Image.new("1", (width+2, height+2))

            # Set the pixel value to 1 (default is 0), accounting for padding
            sub_masks[pixel_str].putpixel((x+1, y+1), 1)

    return sub_masks

def create_sub_mask_annotation(sub_mask):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(np.array(sub_mask), 0.5, positive_orientation="low")

    polygons = []
    segmentations = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        
        if(poly.is_empty):
            # Go to next iteration, dont save empty values in list
            continue

        polygons.append(poly)
        if poly.geom_type == 'Polygon':
            segmentation = np.array(poly.exterior.coords).ravel().tolist()
            segmentations.append(segmentation)
        elif poly.geom_type == 'MultiPolygon':
            for x in poly.geoms:
                segmentation = np.array(x.exterior.coords).ravel().tolist()
                segmentations.append(segmentation)
    return polygons, segmentations

def create_category_annotation(category_dict):
    category_list = []

    for key, value in category_dict.items():
        category = {
            "supercategory": key,
            "id": value,
            "name": key
        }
        category_list.append(category)

    return category_list

def create_image_annotation(file_name, width, height, image_id):
    images = {
        "file_name": file_name + '.jpg',
        "height": height,
        "width": width,
        "id": image_id
    }

    return images

def create_panoptic_image_annotation(file_name, width, height, image_id):
    images = {
        "file_name": file_name + '.jpg',
        "height": height,
        "width": width,
        "id": image_id
    }
    return images

def create_annotation_format(polygon, segmentation, image_id, category_id, annotation_id):
    # print("multi_poly.bounds inside", polygon.bounds)
    min_x, min_y, max_x, max_y = polygon.bounds
    width = max_x - min_x
    height = max_y - min_y
    bbox = (int(min_x), int(min_y), int(width), int(height))
    bbox = np.clip(bbox, 0, np.inf).astype(int).tolist()
    area = polygon.area

    annotation = {
        "segmentation": segmentation,
        "area": area,
        "iscrowd": 0,
        "image_id": image_id,
        "bbox": bbox,
        "category_id": category_id,
        "id": annotation_id
    }

    return annotation

def create_panoptic_annotation_format(image_id, file_name):
    # print("multi_poly.bounds inside", polygon.bounds)
    # min_x, min_y, max_x, max_y = polygon.bounds
    # width = max_x - min_x
    # height = max_y - min_y
    # bbox = (int(min_x), int(min_y), int(width), int(height))
    # bbox = np.clip(bbox, 0, np.inf).astype(int).tolist()
    # area = polygon.area
    segments_infos = [{}]
    # segments_info = {
    #     "id": annotation_id,
    #     "category_id": category_id,
    #     "iscrowd": 0,
    #     "bbox": bbox,
    #     "area": area,
    # }
    annotation = {
        "segments_info": segments_infos,
        "file_name": file_name,
        "image_id": image_id,
    }
    
    return annotation

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

def get_coco_panoptic_json_format():
    # Standard COCO format 
    coco_format = {
        "info": {},
        "licenses": [],
        "images": [{}],
        "categories": [{}],
        "annotations": [{}]
    }

    return coco_format