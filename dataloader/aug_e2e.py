import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
from shapely.geometry import box, Polygon
import math
from math import fabs, sin, cos, radians
from shapely import affinity


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, tags=None, text=None):
        for t in self.transforms:
            img, boxes, tags, text = t(img, boxes, tags, text)
        return img, boxes, tags, text


class RandomResize(object):
    def __init__(self, ratio=[0.5, 1, 2, 3]):
        self.ratio = ratio

    def __call__(self, image, boxes, tags, text):
        scale_ratio = random.choice(self.ratio)
        new_img = cv2.resize(image, None, fx=scale_ratio, fy=scale_ratio)
        new_boxes = []
        for box in boxes:
            new_boxes.append(box * scale_ratio)
        return new_img, new_boxes, tags, text


class RandomResizeShort(object):
    def __init__(self, stride=32, min=10, max=40):
        self.stride = stride
        self.min = min
        self.max = max

    def __call__(self, image, boxes, tags,text):
        min_size = self.stride * random.randint(self.min, self.max)
        h, w = image.shape[:2]
        scale_ratio = min_size * 1.0 / min(h, w)

        new_img = cv2.resize(image, None, fx=scale_ratio, fy=scale_ratio)
        new_boxes = []
        for box in boxes:
            new_boxes.append(box * scale_ratio)
        return new_img, new_boxes, tags,text


class RandomResizeShort_(object):
    def __init__(self, stride=32, min=10, max=40):
        self.stride = stride
        self.min = min
        self.max = max

    def __call__(self, image, boxes, tags,text):
        min_size = random.randint(640, 896)
        h, w = image.shape[:2]
        scale_ratio = min_size * 1.0 / min(h, w)

        new_img = cv2.resize(image, None, fx=scale_ratio, fy=scale_ratio)
        new_boxes = []
        for box in boxes:
            new_boxes.append(box * scale_ratio)
        return new_img, new_boxes, tags,text
class RandomFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, boxes, tags, text):
        if random.random() > self.prob:
            ## flip img
            h, w, c = image.shape
            image = image[:, ::-1, :]
            new_boxes = []
            for box in boxes:
                box = box.copy()
                box[:, 0] = w - box[:, 0]
                new_boxes.append(box)
            return image, new_boxes, tags, text
        else:
            return image, boxes, tags, text


class RandomRotate(object):
    def __init__(self, prob, max_theta=10):
        self.prob = prob
        self.max_theta = max_theta

    def __call__(self, image, boxes, tags, text):
        if random.random() < self.prob:
            degree = random.uniform(-1 * self.max_theta, self.max_theta)
            height, width, _ = image.shape
            heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
            widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
            matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

            matRotation[0, 2] += (widthNew - width) / 2
            matRotation[1, 2] += (heightNew - height) / 2
            imgRotation = cv2.warpAffine(image, matRotation, (widthNew, heightNew), borderValue=(0, 0, 0))

            # start_h, start_w = int((heightNew - height)/2.0), int((widthNew - width)/2.0)
            # boxes = _rotate_segms(boxes, -1*degree, (widthNew/2, heightNew/2), start_h, start_w)
            # boxes = np.array(boxes).reshape((-1, 4, 2))
            boxes = _rotate_segms_v1(boxes, matRotation)

            return imgRotation, boxes, tags, text

        else:
            return image, boxes, tags


def _rect2quad(boxes):
    x_min, y_min, x_max, y_max = boxes[:, 0].reshape((-1, 1)), boxes[:, 1].reshape((-1, 1)), boxes[:, 2].reshape(
        (-1, 1)), boxes[:, 3].reshape((-1, 1))
    return np.hstack((x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max))


def _quad2rect(boxes):
    ## only support rectangle
    return np.hstack((boxes[:, 0].reshape((-1, 1)), boxes[:, 1].reshape((-1, 1)), boxes[:, 4].reshape((-1, 1)),
                      boxes[:, 5].reshape((-1, 1))))


def _quad2minrect(boxes):
    ## trans a quad(N*4) to a rectangle(N*4) which has miniual area to cover it
    return np.hstack((boxes[:, ::2].min(axis=1).reshape((-1, 1)), boxes[:, 1::2].min(axis=1).reshape((-1, 1)),
                      boxes[:, ::2].max(axis=1).reshape((-1, 1)), boxes[:, 1::2].max(axis=1).reshape((-1, 1))))


def _quad2boxlist(boxes):
    res = []
    for i in range(boxes.shape[0]):
        res.append([[boxes[i][0], boxes[i][1]], [boxes[i][2], boxes[i][3]], [boxes[i][4], boxes[i][5]],
                    [boxes[i][6], boxes[i][7]]])
    return res


def _boxlist2quads(boxlist):
    res = np.zeros((len(boxlist), 8))
    for i, box in enumerate(boxlist):
        # print(box)
        res[i] = np.array([box[0][0], box[0][1], box[1][0], box[1][1], box[2][0], box[2][1], box[3][0], box[3][1]])
    return res


def _rotate_polygons(polygons, angle, r_c):
    ## polygons: N*8
    ## r_x: rotate center x
    ## r_y: rotate center y
    ## angle: -15~15

    poly_list = _quad2boxlist(polygons)
    rotate_boxes_list = []
    for poly in poly_list:
        box = Polygon(poly)
        rbox = affinity.rotate(box, angle, r_c)
        if len(list(rbox.exterior.coords)) < 5:
            print(poly)
            print(rbox)
        # assert(len(list(rbox.exterior.coords))>=5)
        rotate_boxes_list.append(rbox.boundary.coords[:-1])
    res = _boxlist2quads(rotate_boxes_list)
    return res


def _rotate_segms(polygons, angle, r_c, start_h, start_w):
    ## polygons: N*8
    ## r_x: rotate center x
    ## r_y: rotate center y
    ## angle: -15~15
    poly_list = []
    for polygon in polygons:
        tmp = []
        polygon = polygon.reshape((-1))
        for i in range(int(len(polygon) / 2)):
            tmp.append([polygon[2 * i] + start_w, polygon[2 * i + 1] + start_h])
        poly_list.append(tmp)

    rotate_boxes_list = []
    for poly in poly_list:
        box = Polygon(poly)
        rbox = affinity.rotate(box, angle, r_c)
        if len(list(rbox.exterior.coords)) < 5:
            print(poly)
            print(rbox)
        # assert(len(list(rbox.exterior.coords))>=5)
        rotate_boxes_list.append(rbox.boundary.coords[:-1])
    res = []
    for i, box in enumerate(rotate_boxes_list):
        tmp = []
        for point in box:
            tmp.append(point[0])
            tmp.append(point[1])
        res.append(np.array(tmp).reshape((-1, 2)))

    return res


def _rotate_segms_v1(polygons, matRotation):
    poly_list = []
    for polygon in polygons:
        poly = cv2.transform(polygon.reshape(1, -1, 2), matRotation)[0]
        poly_list.append(poly)
    return poly_list


class RandomCrop(object):
    def __init__(self, crop_size=640, max_tries=50, min_crop_side_ratio=0.1):
        self.crop_size = crop_size
        self.max_tries = max_tries
        self.min_crop_side_ratio = min_crop_side_ratio

    def __call__(self, image, boxes, tags, text):
        h, w, _ = image.shape
        h_array = np.zeros((h), dtype=np.int32)
        w_array = np.zeros((w), dtype=np.int32)

        for box in boxes:
            box = np.round(box, decimals=0).astype(np.int32)
            minx = np.min(box[:, 0])
            maxx = np.max(box[:, 0])
            w_array[minx:maxx] = 1
            miny = np.min(box[:, 1])
            maxy = np.max(box[:, 1])
            h_array[miny:maxy] = 1

        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]
        if len(h_axis) == 0 or len(w_axis) == 0:
            ## resize image
            return regular_resize(image, boxes, tags,text, self.crop_size)

        if h <= self.crop_size + 1 or w <= self.crop_size + 1:
            return random_crop(image, boxes, tags, text, self.crop_size, self.max_tries, w_axis, h_axis,
                               self.min_crop_side_ratio)
        else:
            return regular_crop(image, boxes, tags, text,self.crop_size, self.max_tries, w_array, h_array, w_axis, h_axis,
                                self.min_crop_side_ratio)


def regular_resize(image, boxes, tags, text, crop_size):
    h, w, c = image.shape
    if max(h, w) < crop_size:
        scale_ratio = 1
        new_img = np.zeros((crop_size, crop_size, 3))
        new_img[:h, :w, :] = image
    else:
        if h < w:
            scale_ratio = crop_size * 1.0 / w
            new_h = int(round(crop_size * h * 1.0 / w))
            if new_h > crop_size:
                new_h = crop_size
            image = cv2.resize(image, (crop_size, new_h))
            new_img = np.zeros((crop_size, crop_size, 3))
            new_img[:new_h, :, :] = image
        else:
            scale_ratio = crop_size * 1.0 / h
            new_w = int(round(crop_size * w * 1.0 / h))
            if new_w > crop_size:
                new_w = crop_size
            image = cv2.resize(image, (new_w, crop_size))
            new_img = np.zeros((crop_size, crop_size, 3))
            new_img[:, :new_w, :] = image
    new_boxes = []
    for box in boxes:
        new_boxes.append(box * scale_ratio)
    return new_img, new_boxes, tags ,text

def random_crop(image, boxes, tags, text, crop_size, max_tries, w_axis, h_axis, min_crop_side_ratio):
    h, w, c = image.shape
    selected_boxes = []
    for i in range(max_tries):
        xx = np.random.choice(w_axis, size=2)
        xmin = np.min(xx)
        xmax = np.max(xx)
        xmin = np.clip(xmin, 0, w - 1)
        xmax = np.clip(xmax, 0, w - 1)
        yy = np.random.choice(h_axis, size=2)
        ymin = np.min(yy)
        ymax = np.max(yy)
        ymin = np.clip(ymin, 0, h - 1)
        ymax = np.clip(ymax, 0, h - 1)
        if xmax - xmin < min_crop_side_ratio * w or ymax - ymin < min_crop_side_ratio * h:
            # area too small
            continue
        if len(boxes) != 0:
            # box_axis_in_area = (boxes[:, :, 0] >= xmin) & (boxes[:, :, 0] <= xmax) \
            #                 & (boxes[:, :, 1] >= ymin) & (boxes[:, :, 1] <= ymax)
            box_axis_in_area = [(box[:, 0] >= xmin) & (box[:, 0] <= xmax) \
                                & (box[:, 1] >= ymin) & (box[:, 1] <= ymax) for box in boxes]

            selected_boxes = []
            for tindex, tbox in enumerate(box_axis_in_area):
                if tbox.sum() == boxes[tindex].shape[0]:
                    selected_boxes.append(tindex)

            selected_boxes = np.array(selected_boxes)
            # selected_boxes = np.where(np.sum(box_axis_in_area, axis=1) == 4)[0]
            if len(selected_boxes) > 0:
                break
        else:
            selected_boxes = []
            break
    if i == max_tries - 1:
        return regular_resize(image, boxes, tags, text, crop_size)

    image = image[ymin:ymax + 1, xmin:xmax + 1, :]
    temp_boxes = []
    temp_tags = []
    temp_text = []
    for tindex in selected_boxes:
        temp_boxes.append(boxes[tindex])
        temp_tags.append(tags[tindex])
        temp_text.append(text[tindex])
    # boxes = boxes[selected_boxes]
    # tags = tags[selected_boxes]
    boxes = temp_boxes
    tags = temp_tags
    text = temp_text
    new_boxes = []
    for box in boxes:
        box[:, 0] -= xmin
        box[:, 1] -= ymin
        new_boxes.append(box)
    return regular_resize(image, new_boxes, tags, text, crop_size)


def regular_crop(image, boxes, tags, text, crop_size, max_tries, w_array, h_array, w_axis, h_axis, min_crop_side_ratio):
    h, w, c = image.shape
    mask_w = np.arange(w - crop_size)
    mask_h = np.arange(h - crop_size)
    keep_w = np.where(np.logical_and(w_array[mask_w] == 0, w_array[mask_w + crop_size - 1] == 0))[0]
    keep_h = np.where(np.logical_and(h_array[mask_h] == 0, h_array[mask_h + crop_size - 1] == 0))[0]

    if keep_w.size > 0 and keep_h.size > 0:
        for i in range(max_tries):
            xmin = np.random.choice(keep_w, size=1)[0]
            xmax = xmin + crop_size
            ymin = np.random.choice(keep_h, size=1)[0]
            ymax = ymin + crop_size

            if len(boxes) != 0:
                # box_axis_in_area = (boxes[:, :, 0] >= xmin) & (boxes[:, :, 0] <= xmax) \
                #                 & (boxes[:, :, 1] >= ymin) & (boxes[:, :, 1] <= ymax)
                box_axis_in_area = [(box[:, 0] >= xmin) & (box[:, 0] <= xmax) \
                                    & (box[:, 1] >= ymin) & (box[:, 1] <= ymax) for box in boxes]

                selected_boxes = []
                for tindex, tbox in enumerate(box_axis_in_area):
                    if tbox.sum() == boxes[tindex].shape[0]:
                        selected_boxes.append(tindex)

                selected_boxes = np.array(selected_boxes)
                # selected_boxes = np.where(np.sum(box_axis_in_area, axis=1) == 4)[0]
                if len(selected_boxes) > 0:
                    break
            else:
                selected_boxes = []
                break

        image = image[ymin:ymax, xmin:xmax, :]
        temp_boxes = []
        temp_tags = []
        temp_text = []
        for tindex in selected_boxes:
            temp_boxes.append(boxes[tindex])
            temp_tags.append(tags[tindex])
            temp_text.append(text[tindex])
        # boxes = boxes[selected_boxes]
        # tags = tags[selected_boxes]
        boxes = temp_boxes
        tags = temp_tags
        text = temp_text
        new_boxes = []
        for box in boxes:
            box[:, 0] -= xmin
            box[:, 1] -= ymin
            new_boxes.append(box)
        return image, new_boxes, tags, text
    else:
        return random_crop(image, boxes, tags, text, crop_size, max_tries, w_axis, h_axis, min_crop_side_ratio)


class RandomBlur(object):
    def __init__(self, prob=0.1):
        self.prob = prob
        self.blur_methods = ['GaussianBlur', 'AverageBlur', 'MedianBlur', 'BilateralBlur', 'MotionBlur']

    def __call__(self, image, boxes, tags, text):
        ## aug1 0-3.0  3-9 3-9 3-9 3-9
        try:
            if random.random() < self.prob:
                blur = random.choice(self.blur_methods)
                if blur == 'GaussianBlur':
                    seq = iaa.Sequential([iaa.GaussianBlur(sigma=(0, 3.0))])
                elif blur == 'AverageBlur':
                    seq = iaa.Sequential([iaa.AverageBlur(k=(3, 9))])
                elif blur == 'MedianBlur':
                    seq = iaa.Sequential([iaa.MedianBlur(k=(3, 9))])
                elif blur == 'BilateralBlur':
                    seq = iaa.Sequential([iaa.BilateralBlur((3, 9), sigma_color=250, sigma_space=250)])
                else:
                    seq = iaa.Sequential([iaa.MotionBlur(k=(3, 9), angle=0, direction=0.0)])

                images = np.expand_dims(image, axis=0)
                images_aug = seq.augment_images(images)
                image = images_aug[0]
            return image, boxes, tags, text
        except:
            return image, boxes, tags, text


class RandomColorJitter(object):
    def __init__(self, prob):
        self.prob = prob
        self.func = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)

    def __call__(self, image, boxes, tags, text):
        if random.random() < self.prob:
            image = self.func(image)
            return image, boxes, tags, text
        return image, boxes, tags, text


class ToPIL(object):
    def __init__(self):
        self.converter = transforms.ToPILImage()

    def __call__(self, image, boxes, tags, text):
        return self.converter(image.astype(np.uint8)), boxes, tags, text


class ToNP(object):
    def __init__(self):
        pass

    def __call__(self, image, boxes, tags, text):
        return np.array(image), boxes, tags, text



class PSSAugmentation_e2e(object):
    def __init__(self, size):
        self.augment = Compose([
            #RandomFlip(),
            RandomRotate(1, 10),
            # RandomResize(),
            RandomResizeShort(),
            RandomCrop(size),
            RandomBlur(0.1),
            ## cv2 to PIL
            ToPIL(),
            ## apply ColorJitter
            RandomColorJitter(0.5),
            ToNP(),
        ])

    def __call__(self, img, boxes, tags,text):
        return self.augment(img, boxes, tags,text)
class PSSAugmentation_e2e_(object):
    def __init__(self, size):
        self.augment = Compose([
            #RandomFlip(),
            RandomRotate(1, 10),
            RandomResize(),
            #RandomResizeShort(),
            RandomCrop(size),
            RandomBlur(0.1),
            ## cv2 to PIL
            ToPIL(),
            ## apply ColorJitter
            RandomColorJitter(0.5),
            ToNP(),
        ])

    def __call__(self, img, boxes, tags,text):
        return self.augment(img, boxes, tags,text)
