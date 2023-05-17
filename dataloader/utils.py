import os
import numpy as np
import math
import cv2
from shapely.geometry import Polygon
from PIL import Image, ImageDraw
import random
import pyclipper
import Polygon as plg
import re
def load_ann(gt_paths,max_text_length):
    res = []
    i=0
    for gt in gt_paths:

        i+=1
        item = {}
        item['polys'] = []
        item['tags'] = []
        item['label'] = []

        reader = open(gt, 'r', encoding='utf-8').readlines()
        # reader = open(gt, 'r').readlines()

        for line in reader:
            parts = line.strip().split('\t')[0].split(',')
            label = parts[-1]
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in parts]
            if len(line) < 9:
                continue
            loc = np.array(list(map(float, line[:8]))).reshape((-1, 2)).astype(np.float32)
            item['polys'].append(loc)
            if label == '###':
                item['tags'].append(True)
            else:
                item['tags'].append(False)
            label = re.sub('[^0-9a-zA-Z]+', '', label)
            label = label[:max_text_length]
            item['label'].append(label)

        # for line in reader:
        #     loc, label = line.strip().split('\t')
        #     parts = loc.split(',')
        #     line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in parts + [label]]
        #     if len(line) < 9:
        #         continue
        #     loc = np.array(list(map(float, line[:8]))).reshape((-1, 2)).astype(np.float32)
        #     item['polys'].append(loc)
        #     if label == '###':
        #         item['tags'].append(True)
        #     else:
        #         item['tags'].append(False)
        res.append(item)

    return res
def load_single_ann_ic(gt_path,max_text_length):
    item = {}
    item['polys'] = []
    item['tags'] = []
    item['label'] = []
    reader = open(gt_path, 'r', encoding='utf-8').readlines()
    # reader = open(gt, 'r').readlines()

    for line in reader:
        parts = line.strip().split('\t')[0].split(',')
        label = parts[-1]
        line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in parts]
        if len(line) < 9:
            continue
        loc = np.array(list(map(float, line[:8]))).reshape((-1, 2)).astype(np.float32)
        item['polys'].append(loc)
        if label == '###':
            item['tags'].append(True)
        else:
            item['tags'].append(False)
        label = re.sub('[^0-9a-zA-Z]+', '', label)
        label = label[:max_text_length]
        item['label'].append(label)

    return item
def load_single_ann_ctw(gt_path,max_text_length):
    item = {}
    item['polys'] = []
    item['tags'] = []
    item['label'] = []
    reader = open(gt_path, 'r', encoding='utf-8').readlines()
    line_num = -1

    for line in reader:
        line_num += 1
        if line_num == 0:
            continue
        parts = line.strip().split('\t')[0].split(',')
        line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in parts]
        if len(line) < 29:
            continue
        loc = np.array(list(map(float, line[:28]))).reshape((-1, 2)).astype(np.float32)
        label = line[28:]

        item['polys'].append(loc)
        if label == '###':
            item['tags'].append(True)
        else:
            item['tags'].append(False)
        l = ""
        for i in label:
            l += i
        label = l
        label = re.sub('[^0-9a-zA-Z]+', '', label)
        label = label[:max_text_length]

        item['label'].append(label)

    return item
def load_ann_tt(gt_paths,max_text_length):
    res = []
    for gt in gt_paths:
        item = {}
        item['polys'] = []
        item['tags'] = []
        item['label'] = []

        reader = open(gt, 'r', encoding='utf-8').readlines()
        # reader = open(gt, 'r').readlines()

        for line in reader:
            #parts = line.strip().split('\t')[0].split(',')
            parts = line.strip().split(',')
            raw_label = parts[-1]
            points_coors=[]
            for part in parts[:2]:
                piece = part.strip().split(',')
                numberlist = re.findall(r'\d+', piece[0])
                points_coors.append([int(n) for n in numberlist])
            if np.array(points_coors).transpose(1, 0).shape[0] > 3:
                item['polys'].append(np.array(points_coors).transpose(1,0))
            else:
                continue
            if len(raw_label.split('transcriptions: [u\'')) <= 1 and len(raw_label.split('transcriptions: [u\"')) <= 1:
                item['label'].append("")
                item['tags'].append(True)
            else:
                if len(raw_label.split('transcriptions: [u\'')) <= 1:
                    label = raw_label.split('transcriptions: [u\"')[1].split('\"]')[0]
                else:
                    label = raw_label.split('transcriptions: [u\'')[1].split('\']')[0]
                if label=='#':
                    item['tags'].append(True)
                else:
                    item['tags'].append(False)
                label = re.sub('[^0-9a-zA-Z]+', '', label)
                item['label'].append(label[:max_text_length])
        res.append(item)
    return res
def load_single_ann_tt(gt_path,max_text_length):
    item = {}
    item['polys'] = []
    item['tags'] = []
    item['label'] = []

    reader = open(gt_path, 'r', encoding='utf-8').readlines()

    for line in reader:
        # parts = line.strip().split('\t')[0].split(',')
        parts = line.strip().split(',')
        raw_label = parts[-1]
        points_coors = []
        for part in parts[:2]:
            piece = part.strip().split(',')
            numberlist = re.findall(r'\d+', piece[0])
            points_coors.append([int(n) for n in numberlist])
        if np.array(points_coors).transpose(1, 0).shape[0] > 3:
            item['polys'].append(np.array(points_coors).transpose(1, 0))
        else:
            continue
        if len(raw_label.split('transcriptions: [u\'')) <= 1 and len(raw_label.split('transcriptions: [u\"')) <= 1:
            item['label'].append("")
            item['tags'].append(True)
        else:
            if len(raw_label.split('transcriptions: [u\'')) <= 1:
                label = raw_label.split('transcriptions: [u\"')[1].split('\"]')[0]
            else:
                label = raw_label.split('transcriptions: [u\'')[1].split('\']')[0]

            if label == '#':
                item['tags'].append(True)
            else:
                item['tags'].append(False)
            label = re.sub('[^0-9a-zA-Z]+', '', label)
            if len(label) == 0:
                item['tags'].append(True)
            else:
                item['tags'].append(False)
            item['label'].append(label[:max_text_length])
    return item
def load_single_ann_ic19(target,max_text_length):
    item = {}
    item['polys'] = []
    item['tags'] = []
    item['label'] = []
    for ins in target:
        lan = ins['language']
        label = ins['transcription']
        label = re.sub('[^0-9a-zA-Z]+', '', label)
        if lan!="Latin" or label=="###" or len(label)==0:
            item['tags'].append(True)
        else:
            item['tags'].append(False)
        poly = ins['points']
        item['polys'].append(np.array(poly))
        item['label'].append(label[:max_text_length])
    return item

def load_single_ann_bs(gt_path,max_text_length):
    item = {}
    item['polys'] = []
    item['tags'] = []
    item['label'] = []
    reader = open(gt_path, 'r', encoding='utf-8').readlines()
    # reader = open(gt, 'r').readlines()

    for line in reader:
        parts = line.strip().split('\t')[0].split(',')
        label = parts[-1]
        line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in parts]
        if len(line) < 9:
            continue
        loc = np.array(list(map(float, line[:-1]))).reshape((-1, 2)).astype(np.float32)
        item['polys'].append(loc)
        if label == '###':
            item['tags'].append(True)
        else:
            item['tags'].append(False)
        label = re.sub('[^0-9a-zA-Z]+', '', label)
        label = label[:max_text_length]
        item['label'].append(label)

    return item
def gen_quad_from_poly(poly):
    point_num = poly.shape[0]
    min_area_quad = np.zeros((4, 2), dtype=np.float32)
    rect = cv2.minAreaRect(poly.astype(
            np.int32))  # (center (x,y), (width, height), angle of rotation)
    box = np.array(cv2.boxPoints(rect))

    first_point_idx = 0
    min_dist = 1e4
    for i in range(4):
        dist = np.linalg.norm(box[(i + 0) % 4] - poly[0]) + \
                   np.linalg.norm(box[(i + 1) % 4] - poly[point_num // 2 - 1]) + \
                   np.linalg.norm(box[(i + 2) % 4] - poly[point_num // 2]) + \
                   np.linalg.norm(box[(i + 3) % 4] - poly[-1])
        if dist < min_dist:
            min_dist = dist
            first_point_idx = i
    for i in range(4):
        min_area_quad[i] = box[(first_point_idx + i) % 4]
    return min_area_quad
def quad_area(poly):
    edge = [(poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
                (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
                (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
                (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])]
    return np.sum(edge) / 2.
def check_and_validate_polys( polys, tags, texts,im_size):

    (h, w) = im_size
    if len(polys) == 0:
        return polys, np.array([]),[]
    validated_polys = []
    validated_tags = []
    validated_texts = []
    for poly, tag,text in zip(polys, tags,texts):
        if len(poly)>16:
            print(len(poly))
            keep_ids = [int((len(poly) * 1.0 / 16) * x)for x in range(16)]
            poly = poly[keep_ids, :]
        poly[:,0] = np.clip(poly[:,0], 0, w - 1)
        poly[:,1] = np.clip(poly[:,1], 0, h - 1)
        quad = gen_quad_from_poly(poly)
        p_area = quad_area(quad)
        if abs(p_area) < 1:
            print('invalid poly')
            continue
        if p_area > 0:
            if tag == False:
                print('poly in wrong direction')
                tag = True  # reversed cases should be ignore
            poly = poly[(0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2,1), :]
        validated_polys.append(poly)
        validated_tags.append(tag)
        validated_texts.append(text)

    return validated_polys, validated_tags,validated_texts
def part_poly(poly,rate,mid,lindex,rindex):
    ratio_pair = np.array([[0], [1]], dtype=np.float32)
    tcl_poly = np.zeros((4, 2), dtype=np.float32)
    point_num=len(poly)
    ratio = min(1.0 / rate * 0.1 * (point_num // 2 - 1), 1)
    if mid is not None:
        p_mid=poly[mid]
        p_mid_r=poly[point_num - 1 - mid]
    else:
        p_mid=(poly[lindex]+poly[rindex])/2
        p_mid_r=(poly[point_num - 1 - lindex]+poly[point_num - 1 - rindex])/2

    top = (1 - ratio) * p_mid + ratio * poly[lindex]  # (poly[lindex] + poly[mid]) / 2
    down = (1 - ratio) * p_mid_r + ratio * poly[
        point_num - 1 - lindex]  # (poly[point_num - 1 - lindex] + poly[point_num - 1 - mid]) / 2
    point_pair = top + (down - top) * ratio_pair
    tcl_poly[0] = point_pair[0]
    tcl_poly[3] = point_pair[1]

    top = (1 - ratio) * p_mid + ratio * poly[rindex]  # (poly[rindex] + poly[mid]) / 2
    down = (1 - ratio) * p_mid_r + ratio * (
    poly[point_num - 1 - rindex])  # (poly[point_num - 1 - rindex] + poly[point_num - 1 - mid]) / 2
    point_pair = top + (down - top) * ratio_pair
    tcl_poly[1] = point_pair[0]
    tcl_poly[2] = point_pair[1]
    rect = cv2.minAreaRect(tcl_poly.astype(np.int32))
    bw, bh = rect[1]
    bw_pos = bw * rate
    bh_pos = bh * rate
    pos_poly = cv2.boxPoints((rect[0], (bw_pos, bh_pos), rect[2]))
    return pos_poly
def part_poly_(poly,rate,mid,lindex,rindex):

    ratio_pair = np.array([[0.5-rate/2], [0.5+rate/2]], dtype=np.float32)
    tcl_poly = np.zeros((4, 2), dtype=np.float32)
    point_num=len(poly)
    ratio = min(rate/2 * (point_num // 2 - 1), 1)
    if mid is not None:
        p_mid=poly[mid]
        p_mid_r=poly[point_num - 1 - mid]
    else:
        p_mid=(poly[lindex]+poly[rindex])/2
        p_mid_r=(poly[point_num - 1 - lindex]+poly[point_num - 1 - rindex])/2

    top = (1 - ratio) * p_mid + ratio * poly[lindex]  # (poly[lindex] + poly[mid]) / 2
    down = (1 - ratio) * p_mid_r + ratio * poly[
        point_num - 1 - lindex]  # (poly[point_num - 1 - lindex] + poly[point_num - 1 - mid]) / 2
    point_pair = top + (down - top) * ratio_pair
    tcl_poly[0] = point_pair[0]
    tcl_poly[3] = point_pair[1]

    top = (1 - ratio) * p_mid + ratio * poly[rindex]  # (poly[rindex] + poly[mid]) / 2
    down = (1 - ratio) * p_mid_r + ratio * (
    poly[point_num - 1 - rindex])  # (poly[point_num - 1 - rindex] + poly[point_num - 1 - mid]) / 2
    point_pair = top + (down - top) * ratio_pair
    tcl_poly[1] = point_pair[0]
    tcl_poly[2] = point_pair[1]
    rect = cv2.minAreaRect(tcl_poly.astype(np.int32))
    bw, bh = rect[1]
    pos_poly = cv2.boxPoints((rect[0], (bw, bh), rect[2]))
    return pos_poly
def gen_polys(poly,rate):
    if poly.shape[0]==4:
        new_poly=np.zeros((6,2),np.float32)
        new_poly[0]=poly[0]
        new_poly[1]=(poly[0]+poly[1])/2
        new_poly[2]=poly[1]
        new_poly[3]=poly[2]
        new_poly[4]=(poly[2]+poly[3])/2
        new_poly[5]=poly[3]
        poly_=new_poly
    elif poly.shape[0] == 6:
        new_poly = np.zeros((10, 2), np.float32)
        new_poly[0] = poly[0]
        new_poly[1] = (poly[0] + poly[1]) / 2
        new_poly[2] = poly[1]
        new_poly[3] = (poly[1] + poly[2]) / 2
        new_poly[4] = poly[2]

        new_poly[5] = poly[3]
        new_poly[6] = (poly[3] + poly[4]) / 2
        new_poly[7] = poly[4]
        new_poly[8] = (poly[4] + poly[5]) / 2
        new_poly[9] = poly[5]
        poly_ = new_poly
    else:
        poly_ = poly
    point_num = poly_.shape[0]
    ratio_pair = np.array(
        [[0], [1]], dtype=np.float32)
    if point_num%4==0:
        lindex=int(point_num/4-1)
        rindex=int(point_num/4)
        mid=None
    else:
        lindex=int(point_num/4-1)
        mid=int(point_num/4)
        rindex=int(point_num/4+1)
    return part_poly_(poly_, rate[0], mid, lindex, rindex), part_poly_(poly_, rate[1], mid, lindex, rindex)
def dist(a, b):
    return np.linalg.norm((a - b), ord=2, axis=0)

def judge_scale(poly,scale_range):
    if len(scale_range)==0:
        return 0
    heights = []
    poly_num = len(poly)
    for i in range(poly_num // 2):
        heights.append(dist(poly[i], poly[poly_num - 1 - i]))
    ave_height = sum(heights) / len(heights)
    rect = cv2.minAreaRect(poly.astype(np.int32))
    bw, bh = rect[1]
    lengths=[ave_height, bw, bh]
    index=np.nonzero(min(lengths)>scale_range)[0]

    if len(index)==0:
        return 0
    else:
        return max(index)+1

def get_centerline(img,poly,fixed_point_num):
    """
    Generate center line by poly clock-wise point.
    """
    point_num = poly.shape[0]
    center_line=[]
    for idx in range(point_num // 2):
        point_center = (poly[idx] + poly[point_num - 1 - idx]) * 0.5
        center_line.append(point_center)
    min_area_quad=gen_quad_from_poly(poly)
    tmp_image = np.zeros((img.shape[0], img.shape[1]), dtype='float32')
    cv2.polylines(tmp_image, [np.array(center_line).astype('int32')], False, 1.0)
    ys, xs = np.where(tmp_image > 0)
    xy_text = list(zip(xs, ys))
    if len(xy_text)<fixed_point_num:
        sample_num = fixed_point_num - len(xy_text)
        sample_index=np.arange(len(xy_text))
        chosed_index=np.random.choice(sample_index, sample_num)
        sample_list = np.array(xy_text)[chosed_index]
        xy_text.extend(sample_list)
    xy_text=np.array(xy_text,dtype=np.float32)
    left_center_pt = ((min_area_quad[0] - min_area_quad[1]) / 2.0).reshape(1, 2)
    right_center_pt = ((min_area_quad[1] - min_area_quad[2]) / 2.0).reshape(1, 2)
    proj_unit_vec = (right_center_pt - left_center_pt) / (
            np.linalg.norm(right_center_pt - left_center_pt) + 1e-6)
    proj_unit_vec_tile = np.tile(proj_unit_vec, (xy_text.shape[0], 1))  # (n, 2)
    left_center_pt_tile = np.tile(left_center_pt, (xy_text.shape[0], 1))  # (n, 2)
    xy_text_to_left_center = xy_text - left_center_pt_tile
    proj_value = np.sum(xy_text_to_left_center * proj_unit_vec_tile, axis=1)
    xy_text = xy_text[np.argsort(proj_value)]
    pos_info = np.array(xy_text).reshape(-1, 2)#[:, ::-1]
    point_num = len(pos_info)
    if point_num > fixed_point_num:
        keep_ids = [
            int((point_num * 1.0 / fixed_point_num) * x)
            for x in range(fixed_point_num)
        ]
        pos_info = pos_info[keep_ids, :]
    pos_info[:, 0] = np.clip(pos_info[:, 0], 0, img.shape[1] - 1)
    pos_info[:, 1] = np.clip(pos_info[:, 1], 0, img.shape[0] - 1)
    return pos_info

def generate_gt(im, boxes, tags, text, converter,args,flag=0):
    h, w, _ = im.shape
    text_polys, text_tags, text = check_and_validate_polys( boxes, tags,text, (h,w))#text_polys, text_tags = boxes, tags
    text_labels = []
    text_lengths = []
    keep_polys = []
    ins_masks = []
    training_mask=[]
    score_map=[]
    loc_map=[]
    sampled_map=[]
    center_points=[]
    num_points=[]
    fpn_strides = args.fpn_strides#[4, 8, 16, 32, 64] #, 16, 32]#[64, 128, 256, 512]
    scale_range =np.array([40, 80, 160, 320,640])[:len(fpn_strides)-1] #(0,40),(40,80),(80,160),(160,320),(320,640)
    for i in range(len(fpn_strides)):
        t_t_mask = np.ones((h // fpn_strides[i], w // fpn_strides[i]), dtype=np.uint8)
        training_mask.append(t_t_mask)
        t_s_mask = np.zeros((h // fpn_strides[i], w // fpn_strides[i]), dtype=np.uint8)
        score_map.append(t_s_mask)
        t_loc_map = np.zeros((4, h  // fpn_strides[i], w  // fpn_strides[i]), dtype=np.float32)
        loc_map.append(t_loc_map)
        t_sampled_points = np.zeros((args.nums_sample_point * 2, h // fpn_strides[i], w  // fpn_strides[i]), dtype=np.float32)
        sampled_map.append(t_sampled_points)
        text_labels.append([])
        text_lengths.append([])
        center_points.append([])
        num_points.append([])
    if len(text_polys) > 0:
        for poly_idx, poly_tag_text in enumerate(zip(text_polys, text_tags, text)):
            poly = poly_tag_text[0]
            tag = poly_tag_text[1]
            t_text = poly_tag_text[2]
            pos_poly,ignore_poly=gen_polys(poly,[0.2,0.5])
            t_index = judge_scale(poly, scale_range)
            '''
            rect = cv2.minAreaRect(poly.astype(np.int32))
            bw, bh = rect[1]
            bw_pos = bw*args.pos_scale
            bh_pos = bh*args.pos_scale
            bw_ignore = bw*args.ignore_scale
            bh_ignore = bh*args.ignore_scale
            pos_poly = cv2.boxPoints((rect[0], (bw_pos, bh_pos), rect[2]))
            ignore_poly = cv2.boxPoints((rect[0], (bw_ignore, bh_ignore), rect[2]))'''

            if tag == True:
                cv2.fillPoly(training_mask[t_index],(ignore_poly / fpn_strides[t_index]).astype(np.int32)[np.newaxis, :, :], 0)
            else:
                temp = np.zeros((h // fpn_strides[t_index], w // fpn_strides[t_index]), dtype=np.uint8)
                cv2.fillPoly(temp, (pos_poly / fpn_strides[t_index]).astype(np.int32)[np.newaxis, :, :], 1)
                if len(np.argwhere(temp == 1)) == 0:
                    cv2.fillPoly(training_mask[t_index],(ignore_poly / fpn_strides[t_index]).astype(np.int32)[np.newaxis, :, :], 0)
                    continue

                ins_masks.append(temp)
                keep_polys.append(poly)
                xmin, xmax = poly[:, 0].min(), poly[:, 0].max()
                ymin, ymax = poly[:, 1].min(), poly[:, 1].max()
                cur_sampled_points = get_centerline(im,poly / fpn_strides[t_index], args.max_text_length)
                norm_sampled_x = 2.0 * cur_sampled_points[:, 0] / (w / fpn_strides[t_index] - 1) - 1.0
                norm_sampled_y = 2.0 * cur_sampled_points[:, 1] / (h / fpn_strides[t_index] - 1) - 1.0
                text_label, length = converter.encode(t_text)

                xy_in_poly = np.argwhere(temp == 1)
                for y, x in xy_in_poly:
                    ori_x = x * fpn_strides[t_index] + fpn_strides[t_index] // 2
                    ori_y = y * fpn_strides[t_index] + fpn_strides[t_index] // 2
                    loc_map[t_index][0, y, x] = (ori_x - xmin)  # /fpn_strides[t_index]
                    loc_map[t_index][1, y, x] = (ori_y - ymin)  # /fpn_strides[t_index]
                    loc_map[t_index][2, y, x] = (xmax - ori_x)  # /fpn_strides[t_index]
                    loc_map[t_index][3, y, x] = (ymax - ori_y)  # /fpn_strides[t_index]
                    text_labels[t_index].append(text_label)
                    text_lengths[t_index].append(length)
                    center_points[t_index].append([x, y])
                    for j in range(len(norm_sampled_x)):
                        norm_center_x = 2.0 * x / (w / fpn_strides[t_index] - 1) - 1.0
                        norm_center_y = 2.0 * y / (h / fpn_strides[t_index] - 1) - 1.0
                        sampled_map[t_index][2 * j][y][x] = (norm_sampled_x[j] - norm_center_x) * w/fpn_strides[t_index]
                        sampled_map[t_index][2 * j + 1][y][x] = (norm_sampled_y[j] - norm_center_y) * h/fpn_strides[t_index]
                cv2.fillPoly(training_mask[t_index],(ignore_poly / fpn_strides[t_index]).astype(np.int32)[np.newaxis, :, :], 0)
                cv2.fillPoly(training_mask[t_index],(pos_poly / fpn_strides[t_index]).astype(np.int32)[np.newaxis, :, :], 1)
                cv2.fillPoly(score_map[t_index], (pos_poly / fpn_strides[t_index]).astype(np.int32)[np.newaxis, :, :],1)
                num_points[t_index].append(len(xy_in_poly))
    num_of_polys = len(keep_polys)
    if num_of_polys == 0:
        keep_polys.append(np.zeros((4, 2)))
        ins_masks.append(np.zeros((h//4, w//4)))

    if args.vis_gt and flag==1:
        img = im.copy()[:, :, (2, 1, 0)].astype(np.uint8)
        img1 = img.copy()
        h, w, c = img.shape
        img = Image.fromarray(img)
        img1 = Image.fromarray(img1)
        img_draw = ImageDraw.Draw(img)

        for ind, poly_ in enumerate(keep_polys):

            poly = list(poly_.reshape((-1)))
            img_draw.line(poly + poly[:2], width=4, fill=(0, 255, 0))
            cur_sampled_points = get_centerline(im, poly_, args.max_text_length)
            img_draw.line(cur_sampled_points[:10], width=4, fill=(255, 0, 0))

        m = score_map+ training_mask+ ins_masks
        m1 = []
        for tm in m:
            m1.append(Image.blend(img1, Image.fromarray(tm*255).convert('RGB').resize((w, h)), 0.5))

        new_im = Image.new('RGB', (w*(len(m1) + 1), h))

        x_offset = 0
        for tm in [img] + m1:
            new_im.paste(tm, (x_offset,0))
            x_offset += w
        if os.path.exists('./outputs') == False:
            os.mkdir('./outputs')
        if os.path.exists('./outputs/eval_gt') == False:
            os.mkdir('./outputs/eval_gt')
        new_im.save('./outputs/eval_gt/' + str(random.randint(0, 10000))+ '.jpg')

    return im,  keep_polys, ins_masks, num_of_polys,training_mask,score_map,loc_map,sampled_map,text_labels,text_lengths,center_points,num_points




