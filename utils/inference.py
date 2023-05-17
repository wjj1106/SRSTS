import os
import re
import cv2
import torch
import logging
import numpy as np
import editdistance
from lib import cpu_nms
import torch.nn.functional as F
from PIL import Image, ImageFont
from skimage.measure import regionprops
from shapely.geometry import box, Polygon
from torch.autograd import Variable
import torch.utils.data as data
from models.model_v1 import SRSTS_v1
from skimage.morphology import label as bwlabel
from utils.str_label_converter import StrLabelConverter
from dataloader.dataloader import MultiStageTextLoader
from utils.weighted_edit_distance import weighted_edit_distance

def norm(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor
def build_model(config, converter):
    if config.MODEL.TYPE == "v1":
        return SRSTS_v1(config,converter)
    pass
def build_dataset(config,converter):
    return MultiStageTextLoader(config,converter,False)

def build_str_converter(config):
    if config.TEST.NAME == "ctw":
        ctw_alphabet = '0123456789abcdefghijklmnopqrstuvwxyz !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        return StrLabelConverter(ctw_alphabet, config.IGNORE_CASE,config.MAX_LENGTH)
    return StrLabelConverter(config.ALPHABET, config.IGNORE_CASE,config.MAX_LENGTH)

def clip_boxes(boxes, h, w):
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], w - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], h - 1), 0)
    # x2 < w
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], w - 1), 0)
    # y2 < h
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], h - 1), 0)
    return boxes


def gen_boxes_from_score_pred(seg_map, box, config):
    height, width = seg_map.shape

    text = seg_map > 0.5
    bwtext, nb_regs = bwlabel(text, return_num=True)  # bwlabel(text, return_num=True)

    conts, _ = cv2.findContours(bwtext.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    conts = list(conts)
    if len(conts) > 1:
        conts.sort(key=lambda x: cv2.contourArea(x), reverse=True)
    elif not conts:
        return []
    mask = np.zeros_like(seg_map, dtype=np.uint8)
    cv2.fillPoly(mask, conts[0].reshape(1, -1, 2), 1)
    proba = cv2.mean(seg_map, mask)[0]
    cont = np.array(conts[0][:, 0, :]).astype(np.float32)
    cont[:, 0] = np.clip(cont[:, 0], 0, width - 1) + box[0]
    cont[:, 1] = np.clip(cont[:, 1], 0, height - 1) + box[1]

    if (proba >= config.TEST.TH_PROB):
        return [cont]
    else:
        return []

def gen_boxes_from_score_pred_ic15(seg_map, box, config):
    height, width = seg_map.shape

    text = seg_map > 0.5

    bwtext, nb_regs = bwlabel(text, return_num=True)
    regions = regionprops(bwtext)

    res = None
    max_area = 0
    for region in regions:
        rect = cv2.minAreaRect(region.coords[:, ::-1])
        raw_bbox = cv2.boxPoints(rect).astype(np.int32)
        bw, bh = rect[1]
        area = bw * bh
        d = 0#np.ceil(min(bw, bh) * (1. / (1 - args.border_prec * 2) - 1) * 0.5)
        bw = bw + 2 * d
        bh = bh + 2 * d
        bbox = cv2.boxPoints((rect[0], (bw, bh), rect[2]))
        bbox[:, 0] = np.clip(bbox[:, 0], 0, width - 1) + box[0]
        bbox[:, 1] = np.clip(bbox[:, 1], 0, height - 1) + box[1]

        # estimate text proba
        mask = np.zeros_like(seg_map, dtype=np.uint8)

        # cv2.fillPoly(mask, raw_bbox.reshape((-1, 4, 2)), 1)
        cv2.fillPoly(mask, region.coords[:, ::-1].reshape((1, -1, 2)), 1)
        proba = cv2.mean(seg_map, mask)[0]

        if (proba >= config.TEST.TH_PROB) and (min(bw, bh) >= 8) and area > max_area:
            res = bbox
            max_area = area
    if res is not None:
        return [res]
    else:
        return []

def packing(save_dir, pack_dir, pack_name):
    files = os.listdir(save_dir)
    if not os.path.exists(pack_dir):
        os.mkdir(pack_dir)
    os.system('zip -r -j -q ' + os.path.join(pack_dir, pack_name + '.zip') + ' ' + save_dir + '/*')

def get_result_faster(score_preds, loc_preds, weight_preds, mask_preds, sampler_preds, rec_preds, img, vis_save_name, config,
               converter, save_name,lexicon_type, vis):
    """
    推理速度快一些，因为把相同stride的不同text instance的grid sample操作放到一块进行了。
    """
    fpn_strides = config.MODEL.FPN_STRIDES
    _, _, H, W = img.size()

    boxes_list = []
    weight_list = []
    index_list = []
    xy_list = []
    for lindex in range(len(fpn_strides)):
        seg_pred = score_preds[lindex].data.cpu().numpy()[0][0]
        loc_pred = loc_preds[lindex].data.cpu().numpy()[0]
        weight_pred = weight_preds[lindex].data.cpu().numpy()[0]
        xy_text = np.argwhere(seg_pred > config.TEST.SCORE_THRESH)
        ori_x = xy_text[:, 1] * fpn_strides[lindex] + fpn_strides[lindex] // 2
        ori_y = xy_text[:, 0] * fpn_strides[lindex] + fpn_strides[lindex] // 2
        geo = loc_pred[:, xy_text[:, 0], xy_text[:, 1]]
        score = seg_pred[xy_text[:, 0], xy_text[:, 1]]
        weight = weight_pred[:, xy_text[:, 0], xy_text[:, 1]]

        x_min = (ori_x - geo[0, :]).reshape((-1, 1))  # * fpn_strides[lindex]
        y_min = (ori_y - geo[1, :]).reshape((-1, 1))
        x_max = (ori_x + geo[2, :]).reshape((-1, 1))
        y_max = (ori_y + geo[3, :]).reshape((-1, 1))
        boxes = np.hstack((x_min, y_min, x_max, y_max, score.reshape((-1, 1))))
        boxes_list.append(boxes)
        weight_list.append(weight)
        temp_index_list = [lindex for i in range(boxes.shape[0])]
        index_list.extend(temp_index_list)
        xy_list.append(xy_text)

    boxes = np.concatenate(boxes_list, axis=0)
    weights = np.concatenate(weight_list, axis=1)
    index_list = np.array(index_list)
    xy_list = np.concatenate(xy_list, axis=0)

    keep = cpu_nms(boxes.astype(np.float32), config.TEST.NMS_THRESH)
    keep_boxes = boxes[keep, :]
    weights = weights[:, keep]

    index_list = index_list[keep]
    xy_list = xy_list[keep, :]
    keep_boxes = clip_boxes(keep_boxes, H, W)



    boxes_l = []
    boxes_h = []
    center_l = []
    center_h = []
    for tindex, box in enumerate(keep_boxes):
        box_temp = box.astype(np.int16)  #
        w = torch.from_numpy(weights[:, tindex]).view((1, -1)).cuda()  #
        m = mask_preds[index_list[tindex]][0, :, box_temp[1]:box_temp[3], box_temp[0]:box_temp[2]]  #
        mask_p = F.sigmoid(torch.mm(w, m.contiguous().view(config.MODEL.WEIGHTS_NUM, -1)))
        mask_p = mask_p.view(int(box_temp[3] - box_temp[1]), int(box_temp[2] - box_temp[0])).data.cpu().numpy()

        poly = gen_boxes_from_score_pred(mask_p, box, config)  #
        if len(poly) == 0:
            continue
        
        poly_mask = np.zeros((int(H / fpn_strides[index_list[tindex]]), int(W / fpn_strides[index_list[tindex]])),
                             np.uint8)
        cv2.fillPoly(poly_mask, (poly[0] / fpn_strides[index_list[tindex]]).astype(np.int32)[np.newaxis, :, :], 1)
        score_mask = score_preds[index_list[tindex]].data.cpu().numpy()[0][0] > config.TEST.SCORE_THRESH
        in_poly, _ = cv2.findContours((score_mask * poly_mask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(in_poly) == 0:
            continue
        if index_list[tindex] == 0:
            boxes_l.append(poly)
        if index_list[tindex] == 1:
            boxes_h.append(poly)
        rect = cv2.minAreaRect(in_poly[0].astype(np.int32))

        in_cur_poly = np.array([[rect[0][1], rect[0][0]]], np.int32)
        center_points = torch.from_numpy(in_cur_poly[:, (1, 0)]).cuda()

        center_points = center_points.unsqueeze(dim=0)
        if index_list[tindex] == 0:
            center_l.append(center_points.float().cpu().numpy())
        if index_list[tindex] == 1:
            center_h.append(center_points.float().cpu().numpy())
    
    polys = []
    strs = []
    if len(center_l)==0 and len(center_h)==0:
        return None, None
    if len(center_l)>0:
        center_points_l = np.array(center_l)
    
        norm_center_points = torch.from_numpy(center_points_l).cuda().permute(2,1,0,3).squeeze(dim=1)
    
        norm_center_points[:, :, 0] = (norm_center_points[:, :, 0] / (W / 4 - 1) - 0.5) / 0.5
        norm_center_points[:, :, 1] = (norm_center_points[:, :, 1] / (H / 4 - 1) - 0.5) / 0.5
        center_sampled_feature = F.grid_sample(sampler_preds[0],norm_center_points.unsqueeze(1).float(),mode="nearest").squeeze(2).permute(0, 2, 1)
        N, nums_text, xy_2 = center_sampled_feature.size()
        temp = center_sampled_feature.view(N, nums_text, int(xy_2 / 2), 2)
        center_sampled_feature_ = torch.zeros_like(temp)
        center_sampled_feature_[:, :, :, 0] = temp[:, :, :, 0] / (W / 4)
        center_sampled_feature_[:, :, :, 1] = temp[:, :, :, 1] / (H / 4)
        center_sampled_points = norm_center_points.unsqueeze(2).expand_as(center_sampled_feature_) + center_sampled_feature_
        sampled_feature_seq = F.grid_sample(rec_preds[0], center_sampled_points)  ## N*C*256*50
        rec_pred = sampled_feature_seq[0]  # net.rec_tail(sampled_feature_seq)

        #print(rec_pred.shape)
        logits = F.softmax(rec_pred.permute(1, 2, 0), dim=2)
        for i in range(rec_pred.size()[1]):
            logit = logits[i]
        
            indexes = logit.argmax(dim=-1)
            pred_str, temp_score, scores = converter.decode(indexes, torch.IntTensor([25]), logit)
            #print(pred_str)
            if temp_score > config.TEST.REC_THRESH:# and box[-1]>args.th_det_score:  # and len(pred_str)>2:
                if lexicon_type == 1:  # generic
                    generic_txt = "/data/wujingjing/share_mask_41/tt_word_spotting/full_lexicons.txt"#"/data/wujingjing/data/icdar2015/generic.txt"
                    pred_str = search_lexicon(generic_txt, pred_str, temp_score, 1, scores, weighted_eidt=True)
                polys.append(boxes_l[i][0])
                strs.append(pred_str)
           
    if len(center_h)>0:  
        center_points_h = np.array(center_h)
    
        norm_center_points = torch.from_numpy(center_points_h).cuda().permute(2,1,0,3).squeeze(dim=1)
   
        norm_center_points[:, :, 0] = (norm_center_points[:, :, 0] / (W / 8 - 1) - 0.5) / 0.5
        norm_center_points[:, :, 1] = (norm_center_points[:, :, 1] / (H / 8 - 1) - 0.5) / 0.5
        center_sampled_feature = F.grid_sample(sampler_preds[1],norm_center_points.unsqueeze(1).float(),mode="nearest").squeeze(2).permute(0, 2, 1)
        N, nums_text, xy_2 = center_sampled_feature.size()
        temp = center_sampled_feature.view(N, nums_text, int(xy_2 / 2), 2)
        center_sampled_feature_ = torch.zeros_like(temp)
        center_sampled_feature_[:, :, :, 0] = temp[:, :, :, 0] / (W / 8)
        center_sampled_feature_[:, :, :, 1] = temp[:, :, :, 1] / (H / 8)
        center_sampled_points = norm_center_points.unsqueeze(2).expand_as(center_sampled_feature_) + center_sampled_feature_
        sampled_feature_seq = F.grid_sample(rec_preds[1], center_sampled_points)  ## N*C*256*50
        rec_pred = sampled_feature_seq[0]  # net.rec_tail(sampled_feature_seq)
  
        logits = F.softmax(rec_pred.permute(1, 2, 0), dim=2)
        for i in range(rec_pred.size()[1]):
            logit = logits[i]#F.softmax(rec_pred.permute(1, 2, 0), dim=2)[i]

            indexes = logit.argmax(dim=-1)
            pred_str, temp_score, scores = converter.decode(indexes, torch.IntTensor([25]), logit)
      
            if temp_score > config.TEST.REC_THRESH:# and box[-1]>args.th_det_score:  # and len(pred_str)>2:
                if lexicon_type == 1:  # generic
                    generic_txt = "/data/wujingjing/share_mask_41/tt_word_spotting/full_lexicons.txt"#"/data/wujingjing/data/icdar2015/generic.txt"
                    pred_str = search_lexicon(generic_txt, pred_str, temp_score, 1, scores, weighted_eidt=True)
                polys.append(boxes_h[i][0])
                strs.append(pred_str)

    if vis:
        im = (norm(img[0].data.cpu()).numpy() * 255).astype(np.uint8).transpose((1, 2, 0))
        img = Image.fromarray(im).convert('RGB')
        img1 = Image.fromarray(im).convert('RGB')
        img2 = Image.fromarray(im).convert('RGB')
        w, h = img.size
        seg_pred_4 = score_preds[0].data.cpu().numpy()[0][0]
        mask = Image.fromarray((seg_pred_4 * 255).astype(np.uint8), 'L').convert('RGB').resize((w, h))
        img_mask_4 = Image.blend(img, mask, 0.5)

        seg_pred_8 = score_preds[1].data.cpu().numpy()[0][0]
        mask = Image.fromarray((seg_pred_8 * 255).astype(np.uint8), 'L').convert('RGB').resize((w, h))
        img_mask_8 = Image.blend(img, mask, 0.5)

        seg_pred_16 = score_preds[2].data.cpu().numpy()[0][0]
        mask = Image.fromarray((seg_pred_16 * 255).astype(np.uint8), 'L').convert('RGB').resize((w, h))
        img_mask_16 = Image.blend(img, mask, 0.5)
        img_draw = ImageDraw.Draw(img)
        for box in boxes:
            box = list(box.reshape(-1))[:-1]
            img_draw.rectangle(box, outline=(255, 0, 0))

        img_draw1 = ImageDraw.Draw(img1)
        for box in keep_boxes:
            box = list(box.reshape(-1))[:-1]
            img_draw1.rectangle(box, outline=(255, 0, 0))

        img_draw2 = ImageDraw.Draw(img2)
        for box in polys:
            box = list(box.reshape(-1))
            img_draw2.line(box + box[:2], fill=(0, 255, 0), width=3)

        new_img = Image.new('RGB', (w * 6, h), (0, 0, 0))
        new_img.paste(img_mask_4, (0, 0))
        new_img.paste(img_mask_8, (w, 0))
        new_img.paste(img_mask_16, (w * 2, 0))
        new_img.paste(img, (w * 3, 0))
        new_img.paste(img1, (w * 4, 0))
        new_img.paste(img2, (w * 5, 0))
        new_img.save(vis_save_name)

    if len(polys) > 0:
        return np.array(polys), strs
    else:
        return None, None
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False
def search_lexicon(lexicon_path, word, score, score_thre=1, scores_numpy=None, weighted_eidt=False,test_name="ic15"):

    if score == 0 or is_number(word):
        return word
    lexicon_fid = open(lexicon_path, 'r')
    lexicon = []
    for line in lexicon_fid.readlines():
        line = line.strip()
        if test_name!="ctw":
            line = re.sub('[^0-9a-zA-Z]+', '', line)
        lexicon.append(line)

    word = word.lower()
    dist_min = 100

    dist_min_pre = 100
    match_dist = 100
    match_word = word
    if weighted_eidt is False:
        for voc_word in lexicon:
            voc_word = voc_word.lower()
            ed = editdistance.eval(word, voc_word)
            dist = ed
            if dist < dist_min:
                dist_min = dist
                match_word = voc_word
                match_dist = dist
            if match_dist == 0:
                break
    else:
        small_lexicon_dict = dict()
        for voc_word in lexicon:
            voc_word = voc_word.lower()
            ed = editdistance.eval(word, voc_word)

            small_lexicon_dict[voc_word] = ed
            dist = ed
            if dist < dist_min_pre:
                dist_min_pre = dist
        small_lexicon = []
        for voc_word in small_lexicon_dict:
            if small_lexicon_dict[voc_word] <= dist_min_pre + 2:
                small_lexicon.append(voc_word)

        for voc_word in small_lexicon:
            voc_word = voc_word.lower()
            ed = weighted_edit_distance(word, voc_word, scores_numpy[:len(word), :])
            dist = ed
            if dist < dist_min:
                dist_min = dist
                match_word = voc_word  # pairs[word]
                match_dist = dist

    return match_word

def search_lexicon_new(lexicon, word, score, score_thre=1, scores_numpy=None, weighted_eidt=False):
    lexicon_path, pair_path = lexicon[0], lexicon[1]
    if score == 0 or is_number(word):# or score > score_thre:
        return word
    lexicon_fid = open(lexicon_path, 'r')
    lexicon = []
    pair_list = open(pair_path, 'r')
    pairs = dict()
    for line in pair_list.readlines():
        line = line.strip()
        word_ = line.split(' ')[0].lower()
        word_gt = line[len(word_) + 1:]
        pairs[word_] = word_gt
        lexicon.append(line)
    lexicon = []
    for line in lexicon_fid.readlines():
        line = line.strip()
        lexicon.append(line.lower())
    word = word.lower()
    dist_min = 100
    dist_min_pre = 100
    match_dist = 100
    match_word = word
    if weighted_eidt is False:
        for voc_word in lexicon:
            voc_word = voc_word.lower()
            ed = editdistance.eval(word, voc_word)
            dist = ed
            if dist < dist_min:
                dist_min = dist
                match_word = pairs[word]  # voc_word
                match_dist = dist

            if match_dist == 0:
                break
    else:
        small_lexicon_dict = dict()
        for voc_word in lexicon:
            voc_word = voc_word.lower()
            ed = editdistance.eval(word, voc_word)

            small_lexicon_dict[voc_word] = ed
            dist = ed
            if dist < dist_min_pre:
                dist_min_pre = dist
        small_lexicon = []
        for voc_word in small_lexicon_dict:
            if small_lexicon_dict[voc_word] <= dist_min_pre + 2:
                small_lexicon.append(voc_word)

        for voc_word in small_lexicon:
            voc_word = voc_word.lower()
            ed = weighted_edit_distance(word, voc_word, scores_numpy[:len(word), :])
            dist = ed
            if dist < dist_min:
                dist_min = dist
                match_word = pairs[voc_word]
                match_dist = dist


    return match_word

def get_lexicon(config, lexicon_type,lexicon_is_official=True):
    if config.TEST.NAME == "tt" and lexicon_type==1:
        return "evaluation/lexicons/tt/full_lexicons.txt"
    elif config.TEST.NAME == "ctw" and lexicon_type == 1:
        return "evaluation/lexicons/ctw/weak_voc_new.txt"
    elif config.TEST.NAME == "ic15":
        if lexicon_is_official == True:
            if lexicon_type == 1:
                return "evaluation/lexicons/ic15/generic.txt"
            elif lexicon_type == 2:
                return "evaluation/lexicons/ic15/weak.txt"
            elif lexicon_type == 3:
                return "evaluation/lexicons/ic15/strong"
        else:
            if lexicon_type == 1:
                generic_txt = "evaluation/lexicons/ic15/generic_new.txt"
                pair_path = "evaluation/lexicons/ic15/generic_new_pair_list.txt"
                
            elif lexicon_type == 2:
                generic_txt = "evaluation/lexicons/ic15/weak_new.txt"
                pair_path = "evaluation/lexicons/ic15/weak_new_pair_list.txt"
            elif lexicon_type == 3:
                generic_txt = "evaluation/lexicons/ic15/strong_new"   
                pair_path = "evaluation/lexicons/ic15/strong_new"   
            return generic_txt, pair_path 
    


def get_result(score_preds, loc_preds, weight_preds, mask_preds, sampler_preds, rec_preds, img, vis_save_name, config,
               converter, save_name,lexicon_type,vis):
    #(score_preds, loc_preds, weight_preds, mask_preds, sampler_preds, rec_preds, img, vis_save_name, config, converter, save_name, vis)

    fpn_strides = config.MODEL.FPN_STRIDES
    _, _, H, W = img.size()

    boxes_list = []
    weight_list = []
    index_list = []
    xy_list = []
    for lindex in range(len(fpn_strides)):
        seg_pred = score_preds[lindex].data.cpu().numpy()[0][0]
        loc_pred = loc_preds[lindex].data.cpu().numpy()[0]
        weight_pred = weight_preds[lindex].data.cpu().numpy()[0]
        xy_text = np.argwhere(seg_pred > config.TEST.SCORE_THRESH)
        ori_x = xy_text[:, 1] * fpn_strides[lindex] + fpn_strides[lindex] // 2
        ori_y = xy_text[:, 0] * fpn_strides[lindex] + fpn_strides[lindex] // 2
        geo = loc_pred[:, xy_text[:, 0], xy_text[:, 1]]
        score = seg_pred[xy_text[:, 0], xy_text[:, 1]]
        weight = weight_pred[:, xy_text[:, 0], xy_text[:, 1]]

        x_min = (ori_x - geo[0, :]).reshape((-1, 1))  # * fpn_strides[lindex]
        y_min = (ori_y - geo[1, :]).reshape((-1, 1))
        x_max = (ori_x + geo[2, :]).reshape((-1, 1))
        y_max = (ori_y + geo[3, :]).reshape((-1, 1))
        boxes = np.hstack((x_min, y_min, x_max, y_max, score.reshape((-1, 1))))
        boxes_list.append(boxes)
        weight_list.append(weight)
        temp_index_list = [lindex for i in range(boxes.shape[0])]
        index_list.extend(temp_index_list)
        xy_list.append(xy_text)

    boxes = np.concatenate(boxes_list, axis=0)
    weights = np.concatenate(weight_list, axis=1)
    index_list = np.array(index_list)
    xy_list = np.concatenate(xy_list, axis=0)

    keep = cpu_nms(boxes.astype(np.float32), config.TEST.NMS_THRESH)
    keep_boxes = boxes[keep, :]
    weights = weights[:, keep]

    index_list = index_list[keep]
    xy_list = xy_list[keep, :]
    keep_boxes = clip_boxes(keep_boxes, H, W)

    polys = []
    strs = []


    for tindex, box in enumerate(keep_boxes):
        box_temp = box.astype(np.int16)  #
        w = torch.from_numpy(weights[:, tindex]).view((1, -1)).cuda()  #
        m = mask_preds[index_list[tindex]][0, :, box_temp[1]:box_temp[3], box_temp[0]:box_temp[2]]  #
        mask_p = F.sigmoid(torch.mm(w, m.contiguous().view(config.MODEL.WEIGHTS_NUM, -1)))
        mask_p = mask_p.view(int(box_temp[3] - box_temp[1]), int(box_temp[2] - box_temp[0])).data.cpu().numpy()

        if config.TEST.NAME == "ic15":
            poly = gen_boxes_from_score_pred_ic15(mask_p, box, config)
            if len(poly) == 0:
                continue
            rect = cv2.minAreaRect(poly[0].astype(np.int32))
 
            in_cur_poly = np.array([[rect[0][1] / fpn_strides[index_list[tindex]], rect[0][0] / fpn_strides[index_list[tindex]]]], np.int32)
        else:
            poly = gen_boxes_from_score_pred(mask_p, box, config)
            if len(poly) == 0:
                continue
            poly_mask = np.zeros((int(H / fpn_strides[index_list[tindex]]), int(W / fpn_strides[index_list[tindex]])),np.uint8)
            cv2.fillPoly(poly_mask, (poly[0] / fpn_strides[index_list[tindex]]).astype(np.int32)[np.newaxis, :, :], 1)
            score_mask = score_preds[index_list[tindex]].data.cpu().numpy()[0][0] > config.TEST.SCORE_THRESH
            in_poly, _ = cv2.findContours((score_mask * poly_mask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(in_poly) == 0:
                continue
            rect = cv2.minAreaRect(in_poly[0].astype(np.int32))
            in_cur_poly = np.array([[rect[0][1], rect[0][0]]], np.int32)
        
        center_points = torch.from_numpy(in_cur_poly[:, (1, 0)]).cuda()
        center_points = center_points.unsqueeze(dim=0)
        norm_center_points = center_points.float()


        norm_center_points[:, :, 0] = (norm_center_points[:, :, 0] / (
                W / fpn_strides[index_list[tindex]] - 1) - 0.5) / 0.5
        norm_center_points[:, :, 1] = (norm_center_points[:, :, 1] / (
                H / fpn_strides[index_list[tindex]] - 1) - 0.5) / 0.5
        center_sampled_feature = F.grid_sample(sampler_preds[index_list[tindex]],
                                               norm_center_points.unsqueeze(1).float(),
                                               mode=config.TEST.GRID_MODE).squeeze(2).permute(0, 2, 1)
        N, nums_text, xy_2 = center_sampled_feature.size()
        temp = center_sampled_feature.view(N, nums_text, int(xy_2 / 2), 2)
        center_sampled_feature_ = torch.zeros_like(temp)
        center_sampled_feature_[:, :, :, 0] = temp[:, :, :, 0] / (W / fpn_strides[index_list[tindex]])
        center_sampled_feature_[:, :, :, 1] = temp[:, :, :, 1] / (H / fpn_strides[index_list[tindex]])
        center_sampled_points = norm_center_points.unsqueeze(2).expand_as(
            center_sampled_feature_) + center_sampled_feature_
        sampled_feature_seq = F.grid_sample(rec_preds[index_list[tindex]], center_sampled_points)  ## N*C*256*50
        rec_pred = sampled_feature_seq[0]

        logits = F.softmax(rec_pred.permute(1, 2, 0), dim=2)[0]

        indexes = logits.argmax(dim=-1)
        pred_str, temp_score, scores = converter.decode(indexes, torch.IntTensor([config.MAX_LENGTH]), logits)
        det_rec_score = 2 * (np.exp(temp_score + box[-1])) / (np.exp(temp_score) + np.exp(box[-1]))
        # if temp_score > 0.70:#args.th_rec_score:
        #if det_rec_score > args.th_rec_score:  # and len(pred_str)>2:
        #get_lexicon(config, lexicon_type,lexicon_is_official=True)
        
        if temp_score  > config.TEST.REC_THRESH:# and box[-1]>args.th_det_score:  # and len(pred_str)>2:
            if len(pred_str) < config.TEST.MIN_LEN :  # or partten.findall(pred_str):  
                continue
            if lexicon_type != 0:
                lexicon = get_lexicon(config, lexicon_type,config.TEST.OFFICIAL_LEXICON)
            if config.TEST.OFFICIAL_LEXICON:
                if lexicon_type == 1 or lexicon_type == 2:  # generic or full or weak
                    pred_str = search_lexicon(lexicon, pred_str, temp_score, 1, scores, weighted_eidt=True,test_name=config.TEST.NAME)
                elif lexicon_type == 3: # strong
                    strong_txt = "{}/voc_{}.txt".format(lexicon,save_name)
                    pred_str = search_lexicon(strong_txt, pred_str, temp_score, 1, scores, weighted_eidt=True)
            else:
                if lexicon_type == 1 or lexicon_type == 2:  # generic or full or weak
                    pred_str = search_lexicon_new(lexicon, pred_str, temp_score, 1, scores, weighted_eidt=True)
                elif lexicon_type == 3: # strong
                    lexicon_txt = "{}/new_voc_{}.txt".format(lexicon[0],save_name)
                    pair_txt = "{}/pair_voc_{}.txt".format(lexicon[1],save_name)
                    pred_str = search_lexicon_new([lexicon_txt, pair_txt], pred_str, temp_score, 1, scores, weighted_eidt=True)
            strs.append(pred_str)
            polys += poly



    if len(polys) > 0:
        return np.array(polys), strs
    else:
        return None, None


def polygon_area(poly):
    edge = [
        (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
        (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
        (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
        (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
    ]
    return np.sum(edge) / 2.

def sort_poly(points):
    points = list(points)
    ps = sorted(points, key=lambda x: x[0])

    if ps[1][1] > ps[0][1]:
        px1 = ps[0][0]
        py1 = ps[0][1]
        px4 = ps[1][0]
        py4 = ps[1][1]
    else:
        px1 = ps[1][0]
        py1 = ps[1][1]
        px4 = ps[0][0]
        py4 = ps[0][1]
    if ps[3][1] > ps[2][1]:
        px2 = ps[2][0]
        py2 = ps[2][1]
        px3 = ps[3][0]
        py3 = ps[3][1]
    else:
        px2 = ps[3][0]
        py2 = ps[3][1]
        px3 = ps[2][0]
        py3 = ps[2][1]

    poly = np.array([[px1, py1], [px2, py2], [px3, py3], [px4, py4]])
    if polygon_area(poly) < 0:
        return poly
    else:
        return poly[(0, 3, 2, 1), :]



def write_to_file_det(contours, strs, file_path):
    """
    :param contours: [[x1, y1], [x2, y2]... [xn, yn]]
    :param file_path: target file path
    """
    # according to total-text evaluation method, output file shoud be formatted to: y0,x0, ..... yn,xn
    with open(file_path, 'w') as f:
        i = 0
        for cont in contours:
            cont = np.stack([cont[:, 1], cont[:, 0]], 1)
            cont = cont.flatten().astype(str).tolist()
            cont = ','.join(cont)
            f.write(cont + '\n')
            i += 1


def write_to_file(contours, strs, file_path):
    """
    :param contours: [[x1, y1], [x2, y2]... [xn, yn]]
    :param file_path: target file path
    """
    # according to total-text evaluation method, output file shoud be formatted to: y0,x0, ..... yn,xn
    with open(file_path, 'w') as f:
        i = 0
        for cont in contours:
            cont = np.stack([cont[:, 0], cont[:, 1]], 1)
            cont = cont.flatten().astype(str).tolist()
            cont = ','.join(cont)
            f.write(cont + ',' + strs[i] + '\n')
            i += 1
def write_to_file_ctw(contours, strs, file_path):
    """
    :param contours: [[x1, y1], [x2, y2]... [xn, yn]]
    :param file_path: target file path
    """
    # according to total-text evaluation method, output file shoud be formatted to: y0,x0, ..... yn,xn
    with open(file_path, 'w') as f:
        i = 0
        for cont in contours:
            cont = np.stack([cont[:, 0], cont[:, 1]], 1)
            cont = cont.flatten().astype(str).tolist()
            cont = ','.join(cont)
            # f.write(cont +  + strs[i] + '\n')

            if i < len(contours) - 1:
                f.write(cont + ',####' + strs[i] + '\n')
            else:
                f.write(cont + ',####' + strs[i])
            i += 1


def inference(config, args, save_dir, lexicon_type):
    # build model
    #config.ALPHABET = 
    print(config.ALPHABET)
    converter = build_str_converter(config)
    model = build_model(config, converter).cuda()
    # build dataset
    dataset = build_dataset(config, converter)
    # mkdir and prepare


    save_dir = save_dir + "/lexicon_{}".format(lexicon_type)
    if config.TEST.OFFICIAL_LEXICON == False:
        save_dir = save_dir + "_new"
    res_path = save_dir + "/res/"
    vis_path = save_dir + "/vis"
    if os.path.exists(res_path) == False:
        os.makedirs(res_path)
    if config.TEST.NAME == "tt" and os.path.exists(save_dir + "/res_det") == False:
        os.makedirs(save_dir + "/res_det")
    if args.vis == True:
        if os.path.exists(vis_path) == False:
            os.makedirs(vis_path)

    data_loader = data.DataLoader(dataset, 1, num_workers=1, shuffle=False, pin_memory=True)
    logging.info('dataset initialize done.')
    net = torch.nn.DataParallel(model).cuda()
    net.load_state_dict(torch.load(args.ckpt))
    net = net.module
    net.eval()

    # cudnn.benchmark = True
    logging.info('begin')
    for i, sample in enumerate(data_loader, 0):    
        img, image_name, ori_h, ori_w, test_height, test_width = sample
        ratio_h = float(test_height.data.numpy()[0]) / float(ori_h.data.numpy()[0])
        ratio_w = float(test_width.data.numpy()[0]) / float(ori_w.data.numpy()[0])
        save_name = image_name[0].split('/')[-1].split('.')[0]
   
        if i % 100 == 0:
            print(i, len(data_loader))
        h, w = img.size(2), img.size(3)
        img = img.cuda()
 
        img = Variable(img)
        
        score_pred, loc_pred, weight_pred, mask_pred, proj_fuse, sampler_pred = net(img)


        vis_save_name = os.path.join(vis_path, save_name + '.jpg')
        boxes, strs = get_result(score_pred, loc_pred, weight_pred, mask_pred,sampler_pred, proj_fuse, img, vis_save_name, config,
                                                                    converter, save_name, lexicon_type,args.vis)

        if boxes is None:
            continue
        
        if config.TEST.NAME == "tt":
            new_boxes = []
            if boxes is not None:
                for ib in range(len(boxes)):    
                    boxes[ib][:, 0] /= ratio_w
                    boxes[ib][:, 1] /= ratio_h
                    new_boxes.append(boxes[ib].astype(np.int32))
            res_name_det =  save_dir + "/res_det/" + "res_"+save_name + '.txt'
            res_name_e2e = res_path + '/' + 'res_' + save_name + '.txt'
            if boxes is not None:
                write_to_file_det(new_boxes, strs, res_name_det)
                write_to_file(new_boxes, strs, res_name_e2e)
            
        
        elif config.TEST.NAME == "ic15":
            
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h
            res_name = res_path + '/' + 'res_' + save_name + '.txt'
            if boxes is not None:
                fp = open(res_name, 'w')
                if boxes is not None:
                    k = -1
                    for box in boxes:
                        box = sort_poly(box.astype(np.int32))
                        k += 1
                        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                            continue
                        fp.write(
                    '{},{},{},{},{},{},{},{},{}\r\n'.format(box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0],
                                                            box[2, 1], box[3, 0], box[3, 1], strs[k]))
                fp.close()
        elif config.TEST.NAME == "ctw":
            new_boxes = []
            if boxes is not None:
                for ib in range(len(boxes)):
                    boxes[ib][:, 0] /= ratio_w
                    boxes[ib][:, 1] /= ratio_h
                    # boxes[ib] = boxes[ib].astype(np.int32)
                    # print(boxes[ib].astype(np.int32))
                    new_boxes.append(boxes[ib].astype(np.int32))

            res_name_e2e = res_path + '/' +'res_000' + save_name + '.txt'
            if boxes is not None:
                write_to_file_ctw(new_boxes, strs, res_name_e2e)

    if config.TEST.NAME == "tt":
        packing(res_path, save_dir, 'submit')
        return save_dir+"/submit.zip", save_dir + "/res_det/"
    elif config.TEST.NAME == "ic15":
        packing(res_path, save_dir, 'submit')
        return save_dir+"/submit.zip", None
    elif config.TEST.NAME == "ctw":
        packing(res_path, save_dir, 'submit')
        return save_dir+"/submit.zip", None