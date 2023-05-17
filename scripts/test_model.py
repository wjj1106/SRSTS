"""
根据args选择对应的yaml进行测评
usage: python scripts/test_model.py --config-file="" --ckpt=""
tt : python scripts/test_model.py --config-file="./configs/evaluation/v1/tt.yaml" --ckpt="/data/wujingjing/save_models/tt_best.pth"
ic15 : python scripts/test_model.py --config-file="./configs/evaluation/v1/ic15.yaml" --ckpt="/data/wujingjing/save_models/ori_finetune_90_0.pth"
ctw : python scripts/test_model.py --config-file="./configs/evaluation/v1/ctw.yaml" --ckpt="/data/wujingjing/save_models/ctw_finetune_v1_0_800.pth"

"""
import argparse
import os
import sys
import time
sys.path.append("/data/wujingjing/2023_05/srsts") # path to current project
from configs import config as cfg
from utils.logger import setup_logger,print_args
from utils.inference import inference
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

def main():
    parser = argparse.ArgumentParser(description="Inferece Procedure of SRSTS")
    parser.add_argument("--config-file",default="./configs/evaluation/v1/ic15.yaml", help="path to config")
    parser.add_argument("--ckpt", default="/data/wujingjing/save_models/ori_finetune_90_0.pth",help="path to model weight")
    parser.add_argument("--out", default="output/",help="path to model weight")
    parser.add_argument("--vis", action="store_true",help="whether to visualize the results")
    parser.add_argument("--lexicon_type", default=0,type=int,help="whether to visualize the results")
    args = parser.parse_args()
    config = cfg
    config.merge_from_file(args.config_file)
    #config.ALPHABET = '0123456789abcdefghijklmnopqrstuvwxyz !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    save_dir = args.out + '/' + config.TEST.NAME
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    log_file_path = save_dir + '/eval_' + time.strftime('%Y%m%d_%H%M%S') + '.log'
    setup_logger(log_file_path)
    print_args(args)
    # inference
    res_path,res_path_det = inference(config, args,save_dir,args.lexicon_type) 
    # evaluation
    if config.TEST.NAME == "tt":
        os.system('/home/ubuntu/anaconda3/envs/wjj_ocr/bin/python evaluation/protocols/tt/Python_scripts/Deteval.py --exp_name={} --res_dir={}'.format(args.out, res_path_det) + '|tee -a ' + log_file_path)
        os.system('/home/ubuntu/anaconda3/envs/wjj_ocr/bin/python evaluation/protocols/tt/tt_word_spotting/script.py -g=evaluation/gts/total-text-gt.zip -s=' + res_path + '|tee -a ' + log_file_path)
    elif config.TEST.NAME == "ic15":
        os.system('/home/ubuntu/anaconda3/envs/wjj_ocr/bin/python evaluation/protocols/ic15/script.py -g=evaluation/gts/icdar-2015-gt.zip -s=' +  res_path + '|tee -a ' + log_file_path)
        os.system('/home/ubuntu/anaconda3/envs/wjj_ocr/bin/python evaluation/protocols/ic15/ic15_word_spotting/script.py -g=evaluation/gts/icdar-2015-gt.zip -s=' + res_path + '|tee -a ' + log_file_path)
        os.system('/home/ubuntu/anaconda3/envs/wjj_ocr/bin/python evaluation/protocols/ic15/ic15_end_to_end/script.py -g=evaluation/gts/icdar-2015-gt.zip -s=' + res_path + '|tee -a ' + log_file_path)
    elif config.TEST.NAME == "ctw":
        os.system('/home/ubuntu/anaconda3/envs/wjj_ocr/bin/python evaluation/protocols/ctw/ctw_end_to_end_abc/text_eval_script.py -g=evaluation/gts/ctw-1500-gt.zip -s=' + res_path + '|tee -a ' + log_file_path)


    
if __name__ == "__main__":
    main()