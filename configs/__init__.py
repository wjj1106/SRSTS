from yacs.config import CfgNode as CN
config = CN()
config.ALPHABET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
config.IGNORE_CASE = True
config.MAX_LENGTH = 25

## model
config.MODEL = CN()
config.MODEL.TYPE = "v1"
config.MODEL.BACKBONE = "resnet50"
config.MODEL.DEFORMALBLE_LAYERS = 6
config.MODEL.SELF_ATTENTION_LAYERS = 4
config.MODEL.FPN_STRIDES = [4,8]
config.MODEL.WEIGHTS_NUM = 4
config.MODEL.SAMPLE_POINT_NUMS = 25

## config in the test phrase
config.TEST = CN()
config.TEST.NAME = "tt"
config.TEST.ROOT = ""
config.TEST.FIX_MAX = False
config.TEST.MAX_SIZE = 1920
config.TEST.MIN_SIZE = 640
config.TEST.SCORE_THRESH = 0.85
config.TEST.REC_THRESH = 0.75
config.TEST.NMS_THRESH = 0.3
config.TEST.TH_PROB = 0.85
config.TEST.LEXICON = 0
config.TEST.GRID_MODE = "nearest"
config.TEST.MIN_LEN = 0
config.TEST.OFFICIAL_LEXICON = True


## config in the training phrase
config.TRAIN = CN()
config.TRAIN.BATCH_SIZE = 16
config.TRAIN.LEARNING_RATE = 1e-4
config.TRAIN.SIZE = 640


