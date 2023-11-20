from dataclasses import dataclass

from detectron2.config import get_cfg

from config import get_settings

from .ditod import add_vit_config
from .ditod.tokenization_bros import BrosTokenizer
from .predictor import DefaultPredictor


@dataclass
class Arguments:
    dataset = "doclaynet"
    config_file_path = "ai/Configs/doclaynet_VGT_cascade_PTM.yaml"
    model_weights_path = "./models/model_final.pth"


args = Arguments()
cfg = get_cfg()
add_vit_config(cfg)
cfg.merge_from_file(args.config_file_path)

settings = get_settings()
print(f"Using {settings.DEVICE} for Inference")
device = settings.DEVICE
cfg.MODEL.DEVICE = "cpu"  # device

print("Loading Tokenizer....")
tokenizer = BrosTokenizer.from_pretrained("naver-clova-ocr/bros-base-uncased")
print("Loading Model weights....")
predictor = DefaultPredictor(cfg, args.model_weights_path)
print(settings)
