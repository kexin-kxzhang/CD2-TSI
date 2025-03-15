import argparse
import torch
import time
import datetime
import json
import yaml
import os
from torch import nn
from dataset_process.dataset_air_quality_Beijing import get_dataloader_Beijing
from dataset_process.dataset_air_quality_Tianjin import get_dataloader_Tianjin
from dataset_process.dataset_discharge import get_dataloader_discharge
from dataset_process.dataset_pooled import get_dataloader_pooled
from dataset_process.dataset_electricity_etth1 import get_dataloader_etth1
from dataset_process.dataset_electricity_etth2 import get_dataloader_etth2
from models.main_model import CD2_TSI
from models.shared_components import SharedEmbedding, Conv1d_with_init
from util.utils import train_da, evaluate_da
import logging

########## args defination
parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument(
    "--train_missing_pattern", type=str, default="block", choices=["point", "block"]
)
parser.add_argument(
    "--eval_missing_pattern", type=str, default="block", choices=["point", "block"]
)
parser.add_argument(
    "--dataset", type=str, default="hydrology", choices=["air_quality", "hydrology", "etth"]
)
parser.add_argument("--nsample", type=int, default=100)

# DA and Fmixup
parser.add_argument("--miu_align", type=float, default=5)
parser.add_argument("--taul", type=float, default=0.2)
parser.add_argument("--tauh", type=float, default=0.7)
parser.add_argument("--freq_interpolation", type=int, default=1)
parser.add_argument("--mixup_lambda", type=float, default=0.2)
args = parser.parse_args()
print(args)


########## load config
path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)
print(json.dumps(config, indent=4))

config["model"]["freq_interpolation"] = args.freq_interpolation
########## model location
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 
foldername = (
    "./save/"
    + str(args.dataset)
    + "/CD2-TSI/"
    + f"train_missing_pattern_{args.train_missing_pattern}"
    + f"-eval_missing_pattern_{args.eval_missing_pattern}"
    + f"-miu_align_{args.miu_align}"
    + f"-taul_{args.taul}"
    + f"-tauh_{args.tauh}"
    + f"-freq_interpolation_{args.freq_interpolation}"
    + f"-mixup_lambda_{args.mixup_lambda}"
    + f"-{current_time}/"
)
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

if args.dataset == "air_quality":
    get_dataloader_src = get_dataloader_Beijing
    get_dataloader_tgt = get_dataloader_Tianjin
elif args.dataset == "hydrology":
    get_dataloader_src = get_dataloader_discharge
    get_dataloader_tgt = get_dataloader_pooled
elif args.dataset == "electricity":
    get_dataloader_src = get_dataloader_etth1
    get_dataloader_tgt = get_dataloader_etth2

########## src and tgt dataloader
src_train_loader, src_valid_loader, src_test_loader, src_scaler, src_mean_scaler = get_dataloader_src(
    device=args.device,
    train_missing_pattern=args.train_missing_pattern,
    eval_missing_pattern=args.eval_missing_pattern,
    seed=args.seed,
    batch_size=config["train"]["batch_size"],
)

tgt_train_loader, tgt_valid_loader, tgt_test_loader, tgt_scaler, tgt_mean_scaler = get_dataloader_tgt(
    device=args.device,
    train_missing_pattern=args.train_missing_pattern,
    eval_missing_pattern=args.eval_missing_pattern,
    seed=args.seed,
    batch_size=config["train"]["batch_size"],
)


########## src and tgt model
if args.dataset == "air_quality":
    target_dim = 27
elif args.dataset == "hydrology":
    target_dim = 20
elif args.dataset == "electricity":
    target_dim = 7


shared_layer = Conv1d_with_init(2, config["diffusion"]["channels"], 1)
shared_side_info = SharedEmbedding(config, args.device, target_dim).to(args.device)
src_model = CD2_TSI(config, args.device, shared_input_projection=shared_layer, shared_time_feature_embed=shared_side_info, target_dim=target_dim).to(args.device)
tgt_model = CD2_TSI(config, args.device, shared_input_projection=shared_layer, shared_time_feature_embed=shared_side_info, target_dim=target_dim).to(args.device)

########## training process
start_time = time.time()
print("Start time:", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

train_da(
    src_model,
    tgt_model,
    config["train"],
    src_train_loader,
    tgt_train_loader,
    tgt_valid_loader=tgt_valid_loader,
    src_valid_loader=src_valid_loader,
    foldername=foldername,
    freq_interpolation=args.freq_interpolation,
    mixup_lambda=args.mixup_lambda,
    miu_align=args.miu_align,
    taul=args.taul,
    tauh=args.tauh,
    device=args.device,
)

torch.save(tgt_model.state_dict(), foldername + "/tgt_model.pth")
logging.basicConfig(filename=foldername + '/tgt_model.log', level=logging.DEBUG)

########## evaluation process
evaluate_da(
    tgt_model,
    tgt_test_loader,
    src_test_loader,
    nsample=args.nsample,
    scaler=tgt_scaler,
    mean_scaler=tgt_mean_scaler,
    foldername=foldername,
    freq_interpolation=args.freq_interpolation,
    mixup_lambda=args.mixup_lambda,
    device=args.device,
)

end_time = time.time()
print("End time:", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
total_seconds = end_time - start_time
hours = total_seconds // 3600
remaining_seconds = total_seconds % 3600
minutes = remaining_seconds // 60
seconds = remaining_seconds % 60
print(
    "First Stage Total time: {:.0f}h {:.0f}min {:.4f}s".format(
        hours, minutes, seconds
    )
)