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
from dataset_process.dataset_electricity_etth1 import get_dataloader_etth1
from dataset_process.dataset_electricity_etth2 import get_dataloader_etth2
from models.main_model import CD2_TSI
from util.utils import train_da, evaluate_da
import logging


########## args defination
parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument(
    "--targetstrategy", type=str, default="hybrid", choices=["hybrid", "random", "block"]
)
parser.add_argument("--missing_pattern", type=str, default="block")  # block|point
parser.add_argument(
    "--dataset", type=str, default="air_quality", choices=["air_quality", "hydrology", "electricity"]
)
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--testmissingratio", type=float, default=0.1)
# DA
parser.add_argument("--lambda_T", type=float, default=0)
parser.add_argument("--lambda_C", type=float, default=0)
parser.add_argument("--freq_interpolation", type=int, default=0)
parser.add_argument("--ratio", type=float, default=0.2)
args = parser.parse_args()
print(args)


########## load config
path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["target_strategy"] = args.targetstrategy
config["model"]["missing_pattern"] = args.missing_pattern

config["train"]["dataset"] = args.dataset
config["train"]["seed"] = args.seed
print(json.dumps(config, indent=4))


########## model location
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 
foldername = (
    "./save/" + str(args.dataset) + "/" + "CD2-TSI" + "/"  + "transfer-" + str(args.lambda_T) + "_" + "consistency-" + str(args.lambda_T) + "_" + current_time + "/"
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
    target_strategy=args.targetstrategy,
    missing_pattern=args.missing_pattern,
    seed=args.seed,
    batch_size=config["train"]["batch_size"],
    missing_ratio=args.testmissingratio,
)

tgt_train_loader, tgt_valid_loader, tgt_test_loader, tgt_scaler, tgt_mean_scaler = get_dataloader_tgt(
    device=args.device,
    target_strategy=args.targetstrategy,
    missing_pattern=args.missing_pattern,
    seed=args.seed,
    batch_size=config["train"]["batch_size"],
    missing_ratio=args.testmissingratio,
)

########## src and tgt model
if args.dataset == "air_quality":
    target_dim = 27
elif args.dataset == "hydrology":
    target_dim = 20
elif args.dataset == "electricity":
    target_dim = 7

config["model"]["freq_interpolation"] = args.freq_interpolation

src_model = CD2_TSI(config, args.device, target_dim=target_dim).to(args.device)
tgt_model = CD2_TSI(config, args.device, target_dim=target_dim).to(args.device)


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
    lambda_T=args.lambda_T,
    lambda_C=args.lambda_C,
    freq_interpolation=args.freq_interpolation,
    device=args.device,
    ratio=args.ratio,
)

logging.basicConfig(filename=foldername + '/test_model.log', level=logging.DEBUG)
logging.info("model_name={}".format(args.modelfolder))
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
    device=args.device,
    ratio=args.ratio,
)
end_time = time.time()
print("End time:", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
total_seconds = end_time - start_time
hours = total_seconds // 3600
remaining_seconds = total_seconds % 3600
minutes = remaining_seconds // 60
seconds = remaining_seconds % 60
print(
    "Total time: {:.0f}h {:.0f}min {:.4f}s".format(
        hours, minutes, seconds
    )
)
