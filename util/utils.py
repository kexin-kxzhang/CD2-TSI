import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import pickle
from torch import nn
import copy

from util.FMixup import freq_mixup_interpolation
from util.DA import discrepancy_alignment_loss    

def train_da(
    src_model,
    tgt_model,
    config,
    src_train_loader,
    tgt_train_loader,
    tgt_valid_loader=None,
    src_valid_loader=None,
    valid_epoch_interval=20,
    foldername="",
    freq_interpolation=1,
    mixup_lambda=0.2,
    miu_align=5,
    taul=0.2,
    tauh=0.7,
    device=None,
):
    src_optimizer = Adam(src_model.parameters(), lr=config["lr"], weight_decay=1e-6)
    tgt_optimizer = Adam(tgt_model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        tgt_output_path = foldername + "/tgt_model.pth"

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    src_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        src_optimizer, milestones=[p1, p2], gamma=0.1
    )

    tgt_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        tgt_optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        total_loss = 0
        src_model.train()
        tgt_model.train()
        tgt_batch_no = 0

        with tqdm(src_train_loader, mininterval=5.0, maxinterval=50.0) as src_it, \
            tqdm(tgt_train_loader, mininterval=5.0, maxinterval=50.0) as tgt_it:
            for src_batch, tgt_batch in zip(src_it, tgt_it):
                if freq_interpolation==1:
                    tgt_cond_data = tgt_batch["observed_data"].to(device).float()*tgt_batch["cond_mask"].to(device)
                    src_cond_data = src_batch["observed_data"].to(device).float()*src_batch["cond_mask"].to(device)
                    mixup_tgt = freq_mixup_interpolation(tgt_cond_data, src_cond_data, alpha=0.003, ratio=mixup_lambda)
                    tgt_batch["freq_itp"] = torch.tensor(mixup_tgt).to(device).float()
                    mixup_src = freq_mixup_interpolation(src_cond_data, tgt_cond_data, alpha=0.003, ratio=mixup_lambda)
                    src_batch["freq_itp"] = torch.tensor(mixup_src).to(device).float()

                src_optimizer.zero_grad()
                tgt_optimizer.zero_grad()

                src_loss, src_pred = src_model(src_batch)  
                tgt_loss, tgt_pred = tgt_model(tgt_batch) 

                total_loss = 1 * src_loss + 1 * tgt_loss
                if miu_align != 0:
                    with torch.no_grad():
                        src_model.eval()
                        _, tgt_pred_from_src_model = src_model(tgt_batch)
                        l_align = discrepancy_alignment_loss(tgt_pred, tgt_pred_from_src_model, taul, tauh)
                        total_loss += miu_align * l_align
 
                total_loss.backward()
                avg_loss += total_loss.item()
                src_optimizer.step()
                tgt_optimizer.step()

                tgt_batch_no += 1

                tgt_it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / tgt_batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
                if tgt_batch_no >= config["itr_per_epoch"]:
                    break

            src_lr_scheduler.step()
            tgt_lr_scheduler.step()

        if tgt_valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            tgt_model.eval()
            avg_loss_valid = 0
            tgt_valid_batch_no = 0

            with torch.no_grad():   
                with tqdm(src_valid_loader, mininterval=5.0, maxinterval=50.0) as valid_src_it, \
                    tqdm(tgt_valid_loader, mininterval=5.0, maxinterval=50.0) as valid_tgt_it:
                    for valid_src_batch, valid_tgt_batch in zip(valid_src_it, valid_tgt_it):
                        if freq_interpolation==1:
                            valid_tgt_cond_data = valid_tgt_batch["observed_data"].to(device).float()*valid_tgt_batch["cond_mask"].to(device)
                            valid_src_cond_data = valid_src_batch["observed_data"].to(device).float()*valid_src_batch["cond_mask"].to(device)
                            valid_mixup_tgt = freq_mixup_interpolation(valid_tgt_cond_data, valid_src_cond_data, alpha=0.003, ratio=mixup_lambda)
                            valid_tgt_batch["freq_itp"] = torch.tensor(valid_mixup_tgt).to(device).float()

                        loss, _, _, _, _ = tgt_model(valid_tgt_batch, is_train=0)
                        avg_loss_valid += loss.item()
                        tgt_valid_batch_no+=1
                        valid_tgt_it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / tgt_valid_batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / tgt_valid_batch_no,
                    "at",
                    epoch_no,
                )

    if foldername != "":
        torch.save(tgt_model.state_dict(), tgt_output_path)


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )

def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))

def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def calc_quantile_CRPS_sum(target, forecast, eval_points, mean_scaler, scaler):

    eval_points = eval_points.mean(-1)
    target = target * scaler + mean_scaler
    target = target.sum(-1)
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = torch.quantile(forecast.sum(-1),quantiles[i],dim=1)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def evaluate_da(model, tgt_test_loader, src_test_loader, nsample=100, scaler=1, mean_scaler=0, foldername="", 
                freq_interpolation=1, mixup_lambda=0.2, device=None):

    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []

        tgt_test_batch_no = 0
        with tqdm(src_test_loader, mininterval=5.0, maxinterval=50.0) as src_test_it, \
            tqdm(tgt_test_loader, mininterval=5.0, maxinterval=50.0) as tgt_test_it:
            for src_test_batch, tgt_test_batch in zip(src_test_it, tgt_test_it):
                output = model.evaluate(tgt_test_batch, nsample)

                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  
                c_target = c_target.permute(0, 2, 1)  
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                mse_current = (
                    ((samples_median.values - c_target) * eval_points) ** 2
                ) * (scaler ** 2)
                mae_current = (
                    torch.abs((samples_median.values - c_target) * eval_points) 
                ) * scaler

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                tgt_test_batch_no += 1 

                tgt_test_it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "batch_no": tgt_test_batch_no,
                    },
                    refresh=True,
                )

            with open(
                foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                all_target = torch.cat(all_target, dim=0)
                all_evalpoint = torch.cat(all_evalpoint, dim=0)
                all_observed_point = torch.cat(all_observed_point, dim=0)
                all_observed_time = torch.cat(all_observed_time, dim=0)
                all_generated_samples = torch.cat(all_generated_samples, dim=0)

                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                        all_evalpoint,
                        all_observed_point,
                        all_observed_time,
                        scaler,
                        mean_scaler,
                    ],
                    f,
                )

            CRPS = calc_quantile_CRPS(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )

            with open(
                foldername + "/target_model_result_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                pickle.dump(
                    [
                        np.sqrt(mse_total / evalpoints_total),
                        mae_total / evalpoints_total,
                        CRPS,
                    ],
                    f,
                )
                print("target_model_RMSE:", np.sqrt(mse_total / evalpoints_total))
                print("target_model_MAE:", mae_total / evalpoints_total)
                print("target_model_CRPS:", CRPS)
