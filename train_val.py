import argparse
import logging
import os
import pprint
import shutil
import time
import warnings

import torch
import torch.distributed as dist
import torch.optim
import yaml
from easydict import EasyDict
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets.data_builder import build_dataloader
from modeling.model import UniAS

# from modeling.criterion import build_criterion
from utils.eval_helper import dump, log_metrics, merge_together, performances
from utils.misc_helper import (
    AverageMeter,
    create_logger,
    get_current_time,
    load_state,
    save_checkpoint,
    set_random_seed,
)
from utils.training_utils import get_optimizer, get_scheduler, setup_distributed
from utils.vis_helper import visualize_compound, visualize_single


# 忽略所有警告
warnings.filterwarnings("ignore")
cpu_num = 4
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)

parser = argparse.ArgumentParser(description="UniAS Framework")
parser.add_argument("--config", default="./configs/mvtec_ad_config.yaml")
parser.add_argument("--exp-path", default="./experiments/mvtec_ad")
parser.add_argument("-e", "--evaluate", action="store_true")
parser.add_argument("--local-rank", default=None, help="local rank for dist")


def main():
    global args, config, key_metric, best_metric
    args = parser.parse_args()

    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    # set up distributed and random seed
    config.PORT = config.get("PORT", None)
    rank, world_size = setup_distributed(port=config.PORT)
    random_seed = config.get("RANDOM_SEED", None)
    reproduce = config.get("REPRODUCE", None)
    if random_seed:
        set_random_seed(random_seed, reproduce)

    # update save path config
    ckpt_name = config.SAVER.EXP_NAME + "_ckpt.pth.tar"
    exp_name = config.SAVER.EXP_NAME.split("_")[0]
    config.EXP_PATH = os.path.join(
        args.exp_path, exp_name
    )  # ./experiments/mvtec_ad/run1
    config.LOG_PATH = os.path.join(
        config.EXP_PATH, config.SAVER.LOG_DIR
    )  # ./experiments/mvtec_ad/run1/log
    config.EVALUATOR.eval_dir = os.path.join(
        config.EXP_PATH, config.SAVER.EXP_NAME
    )  # ./experiments/mvtec_ad/run1/result_eval_temp
    # create logger
    if rank == 0:
        os.makedirs(config.EXP_PATH, exist_ok=True)
        os.makedirs(config.LOG_PATH, exist_ok=True)

        current_time = get_current_time()
        if not args.evaluate:
            tb_logger = SummaryWriter(config.LOG_PATH + "/events_dec/" + current_time)
        else:
            tb_logger = None
        logger = create_logger(
            "global_logger", config.LOG_PATH + "/dec_{}.log".format(current_time)
        )
        logger.info("args: {}".format(pprint.pformat(args)))
        logger.info("config: {}".format(pprint.pformat(config)))
    else:
        tb_logger = None

        def print(*args, **kwargs):
            pass

    # create model
    model = UniAS(config.MODEL)
    model.cuda()
    local_rank = int(os.environ["LOCAL_RANK"])
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        # find_unused_parameters=True,
    )

    layers = []
    for module in config.MODEL.keys():
        layers.append(module)

    # parameters needed to be updated
    parameters = []
    for param in model.parameters():
        if param.requires_grad:
            parameters.append(param)

    optimizer = get_optimizer(parameters, config.TRAINER.OPTIMIZER)
    lr_scheduler = get_scheduler(optimizer, config.TRAINER.LR_SCHEDULER)

    key_metric = config.EVALUATOR["KEY_METRIC"]
    best_metric = 0
    last_epoch = 0

    # load model: auto_resume > resume_model > load_path
    auto_resume = config.SAVER.get("AUTO_RESUME", True)
    resume_model = config.SAVER.get("RESUME_MODEL", None)
    load_path = config.SAVER.get("LOAD_PATH", None)

    lastest_model = os.path.join(config.EXP_PATH, ckpt_name)
    if not resume_model and (
        args.evaluate or (auto_resume and os.path.exists(lastest_model))
    ):
        resume_model = lastest_model
    if resume_model:
        best_metric, last_epoch = load_state(resume_model, model, optimizer=optimizer)
    elif load_path:
        assert os.path.exists(load_path)
        best_metric, last_epoch = load_state(load_path, model)

    train_loader, val_loader = build_dataloader(
        config.DATASET, distributed=True, train=not args.evaluate
    )

    if args.evaluate:
        validate(val_loader, model, tb_logger, last_epoch, key_metric)
        return

    # criterion = build_criterion(config.CRITERION)

    for epoch in range(last_epoch, config.TRAINER.MAX_EPOCH):
        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)
        last_iter = epoch * len(train_loader)
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            lr_scheduler,
            epoch,
            last_iter,
            tb_logger,
            # criterion,
        )
        lr_scheduler.step(epoch)

        if (epoch + 1) % config.TRAINER.VAL_FREQ_EPOCH == 0:
            ret_metrics = validate(val_loader, model, tb_logger, epoch, key_metric)
            # only ret_metrics on rank0 is not empty
            if rank == 0:
                ret_key_metric = ret_metrics[key_metric]
                is_best = ret_key_metric >= best_metric
                best_metric = max(ret_key_metric, best_metric)
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "arch": config.MODEL,
                        "state_dict": model.state_dict(),
                        "best_metric": best_metric,
                        "optimizer": optimizer.state_dict(),
                    },
                    is_best,
                    config,
                    name=ckpt_name,
                )


def train_one_epoch(
    train_loader,
    model,
    optimizer,
    lr_scheduler,
    epoch,
    start_iter,
    tb_logger,
    # criterion,
):
    batch_time = AverageMeter(config.TRAINER.PRINT_FREQ_STEP)
    data_time = AverageMeter(config.TRAINER.PRINT_FREQ_STEP)
    losses = AverageMeter(config.TRAINER.PRINT_FREQ_STEP)

    model.train()
    # freeze selected layers
    for layer in ["backbone"]:
        module = getattr(model.module, layer)
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    logger = logging.getLogger("global_logger")
    end = time.time()

    for i, input in enumerate(train_loader):
        curr_step = start_iter + i
        current_lr = lr_scheduler.get_lr()[0]

        # measure data loading time
        data_time.update(time.time() - end)

        # forward
        recon_feature = model(input, training=True)
        loss = recon_feature["loss"]
        reduced_loss = loss.clone()
        dist.all_reduce(reduced_loss)
        reduced_loss = reduced_loss / world_size
        losses.update(reduced_loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()
        # update
        if config.TRAINER.get("CLIP_MAX_NORM", None):
            max_norm = config.TRAINER.CLIP_MAX_NORM
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)

        if (curr_step + 1) % config.TRAINER.PRINT_FREQ_STEP == 0 and rank == 0:
            tb_logger.add_scalar("loss_train", losses.avg, curr_step + 1)
            tb_logger.add_scalar("lr", current_lr, curr_step + 1)
            tb_logger.flush()
            logger.info(
                "Epoch: [{0}/{1}]\t"
                "Iter: [{2}/{3}]\t"
                "Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                "Data {data_time.val:.2f} ({data_time.avg:.2f})\t"
                "Loss {loss.val:.5f} ({loss.avg:.5f})\t"
                "LR {lr:.5f}\t".format(
                    epoch + 1,
                    config.TRAINER.MAX_EPOCH,
                    curr_step + 1,
                    len(train_loader) * config.TRAINER.MAX_EPOCH,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    lr=current_lr,
                )
            )

        end = time.time()


def validate(val_loader, model, tb_logger, current_epoch, key_metric):
    batch_time = AverageMeter(0)
    losses = AverageMeter(0)

    model.eval()
    rank = dist.get_rank()
    logger = logging.getLogger("global_logger")
    # criterion = build_criterion(config.CRITERION)
    end = time.time()

    if rank == 0:
        os.makedirs(config.EVALUATOR.eval_dir, exist_ok=True)
        logger.info("evaluator saving to {}".format(config.EVALUATOR.eval_dir))
    # all threads write to config.EVALUATOR.eval_dir, it must be made before every thread begin to write
    dist.barrier()

    with torch.no_grad():
        for i, input in enumerate(val_loader):
            # forward
            recon_feature = model(input, training=False)
            dump(config.EVALUATOR.eval_dir, recon_feature, input)

            # record loss
            loss = recon_feature["loss"]
            num = len(input["filename"])
            losses.update(loss.item(), num)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % config.TRAINER.PRINT_FREQ_STEP == 0 and rank == 0:
                if tb_logger:
                    tb_logger.flush()
                logger.info(
                    "Test: [{0}/{1}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})".format(
                        i + 1, len(val_loader), batch_time=batch_time
                    )
                )

    # gather final results
    dist.barrier()
    total_num = torch.Tensor([losses.count]).cuda()
    loss_sum = torch.Tensor([losses.avg * losses.count]).cuda()
    dist.all_reduce(total_num, async_op=True)
    dist.all_reduce(loss_sum, async_op=True)
    final_loss = loss_sum.item() / total_num.item()

    ret_metrics = {}  # only ret_metrics on rank0 is not empty
    if rank == 0:
        logger.info("Gathering final results ...")
        # total loss
        logger.info(" * Loss {:.5f}\ttotal_num={}".format(final_loss, total_num.item()))
        fileinfos, preds, masks = merge_together(config.EVALUATOR.eval_dir)
        # if not args.evaluate:
        shutil.rmtree(config.EVALUATOR.eval_dir)
        # evaluate, log & vis
        ret_metrics = performances(
            fileinfos,
            preds,
            masks,
            config.EVALUATOR.METRICS,
            save_dir=config.EVALUATOR.eval_dir,
        )
        log_metrics(ret_metrics, config.EVALUATOR.METRICS)
        if tb_logger:
            tb_logger.add_scalar("loss_val", losses.avg, current_epoch + 1)
            tb_logger.add_scalar(
                "key_metric", ret_metrics[key_metric], current_epoch + 1
            )
            tb_logger.add_scalar(
                "dice", ret_metrics["mean_dice_auc"], current_epoch + 1
            )
            tb_logger.add_scalar("ap", ret_metrics["mean_ap_auc"], current_epoch + 1)

        if rank == 0:
            os.makedirs(config.EVALUATOR.eval_dir, exist_ok=True)

        if args.evaluate and config.EVALUATOR.get("VIS_COMPOUND", None):
            visualize_compound(
                fileinfos,
                preds,
                masks,
                config.EVALUATOR.eval_dir,
                config.EVALUATOR.VIS_COMPOUND,
                config.DATASET.IMAGE_READER,
            )
        if args.evaluate and config.EVALUATOR.get("VIS_SINGLE", None):
            visualize_single(
                fileinfos,
                preds,
                config.EVALUATOR.eval_dir,
                config.EVALUATOR.VIS_SINGLE,
                config.DATASET.IMAGE_READER,
            )
    model.train()
    return ret_metrics


if __name__ == "__main__":
    main()
