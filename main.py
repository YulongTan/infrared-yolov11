import copy
import csv
import os
import warnings
from argparse import ArgumentParser
# from ultralytics import YOLO
import torch
import tqdm
import yaml
from torch.utils import data
import torch.quantization
import cv2
import sys
import numpy as np

import torchvision.transforms.functional as TF


# 将绝对路径添加到 sys.path
sys.path.append("D:/pycharm_proj/yolov11/nets")
from nets import nn
from utils import util
from utils.dataset import Dataset

warnings.filterwarnings("ignore")

data_dir = 'D:/pycharm_proj/yolov11/kaist'


def train(args, params):
    # Model
    model = nn.yolo_v11_n(len(params['names']))
    model.cuda()
    # Optimizer
    optimizer = torch.optim.SGD(util.set_params(model, params['weight_decay']),
                                params['min_lr'], params['momentum'], nesterov=True)

    # EMA
    ema = util.EMA(model) if args.local_rank == 0 else None
    filenames = []
    with open(f'{data_dir}/train.txt') as reader:
        for filename in reader.readlines():
            filename = os.path.basename(filename.rstrip())
            # filenames.append(f'{data_dir}/kaist_wash_picture_train/visible/' + filename)
            filenames.append(f'{data_dir}/kaist_wash_picture_train/RGBA/' + filename)  # RGBA
    sampler = None
  
    dataset = Dataset(filenames, args.input_size, params, augment=True)
    print("okhere") 
    if args.distributed:
        sampler = data.distributed.DistributedSampler(dataset)

    loader = data.DataLoader(dataset, args.batch_size, sampler is None, sampler,
                             num_workers=8, pin_memory=True, collate_fn=Dataset.collate_fn)

    # Scheduler
    num_steps = len(loader)
    scheduler = util.LinearLR(args, params, num_steps)

    if args.distributed:
        # DDP mode
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

    best = 0
    criterion = util.ComputeLoss(model, params)

    with open('weights/step.csv', 'w') as log:
        if args.local_rank == 0:
            logger = csv.DictWriter(log, fieldnames=['epoch',
                                                     'box', 'cls', 'dfl',
                                                     'Recall', 'Precision', 'mAP@50', 'mAP'])
            logger.writeheader()

        for epoch in range(args.epochs):
            model.train()
            if args.distributed:
                sampler.set_epoch(epoch)
            if args.epochs - epoch == 10:
                loader.dataset.mosaic = False

            p_bar = enumerate(loader)

            if args.local_rank == 0:
                print(('\n' + '%10s' * 5) % ('epoch', 'memory', 'box', 'cls', 'dfl'))
                p_bar = tqdm.tqdm(p_bar, total=num_steps)

            optimizer.zero_grad()
            avg_box_loss = util.AverageMeter()
            avg_cls_loss = util.AverageMeter()
            avg_dfl_loss = util.AverageMeter()

            for i, (samples, targets) in p_bar:
                # 打印 targets
                # print(f"\n[Debug] Batch {i} targets shape: {targets.shape}")
                # print(f"[Debug] targets 内容（前3个）:\n{targets[:3]}")

                # 显示每张图片（只显示第一个样本，防止弹太多窗口）

                # 假设 samples 是 [B, C, H, W] 且为 torch.Tensor
                image_tensor = samples[0].cpu()
                # print(image_tensor.shape)
                image_np = image_tensor.numpy().transpose(1, 2, 0)  # CHW -> HWC

                if image_np.shape[2] == 1:
                    image_np = np.repeat(image_np, 3, axis=2)  # 单通道转三通道
                elif image_np.shape[2] == 4:
                    image_np = image_np[:, :, :3]  # 丢掉 alpha 通道

                # 若是归一化图像（0~1），恢复到0~255
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)

                # cv2.imshow(f"Batch {i} Sample 0", image_np)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # breakpoint()

                step = i + num_steps * epoch
                scheduler.step(step, optimizer)

                samples = samples.cuda().float() / 255
                # Forward

                outputs = model(samples)  # forward


                loss_box, loss_cls, loss_dfl = criterion(outputs, targets)

                avg_box_loss.update(loss_box.item(), samples.size(0))
                avg_cls_loss.update(loss_cls.item(), samples.size(0))
                avg_dfl_loss.update(loss_dfl.item(), samples.size(0))

                loss_box *= args.batch_size  # loss scaled by batch_size
                loss_cls *= args.batch_size  # loss scaled by batch_size
                loss_dfl *= args.batch_size  # loss scaled by batch_size
                loss_box *= args.world_size  # gradient averaged between devices in DDP mode
                loss_cls *= args.world_size  # gradient averaged between devices in DDP mode
                loss_dfl *= args.world_size  # gradient averaged between devices in DDP mode

                # Backward
                (loss_box + loss_cls + loss_dfl).backward()

                # Optimize
                if step % 1 == 0:
                    # amp_scale.step(optimizer)  # optimizer.step
                    # amp_scale.update()
                    optimizer.step()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)

                torch.cuda.synchronize()

                # Log
                if args.local_rank == 0:
                    memory = f'{torch.cuda.memory_reserved() / 1E9:.4g}G'  # (GB)
                    s = ('%10s' * 2 + '%10.3g' * 3) % (f'{epoch + 1}/{args.epochs}', memory,
                                                       avg_box_loss.avg, avg_cls_loss.avg, avg_dfl_loss.avg)
                    p_bar.set_description(s)

            if args.local_rank == 0:
                # mAP
                last = test(args, params, ema.ema)

                logger.writerow({'epoch': str(epoch + 1).zfill(3),
                                 'box': str(f'{avg_box_loss.avg:.3f}'),
                                 'cls': str(f'{avg_cls_loss.avg:.3f}'),
                                 'dfl': str(f'{avg_dfl_loss.avg:.3f}'),
                                 'mAP': str(f'{last[0]:.3f}'),
                                 'mAP@50': str(f'{last[1]:.3f}'),
                                 'Recall': str(f'{last[2]:.3f}'),
                                 'Precision': str(f'{last[3]:.3f}')})
                log.flush()

                # Update best mAP
                if last[0] > best:
                    best = last[0]

                # Save model
                save = {'epoch': epoch + 1,
                        'model': copy.deepcopy(ema.ema)}

                # Save last, best and delete
                torch.save(model.state_dict(), './weights/last.pt')
                if best == last[0]:
                    torch.save(model.state_dict(), './weights/last.pt')
                del save

    if args.local_rank == 0:
        util.strip_optimizer('./weights/best.pt')  # strip optimizers
        util.strip_optimizer('./weights/last.pt')  # strip optimizers

    torch.cuda.empty_cache()


@torch.no_grad()
def test(args, params, model=None):
    filenames = []
    with open(f'D:/pycharm_proj/yolov11/kaist/validate.txt') as reader:
        for filename in reader.readlines():
            filename = os.path.basename(filename.rstrip())
            # filenames.append(f'{data_dir}/kaist_wash_picture_test/visible/' + filename)
            filenames.append(f'{data_dir}/kaist_wash_picture_test/RGBA/' + filename)#RGBA

    dataset = Dataset(filenames, args.input_size, params, augment=False)
    loader = data.DataLoader(dataset, batch_size=96, shuffle=False, num_workers=8,
                             pin_memory=True, collate_fn=Dataset.collate_fn)

    if model is None:
        # 加载模型权重并确保完整加载模型
        with torch.serialization.safe_globals([np._core.multiarray._reconstruct]):  # 允许加载 numpy 类型
            model = torch.load(f='./weights/best.pt', map_location='cuda')
        model = model['model'].float().fuse()
    model.eval()

    # Configure
    iou_v = torch.linspace(start=0.5, end=0.95, steps=10).cuda()  # iou vector for mAP@0.5:0.95
    n_iou = iou_v.numel()

    m_pre = 0
    m_rec = 0
    map50 = 0
    mean_ap = 0
    metrics = []
    p_bar = tqdm.tqdm(loader, desc=('%10s' * 5) % ('', 'precision', 'recall', 'mAP50', 'mAP'))
    for samples, targets in p_bar:
        samples = samples.cuda().float()

        samples = samples / 255.  # 0 - 255 to 0.0 - 1.0
        _, _, h, w = samples.shape  # batch-size, channels, height, width
        scale = torch.tensor((w, h, w, h)).cuda()
        # Inference
        outputs = model(samples)
        print(outputs[0][4:6].max())


        # NMS
        outputs = outputs[0]
        outputs = util.non_max_suppression(outputs)
        # Metrics
        for i, output in enumerate(outputs):
            idx = targets['idx'] == i
            cls = targets['cls'][idx]
            box = targets['box'][idx]

            cls = cls.cuda()
            box = box.cuda()

            metric = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()

            if output.shape[0] == 0:
                if cls.shape[0]:
                    metrics.append((metric, *torch.zeros((2, 0)).cuda(), cls.squeeze(-1)))
                continue
            # Evaluate
            if cls.shape[0]:
                target = torch.cat(tensors=(cls, util.wh2xy(box) * scale), dim=1)
                metric = util.compute_metric(output[:, :6], target, iou_v)
            # Append
            metrics.append((metric, output[:, 4], output[:, 5], cls.squeeze(-1)))

    # Compute metrics
    metrics = [torch.cat(x, dim=0).cpu().numpy() for x in zip(*metrics)]  # to numpy
    if len(metrics) and metrics[0].any():
        tp, fp, m_pre, m_rec, map50, mean_ap = util.compute_ap(*metrics)
    # Print results
    print(('%10s' + '%10.3g' * 4) % ('', m_pre, m_rec, map50, mean_ap))
    # Return results
    model.float()  # for training
    return mean_ap, map50, m_rec, m_pre


def profile(args, params):
    import thop
    shape = (1, 3, args.input_size, args.input_size)
    # shape = (1, 4, args.input_size, args.input_size) #RGBA
    model = nn.yolo_v11_n(len(params['names'])).fuse()

    model.eval()
    model(torch.zeros(shape))

    x = torch.empty(shape)
    flops, num_params = thop.profile(model, inputs=[x], verbose=False)
    flops, num_params = thop.clever_format(nums=[2 * flops, num_params], format="%.3f")

    if args.local_rank == 0:
        print(f'Number of parameters: {num_params}')
        print(f'Number of FLOPs: {flops}')


def main():
    parser = ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--epochs', default=600, type=int)
    parser.add_argument('--train',default=True, action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    if args.distributed:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')


    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')

    with open('utils/kitti.yaml', errors='ignore') as f:
        params = yaml.safe_load(f)

    util.setup_seed()
    util.setup_multi_processes()

    profile(args, params)


    if args.train:
        train(args, params)
    if args.test:
        test(args, params)


if __name__ == "__main__":
    main()
