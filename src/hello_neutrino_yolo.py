import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from deeplite.torch_profiler.torch_data_loader import TorchForwardPass
from deeplite.torch_profiler.torch_inference import TorchEvaluationFunction
from deeplite_torch_zoo import (get_data_splits_by_name, get_eval_function,
                                get_model_by_name)
from deeplite_torch_zoo.src.objectdetection.yolov5.models.yolov5_loss import \
    YoloV5Loss
from neutrino.framework.functions import LossFunction
from neutrino.framework.nn import NativeOptimizerFactory
from neutrino.framework.torch_framework import TorchFramework
from neutrino.framework.torch_nn import DefaultTorchNativeSchedulerFactory
from neutrino.job import Neutrino
from neutrino.nlogger import getLogger
from pycocotools.coco import COCO

try:
    from external_training import YoloTrainingLoop
except ModuleNotFoundError as e:
    print('Missing required packages for external training loop: ', e)

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
# WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

logger = getLogger(__name__)


class YOLOEval(TorchEvaluationFunction):
    def __init__(self, model_name, dataset_name, data_root, num_classes, test_img_size):
        self._data_root = data_root
        self._evaluation_fn = get_eval_function(dataset_name=dataset_name, model_name=model_name)
        self._gt = None
        if dataset_name == 'voc':
            self._data_root = Path(self._data_root) / "VOC2007"
        if dataset_name == 'coco':
            self._gt = COCO(Path(self._data_root) / "annotations/instances_val2017.json")
        self._num_classes = num_classes
        self._test_img_size = test_img_size

    def _compute_inference(self, model, data_loader, **kwargs):
        # silent **kwargs
        return self._evaluation_fn(model=model, data_root=self._data_root, gt=self._gt,
                                   num_classes=self._num_classes, img_size=self._test_img_size)


class YOLOLoss(LossFunction):
    def __init__(self, num_classes=20, device='cuda', model=None):
        self.criterion = YoloV5Loss(num_classes=num_classes, model=model, device=device)
        self.torch_device = device

    def __call__(self, model, data):
        imgs, targets, labels_length, imgs_id = data
        _img_size = imgs.shape[-1]

        imgs = imgs.to(self.torch_device)
        pred = model(imgs)
        _, loss_giou, loss_conf, loss_cls = self.criterion(pred, targets, labels_length, _img_size)

        return {'lgiou': loss_giou, 'lconf': loss_conf, 'lcls': loss_cls}


class YoloOptimizerFactory(NativeOptimizerFactory):
    def __init__(self, name, weight_decay=0.0, **kwargs):
        self.optimizer_cls = getattr(torch.optim, name)
        self.optimizer_kwargs = kwargs
        self.weight_decay = weight_decay

    def make(self, native_model):
        g0, g1, g2 = self.get_param_groups(native_model)
        optimizer = self.optimizer_cls(g0, **self.optimizer_kwargs)
        optimizer.add_param_group({'params': g1, 'weight_decay': self.weight_decay})
        optimizer.add_param_group({'params': g2})  # add g2 (biases)
        return optimizer

    def get_param_groups(self, native_model):
        g0, g1, g2 = [], [], []  # optimizer parameter groups
        for v in native_model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
                g2.append(v.bias)
            if isinstance(v, (torch.nn.BatchNorm2d)):  # weight (no decay)
                g0.append(v.weight)
            elif hasattr(v, 's'):
                g0.append(v.s)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                g1.append(v.weight)
        return g0, g1, g2


def get_yolo5_6_config(loop_params, lr):
    cfg = yolo5_6_finetune_config
    lr0 = lr if lr else cfg['lr0']
    default_opt_kwargs = {'name': 'SGD', 'lr': lr0,
                          'momentum': cfg['momentum'], 'nesterov': True}
    if 'epochs' not in loop_params:
        raise KeyError("epochs must be defined by config file or command line arg")

    if not loop_params.get('optimizer', None):
        opt_kwargs = default_opt_kwargs
    else:
        opt_kwargs = loop_params['optimizer']
        # overwrite opt lr with command line lr
        opt_kwargs['lr'] = lr if lr else opt_kwargs['lr']

    sched_kwargs = {'name': 'CosineAnnealingLR', 'T_max':  loop_params['epochs'],
                    'eta_min': cfg['lrf'] * lr0}
    scheduler = DefaultTorchNativeSchedulerFactory(**sched_kwargs)
    loop_params['scheduler'] = {'factory': scheduler,
                                'triggers': {'after_epoch': True},
                                'eval_based': False}

    optimizer = YoloOptimizerFactory(weight_decay=cfg['weight_decay'], **opt_kwargs)
    loop_params['optimizer'] = optimizer
    return loop_params


# from https://github.com/ultralytics/yolov5/blob/956be8e642b5c10af4a1533e09084ca32ff4f21f/data/hyps/hyp.scratch.yaml

yolo5_6_finetune_config = {
    'lr0': 0.0032,
    'lrf': 0.12,
    'momentum': 0.843,
    'weight_decay': 0.00036,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model/dataset args
    parser.add_argument('--dataset', metavar='DATASET', default='voc', help='dataset to use')
    parser.add_argument('-r', '--data_root', metavar='PATH', default='/neutrino/datasets/VOCdevkit',
                        help='dataset data root path')
    parser.add_argument('-b', '--batch_size', type=int, metavar='N', default=8, help='mini-batch size')
    parser.add_argument('-j', '--workers', type=int, metavar='N', default=4, help='number of data loading workers')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='yolo3', help='model architecture',
        choices=['yolo3', 'yolo4s', 'yolo4m', 'yolo5_6n', 'yolo5_6s', 'yolo5_6m', 'yolo5_6l', 'yolo5_6x'])

    # neutrino args
    parser.add_argument('--lr', default=None, type=float, help='learning rate for training Yolo model')
    parser.add_argument('-d', '--delta', type=float, metavar='DELTA', default=0.05, help='accuracy drop tolerance')
    parser.add_argument('--deepsearch', action='store_true', help="to consume the delta as much as possible")
    parser.add_argument('--dryrun', action='store_true', help="force all loops to early break")
    parser.add_argument('--fp16', action='store_true', help="export to fp16 as well if it is possible")
    parser.add_argument('--horovod', action='store_true', help="activate horovod")
    parser.add_argument('--device', type=str, metavar='DEVICE', default='GPU', help='Device to use, CPU or GPU',
                        choices=['GPU', 'CPU'])
    parser.add_argument('--epochs', default=50, type=int, help='number of fine-tuning epochs')
    parser.add_argument('--ft_epochs', default=1, type=int, help='number of fine-tuning epochs')
    parser.add_argument('--eval_freq', type=int, default=2,
                        help='frequency at which perform evaluation')
    parser.add_argument('--img_size', type=int, metavar='XXX', default=None,
                        help='img size for train dataset')
    parser.add_argument('--test_img_size', type=int, metavar='XXX', default=None,
                        help='img size for test dataset')
    parser.add_argument('--external_tl', action='store_true')

    # external args
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--hp_config', type=str, default='finetune',
                        help='hyper-parameters config only recognized by the train_yolo.py script')

    # DDP mode
    if LOCAL_RANK != -1:
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        torch.distributed.init_process_group(backend="nccl" if torch.distributed.is_nccl_available() else "gloo")

    args = parser.parse_args()

    if LOCAL_RANK != -1 and args.horovod:
        raise RuntimeError("Do not run horovod and torch.distributed!")

    if args.external_tl:
        # litle hack for yolo ddp
        from external_training.yolo_utils.general import set_logging
        set_logging(RANK)

    device_map = {'CPU': 'cpu', 'GPU': 'cuda'}
    data_splits_kwargs = {}
    if args.img_size is not None:
        data_splits_kwargs = {'img_size': args.img_size}
    data_splits = get_data_splits_by_name(
        data_root=args.data_root,
        dataset_name=args.dataset,
        model_name=args.arch,
        batch_size=args.batch_size,
        num_workers=args.workers,
        device=device_map[args.device],
        **data_splits_kwargs,
    )
    num_classes = data_splits['train'].dataset.num_classes

    if args.test_img_size is None:
        args.test_img_size = data_splits["test"].dataset._img_size

    fp = TorchForwardPass(model_input_pattern=(0, '_', '_', '_'))

    reference_model = get_model_by_name(model_name=args.arch,
                                        dataset_name=args.dataset,
                                        pretrained=True,
                                        progress=True,
                                        device=device_map[args.device])

    # eval func
    eval_key = 'mAP'
    if args.dryrun:
        def eval_func(model, data_splits, device=None):
            return {eval_key: 1}
    else:
        eval_func = YOLOEval(model_name=args.arch, dataset_name=args.dataset,
                             data_root=args.data_root, num_classes=num_classes,
                             test_img_size=args.test_img_size)

    # loss
    loss_cls = YOLOLoss
    loss_kwargs = {'device': device_map[args.device], 'model': reference_model, 'num_classes': num_classes}

    # custom config
    config = {
        'task_type': 'object_detection',
        'export': {
            'format': 'onnx',
            'kwargs': {'precision': 'fp16' if args.fp16 else 'fp32'}
        },
        'deepsearch': args.deepsearch,
        'delta': args.delta,
        'device': args.device,
        'use_horovod': args.horovod,
        'full_trainer': {
            'epochs': args.epochs,
            'eval_key': eval_key,
            'early_stopping_patience': 20,
            'loop_test': args.dryrun,
            'eval_freq': args.eval_freq
        },
        'fine_tuner': {
            'loop_params': {
                'epochs': args.ft_epochs,
                'loop_test': args.dryrun
            }
        },
    }

    config['full_trainer'] = get_yolo5_6_config(config['full_trainer'], args.lr)
    if args.external_tl:
        trainer = YoloTrainingLoop(args)
        config['external_training_loop'] = trainer

    optimized_model = Neutrino(TorchFramework(),
                               data=data_splits,
                               model=reference_model,
                               config=config,
                               eval_func=eval_func,
                               forward_pass=fp,
                               loss_function_cls=loss_cls,
                               loss_function_kwargs=loss_kwargs).run(dryrun=args.dryrun)
