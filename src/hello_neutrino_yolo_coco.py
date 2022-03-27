import argparse
from pathlib import Path

import torch
import torch.nn as nn
from pycocotools.coco import COCO

from neutrino.job import Neutrino
from neutrino.nlogger import getLogger
from neutrino.framework.functions import LossFunction
from neutrino.framework.torch_framework import TorchFramework
from neutrino.framework.nn import NativeOptimizerFactory
from neutrino.framework.torch_nn import DefaultTorchNativeSchedulerFactory

from deeplite.torch_profiler.torch_data_loader import TorchForwardPass
from deeplite.torch_profiler.torch_inference import TorchEvaluationFunction

from deeplite_torch_zoo import get_data_splits_by_name, get_model_by_name
from deeplite_torch_zoo.wrappers.eval import yolo_eval_coco
from deeplite_torch_zoo.src.objectdetection.yolov5.models.yolov5_loss import \
    YoloV5Loss

logger = getLogger(__name__)


class YOLOEval(TorchEvaluationFunction):
    def __init__(self, net, data_root):
        self.net = net
        self.data_root = data_root
        self._gt = COCO(Path(self.data_root) / "annotations/instances_val2017.json")

    def _compute_inference(self, model, data_loader, **kwargs):
        # silent **kwargs
        return yolo_eval_coco(model=model, gt=self._gt, data_root=self.data_root, net=self.net)


class YOLOLoss(LossFunction):
    def __init__(self, num_classes=80, device='cuda', model=None):
        self.criterion = YoloV5Loss(num_classes=num_classes, model=model, device=device)
        self.torch_device = device

    def __call__(self, model, data):
        imgs, targets, labels_length, imgs_id = data
        _img_size = imgs.shape[-1]

        imgs = imgs.to(self.torch_device)
        p, p_d = model(imgs)
        _, loss_giou, loss_conf, loss_cls = self.criterion(p, p_d, targets, labels_length, _img_size)

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
    parser.add_argument('--coco_path', default='/neutrino/datasets/coco2017/',
                        help='coco data path contains train2017 and val2017')
    parser.add_argument('-b', '--batch_size', type=int, metavar='N', default=8, help='mini-batch size')
    parser.add_argument('-j', '--workers', type=int, metavar='N', default=4, help='number of data loading workers')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='yolo4m', help='model architecture',
        choices=['yolo4s', 'yolo4m', 'yolo5_6n', 'yolo5_6s', 'yolo5_6m'])

    # neutrino args
    parser.add_argument('-d', '--delta', type=float, metavar='DELTA', default=0.05, help='accuracy drop tolerance')
    parser.add_argument('--deepsearch', action='store_true', help="to consume the delta as much as possible")
    parser.add_argument('--dryrun', action='store_true', help="force all loops to early break")
    parser.add_argument('--horovod', action='store_true', help="activate horovod")
    parser.add_argument('--device', type=str, metavar='DEVICE', default='GPU', help='Device to use, CPU or GPU',
                        choices=['GPU', 'CPU'])
    parser.add_argument('--bn_fuse', action='store_true', help="fuse batch normalization layers")
    args = parser.parse_args()
    device_map = {'CPU': 'cpu', 'GPU': 'cuda'}

    data_splits = get_data_splits_by_name(
        data_root=args.coco_path,
        dataset_name='coco',
        model_name=args.arch,
        batch_size=args.batch_size,
        num_workers=args.workers,
        device=device_map[args.device],

    )
    fp = TorchForwardPass(model_input_pattern=(0, '_', '_', '_'))

    reference_model = get_model_by_name(model_name=args.arch,
                                        dataset_name='coco_80',
                                        pretrained=True,
                                        progress=True,
                                        device=device_map[args.device],)

    # eval func
    eval_key = 'mAP'
    if args.dryrun:
        def eval_func(model, data_splits):
            return {eval_key: 1}
    else:
        eval_func = YOLOEval(net=args.arch, data_root=args.coco_path)

    # loss
    loss_cls = YOLOLoss
    loss_kwargs = {'device': device_map[args.device], 'model': reference_model}

    # custom config
    config = {
        'task_type': 'object_detection',
        'export': {'format': ['onnx']},
        'deepsearch': args.deepsearch,
        'delta': args.delta,
        'device': args.device,
        'use_horovod': args.horovod,
        'bn_fusion': args.bn_fuse,
        'full_trainer': {
            'epochs': args.epochs,
            'eval_key': eval_key,
            'early_stopping_patience': 20,
            'loop_test': args.dryrun
        },
        'fine_tuner': {
            'loop_params': {
                'epochs': 1,
                'loop_test': args.dryrun
            }
        },
    }

    # get default yolo optimizer, scheduler
    config['full_trainer'] = get_yolo5_6_config(config['full_trainer'], args.lr)

    optimized_model = Neutrino(TorchFramework(),
                               data=data_splits,
                               model=reference_model,
                               config=config,
                               eval_func=eval_func,
                               forward_pass=fp,
                               loss_function_cls=loss_cls,
                               loss_function_kwargs=loss_kwargs).run(dryrun=args.dryrun)
