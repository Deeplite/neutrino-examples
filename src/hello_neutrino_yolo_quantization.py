import argparse

from neutrino.framework.torch_framework import TorchFramework
from deeplite.torch_profiler.torch_data_loader import TorchForwardPass
from neutrino.job import Neutrino

from deeplite_torch_zoo import get_data_splits_by_name, get_model_by_name, get_eval_function
from deeplite_torch_zoo.src.objectdetection.yolov5.models.yolov5_6 import Detect

from pathlib import Path
from pycocotools.coco import COCO
import torch
import torch.nn as nn

from neutrino.framework.functions import LossFunction
from neutrino.framework.nn import NativeOptimizerFactory
from deeplite.torch_profiler.torch_inference import TorchEvaluationFunction

from deeplite_torch_zoo.src.objectdetection.yolov5.models.yolov5_loss import \
    YoloV5Loss


class YOLOEval(TorchEvaluationFunction):
    def __init__(self, model_name, data_root, dataset_type, test_img_size):
        self._gt = None
        self._data_root = data_root
        self._dataset_type = dataset_type
        self._test_img_size = test_img_size
        self._evaluation_fn = get_eval_function(model_name=model_name, dataset_name=self._dataset_type)

        if self._dataset_type in ("voc", "voc07"):
            self._data_root = Path(self._data_root) / "VOC2007"
        if self._dataset_type == "coco":
            self._gt = COCO(Path(self._data_root) / "annotations/instances_val2017.json")

    def _compute_inference(self, model, data_loader, **kwargs):
        # silent **kwargs
        return self._evaluation_fn(model=model, data_root=self._data_root,
                                   gt=self._gt, img_size=self._test_img_size)


class YOLOLoss(LossFunction):
    def __init__(self, model=None, num_classes=None, device='cuda'):
        self.criterion = YoloV5Loss(model=model, num_classes=num_classes, device=device)
        self.torch_device = device

    def __call__(self, model, data):
        imgs, targets, labels_length, imgs_id = data
        _img_size = imgs.shape[-1]

        imgs = imgs.to(self.torch_device)
        pred = model(imgs)
        _, loss_giou, loss_conf, loss_cls = self.criterion(pred, targets,
                                                           labels_length, _img_size)
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
    default_sched_kwargs = {
        'factory': {
            'name': 'CosineAnnealingLR',
            'T_max': loop_params['epochs'],
            'eta_min': cfg['lrf'] * lr0
        },
        'triggers': {
            'after_epoch': True
        },
        'eval_based': False
    }

    if not loop_params.get('optimizer', None):
        opt_kwargs = default_opt_kwargs
    else:
        opt_kwargs = loop_params['optimizer']
        # overwrite opt lr with command line lr
        opt_kwargs['lr'] = lr if lr else opt_kwargs['lr']

    if not loop_params.get('scheduler', None):
        loop_params['scheduler'] = default_sched_kwargs
    optimizer = YoloOptimizerFactory(weight_decay=cfg['weight_decay'], **opt_kwargs)
    loop_params['optimizer'] = optimizer
    return loop_params


yolo5_6_finetune_config = {
    'lr0': 0.0032,
    'lrf': 0.12,
    'momentum': 0.843,
    'weight_decay': 0.00036,
}


def get_runtime_resolution(runtime_resolution):
    if runtime_resolution:
        resolutions = []
        for res_str in runtime_resolution:
            dim_strs = res_str.split('x')
            resolutions.append((int(dim_strs[0]), int(dim_strs[1])))
        return resolutions
    return []


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model/dataset args
    parser.add_argument('--dataset', type=str, default='voc', help='voc, coco, other obj detection dataset')
    parser.add_argument('-r', '--data_root', metavar='PATH', default='/neutrino/datasets/VOCdevkit/', help='dataset data root path')
    parser.add_argument('-b', '--batch_size', type=int, metavar='N', default=8, help='mini-batch size')
    parser.add_argument('-j', '--workers', type=int, metavar='N', default=4, help='number of data loading workers')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='yolo3', help='model architecture',
        choices=['yolo3', 'yolo4s', 'yolo4m', 'yolo5_6n', 'yolo5_6s', 'yolo5_6m'])
    parser.add_argument('--test_image_size', type=int, default=448, help='image resolution for evaluation')

    # neutrino args
    parser.add_argument('-d', '--delta', type=float, metavar='DELTA', default=0.05, help='accuracy drop tolerance')
    parser.add_argument('--deepsearch', action='store_true', help='to consume the delta as much as possible')
    parser.add_argument('--dryrun', action='store_true', help='force all loops to early break')
    parser.add_argument('--fp16', action='store_true', help="export to fp16 as well if it is possible")
    parser.add_argument('--horovod', action='store_true', help='activate horovod')
    parser.add_argument('--device', type=str, metavar='DEVICE', default='GPU', help='Device to use, CPU or GPU',
                        choices=['GPU', 'CPU'])
    parser.add_argument('--bn_fuse', action='store_true', help='fuse batch normalization layers')
    parser.add_argument('--epochs', default=50, type=int, help='number of fine-tuning epochs')
    parser.add_argument('--lr', default=None, type=float, help='learning rate for training quantized model')
    parser.add_argument('--num_classes', type=int, default=None, help='Number of classes in dataset')
    parser.add_argument('--runtime_resolution', type=str, nargs='*', default=[], help='Image resolution at runtime, if different from train dataset. e.g. \'320x320\'')

    # quant args
    parser.add_argument('--conv11', action='store_true', help='if set true, will quantize conv1x1 layers also')
    parser.add_argument('--group_conv', action='store_true', help='if set true, will quantize group conv layers also')
    parser.add_argument('--skip_layers', nargs='*', default=[], help='indices of layers to be skipped according to model.flat_view(). ex: 2 6 8')
    parser.add_argument('--skip_layers_ratio', type=float, default=0.0, help='skip quantization of the first percentage of layers, 0.0-1.0')

    args = parser.parse_args()
    device_map = {'CPU': 'cpu', 'GPU': 'cuda'}

    arch_name = args.arch
    if not args.num_classes:
        if args.dataset == 'voc':
            args.num_classes = 20
        elif args.dataset == 'coco':
            args.num_classes = 80
    if args.dataset == 'wider_face':
        args.num_classes = 8

    data_splits = get_data_splits_by_name(
        data_root=args.data_root,
        dataset_name=args.dataset,
        model_name=args.arch,
        batch_size=args.batch_size,
        num_workers=args.workers,
        device=device_map[args.device],
        num_classes=args.num_classes
    )
    fp = TorchForwardPass(model_input_pattern=(0, '_', '_', '_'))

    reference_model = get_model_by_name(model_name=arch_name,
                                        dataset_name=args.dataset,
                                        pretrained=True,
                                        progress=True,
                                        device=device_map[args.device])

    for k, m in reference_model.named_modules():
        if isinstance(m, Detect):
            m.onnx_dynamic = True
            m.inplace = False

    # eval func
    eval_key = 'mAP'
    if args.dryrun:
        def eval_func(model, data_splits, device=None):
            return {eval_key: 1}
    else:
        eval_func = YOLOEval(model_name=args.arch, data_root=args.data_root,
                             dataset_type=args.dataset, test_img_size=args.test_image_size)

    # loss
    loss_cls = YOLOLoss
    loss_kwargs = {
        'device': device_map[args.device],
        'model': reference_model,  # subject to dataset
        'num_classes': args.num_classes
    }

    resolutions = get_runtime_resolution(args.runtime_resolution)

    config = {
        'task_type': '__custom__',
        'export': {
            'format': ['dlrt'],
            'kwargs': {'resolutions': resolutions, 'precision': 'fp16' if args.fp16 else 'fp32'}
        },
        'optimization': 'quantization',
        'deepsearch': args.deepsearch,
        'delta': args.delta,
        'device': args.device,
        'use_horovod': args.horovod,
        'bn_fusion': args.bn_fuse,
        'full_trainer': {
            'epochs': args.epochs,
            'eval_skip': 1,
            'eval_freq': 2,
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
        'custom_compression': {
            'quantization_args': {
                'quantize_conv11': args.conv11,
                'quantize_group': args.group_conv,
                'skip_layers': args.skip_layers,
                'skip_layers_ratio': args.skip_layers_ratio,
            }
        }
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
