import argparse
import os
import torch
import torch.nn as nn

from neutrino.framework.functions import LossFunction
from neutrino.framework.torch_framework import TorchFramework
from neutrino.framework.torch_nn import NativeOptimizerFactory, NativeSchedulerFactory
from deeplite.profiler import Device
from deeplite.torch_profiler.torch_data_loader import TorchForwardPass
from deeplite.torch_profiler.torch_inference import TorchEvaluationFunction
from neutrino.job import Neutrino
from neutrino.nlogger import getLogger

from deeplite_torch_zoo import get_data_splits_by_name, get_model_by_name
from deeplite_torch_zoo.wrappers.models.segmentation.unet import unet_carvana
from deeplite_torch_zoo.wrappers.eval import seg_eval_func

from deeplite_torch_zoo.src.segmentation.unet_scse.repo.src.losses.multi import MultiClassCriterion
from deeplite_torch_zoo.src.segmentation.unet_scse.repo.src.utils.scheduler import CosineWithRestarts
from deeplite_torch_zoo.src.segmentation.unet_scse.repo.src.losses.multi import MultiClassCriterion
from deeplite_torch_zoo.src.segmentation.fcn.solver import cross_entropy2d

logger = getLogger(__name__)


class UNetEval(TorchEvaluationFunction):
    def __init__(self, model_type):
        self.model_type = model_type
        self.eval_func = seg_eval_func

    def _compute_inference(self, model, data_loader, **kwargs):
        # silent **kwargs
        data_loader = data_loader.native_dl
        return self.eval_func(model=model, data_loader=data_loader, net=self.model_type)


class UNetLoss(LossFunction):
    def __init__(self, net, device='cuda'):
        self.torch_device = device
        if net == 'unet':
            self.criterion = nn.BCEWithLogitsLoss()
        elif 'unet_scse_resnet18' in net:
            self.criterion = MultiClassCriterion(loss_type='Lovasz', ignore_index=255)
        else:
            raise ValueError

    def __call__(self, model, data):
        imgs, true_masks, _ = data
        true_masks = true_masks.to(self.torch_device)

        imgs = imgs.to(self.torch_device)
        masks_pred = model(imgs)
        loss = self.criterion(masks_pred, true_masks)

        return {'loss': loss}


class UNetNativeOptimizerFactory(NativeOptimizerFactory):
    def __init__(self):
        self.lr = 0.001
        self.momentum = 0.9
        self.weight_decay = 1e-8

    def make(self, native_model):
        return torch.optim.RMSprop(native_model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                                   momentum=self.momentum)


class UNetNativeSchedulerFactory(NativeSchedulerFactory):
    def make(self, native_optimizer):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(native_optimizer, 'max', patience=10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model/dataset args
    parser.add_argument('--dataset', choices={'carvana', 'voc'}, default='voc',
                        help="Choose whether to use carvana or voc dataset. The model's architecture will be chosen accordingly.")
    parser.add_argument('--voc_path', default='/neutrino/datasets/VOCdevkit', help='voc data path.')
    parser.add_argument('--carvana_path', default='/neutrino/datasets/carvana/', help='carvana data path.')
    parser.add_argument('-b', '--batch_size', type=int, metavar='N', default=4, help='mini-batch size')
    parser.add_argument('-j', '--workers', type=int, metavar='N', default=4, help='number of data loading workers')
    parser.add_argument('--num_classes', type=int, default=20, help='number of classes to use (only for voc)')

    # neutrino args
    parser.add_argument('-d', '--delta', type=float, metavar='DELTA', default=0.02, help='metric drop tolerance')
    parser.add_argument('--deepsearch', action='store_true', help="to consume the delta as much as possible")
    parser.add_argument('--dryrun', action='store_true', help="force all loops to early break")
    parser.add_argument('--horovod', action='store_true', help="activate horovod")
    parser.add_argument('--device', type=str, metavar='DEVICE', default='GPU',
                        help='Device to use, CPU or GPU (however locked to GPU for now)',
                        choices=['GPU', 'CPU'])
    parser.add_argument('--bn_fuse', action='store_true', help="fuse batch normalization layers")

    args = parser.parse_args()
    device_map = {'CPU': 'cpu', 'GPU': 'cuda'}

    if args.dataset == 'carvana':
        print("Choosing carvana dataset")
        args.arch = 'unet'

        data_splits = get_data_splits_by_name(
            data_root=args.carvana_path,
            dataset_name='carvana',
            model_name=args.arch,
            batch_size=args.batch_size,
            num_workers=args.workers,
            device=device_map[args.device],
        )
        # data_splits = get_carvana_for_unet(args.carvana, args.batch_size, args.workers)
        teacher = unet_carvana(pretrained=True, progress=True)

        eval_key = 'dice_coeff'
    else:
        print("Choosing voc dataset")
        args.arch = 'unet_scse_resnet18'

        data_splits = get_data_splits_by_name(
            data_root=args.voc_path,
            dataset_name='voc',
            model_name=args.arch,
            batch_size=args.batch_size,
            num_classes=args.num_classes,
            num_workers=args.workers,
            backbone='vgg',
            sbd_root=None,
            device=device_map[args.device]
        )
        teacher = get_model_by_name(
            model_name=args.arch,
            dataset_name='voc',
            pretrained=True,
            progress=True,
            device=device_map[args.device])

        eval_key = 'miou'

    if args.dryrun:
        def eval_func(model, data_splits):
            return {eval_key: 1}
    else:
        eval_func = UNetEval(args.arch)

    framework = TorchFramework()
    fp = TorchForwardPass(model_input_pattern=(0, '_', '_'))

    # loss
    loss_cls = UNetLoss
    loss_kwargs = {'net': args.arch, 'device': device_map[args.device]}

    # custom config
    config = {'deepsearch': args.deepsearch,
              'delta': args.delta,
              'device': args.device,
              'use_horovod': args.horovod,
              'task_type': 'segmentation',
              'bn_fusion': args.bn_fuse,
              'full_trainer': {'eval_key': eval_key,
                               # uncomment these two below if you want to try other optimizer / scheduler
                               # 'optimizer': UNetNativeOptimizerFactory,
                               # 'scheduler': {'factory': UNetNativeSchedulerFactory, 'eval_based': False}
                               },
                'export':{'format': ['onnx']},
              }

    optimized_model = Neutrino(TorchFramework(),
                               data=data_splits,
                               model=teacher,
                               config=config,
                               eval_func=eval_func,
                               forward_pass=fp,
                               loss_function_cls=loss_cls,
                               loss_function_kwargs=loss_kwargs).run(dryrun=args.dryrun)
