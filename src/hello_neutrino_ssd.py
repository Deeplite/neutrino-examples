import argparse
from pathlib import Path

from deeplite.torch_profiler.torch_data_loader import TorchForwardPass
from deeplite.torch_profiler.torch_inference import TorchEvaluationFunction
from deeplite_torch_zoo import (get_data_splits_by_name, get_eval_function,
                                get_model_by_name)
from deeplite_torch_zoo.src.objectdetection.ssd.config.mobilenetv1_ssd_config import \
    MOBILENET_CONFIG
from deeplite_torch_zoo.src.objectdetection.ssd.config.vgg_ssd_config import \
    VGG_CONFIG
from deeplite_torch_zoo.src.objectdetection.ssd.repo.vision.nn.multibox_loss import \
    MultiboxLoss
from neutrino.framework.functions import LossFunction
from neutrino.framework.torch_framework import TorchFramework
from neutrino.job import Neutrino
from neutrino.nlogger import getLogger
from pycocotools.coco import COCO

logger = getLogger(__name__)


class Eval(TorchEvaluationFunction):
    def __init__(self, model_name, dataset_name, data_root, num_classes):
        self._evaluation_fn = get_eval_function(model_name=model_name, dataset_name=dataset_name)
        self._gt = None
        self._data_root = data_root
        if dataset_name == 'coco':
            self._gt = COCO(Path(self._data_root) / "annotations/instances_val2017.json")
        self._num_classes = num_classes

    def _compute_inference(self, model, data_loader, **kwargs):
        # silent **kwargs
        data_loader = data_loader.native_dl
        eval_kwargs = {'model': model, 'data_loader': data_loader}
        if self._gt:
            eval_kwargs['gt'] = self._gt
        return self._evaluation_fn(**eval_kwargs)


class SSDLoss(LossFunction):
    def __init__(self, model_name, device="cuda"):
        if 'vgg' or 'resnet' in model_name:
            config = VGG_CONFIG()
        elif 'mb' in model_name:
            config = MOBILENET_CONFIG()
        else:
            raise ValueError('example only supports mobilenet, vgg based ssd models')
        self.torch_device = device
        self.criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                                      center_variance=0.1, size_variance=0.2, device=device)

    def __call__(self, model, data):
        model.to(self.torch_device)
        images, boxes, labels = data[0], data[1], data[2]
        images, boxes, labels = images.to(self.torch_device), boxes.to(self.torch_device), labels.to(self.torch_device)

        confidence, locations = model(images)
        regression_loss, classification_loss = self.criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        loss = {'regression_loss': regression_loss, 'classification_loss': classification_loss}
        return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model/dataset args
    parser.add_argument('-r', '--data_root', default='/neutrino/datasets/coco2017/', help='path to the dataset')
    parser.add_argument('--dataset', default='coco', choices=['coco', 'voc'])
    # parser.add_argument('--annotation-file', default='annotations/instances_val2017.json',
        # choices=['test_data_COCO.json', 'annotations/instances_val2017.json'])
    parser.add_argument('-b', '--batch_size', type=int, metavar='N', default=8, help='mini-batch size')
    parser.add_argument('-j', '--workers', type=int, metavar='N', default=4, help='number of data loading workers')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='mb2_ssd',
        choices=('mb2_ssd', 'mb1_ssd', 'mb2_ssd_lite', 'vgg16_ssd', 'resnet18_ssd'),
        help='model architecture, coco only supported for mb2_ssd, other choices for VOC')

    # neutrino args
    parser.add_argument('-d', '--delta', type=float, metavar='DELTA', default=0.05, help='accuracy drop tolerance')
    parser.add_argument('--deepsearch', action='store_true', help="to consume the delta as much as possible")
    parser.add_argument('--dryrun', action='store_true', help="force all loops to early break")
    parser.add_argument('--fp16', action='store_true', help="export to fp16 as well if it is possible")
    parser.add_argument('--horovod', action='store_true', help="activate horovod")
    parser.add_argument('--device', type=str, metavar='DEVICE', default='GPU', help='Device to use, CPU or GPU',
                        choices=['GPU', 'CPU'])
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate for training model. This LR is internally scaled by num gpus during distributed training')
    parser.add_argument('--ft_lr', default=0.001, type=float, help='learning rate during fine-tuning iterations')
    parser.add_argument('--ft_epochs', default=1, type=int, help='number of fine-tuning epochs')

    args = parser.parse_args()
    device_map = {'CPU': 'cpu', 'GPU': 'cuda'}

    data_splits = get_data_splits_by_name(
        data_root=args.data_root,
        dataset_name=args.dataset,
        model_name=args.arch,
        batch_size=args.batch_size,
        num_workers=args.workers,
        device=device_map[args.device],
    )

    if args.dataset == 'voc':
        fp = TorchForwardPass(model_input_pattern=(0, '_', '_'))
        num_classes = 20
    else:
        fp = TorchForwardPass(model_input_pattern=(0, '_', '_', '_'))
        num_classes = 80

    reference_model = get_model_by_name(
        model_name=args.arch,
        dataset_name=args.dataset,
        pretrained=True,
        progress=True,
        device=device_map[args.device],
    )

    # eval func
    eval_key = 'mAP'
    if args.dryrun:
        def eval_func(model, data_splits):
            return {eval_key: 1}
    else:
        eval_func = Eval(model_name=args.arch, dataset_name=args.dataset,
                         data_root=args.data_root, num_classes=num_classes)

    # loss
    loss_cls = SSDLoss
    loss_kwargs = {'device': device_map[args.device], 'model_name': args.arch}

    # custom config
    config = {
        'deepsearch': args.deepsearch,
        'delta': args.delta,
        'device': args.device,
        'use_horovod': args.horovod,
        'task_type': 'object_detection',
        'export': {
            'format': 'onnx',
            'kwargs': {'precision': 'fp16' if args.fp16 else 'fp32'}
        },
        'full_trainer': {'optimizer': {'name': 'SGD','lr': args.lr}},
        'fine_tuner': {
            'loop_params': {
                'epochs': args.ft_epochs,
                'optimizer': {'lr': args.ft_lr}
            }
        }
    }

    optimized_model = Neutrino(TorchFramework(),
                               data=data_splits,
                               model=reference_model,
                               config=config,
                               eval_func=eval_func,
                               forward_pass=fp,
                               loss_function_cls=loss_cls,
                               loss_function_kwargs=loss_kwargs).run(dryrun=args.dryrun)
