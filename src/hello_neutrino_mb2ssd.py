import argparse
import os

from pycocotools.coco import COCO

from neutrino.framework.functions import LossFunction
from neutrino.framework.torch_framework import TorchFramework
from neutrino.framework.torch_profiler.torch_data_loader import TorchForwardPass
from neutrino.framework.torch_profiler.torch_inference import TorchEvaluationFunction
from neutrino.job import Neutrino
from neutrino.nlogger import getLogger
from deeplite_torch_zoo.wrappers.wrapper import get_data_splits_by_name, get_model_by_name
from deeplite_torch_zoo.wrappers.eval import mb2_ssd_eval_func
from deeplite_torch_zoo.src.objectdetection.mb_ssd.repo.vision.nn.multibox_loss import MultiboxLoss
from deeplite_torch_zoo.src.objectdetection.mb_ssd.config.mobilenetv1_ssd_config import MOBILENET_CONFIG



logger = getLogger(__name__)


class Eval(TorchEvaluationFunction):
    def _compute_inference(self, model, data_loader, **kwargs):
        # silent **kwargs
        cocoGt = COCO(f"{args.data_root}/{args.annotation_file}")
        data_loader = data_loader.native_dl
        return mb2_ssd_eval_func(model=model, data_loader=data_loader, gt=cocoGt)


class SSDLoss(LossFunction):
    def __init__(self, device="cuda"):
        config = MOBILENET_CONFIG()
        self.torch_device = device
        self.criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                                      center_variance=0.1, size_variance=0.2, device=device)

    def __call__(self, model, data):
        model.to(self.torch_device)
        images, boxes, labels, _ = data
        images, boxes, labels = images.to(self.torch_device), boxes.to(self.torch_device), labels.to(self.torch_device)

        confidence, locations = model(images)
        regression_loss, classification_loss = self.criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        loss = {'regression_loss': regression_loss, 'classification_loss': classification_loss}
        return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model/dataset args
    parser.add_argument('--data-root', default='/home/ehsan/data/', help='path to the dataset')
    parser.add_argument('--dataset-type', default='coco_gm', choices=['coco_gm', 'coco'])
    parser.add_argument('--annotation-file', default='test_data_COCO.json',
        choices=['test_data_COCO.json', 'annotations/instances_val2017.json'])
    parser.add_argument('-b', '--batch_size', type=int, metavar='N', default=8, help='mini-batch size')
    parser.add_argument('-j', '--workers', type=int, metavar='N', default=4, help='number of data loading workers')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='mb2_ssd', help='model architecture')

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
        data_root=args.data_root,
        dataset_name=args.dataset_type,
        model_name=args.arch,
        batch_size=args.batch_size,
        num_workers=args.workers,
        device=device_map[args.device],

    )
    fp = TorchForwardPass(model_input_pattern=(0, '_', '_', '_'))
    num_classes = data_splits['train'].dataset.num_classes - 1
    reference_model = get_model_by_name(model_name=args.arch,
                                        dataset_name=f'{args.dataset_type}_{num_classes}',
                                        pretrained=True,
                                        progress=True,
                                        device=device_map[args.device],)

    # eval func
    eval_key = 'mAP'
    if args.dryrun:
        def eval_func(*args, **kwargs):
            return {eval_key: 1}
    else:
        eval_func = Eval()

    # loss
    loss_cls = SSDLoss
    loss_kwargs = {'device': device_map[args.device]}

    # custom config
    config = {'deepsearch': args.deepsearch,
              'delta': args.delta,
              'device': args.device,
              'use_horovod': args.horovod,
              'task_type': 'object_detection',
              'bn_fusion': args.bn_fuse,
              }

    optimized_model = Neutrino(TorchFramework(),
                               data=data_splits,
                               model=reference_model,
                               config=config,
                               eval_func=eval_func,
                               forward_pass=fp,
                               loss_function_cls=loss_cls,
                               loss_function_kwargs=loss_kwargs).run(dryrun=args.dryrun)
