import argparse
import os

from neutrino.framework.functions import LossFunction
from neutrino.framework.torch_framework import TorchFramework
from deeplite.torch_profiler.torch_data_loader import TorchForwardPass
from deeplite.torch_profiler.torch_inference import TorchEvaluationFunction
from neutrino.job import Neutrino
from neutrino.nlogger import getLogger
from deeplite_torch_zoo.wrappers.wrapper import get_data_splits_by_name, get_model_by_name
from deeplite_torch_zoo.wrappers.eval import yolo_eval_voc
from deeplite_torch_zoo.src.objectdetection.yolov5.models.yolov5_loss import \
    YoloV5Loss


logger = getLogger(__name__)


class YOLOEval(TorchEvaluationFunction):
    def __init__(self, net, data_root):
        self.net = net
        self.data_root = data_root

    def _compute_inference(self, model, data_loader, **kwargs):
        # silent **kwargs
        return yolo_eval_voc(model=model, data_root=self.data_root, net=self.net)


class YOLOLoss(LossFunction):
    def __init__(self, num_classes=20, device='cuda', model=None):
        # when num classes is not provided to YoloV3Loss it uses 20 as the default.
        # that's okay here because the whole file assumes voc dataset for testing.
        self.criterion = YoloV5Loss(num_classes=num_classes, model=model, device=device)
        self.torch_device = device

    def __call__(self, model, data):
        imgs, targets, labels_length, imgs_id = data
        _img_size = imgs.shape[-1]

        imgs = imgs.to(self.torch_device)
        p, p_d = model(imgs)
        _, loss_giou, loss_conf, loss_cls = self.criterion(p, p_d, targets, labels_length, _img_size)

        return {'lgiou': loss_giou, 'lconf': loss_conf, 'lcls': loss_cls}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model/dataset args
    parser.add_argument('--voc_path', default='/neutrino/datasets/VOCdevkit/',
                        help='vockit data path contains VOC2007 and VOC2012.')
    parser.add_argument('-b', '--batch_size', type=int, metavar='N', default=8, help='mini-batch size')
    parser.add_argument('-j', '--workers', type=int, metavar='N', default=4, help='number of data loading workers')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='yolo5m', help='model architecture',
        choices=['yolo4s', 'yolo4m', 'yolo5_6s', 'yolo5_6m', 'yolo5_6l', 'yolo5_6x'])

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
        data_root=args.voc_path,
        dataset_name='voc',
        model_name=args.arch,
        batch_size=args.batch_size,
        num_workers=args.workers,
        device=device_map[args.device],

    )
    fp = TorchForwardPass(model_input_pattern=(0, '_', '_', '_'))

    reference_model = get_model_by_name(model_name=args.arch,
                                        dataset_name='voc_20',
                                        pretrained=True,
                                        progress=True,
                                        device=device_map[args.device],)

    # eval func
    eval_key = 'mAP'
    if args.dryrun:
        def eval_func(model, data_splits):
            return {eval_key: 1}
    else:
        eval_func = YOLOEval(net=args.arch, data_root=os.path.join(args.voc_path, 'VOC2007'))

    # loss
    loss_cls = YOLOLoss
    loss_kwargs = {'device': device_map[args.device], 'model': reference_model}

    # custom config
    config = {
        'deepsearch': args.deepsearch,
        'delta': args.delta,
        'device': args.device,
        'use_horovod': args.horovod,
        'task_type': 'object_detection',
        'bn_fusion': args.bn_fuse,
        'export': {'format': ['onnx']},
    }

    optimized_model = Neutrino(TorchFramework(),
                               data=data_splits,
                               model=reference_model,
                               config=config,
                               eval_func=eval_func,
                               forward_pass=fp,
                               loss_function_cls=loss_cls,
                               loss_function_kwargs=loss_kwargs).run(dryrun=args.dryrun)
