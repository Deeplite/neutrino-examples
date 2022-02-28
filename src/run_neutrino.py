import os
import re
import argparse

from neutrino.job import Neutrino
from neutrino.nlogger import getLogger
from neutrino.framework.torch_framework import TorchFramework
from deeplite.torch_profiler.torch_data_loader import TorchForwardPass

from deeplite_torch_zoo.wrappers.wrapper import (
    get_data_splits_by_name,
    get_model_by_name,
)

from neutrino_wrappers import YOLOEval, YOLOCOCOEval, YOLOLoss, SSDVOCEval, \
    SSDVOCLoss, SSDCOCOEval, SSDCOCOLoss, UNetEval, UNetLoss


logger = getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model/dataset args
    parser.add_argument(
        "--dataset", metavar="DATASET", default="cifar100", help="dataset to use"
    )
    parser.add_argument(
        "-r", "--data_root", metavar="PATH", default="", help="dataset data root path"
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, metavar="N", default=128, help="mini-batch size"
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        metavar="N",
        default=4,
        help="number of data loading workers",
    )
    parser.add_argument(
        "-a", "--arch", metavar="ARCH", default="resnet18", help="model architecture"
    )
    parser.add_argument(
        "--annotation-file", default="annotations/instances_val2017.json"
    )

    # neutrino args
    parser.add_argument(
        "-d",
        "--delta",
        type=float,
        metavar="DELTA",
        default=1,
        help="accuracy drop tolerance",
    )
    parser.add_argument(
        "-l", "--level", type=int, default=1, help="level", choices=(1, 2)
    )
    parser.add_argument(
        "-o",
        "--optimization",
        type=str,
        default="compression",
        choices=("compression", "latency"),
    )
    parser.add_argument(
        "--deepsearch",
        action="store_true",
        help="to consume the delta as much as possible",
    )
    parser.add_argument(
        "--fp16", action="store_true", help="export to fp16 as well if it is possible"
    )
    parser.add_argument(
        "--dryrun", action="store_true", help="force all loops to early break"
    )
    parser.add_argument("--horovod", action="store_true", help="activate horovod")
    parser.add_argument(
        "--bn_fuse", action="store_true", help="fuse batch normalization layers"
    )
    parser.add_argument(
        "--device",
        type=str,
        metavar="DEVICE",
        default="GPU",
        help="Device to use, CPU or GPU",
    )

    args = parser.parse_args()
    device_map = {"CPU": "cpu", "GPU": "cuda"}

    DATASET_NAME_MAP = {
        "voc": "voc_20",
        "coco": "coco_80",
    }

    data_splits = get_data_splits_by_name(
        dataset_name=args.dataset,
        model_name=args.arch,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.workers,
        device=device_map[args.device],
    )

    reference_model = get_model_by_name(
        model_name=args.arch,
        dataset_name=args.dataset
        if args.dataset not in DATASET_NAME_MAP
        else DATASET_NAME_MAP[args.dataset],
        pretrained=True,
        progress=True,
        device=device_map[args.device],
    )

    config = {
        "deepsearch": args.deepsearch,
        "level": args.level,
        "delta": args.delta,
        "device": args.device,
        "optimization": args.optimization,
        "use_horovod": args.horovod,
        "bn_fusion": args.bn_fuse,
        "export": {"format": ["onnx"]},
        "onnx_precision": "fp16" if args.fp16 else "fp32",
    }

    def simplify_model_name(model_name):
        MODEL_NAME_SUBSTRINGS = ["yolo", "unet", "ssd"]
        for substring in MODEL_NAME_SUBSTRINGS:
            if re.search(substring, model_name):
                return substring
        return model_name

    NEUTRINO_KWARG_DICT = {
        ('ssd', 'voc'): {
            'eval_func_cls': SSDVOCEval,
            'eval_func_kwargs': {},
            'forward_pass': TorchForwardPass(model_input_pattern=(0, "_", "_")),
            'loss_function_cls': SSDVOCLoss,
            'loss_function_kwargs': {"device": device_map[args.device]},
            'task_type': 'object_detection',
        },
        ('ssd', 'coco'): {
            'eval_func_cls': SSDCOCOEval,
            'eval_func_kwargs': {'data_root': args.data_root},
            'forward_pass': TorchForwardPass(model_input_pattern=(0, "_", "_", "_")),
            'loss_function_cls': SSDCOCOLoss,
            'loss_function_kwargs': {"device": device_map[args.device]},
            'task_type': 'object_detection',
        },
        ('yolo', 'voc'): {
            'eval_func_cls': YOLOEval,
            'eval_func_kwargs': {'net': args.arch, 'data_root': os.path.join(args.data_root, "VOC2007")},
            'forward_pass': TorchForwardPass(model_input_pattern=(0, "_", "_", "_")),
            'loss_function_cls': YOLOLoss,
            'loss_function_kwargs': {"device": device_map[args.device], "model": reference_model},
            'task_type': 'object_detection',
        },
        ('yolo', 'coco'): {
            'eval_func_cls': YOLOCOCOEval,
            'eval_func_kwargs': {'net': args.arch, 'data_root': args.data_root},
            'forward_pass': TorchForwardPass(model_input_pattern=(0, "_", "_", "_")),
            'loss_function_cls': YOLOLoss,
            'loss_function_kwargs': {"device": device_map[args.device], "model": reference_model},
            'task_type': 'object_detection',
        },
    }

    neutrino_kwargs = {}
    model_dataset_key = (simplify_model_name(args.arch), args.dataset)
    if model_dataset_key in NEUTRINO_KWARG_DICT:
        neutrino_kwargs = NEUTRINO_KWARG_DICT[model_dataset_key]
        neutrino_kwargs['eval_func'] = neutrino_kwargs['eval_func_cls'](**neutrino_kwargs['eval_func_kwargs'])
        config["task_type"] = neutrino_kwargs['task_type']

        del neutrino_kwargs['task_type']
        del neutrino_kwargs['eval_func_cls']
        del neutrino_kwargs['eval_func_kwargs']

    optimized_model = Neutrino(
        framework=TorchFramework(),
        data=data_splits,
        model=reference_model,
        config=config,
        **neutrino_kwargs,
    ).run(dryrun=args.dryrun)
