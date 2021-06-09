import argparse

from neutrino.framework.torch_framework import TorchFramework
from neutrino.job import Neutrino

from deeplite_torch_zoo.wrappers.wrapper import get_data_splits_by_name, get_model_by_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model/dataset args
    parser.add_argument('--dataset', metavar='DATASET', default='cifar100', help='dataset to use')
    parser.add_argument('-r', '--data_root', metavar='PATH', default='', help='dataset data root path')
    parser.add_argument('-b', '--batch_size', type=int, metavar='N', default=128, help='mini-batch size')
    parser.add_argument('-j', '--workers', type=int, metavar='N', default=4, help='number of data loading workers')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', help='model architecture')

    # neutrino args
    parser.add_argument('-d', '--delta', type=float, metavar='DELTA', default=1, help='accuracy drop tolerance')
    parser.add_argument('-l', '--level', type=int, default=1, help='level', choices=(1, 2))
    parser.add_argument('-o', '--optimization', type=str, default='compression', choices=('compression', 'latency'))
    parser.add_argument('--deepsearch', action='store_true', help="to consume the delta as much as possible")
    parser.add_argument('--fp16', action='store_true', help="export to fp16 as well if it is possible")
    parser.add_argument('--dryrun', action='store_true', help="force all loops to early break")
    parser.add_argument('--horovod', action='store_true', help="activate horovod")
    parser.add_argument('--bn_fuse', action='store_true', help="fuse batch normalization layers")
    parser.add_argument('--device', type=str, metavar='DEVICE', default='GPU', help='Device to use, CPU or GPU')

    args = parser.parse_args()
    device_map = {'CPU': 'cpu', 'GPU': 'cuda'}

    data_splits = get_data_splits_by_name(dataset_name=args.dataset,
                                          data_root=args.data_root,
                                          batch_size=args.batch_size,
                                          num_workers=args.workers,
                                          device=device_map[args.device])

    reference_model = get_model_by_name(model_name=args.arch,
                                        dataset_name=args.dataset,
                                        pretrained=True,
                                        progress=True,
                                        device=device_map[args.device])

    config = {'deepsearch': args.deepsearch,
              'level': args.level,
              'delta': args.delta,
              'device': args.device,
              'onnx_precision': 'fp16' if args.fp16 else 'fp32',
              'optimization': args.optimization,
              'use_horovod': args.horovod,
              'bn_fusion': args.bn_fuse,
              }

    optimized_model = Neutrino(framework=TorchFramework(),
                               data=data_splits,
                               model=reference_model,
                               config=config).run(dryrun=args.dryrun)
