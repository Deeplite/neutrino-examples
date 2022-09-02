from neutrino.training.external import ExternalTrainingLoop
from .train_yolo import WORLD_SIZE, train, parse_opt


class YoloTrainingLoop(ExternalTrainingLoop):
    def __init__(self, config, native_data_splits=None):
        opt_parser = parse_opt()
        opt = opt_parser.parse_args(args=[
            '--img-dir', str(config.data_root),
            '--batch-size', str(config.batch_size),
            '--eval-freq', str(config.eval_freq),
            '--weight_path', './quant_yolo_tmp',  # path to dump the loop's intermediate weights
            '--checkpoint_save_freq', str(config.eval_freq),
            '--dataset', str(config.dataset),
            '--net', str(config.arch),
            '--hp_config', str(config.hp_config),
            '--test_img_res', str(config.test_img_size) if isinstance(config.test_img_size, int) else 0,
            '--train_img_res', str(config.img_size) if isinstance(config.img_size, int) else 0,
            '--device=%s' % ','.join(map(lambda s: str(s), range(WORLD_SIZE))),
            '--epochs', str(config.epochs)
        ])
        super().__init__(self._train_yolo, opt)
        self.ft_epochs = config.ft_epochs
        self.native_data_splits = native_data_splits

    def _train_yolo(self, model, args):
        return train(args, model, dataset_splits=None)  # reload data splits for dist traniing

    def modify_args_for_finetuning(self, args):
        args.epochs = self.ft_epochs
        return args

    def modify_args_for_epochs(self, args, epochs):
        args.epochs = epochs
        return args

    def modify_args_for_validation(self, args, validation):
        args.noval = not validation
        args.nosave = not validation
        return args
