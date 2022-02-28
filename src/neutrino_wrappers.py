from pathlib import Path

from pycocotools.coco import COCO
from neutrino.framework.functions import LossFunction
from deeplite.torch_profiler.torch_inference import TorchEvaluationFunction

from deeplite_torch_zoo.wrappers.eval import vgg16_ssd_eval_func
from deeplite_torch_zoo.src.objectdetection.ssd.repo.vision.nn.multibox_loss import (
    MultiboxLoss,
)
from deeplite_torch_zoo.src.objectdetection.ssd.config.vgg_ssd_config import VGG_CONFIG
from deeplite_torch_zoo.src.objectdetection.ssd.config.mobilenetv1_ssd_config import (
    MOBILENET_CONFIG,
)

from deeplite_torch_zoo.wrappers.eval import yolo_eval_voc
from deeplite_torch_zoo.wrappers.eval import mb2_ssd_eval_func
from deeplite_torch_zoo.src.objectdetection.yolov5.models.yolov5_loss import YoloV5Loss
from deeplite_torch_zoo.wrappers.eval import yolo_eval_coco

from deeplite_torch_zoo.wrappers.models.segmentation.unet import unet_carvana
from deeplite_torch_zoo.wrappers.eval import seg_eval_func

from deeplite_torch_zoo.src.segmentation.unet_scse.repo.src.losses.multi import (
    MultiClassCriterion,
)

from deeplite_torch_zoo.src.segmentation.unet_scse.repo.src.losses.multi import (
    MultiClassCriterion,
)
from deeplite_torch_zoo.src.segmentation.fcn.solver import cross_entropy2d


class YOLOEval(TorchEvaluationFunction):
    def __init__(self, net, data_root):
        self.net = net
        self.data_root = data_root

    def _compute_inference(self, model, data_loader, **kwargs):
        # silent **kwargs
        return {k: 100.0 * v for k, v in yolo_eval_voc(model=model, data_root=self.data_root,
            net=self.net).items()}


class YOLOCOCOEval(TorchEvaluationFunction):
    def __init__(self, net, data_root):
        self.net = net
        self.data_root = data_root
        self._gt = COCO(Path(self.data_root) / "annotations/instances_val2017.json")

    def _compute_inference(self, model, data_loader, **kwargs):
        # silent **kwargs
        return {k: 100.0 * v for k, v in yolo_eval_coco(
            model=model, gt=self._gt, data_root=self.data_root, net=self.net
        ).items()}


class YOLOLoss(LossFunction):
    def __init__(self, num_classes=20, model=None, device="cuda"):
        self.criterion = YoloV5Loss(num_classes=num_classes, model=model, device=device)
        self.torch_device = device

    def __call__(self, model, data):
        imgs, targets, labels_length, imgs_id = data
        _img_size = imgs.shape[-1]

        imgs = imgs.to(self.torch_device)
        p, p_d = model(imgs)
        _, loss_giou, loss_conf, loss_cls = self.criterion(
            p, p_d, targets, labels_length, _img_size
        )

        return {"lgiou": loss_giou, "lconf": loss_conf, "lcls": loss_cls}


class SSDVOCEval(TorchEvaluationFunction):
    def _compute_inference(self, model, data_loader, **kwargs):
        # silent **kwargs
        data_loader = data_loader.native_dl
        return {k: 100.0 * v for k, v in vgg16_ssd_eval_func(model=model,
            data_loader=data_loader).items()}


class SSDVOCLoss(LossFunction):
    def __init__(self, device="cuda"):
        config = VGG_CONFIG()
        self.torch_device = device
        self.criterion = MultiboxLoss(
            config.priors,
            iou_threshold=0.5,
            neg_pos_ratio=3,
            center_variance=0.1,
            size_variance=0.2,
            device=device,
        )

    def __call__(self, model, data):
        model.to(self.torch_device)
        images, boxes, labels = data
        images, boxes, labels = (
            images.to(self.torch_device),
            boxes.to(self.torch_device),
            labels.to(self.torch_device),
        )

        confidence, locations = model(images)
        regression_loss, classification_loss = self.criterion(
            confidence, locations, labels, boxes
        )
        loss = {
            "regression_loss": regression_loss,
            "classification_loss": classification_loss,
        }
        return loss


class SSDCOCOEval(TorchEvaluationFunction):
    def __init__(self, net, data_root):
        self.data_root = data_root

    def _compute_inference(self, model, data_loader, **kwargs):
        # silent **kwargs
        cocoGt = COCO(f"{self.data_root}/ annotations/instances_val2017.json")
        data_loader = data_loader.native_dl
        return {k: 100.0 * v for k, v in mb2_ssd_eval_func(model=model,
            data_loader=data_loader, gt=cocoGt).items()}


class SSDCOCOLoss(LossFunction):
    def __init__(self, device="cuda"):
        config = MOBILENET_CONFIG()
        self.torch_device = device
        self.criterion = MultiboxLoss(
            config.priors,
            iou_threshold=0.5,
            neg_pos_ratio=3,
            center_variance=0.1,
            size_variance=0.2,
            device=device,
        )


class UNetEval(TorchEvaluationFunction):
    def __init__(self, model_type):
        self.model_type = model_type
        self.eval_func = seg_eval_func

    def _compute_inference(self, model, data_loader, **kwargs):
        # silent **kwargs
        data_loader = data_loader.native_dl
        return self.eval_func(model=model, data_loader=data_loader, net=self.model_type)


class UNetLoss(LossFunction):
    def __init__(self, net, device="cuda"):
        self.torch_device = device
        if net == "unet":
            self.criterion = nn.BCEWithLogitsLoss()
        elif "unet_scse_resnet18" in net:
            self.criterion = MultiClassCriterion(loss_type="Lovasz", ignore_index=255)
        else:
            raise ValueError

    def __call__(self, model, data):
        imgs, true_masks, _ = data
        true_masks = true_masks.to(self.torch_device)

        imgs = imgs.to(self.torch_device)
        masks_pred = model(imgs)
        loss = self.criterion(masks_pred, true_masks)

        return {"loss": loss}
