
<p align="center">
  <img src="https://github.com/Deeplite/neutrino-examples/raw/master/deeplite-logo-color.png" />
</p>

# Deeplite Neutrino Examples

Public examples to use [Deeplite Neutrino™](https://docs.deeplite.ai/neutrino/index.html) engine for model architecture optimization.

> **_NOTE:_**  Make sure you have obtained either a free community license or a commercial support license, [from here](https://docs.deeplite.ai/neutrino/license.html)

## Classification Models

```{.python}
    python src/hello_neutrino_classifier.py --arch resnet18 --dataset cifar100 --delta 1
```

The pretrained architecture `models` and the `datasets` are loaded from the [Deeplite Torch Zoo](https://github.com/Deeplite/deeplite-torch-zoo). The `delta` of 1% denotes the maximum affordable reduction in the accuracy of the model during optimization. Feel free to play around with different classification models and datasets, along with different `delta` values to get different optimized results. The `arch` and the `datasets` can be customized with any native PyTorch pretrained model.

## Object Detection Models

Different object detection use-cases and examples are provided to play around with.

To optimize a `SSD-300` model on [`VOC 2007 dataset`](http://host.robots.ox.ac.uk/pascal/VOC/), run the following,

```{.python}
    python src/hello_neutrino_ssd.py --arch resnet18_ssd --delta 0.05 --dataset voc -r  ~/PATH/TO/VOCdevkit
```

To optimize a `SSD-300` model with `Mobilenet_v2` backend on [`COCO dataset`](https://cocodataset.org/#home), run the following,

```{.python}
    python src/hello_neutrino_ssd.py --arch mb2_ssd --delta 0.05 --dataset coco -r ~/PATH/TO/coco2017
```

To optimize a `YOLOv5` model on [`VOC 2007 dataset`](http://host.robots.ox.ac.uk/pascal/VOC/), run the following,

```{.python}
    python src/hello_neutrino_yolo.py --arch yolo5_6s --delta 0.05 --dataset voc -r ~/PATH/TO/VOCdevkit
```

To optimize a `YOLOv5` model on [`COCO dataset`](https://cocodataset.org/#home), run the following,

```{.python}
    python src/hello_neutrino_yolo.py --arch yolo5_6s --delta 0.05 --dataset coco -r ~/PATH/TO/coco2017
```

The `delta` of 0.05 denotes the maximum affordable reduction in the mAP (Mean Average Precision) of the model during optimization. Feel free to play around with different object detection models and datasets, along with different `delta` values to get different optimized results. The `arch` and the `datasets` can be customized with any native PyTorch pretrained model.

## Segmentation Models

To optimize a `U-Net` style model backend on [`VOC 2007 dataset`](http://host.robots.ox.ac.uk/pascal/VOC/), run the following,
```{.python}
    python src/hello_neutrino_unet.py --dataset voc --delta 0.02 --voc_path ~/PATH/TO/VOCdevkit
```

The `delta` of 0.02 denotes the maximum affordable reduction in the mIOU (Mean Intersection over Union) of the model during optimization. Feel free to play around with different segmentation models and datasets, along with different `delta` values to get different optimized results. The `arch` and the `datasets` can be customized with any native PyTorch pretrained model.


## Quantizing models for DLRT
To quantize a YOLO model for inference with DeepliteRT, run the following
```
python src/hello_neutrino_yolo_quantization.py --arch yolo5_6s --dataset voc -r ~/PATH/TO/VOCdevkit
```
To improve latency at the risk of weaker model accuracy, add the `--conv11` flag to quantize 1x1 convolutions.
To improve the accuracy, try using the `--skip_layers_ratio` argument to skip quantization of the first x% of the convolution layers

If your model will process a different image resolution at runtime, pass it with `--runtime_resolution HxW` arguemnt. Ex: `--runtime_resolution 320x320`. This way the exported model
will accept images of the correct resolution.


## Distributed Training
To setup Horovod for training with multiple GPUs, see the guide in the [documentation](https://docs.deeplite.ai/neutrino/samples.html?highlight=distributed#running-on-multi-gpu-on-a-single-machine) and make sure to pass argument `--horovod` to the example script. Note that Neutrino scales the learning rate by the number of GPUs, so passing a learning rate of 0.1 to Neutrino with 4 GPUs will apply a learning rate of 0.4 to the optimizer. 
