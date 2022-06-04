# Mean Teacher 3D UNet Model for Semi-supervised Segmentation

This repo is a 3d variant of [Marwan's 2D implementation](https://github.com/marwankefah/mean-teacher-medical-imaging), thanks to his inspiration.

A brief structure of the model is shown in the following figure:
![meanteacher](https://s3.bmp.ovh/imgs/2022/06/04/48b91dc4d17449de.png)

This framework takes labeled and unlabeled images as input and introduces random noise for contamination, respectively.   
The uncontaminated original input images will predict the results by a Student model composed of an ordinary UNet, while the contaminated data will predict the other set of results by a Teacher model with exactly the same structure.   
For the labeled data, the Student model is supervised by ground truth on the one hand and by the consistency constraint of the predicted results of the contaminated data on the other hand, while for the unlabeled data, only their consistency loss is used for supervision.   
The parameters of the $M_T$ are then periodically updated from the $M_S$ by exponential moving average (EMA).

## How to use
For training, you just have to run `train.py`:
```
python train.py
```
But before, you need to change three folder paths:
* `labeled_data_folder`: is where you save your labeled data
* `unlabeled_data_folder`: is where you save your unlabeled data
* `model_folder`: is where you wish to save the model params

And one parameter:
* `num_classes`: depends on your dataset

Oh, you may also have to modify the `CUDA_VISIBLE_DEVICES` at the beginning of the file.

For inference, you just have to run `predict.py` with input and output folders:
```
python predict.py -i YourTestImageFolder -o ResultSaveFolder
```

## Something important
In order to improve the efficiency, we set the target `Spacing` of the images in the pre-processing process to be relatively large and the `patch-size` of the data entering the model to be relatively small in this example model.
This compromise will inevitably lead to a decrease in accuracy, you can adjust these two parameters according to your needs to get better results.

