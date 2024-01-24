# Training-free image style alignment (TISA)
## Introduction
![Alt text](pics/intro.png)
Handheld ultrasound devices face usage limitations due to user inexperience and cannot benefit from supervised deep learning without extensive expert annotations. Moreover, the models trained on standard ultrasound device data are constrained by training data distribution and perform poorly when directly applied to handheld device data. In this study, we propose the Training-free Image Style Alignment (TISA) framework to align the style of handheld device data to those of standard devices. The proposed TISA can directly infer handheld device images without extra training and is suited for clinical applications. We show that TISA performs better and more stably in medical detection and segmentation tasks for handheld device data. We further validate TISA as the clinical model for automatic measurements of spine curvature and carotid intima-media thickness, which agrees well with manual results. We demonstrate the potential for TISA to facilitate automatic diagnosis on handheld ultrasound devices and expedite their eventual widespread use.

## Step 1: Train models on source data
![Alt text](pics/step1.png)
For spine ultrasound data, run
```
python ddpm/ddpm_train.py --data_dir
python controlnet/controlnet_train.py --data_dir
python detection/det_train.py --data_dir --seed RANDOM_SEED
```
For carotid ultrasound data, run
```
python ddpm/ddpm_train.py --data_dir
python segmentation/det_train.py --data_dir --seed RANDOM_SEED
```

## Step 2: Training-free alignment for target data
![Alt text](pics/step2.png)
