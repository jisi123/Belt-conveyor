Accurate location method of idler for belt conveyor faults based on visual and auditory dual mode fusion
As one of the key components of the belt conveyor, the safe and stable operation of the idler plays a vital role in ensuring the continuity and efficiency of the bulk conveying system. Therefore, an accurate fault location method for idler by fusion of dual-modal information of image and audio was proposed. Resnet and VGGish were used to extract image features and audio features respectively. An audiovisual fusion module was designed to realize alignment and fusion of the two features, and finally pixel classification was used to realize pixel-level segmentation of the faulty idler.________________________________________
Key Features
Belt Conveyor, Idler Fault, Multimodal Detectio
The detection effect images can be viewed in \network\avs_ms3\test_logs\MS3_resnet_20241216-154040\pred_masks
________________________________________
Instructions
Test the code by following these steps: Download the data set and store it in the data file, making sure it has a.cvs directory file; Put the backbone extraction network into the pretrained file, the download link is shown below
Data preparation
1.	dataset
This data set was collected from the self-built experimental platform. Please download the public data set from here: https://pan.baidu.com/s/1qMD511uPA6ULsuBPvvhOqQ?pwd=6v6q
The production and labeling of the idler fault data set for training and testing can be downloaded here: https://pan.baidu.com/s/11zCxpBRsnf5P44pGaQUDEQ?pwd=t4tu
Download the data set and place it in the data file directory
2.	backbones
The pretrained ResNet50(vision) and VGGish (audio) backbones can be downloaded from here and placed to the directory pretrained.
https://pan.baidu.com/s/1XCxI3THok2Kdne-t3o5OnA?pwd=22qt
3.	train wights
If you don't want to train the weights from scratch, this location provides the training weights, placed in change test.sh at the specified location –weights
https://pan.baidu.com/s/1GJlftVfbeqALWjDvtnLh3A?pwd=mdtr
________________________________________
4.	setting
•	Train Model
cd network /avs_s3
bash train.sh
python train.py 
--session_name "MS3_resnet"\
--visual_backbone "resnet"\
--max_epoches 30\
--train_batch_size 1\
--lr 0.0001\
--tpavi_stages 0 1 2 3\
--tpavi_va_flag\
--masked_av_flag\
--masked_av_stages 0 1 2 3 \
--lambda_1 0.5 --kl_flag\
•	Test Model
cd network /avs_s3
bash test.sh
python test.py
--session_name "MS3_resnet"\
--visual_backbone "resnet"\
--weights "MS3_resnet_best.pth"\
--test_batch_size 1\
 --tpavi_stages 0 1 2 3\
--tpavi_va_flag\
--save_pred_mask
________________________________________
Result
The test results are in the following path:
cd ..\network\avs_ms3\test_logs
Documentation
For detailed documentation, please refer to the Project Documentation.
Contact
Please contact us if you have any questions
Email: 15303717926@163.com
