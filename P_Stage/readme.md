# Description for the Folder
P_Stage 공유자료 올려놓으면 좋을 것 같아요

# History

['21.08.24] Labeling_train.ipynb, resnet_base.ipynb. train.json updated. </br>
['21.08.25] To_csv_for_dataset.py, train.csv updated

# Explanation for each code

**Labeling_train.ipynb**

각 사진에 대해 Class가 적혀있지 않기 때문에, 사진별 Class를 Labeling하고 이를 Json 파일로 변환합니다.
- 관련 자료 : train.json

**resnet_base.inpynb**

Pre-trained된 ResNet18 Model로 학습하고, 이에 따른 학습결과로 submission.csv 파일을 제출할 수 있도록 하는 파일입니다.

**To_csv_for_dataset.py**

각 사진에 대해 Class가 적혀있지 않기 때문에, 사진별 Class를 Labeling하고 이를 csv 파일로 변환합니다.
- 같은 작업을 수행한 json파일이 있지만, 일부 확장자 문제(png, jpeg)와 csv 사용이 친숙한 경우가 있기 때문에 제공합니다.
- 관련 자료 : train.csv


