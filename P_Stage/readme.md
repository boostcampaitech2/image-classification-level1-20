# Description for the Folder
P_Stage 공유자료 올려놓으면 좋을 것 같아요

# History

['21.08.24] Labeling_train.ipynb, resnet_base.ipynb. train.json updated. </br>
['21.08.25] To_csv_for_dataset.py, train.csv updated
['21.08.26] classificate updated

# Explanation for each code

[**Labeling_train.ipynb**](https://github.com/boostcampaitech2/image-classification-level1-20/blob/main/P_Stage/Labeling_train.ipynb)

각 사진에 대해 Class가 적혀있지 않기 때문에, 사진별 Class를 Labeling하고 이를 Json 파일로 변환합니다.
- 관련 자료 : [train.json](https://github.com/boostcampaitech2/image-classification-level1-20/blob/main/P_Stage/train.json)

[**resnet_base.inpynb**](https://github.com/boostcampaitech2/image-classification-level1-20/blob/main/P_Stage/resnet_base.ipynb)

Pre-trained된 ResNet18 Model로 학습하고, 이에 따른 학습결과로 submission.csv 파일을 제출할 수 있도록 하는 파일입니다.

[**To_csv_for_dataset.py**](https://github.com/boostcampaitech2/image-classification-level1-20/blob/main/P_Stage/To_csv_for_dataset.py)

각 사진에 대해 Class가 적혀있지 않기 때문에, 사진별 Class를 Labeling하고 이를 csv 파일로 변환합니다.
- 같은 작업을 수행한 json파일이 있지만, 일부 확장자 문제(png, jpeg)와 csv 사용이 친숙한 경우가 있기 때문에 제공합니다.
- 관련 자료 : [train.csv](https://github.com/boostcampaitech2/image-classification-level1-20/blob/main/P_Stage/train.csv)

[**classification.py**](https://github.com/boostcampaitech2/image-classification-level1-20/blob/main/P_Stage/classification.py)

각 사진을 마스크, 성별, 나이로 폴더를 만들고 그 안에 적용시켜줌, 3, 2, 3 클래스로 구분되며 세부적인 클래스 분류하는 분류기 만들기에 적합
