 
[**Trial_First.py**](https://github.com/boostcampaitech2/image-classification-level1-20/blob/e8e6f88b596c874ac763d28e97fc2b57c1762ad2/김범찬_T2031/Trial_First.py)

데이터 불균형 해소를 위한 도전

- **main idea** :  batch마다 각 class 별로 사진을 한 장씩 가져와 총 18장을 한 번에 학습하는 방법
- **result** : Evaluation시, label이 편향되는 결과가 나타남. 부족한 label내 data의 overfitting으로 인한 문제라 추정.
- **to be improved**: 다른 방법으로 데이터 증강을 시도해보자!

[**Face Detection Practice**](https://github.com/boostcampaitech2/image-classification-level1-20/blob/e8e6f88b596c874ac763d28e97fc2b57c1762ad2/김범찬_T2031/Trial_First.py)

사진에서 얼굴 검출을 하기 위한 도전

- **main idea** :  <code>cv2.CasacadeClassifier</code>를 이용하여 사진에서 얼굴과 눈을 검출해보자.
- **result** : scale_factor가 작다면 ROI가 많이 생기지만 부정확하며, scale_factior가 크면 정확하지만 ROI가 잘 생성되지 않는다.
  - 적절한 scale_factor를 찾는 것이 중요
- **to be improved**: 눈을 기준으로 ROI를 찾은 후에, 인체의 비례를 활용하여 얼굴 검출을 해보자.

[**mtcnn_in_collate**](https://github.com/boostcampaitech2/image-classification-level1-20/blob/36de3d5b1216004e532635cb94acef9efe981793/김범찬_T2031/mtcnn_in_collate.ipynb)

얼굴 검출을 <code>collate_fn</code> 내에서 실행하도록 하기 위함.

- **main idea** :  <code>MTCNN</code>을 이용하여 <code>collate_fn</code> 내에서 얼굴만 crop할 수 있도록
- **result** : dataloader로 data를 불러올 때, 얼굴만 자른 사진이 나온다.
- **to be improved**: mixup이나 cutmix도 collate_fn 내에서 작업할 수 있도록 하나로 묶으면 좋겠다.