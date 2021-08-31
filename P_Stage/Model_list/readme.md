## Model List

* 한재현

    - DenseNet121 (pretrained, NUM_EPOCH = 10, BATCH_SIZE = 50, LEARNING_RATE = 0.001)
    : 별로..
    
    - EfficientNetb3 (pretrained, NUM_EPOCH = 10, BATCH_SIZE = 32, LEARNING_RATE = 0.002) 
    : Epoch10 - Loss: 0.333, Acc: 0.881, F1: 0.7778   
    -> 리더보드 제출 결과: 42%.....(띠로리..)
    
    - EfficientNetb3 (pretrained, NUM_EPOCH = 30, BATCH_SIZE = 32, LEARNING_RATE = 0.002) 
    : Epoch30 - Loss: 0.018, Acc: 0.994, F1: 0.9892
    -> 제출 결과- ACC: 74.9841%	F1: 0.6824
    
    - EfficientNetb3 + MTCNN을 통한 crop 후 데이터 (pretrained, NUM_EPOCH = 30, BATCH_SIZE = 32, LEARNING_RATE = 0.002) 
    
    -> 제출 결과- ACC: 73.6349%	F1: 0.6371   (ㅠㅡㅠ)

    - EfficientNetb3 + MTCNN을 통한 crop 후 데이터 (pretrained, NUM_EPOCH = 25, BATCH_SIZE = 32, LEARNING_RATE = 0.002) 
    : crop범위를 조금 확장하고, 데이터의 인중부터 입술 부분 cutout
    Epoch25 - Loss: 0.026, Acc: 0.992, F1: 0.9830
    -> 제출 결과- ACC: 74.1905% F1: 0.6824  
    
    - EfficientNetb3 + MTCNN을 통한 crop 후 데이터 (pretrained, NUM_EPOCH = 30, BATCH_SIZE = 32, LEARNING_RATE = 0.002) 
    : crop범위를 조금 확장하고, 데이터의 인중부터 입술 부분 cutout
        Epoch30 - Loss: 0.015, Acc: 0.995, F1: 0.9898
    -> 제출 결과- ACC: 75.6503% F1: 0.6913
    - EfficientNet b4 + MTCNN을 통한 crop 후 데이터  (pretrained, NUM_EPOCH = 30, BATCH_SIZE = 16, LEARNING_RATE = 0.002) 
    -> 제출 결과- ACC: 77.143% F1: 0.717
    
* 김민성
    - vit_tiny_patch16_224
        - epoch=30, batch_size=32, learning_rate=0.002, cross_entropy, adam
            - train
                - ...모르고 지워버림
            - submission
                - accuracy : 0.47, f1_score : 0.3879
        - epoch=30, batch_size=32, learning_rate=0.002, f1_loss+cross_entropy, adam
            - train
                - accuracy : 0.271, f1_score : 0.1067
        - epoch=50, batch_size=32, learning_rate=0.002, f1_loss * 0.8 + cross_entropy * 0.2, adam
            - train
                - best accuracy : 0.939, f1_score : 0.893
            - submission
                - accuracy : 0.433, f1_score : 0.3717
    

* 김정현

    Resnet18(pretrained) -> best result : epoch20, train loss : 0.02, f1 : 0.91 **submission** : acc :0.73 f1 : 0.66

    Resnet18(pretrained)_gender -> best result : epoch1 , train loss =0.03, f1 : 0.03 no submission label [ 0 , 1 ]
    
    Resnet18(pretrained)_age -> ongoing
    
    Resnet18(pretrained)_mask -> ongoing
    
    Resnet34(pretrained) -> best result : epoch50, train loss : 0.01 f1 : 0.93 **submission** : acc : 0.74 f1: 0.64
    
    

* 김재영  

    Resnext50(pretrained) epoch: 3, batch_size:128, lr: 0.005 을 이용해서 마스크, 성별, 나이 각각 학습  
    Linear(8, 18) epoch: 5, batch_size:128, lr:0.005을 이용해서 18가지 클래스에 대해 분류 -> acc: 73%	f1: 0.6454  
