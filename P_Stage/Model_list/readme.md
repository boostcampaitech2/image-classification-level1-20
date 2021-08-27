## Model List

* 한재현
    DenseNet121 (NUM_EPOCH = 10, BATCH_SIZE = 50, LEARNING_RATE = 0.001)
    : 별로..
    
    EfficientNetb3 (NUM_EPOCH = 10, BATCH_SIZE = 32, LEARNING_RATE = 0.002) 
    : Epoch10 - Loss: 0.333, Acc: 0.881, F1: 0.7778   
    -> 리더보드 제출 결과: 42%.....(띠로리..)
    
    EfficientNetb3 (NUM_EPOCH = 30, BATCH_SIZE = 32, LEARNING_RATE = 0.002) 
    : Epoch30 - Loss: 0.018, Acc: 0.994, F1: 0.9892
    -> 제출 결과- ACC: 74.9841%	F1: 0.6824

* 김민성
    - vit_tiny_patch16_224
        - epoch=30, batch_size=32, learning_rate=0.002, cross_entropy, adam
        - accuracy : 47%, f1_score : 0.3879
    

* 김정현

    Resnet18(pretrained) -> best result : epoch20, train loss : 0.02, f1 : 0.91 **submission** : acc :0.73 f1 : 0.66

    Resnet34(pretrained) -> ongoing

* 김재영  

    Resnext50(pretrained) epoch: 3, batch_size:128, lr: 0.005 을 이용해서 마스크, 성별, 나이 각각 학습  
    Linear(8, 18) epoch: 5, batch_size:128, lr:0.005을 이용해서 18가지 클래스에 대해 분류 -> acc: 73%	f1: 0.6454  
