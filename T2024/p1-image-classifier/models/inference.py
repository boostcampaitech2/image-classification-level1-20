import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


# Inference
class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)

test_dir = '/opt/ml/input/data/eval'
submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
image_dir = os.path.join(test_dir, 'images')

# Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
dataset = TestDataset(image_paths, transform)

loader = DataLoader(
    dataset,
    shuffle=False
)


class Inference(nn.Module):
    # 모델을 정의합니다. (학습한 모델이 있다면 torch.load로 모델을 불러주세요!)
    def __init__(self, model, PATH, device):
        self.model.load_state_dict(torch.load(PATH)['model_state_dict'])
        self.model.eval()
        self.PATH = PATH
        self.device = device

    def forward(self):
        # 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
        all_predictions = []
        for images in loader:
            with torch.no_grad():
                images = images.to(self.device)
                pred = self.model(images)
                pred = pred.argmax(dim=-1)
                all_predictions.extend(pred.cpu().numpy())
        submission['ans'] = all_predictions

        # 제출할 파일을 저장합니다.
        submission.to_csv(os.path.join(test_dir, 'submission2.csv'), index=False)
        print('test inference is done!')
