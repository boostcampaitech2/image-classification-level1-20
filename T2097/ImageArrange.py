import os
import shutil

# =================================================================================================

class ClassLabeling:
    def __init__(self):
        self.mask = { 'wear' : 0, 'incorrect' : 1, 'notwear' : 2 }
        self.gender = { 'male' : 0, 'female' : 1 }
    
    def __call__(self, mask, gender, age):
        """
        mask, gender, age를 매개변수로 받아서 label(0~17)을 리턴하는 함수
        mask, gender, age의 타입은 str, str, int

        mask는 'wear', 'incorrect', 'notwear' 중 1
        gender는 'male', 'female' 중 1
        """
        mask_label = self.mask[mask]
        gender_label = self.gender[gender]
        age_label = 0 if age < 30 else 1 if age < 60 else 2
        return mask_label * 6 + gender_label * 3 + age_label
    
    def label_to_feature(self, label):
        """
        label(0~17)을 매개변수로 받아서 (mask, gender, age)를 리턴하는 함수.
        mask, gender, age의 타입은 str, str, int

        단, age는 1, 31, 61 중 1. 만약 정확한 나이가 필요하다면 원본 파일의 정보를 이용해야 함.
        """
        rev = lambda _dict : dict(map(reversed, _dict.items()))
        mask = rev(self.mask)[label // 6]
        gender = rev(self.gender)[label % 6 // 3]
        age = label % 3 * 30 + 1
        return (mask, gender, age)

ClassLabel = ClassLabeling()

# =================================================================================================

class ImageArrange:
    def __init__(self):
        self.ClassLabel = ClassLabeling()

    def _clear_folder(self, path_out):
        for file in os.listdir(path_out):
            shutil.rmtree(os.path.join(path_out, file))

    def _make_class(self, path_out, class_num = 18):
        for i in range(class_num):
            try: os.mkdir(os.path.join(path_out, str(i)))
            except: pass

    def _get_file_name(self, path_out, age, label):
        mask, gender, _ = ClassLabel.label_to_feature(label)
        sz = len(os.listdir(os.path.join(path_out, str(label))))
        return '_'.join([str(sz + 1), mask, gender, str(age)])

    def _copy_and_rename(self, path_image, path_copy, image_name, new_name):
        shutil.copy(path_image, path_copy)
        shutil.move(os.path.join(path_copy, image_name), os.path.join(path_copy, new_name))

    def image_arrange(self, path_in, path_out):
        """
        path_in의 image를 path_out의 class 폴더로 분류
        확장자는 .jpg로 통일

        Args:
            path_in: 제공받은 train.tar/train/images의 디렉토리
            path_out: 분류할 class 폴더가 위치하는 디렉토리 (빈 폴더여야 합니다)
        
        Returns:
            None
        """
        self._clear_folder(path_out)
        self._make_class(path_out)

        folders = sorted([folder for folder in os.listdir(path_in) if folder[0] != '.'])

        for idx, image_folder in enumerate(folders):
            path_folder = os.path.join(path_in, image_folder)
            gender = image_folder.split('_')[1]
            age = int(image_folder.split('_')[-1])
            for image in os.listdir(path_folder):
                if image[0] == '.': continue
                mask = 'incorrect' if image[:2] == 'in' else 'notwear' if image[:2] == 'no' else 'wear'
                label = ClassLabel(mask, gender, age)
                file_name = self._get_file_name(path_out, age, label) + ".jpg"
                self._copy_and_rename(os.path.join(path_folder, image), os.path.join(path_out, str(label)), image, file_name)
            if (idx + 1) % 100 == 0: print(f"{idx + 1} / {len(folders)} !")

# =================================================================================================

if __name__ == "__main__":
    ImgArrange = ImageArrange()
    ImgArrange.image_arrange(r'/opt/ml/mask-classification/data/train.tar/train/images',
                             r'/opt/ml/mask-classification/data/class')