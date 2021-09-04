import numpy as np
import pandas as pd

def compress(csv_mask, csv_gender, csv_age):
    ret = csv_mask
    for i in range(1, 12601):
        ret[i][1] = int(csv_mask[i][1]) * 6 + int(csv_gender[i][1]) * 3 + int(csv_age[i][1])
    return ret

def conv(n):
    mask = n // 6
    gender = n % 6 // 3
    age = n % 3
    return (mask, gender, age)

def vote(p: list, data: list) -> int:
    mask = [0.] * 3
    gender = [0.] * 2
    age = [0.] * 3
    for i in range(3):
        _mask, _gender, _age = conv(data[i])
        mask[_mask] += p[0][i]
        gender[_gender] += p[1][i]
        age[_age] += p[2][i]
    mask_label = mask.index(max(mask))
    gender_label = gender.index(max(gender))
    age_label = age.index(max(age))
    # print(mask, gender, age)
    return mask_label * 6 + gender_label * 3 + age_label

if __name__ == "__main__":
    path_in = ["/opt/ml/input/data/eval/submission_EfficientNet.csv",
     "/opt/ml/input/data/eval/submission_VIT.csv", 
    "/opt/ml/input/data/eval/submission_ResNet_Mask.csv", "/opt/ml/input/data/eval/submission_ResNet_Gender.csv", "/opt/ml/input/data/eval/submission_ResNet_Age.csv"] # Efficient, VIT, Res_mask, Res_gender, Res_age
    path_out = "/opt/ml/input/data/eval/submission.csv"

    csv_data = [np.loadtxt(path, delimiter=',', dtype=str) for path in path_in]
    csv_data[2] = compress(csv_data[2], csv_data[3], csv_data[4])
    csv_data.pop(); csv_data.pop();

    ans = [['', ''] for _ in range(12600)]

    p = [
        [0.2, 0.2, 0.2], # mask
        [0.4, 0.15, 0.15], # gender
        [0.3, 0.3, 0.3] # age
    ]

    for i in range(12600):
        ans[i][0] = csv_data[0][i + 1][0]
        ans[i][1] = vote(p, [int(csv_data[j][i + 1][1]) for j in range(3)])

    df = pd.DataFrame(ans)
    df.columns = ['ImageID', 'ans']
    df.to_csv(path_out, index=False)