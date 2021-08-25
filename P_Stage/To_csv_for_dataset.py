import os
import pandas as pd 

root = "input/data/train/images" # train image folder의 경로를 넣어주시면 됩니다.

dirs_in_root = [x for x in os.listdir(root) if "._" not in x]

total = []

for dir_in_root in dirs_in_root:
    attr = dir_in_root.split("_") # sex : 1, age = -1
    path = os.path.join(root, dir_in_root)
    img_list = [x for x in os.listdir(path) if "._" not in x and "check" not in x]
        
    for img in img_list:
        path_ = os.path.join(path,img)
        if "mask" in img and "incorrect" not in img:
            mask = "Wear"
        elif "incorrect" in img:
            mask = "Incorrect"
        else:
            mask = "Not Wear"
        temp = [path_, attr[1], int(attr[-1]), mask]
        
        total.append(temp)
        
df = pd.DataFrame(total, columns = ["path", "sex", "age", "mask"])
df["age"] = pd.cut(df["age"], bins = [0, 30, 60, 90], right = False, labels = ["young","mid","old"])  
df["sex"] = df["sex"].map({"male":0, "female": 1})
df["mask"] = df["mask"].map({"Wear":0, "Incorrect": 1, "Not Wear" : 2})
df["age"] = df["age"].map({"young":0, "mid":1, "old" : 2}).astype(int)

df["class"] = df["mask"]*6 + df["sex"]*3 + df["age"]
df = df.drop(["sex","age","mask"], axis = 1)

# df 파일은 {path to image, image class} 형태로 저장됩니다.

df.to_csv("train.csv") # csv 파일을 저장할 경로를 적어주시면 됩니다.