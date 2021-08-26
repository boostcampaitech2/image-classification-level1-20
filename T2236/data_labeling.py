import pandas as pd

root_csv_path = 'input/data/train/train.csv'
information = pd.read_csv(root_csv_path)
info_df = pd.DataFrame(information)

# age_class
bins = [0,30,60,80]
bins_label = ["< 30","30 <= age < 60","60 <="]
info_df["age_class"] = pd.cut(info_df.age, bins, right = False, labels = bins_label, include_lowest = True)


# path and class
image_path = "input/data/train/images/"
path_lists = []
class_lists = []
for i in range(len(info_df)):
    path = info_df.iloc[i][4]
    path_list = []
    mask = ["incorrect_mask", "mask1" , "mask2", "mask3", "mask4", "mask5", "normal"]
    for m in mask :
        path_list.append(image_path + path +'/' + m + '.jpg')
    path_lists.append(path_list)

    class_list = []
    class_condition={"incorrect_mask":6, "mask1":0 , "mask2":0, "mask3":0, "mask4":0, "mask5":0, "normal":12, "male":0, "female":3, "< 30":0,"30 <= age < 60":1,"60 <=":2}
    for m in mask:
        class_list.append(class_condition[m] + class_condition[info_df.iloc[i]["gender"]] + class_condition[info_df.iloc[i]["age_class"]])
    class_lists.append(class_list)

info_df["class"] = class_lists
info_df["path_list"] = path_lists
info_df = info_df.drop(["path","age","id","gender","race","age_class"],axis=1)


new_class = []
for i in range(len(info_df)):
    for j in range(len(info_df.iloc[i][0])):
        new_class.append([info_df.iloc[i][1][j],info_df.iloc[i][0][j]])
    
new_data = pd.DataFrame(new_class,columns=['path','label'])

new_data.to_csv("train_data_path_and_class.csv", index = False)
print(new_data)
