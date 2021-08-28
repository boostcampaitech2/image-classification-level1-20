import pandas as pd

root_csv_path = '/opt/ml/train_data_path_and_class.csv'
information = pd.read_csv(root_csv_path)
info_df = pd.DataFrame(information)


for i in range(len(info_df)):
    info_df.loc[i,"path"] = info_df.iloc[i]["path"].replace("/images","/new_imgs")

info_df.to_csv("new_train_data_path_and_class.csv", index = False)
print(info_df)
print(info_df.iloc[i]["path"].replace("/images","/new_imgs"))