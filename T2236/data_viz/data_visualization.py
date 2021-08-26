import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

data = pd.read_csv('./input/data/train/train.csv')
data.describe(include = 'all')

# data age classes
bins = [0,30,60,80]
bins_label = ["< 30","30 <= age < 60","60 <="]
data["age_class"] = pd.cut(data.age, bins, right = False, labels = bins_label, include_lowest = True)
print(data)
print()

# figure 1
gender_age = data.groupby(["gender","age_class"]).size().unstack()
print(gender_age)
gender_age.plot(kind = 'bar', stacked = True)
plt.xticks(rotation = 0)
plt.savefig("team_github/image-classification-level1-20/T2236/figure1")
print()

# figure 2
gender_age = data.groupby(["age_class","gender"]).size().unstack()
print(gender_age)
gender_age.plot(kind = 'bar', stacked = True)
plt.xticks(rotation = 0)
plt.savefig("team_github/image-classification-level1-20/T2236/figure2")
print()


# population percentage
male_percentage = gender_age["male"]/(gender_age["male"].sum()+gender_age["female"].sum())*100
print("male_percentage")
print(male_percentage)
print()
female_percentage = gender_age["female"]/(gender_age["male"].sum()+gender_age["female"].sum())*100
print("female_percentage")
print(female_percentage)
print()


# 경우의 수
mask = ["Correct", "Incorrect", "Not_wear"]
gender = ["Male", "Female"]
age_class = ["< 30","30 <= age < 60","60 <="]

col = ["mask", "gender", "age_class"]
classes = []
for m in mask :
    for g in gender:
        for a in age_class:
            classes.append([m,g,a])

classes_frame = pd.DataFrame(classes, columns = col)
print(classes_frame)