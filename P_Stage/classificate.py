

import os
import shutil

path_dir = './input/data/train/images/'



incorrect_mask_list = []
normal_list = []
mask_list = []
female_list = []
male_list = []
age_30_list = []
age_30_60list = []
age_60_list = []


os.makedirs('./incorrect', exist_ok=True)
os.makedirs('./normal/', exist_ok=True)
os.makedirs('./mask/', exist_ok=True)
os.makedirs('./female/', exist_ok=True)
os.makedirs('./male/', exist_ok=True)
os.makedirs('./age_30/', exist_ok=True)
os.makedirs('./age_60/', exist_ok=True)
os.makedirs('./age_30_60/', exist_ok=True)


#-----------------incorrect--------------------#

for list in os.listdir(path_dir):
    if list[0] != '.':
        path_list_dir = path_dir + list
        for name in os.listdir(path_dir + list):
            if name[0] != '.' and 'incorrect' in name:
                incorrect_mask_list.append(path_list_dir + '/' + name)

for file in incorrect_mask_list:
    shutil.copy(file, './incorrect/' + file[18:].replace('/','_'))
    

#-----------------normal--------------------#

for list in os.listdir(path_dir):
    if list[0] != '.':
        path_list_dir = path_dir + list
        for name in os.listdir(path_dir + list):
            if name[0] != '.' and 'normal' in name:
                normal_list.append(path_list_dir + '/' + name)


for file in normal_list:
    shutil.copy(file, './normal/' + file[18:].replace('/','_'))


#-----------------mask--------------------#

for list in os.listdir(path_dir):
    if list[0] != '.':
        path_list_dir = path_dir + list
        for name in os.listdir(path_dir + list):
            if name[0] != '.' and 'mask' in name and 'incorrect' not in name:
                mask_list.append(path_list_dir + '/' + name)

for file in mask_list:
    shutil.copy(file, './mask/' + file[18:].replace('/','_'))
    



#-----------------female-------------------#

for list in os.listdir(path_dir):
    if list[0] != '.':
        path_list_dir = path_dir + list
        for name in os.listdir(path_dir + list):
            if 'female' in path_list_dir and name[0] != '.':
                female_list.append(path_list_dir + '/' + name)


for file in female_list:
    shutil.copy(file, './female/' + file[18:].replace('/','_'))
    
#-----------------male--------------------#
for list in os.listdir(path_dir):
    if list[0] != '.':
        path_list_dir = path_dir + list
        for name in os.listdir(path_dir + list):
            if '_male' in path_list_dir and name[0] != '.':
                male_list.append(path_list_dir + '/' + name)


for file in male_list:
    shutil.copy(file, './male/' + file[18:].replace('/','_'))

#-----------------age--------------------#
for list in os.listdir(path_dir):
    if list[0] != '.':
        path_list_dir = path_dir + list
        for name in os.listdir(path_dir + list):
            if int(path_list_dir[-2:]) < 30 and name[0] != '.':
                age_30_list.append(path_list_dir + '/' + name)


for file in age_30_list:
    shutil.copy(file, './age_30/' + file[18:].replace('/','_'))


for list in os.listdir(path_dir):
    if list[0] != '.':
        path_list_dir = path_dir + list
        for name in os.listdir(path_dir + list):
            if int(path_list_dir[-2:]) >= 30 and int(path_list_dir[-2:]) < 60 and name[0] != '.':
                age_30_60list.append(path_list_dir + '/' + name)


for file in age_30_60list:
    shutil.copy(file, './age_30_60/' + file[18:].replace('/','_'))


for list in os.listdir(path_dir):
    if list[0] != '.':
        path_list_dir = path_dir + list
        for name in os.listdir(path_dir + list):
            if int(path_list_dir[-2:]) >= 60 and name[0] != '.':
                age_60_list.append(path_list_dir + '/' + name)

for file in age_60_list:
    shutil.copy(file, './age_60/' + file[18:].replace('/','_'))


