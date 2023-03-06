import os
import csv
from sklearn.model_selection import train_test_split
import numpy as np
import cv2

# path = ['/media/hsy/1882C80C82C7EC76/Level-patch-1024-patient/']
# path = ["/media/hsy/1882C80C82C7EC76/Liver-patch-new-normal/", "/media/hsy/1882C80C82C7EC76/Liver-patch-new/", "/media/hsy/1882C80C82C7EC76/Liver-cancer-new-TEST/"]

# path = ["/media/hsy/1882C80C82C7EC76/Liver-cancer-new-TEST/"]
# path = [
    # "/media/hsy/3db0b616-908e-4357-b485-b92ae8b7b8e5/qiu/Lung-cancers/lung_colon_image_set/lung_image_sets/lung_aca",
    # "/media/hsy/3db0b616-908e-4357-b485-b92ae8b7b8e5/qiu/Lung-cancers/lung_colon_image_set/lung_image_sets/lung_n",
    # "/media/hsy/3db0b616-908e-4357-b485-b92ae8b7b8e5/qiu/Lung-cancers/lung_colon_image_set/lung_image_sets/lung_scc",
    # "/media/hsy/3db0b616-908e-4357-b485-b92ae8b7b8e5/qiu/Lung-cancers/lung_colon_image_set/colon_image_sets/colon_aca",
    # "/media/hsy/3db0b616-908e-4357-b485-b92ae8b7b8e5/qiu/Lung-cancers/lung_colon_image_set/colon_image_sets/colon_n"
    # ]
path = ["/media/hsy/3db0b616-908e-4357-b485-b92ae8b7b8e5/qiu/breast-cancers/BreaKHis_v1/histology_slides/breast/benign/SOB/",
        "/media/hsy/3db0b616-908e-4357-b485-b92ae8b7b8e5/qiu/breast-cancers/BreaKHis_v1/histology_slides/breast/malignant/SOB/"]

scale = '200X'

bengin_path = [path[0] + i for i in os.listdir(path[0])]
malignant_path = [path[1] + i for i in os.listdir(path[1])]

# path_list = os.listdir(path[0])
# path_list = [path[0] + i + '/' for i in path_list]

# 创建csv文件
# csv_metadata_rgb = open('metadata_rgb.csv', 'w+', newline='')
# csv_metadata_raw = open('metadata_raw.csv', 'w+', newline='')
csv_train = open("train.csv", 'w+', newline="")
csv_test = open("test.csv", 'w+', newline="")
csv_val = open("val.csv", 'w+', newline="")


# 创建writer
writer_train = csv.writer(csv_train)
writer_test = csv.writer(csv_test)
writer_val = csv.writer(csv_val)


# create list header
writer_train.writerow(['file', "label"])
writer_test.writerow(['file', "label"])
writer_val.writerow(['file', "label"])


# train_images_path = []  # 存储训练集的所有图片路径
# train_images_label = []  # 存储训练集图片对应索引信息
# val_images_path = []  # 存储验证集的所有图片路径
# val_images_label = []  # 存储验证集图片对应索引信息

# 读元数据文件
def read_csv(csv_path):
    csv_data = []
    csv_label = []
    files = csv.DictReader(open(csv_path, 'r'))
    for row in files:
        csv_data.append([row['file']])
        csv_label.append([row['label']])
    # s_data = np.array(csv_data)
    # np.random.shuffle(s_data)
    #
    return csv_data, csv_label


# Counting the number of current dataset
# train_images_path, train_images_label = read_csv('/media/hsy/1882C80C82C7EC76/ZJH/Cross-Validation-Dataset'
#                                                  '/Current_dataset/1/train.csv')
# val_images_path, val_images_label = read_csv('/media/hsy/1882C80C82C7EC76/ZJH/Cross-Validation-Dataset'
#                                              '/Current_dataset/1/val.csv')
# test_images_path, test_images_label = read_csv('/media/hsy/1882C80C82C7EC76/ZJH/Cross-Validation-Dataset'
#                                                '/Current_dataset/1/test.csv')
# total_0 = train_images_label.count(['0']) + test_images_label.count(['0']) + val_images_label.count(['0'])
# total_1 = train_images_label.count(['1']) + test_images_label.count(['1']) + val_images_label.count(['1'])
# total_2 = train_images_label.count(['2']) + test_images_label.count(['2']) + val_images_label.count(['2'])


# 计算数据集的均值和方差
def count_mean_std(data_path):
    mean = [0] * 3
    std = [0] * 3
    total = len(data_path)

    for path in data_path:
        img = cv2.imread(path) / 255
        m = np.mean(img, (0, 1))
        s = np.std(img, (0, 1))
        mean += m
        std += s

    mean = mean / total
    std = std / total

    return mean, std


# Normal = []
# Level_1_2 = []
# Level_3_4 = []

# F_1 = []
# F_2 = []
# F_3 = []
# F_4 = []
# F_5 = []
# F_6 = []
# F_7 = []
# F_8 = []
F = [[] for i in range(8)]

for i in range(len(bengin_path)):
    file_1 = os.listdir(bengin_path[i])
    for patient in file_1:
        file_2 = bengin_path[i] + '/' + patient + '/' + scale + '/'
        file_3 = os.listdir(file_2)
        for file in file_3:
            F[i].append(file_2 + file)
for i in range(len(malignant_path)):
    file_1 = os.listdir(malignant_path[i])
    for patient in file_1:
        file_2 = malignant_path[i] + '/' + patient + '/' + scale + '/'
        file_3 = os.listdir(file_2)
        for file in file_3:
            F[i + 4].append(file_2 + file)

# for i in range(4):
#     ori_path = [path[i] + '/' + file for file in os.listdir(path[i])]
#     
#     for file in ori_path:
#         if i == 0:
#             F_1.append(file)
#         elif i == 1:
#             F_2.append(file)
#         elif i == 2:
#             F_3.append(file)
#         elif i == 3:
#             F_4.append(file)
#         else:
#             F_5.append(file)

# np.random.shuffle(Normal)
# np.random.shuffle(Level_1_2)
# np.random.shuffle(Level_3_4)
for j in range(8):
    np.random.shuffle(F[j])
# np.random.shuffle(F_1)
# np.random.shuffle(F_2)
# np.random.shuffle(F_3)
# np.random.shuffle(F_4)
# np.random.shuffle(F_5)

ratio = 0.6

for i in range(len(F)):

    count = 0
    for patient in F[i]:
    # for patient in F_1:
    #     patient_path = os.listdir(patient)
        if count / len(F[i]) < ratio:
            writer = writer_train
        elif 0.8 <= count / len(F[i]):
            writer = writer_test
        else:
            writer = writer_val
            # writer = writer_test
    # for file in patient_path:
    #     writer.writerow([patient+file, 0])
    #     count += 1
        if i > 3:
            writer.writerow([patient, 1])
        else:
            writer.writerow([patient, 0])
        count += 1

# count = 0
# # for patient in Level_1_2:
# for patient in F_2:
#     # patient_path = os.listdir(patient)
#     if count / len(F_2) < ratio:
#         writer = writer_train
#     elif 0.8 <= count / len(F_2):
#         writer = writer_test
#     else:
#         writer = writer_val
#     writer.writerow([patient, 1])
#     count += 1
#
# count = 0
# # for patient in Level_3_4:
# for patient in F_3:
#     # patient_path = os.listdir(patient)
#     if count / len(F_3) < ratio:
#         writer = writer_train
#     elif 0.8 <= count / len(F_3):
#         writer = writer_test
#     else:
#         writer = writer_val
#     writer.writerow([patient, 2])
#     count += 1
#
# count = 0
# # for patient in Level_3_4:
# for patient in F_4:
#     # patient_path = os.listdir(patient)
#     if count / len(F_4) < ratio:
#         writer = writer_train
#     elif 0.8 <= count / len(F_4):
#         writer = writer_test
#     else:
#         writer = writer_val
#     writer.writerow([patient, 3])
#     count += 1
#
# count = 0
# # for patient in Level_3_4:
# for patient in F_5:
#     # patient_path = os.listdir(patient)
#     if count / len(F_5) < ratio:
#         writer = writer_train
#     elif 0.8 <= count / len(F_5):
#         writer = writer_test
#     else:
#         writer = writer_val
#     writer.writerow([patient, 4])
#     count += 1
# patience_index = ''
# count = 0
# for i in range(len(path)):
#
#     # 顺序打开文件
#     path_temp = os.listdir(path[i])
#     # np.random.shuffle(path_temp)
#     path_temp = [path[i] + f + '/' for f in path_temp]
#
#     # count = 0
#     count = 0
#
#     for j in range(len(path_temp)):
#         count += 1
#         jpg_path = os.listdir(path_temp[j])
#         for file in jpg_path:
#             # path_patient = os.listdir(path_temp[j]+file+'/')
#         #     if (count / len(path_temp)) > 0.8 and i != 0:
#         #         writer = writer_val
#         #     else:
#         #         writer = writer_test
#         #     writer = writer_train
#         #     for pic in path_patient:
#             writer = writer_train
#             if j == 0:
#                 label = 0
#             # elif j == 1 or j == 3:
#             #     label = 1
#             # else:
#             #     label = 2
#             else:
#                 label = 1
#
#             new_file = path_temp[j] + file
#             # new_file = path_temp[j] + file + '/' + pic
#             writer.writerow([new_file, label])
#
#     # for k in range(len(path_temp)):
#     #
#     #     path_patient = os.listdir(path_temp[k])
#     #     for file in path_patient:
#     #
#     #         if file.endswith('.jpg'):
#     #             # if '-M' in file:
#     #             #     writer = writer_train
#     #             #     # count += 1
#     #             # else:
#     #             #     if count//16/len(path_temp) <= 0.2:
#     #             #         writer = writer_test
#     #             #         count += 1
#     #             #     elif 0.2 < count//16/len(path_temp) <= 0.3:
#     #             #         writer = writer_val
#     #             #         count += 1
#     #             #     else:
#     #             #         writer = writer_train
#     #             writer = writer_test
#     #             # writer = writer_train
#     #
#     #         if i == 0:
#     #             label = 0
#     #         elif i == 1:
#     #             label = 1
#     #         else:
#     #             label = 2
#     #
#     #         new_file = path_temp[k] + file
#     #         writer.writerow([new_file, label])

csv_test.close()
csv_train.close()
csv_val.close()
