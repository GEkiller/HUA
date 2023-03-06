import os
import numpy as np
import cv2

# path = ["/media/hsy/3db0b616-908e-4357-b485-b92ae8b7b8e5/qiu/liver-cancer/T16/2/",
#         "/media/hsy/3db0b616-908e-4357-b485-b92ae8b7b8e5/qiu/liver-cancer/T16/3/",
#         "/media/hsy/3db0b616-908e-4357-b485-b92ae8b7b8e5/qiu/liver-cancer/T17/2/",
#         "/media/hsy/3db0b616-908e-4357-b485-b92ae8b7b8e5/qiu/liver-cancer/T17/3/"
#         ]
# path = ["/media/hsy/3db0b616-908e-4357-b485-b92ae8b7b8e5/qiu/Lung-cancers/lung_colon_image_set/lung_image_sets"
#         "/lung_aca/",
#         "/media/hsy/3db0b616-908e-4357-b485-b92ae8b7b8e5/qiu/Lung-cancers/lung_colon_image_set/lung_image_sets/lung_n/",
#         "/media/hsy/3db0b616-908e-4357-b485-b92ae8b7b8e5/qiu/Lung-cancers/lung_colon_image_set/lung_image_sets/lung_scc/"
#         ]
# path = "/media/hsy/3db0b616-908e-4357-b485-b92ae8b7b8e5/qiu/liver-cancer/TEST/0/"
# path = "/media/hsy/3db0b616-908e-4357-b485-b92ae8b7b8e5/qiu/liver-cancer/Normal_new/"
save_path = ["/media/hsy/1882C80C82C7EC76/Liver-cancer-new-TEST/0/",
             "/media/hsy/1882C80C82C7EC76/Liver-cancer-new-TEST/2/",
             "/media/hsy/1882C80C82C7EC76/Liver-cancer-new-TEST/3/",
             ]
# save_path = "/media/hsy/1882C80C82C7EC76/Liver-patch-new/Normal/"
# save_path = ["/media/hsy/1882C80C82C7EC76/Liver-patch-new/Normal/",
#              "/media/hsy/1882C80C82C7EC76/Liver-patch-new/T16-2/",
#              "/media/hsy/1882C80C82C7EC76/Liver-patch-new/T16-3/",
#              "/media/hsy/1882C80C82C7EC76/Liver-patch-new/T17-2/",
#              "/media/hsy/1882C80C82C7EC76/Liver-patch-new/T17-3/"]
for k in range(len(save_path)):

    path_list = os.listdir(save_path[k])
    path_temp = [save_path[k] + i + '/' for i in path_list]
    for j in range(len(path_temp)):
        path_patient = os.listdir(path_temp[j])
        for file in path_patient:
            img = cv2.imread(path_temp[j] + file)
            img_cvt_cat = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(path_temp[j]+file, img_cvt_cat)
            # if np.shape(img)[0] == 1024:
            #     img = cv2.resize(img, (512, 512))
            #     cv2.imwrite(path_temp[j] + file, img)

    # k = 1
    # for i in range(4):
    #     for j in range(4):
    #         if k != 1 and k != 4 and k != 13 and k != 16:
    #             save = save_path + file[:-4] + '-' + str(k) + '.jpg'
    #             cv2.imwrite(save, img[i * 1024:(i + 1) * 1024, j * 1024:(j + 1) * 1024, :])
    #         k += 1

# for w in range(1):
#     path_list = os.listdir(path[w])
#     for file in path_list:
#         img = cv2.imread(path[w] + file)
#         re_img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
#         cv2.imwrite(path[w]+file, re_img)

        # if img.shape[0] == 2048:
        #     k = 1
        #     for i in range(2):
        #         for j in range(2):
        #             save_path = path[w] + file[:-4] + '-' + str(k) + '.jpg'
        #             cv2.imwrite(save_path, img[i * 1024:(i + 1) * 1024, j * 1024:(j + 1) * 1024, :])
        #             k += 1
        # if img.shape[0] == 4096:
        #     k = 1
        #     for i in range(4):
        #         for j in range(4):
        #             save_path = path[w] + file[:-4] + '-' + str(k) + '.jpg'
        #             cv2.imwrite(save_path, img[i * 1024:(i + 1) * 1024, j * 1024:(j + 1) * 1024, :])
        #             k += 1

