import os
import time
import json
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from faster_rcnn_framework import FasterRCNN, FastRCNNPredictor
from models.resnet50_fpn_model import resnet50_fpn_backbone
# from backbone.resnet152_fpn_model import resnet152_fpn_backbone
# from network_files.rpn_function import AnchorsGenerator
# from backbone.mobilenetv2_model import MobileNetV2
# from draw_box_utils import draw_box
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def create_model(num_classes):
    # mobileNetv2+faster_RCNN
    # backbone = MobileNetV2().features
    # backbone.out_channels = 1280
    #
    # anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                     aspect_ratios=((0.5, 1.0, 2.0),))
    #
    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
    #                                                 output_size=[7, 7],
    #                                                 sampling_ratio=2)
    #
    # models = FasterRCNN(backbone=backbone,
    #                    num_classes=num_classes,
    #                    rpn_anchor_generator=anchor_generator,
    #                    box_roi_pool=roi_pooler)

    # resNet50+fpn+faster_RCNN
    backbone = resnet50_fpn_backbone()
    # backbone = resnet152_fpn_backbone()
    model = FasterRCNN(backbone=backbone, num_classes=num_classes)

    return model


def assist(img, device, num_classes=3):
    # get devices
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create models
    model = create_model(num_classes=num_classes)

    # load train weights
    train_weights = "./FPN/resNetFpn-models-99.pth"
    assert os.path.exists(train_weights), "{} file dose not exist.".format(train_weights)
    model.load_state_dict(torch.load(train_weights, map_location=device)["models"])
    model.to(device)

    # read class_indict
    # label_json_path = './pascal_voc_classes.json'
    label_json_path = './class.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    json_file = open(label_json_path, 'r')
    class_dict = json.load(json_file)
    category_index = {v: k for k, v in class_dict.items()}

    # load image
    # img_path = "/home/hsy/PycharmProjects/Test_3"
    # for img in os.listdir(img_path):
    # img_head = img.split(".")[0]
    # original_img = Image.open(os.path.join(img_path, img))

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time.time()
        predictions = model(img.to(device))[0]
        print("inference+NMS time: {}".format(time.time() - t_start))

        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()

        increase_factor = 0

        for index in range(len(predict_classes)):
            if predict_classes[index] == 1 and predict_scores[index] >= 0.6:
                increase_factor += 2
            elif predict_classes[index] == 2 and predict_scores[index] >= 0.6:
                increase_factor += 4
            else:
                pass

        if len(predict_boxes) == 0:
            print("没有检测到任何目标!")

        return increase_factor

        # draw_box(original_img,
        #          predict_boxes,
        #          predict_classes,
        #          predict_scores,
        #          category_index,
        #          thresh=0.8,
        #          line_thickness=1)
        # plt.imshow(original_img)
        # plt.show()

        # 保存预测的图片结果
        # if not os.path.exists(os.path.join(img_path, "Result")):
        #     os.makedirs(os.path.join(img_path, "Result"))
        # save_path = os.path.join(img_path, "Result")
        # original_img.save(os.path.join(save_path, f"{img_head}_test_result.jpg"))


