# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import argparse
import os
import paddle
from paddleseg.cvlibs import manager, Config
from paddleseg.utils import get_sys_env, logger
import math
import cv2
import numpy as np
from paddleseg import utils
from paddleseg.core import infer
from paddleseg.utils import progbar


def mkdir(path):
    sub_dir = os.path.dirname(path)  # 去掉文件名，返回目录
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)


def partition_list(arr, m):
    """split the list 'arr' into m pieces"""
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]


def parse_args():
    parser = argparse.ArgumentParser(description='Model prediction')

    # params of prediction
    parser.add_argument(
        "--config", dest="cfg", help="The config file.", default=None, type=str)
    parser.add_argument(
        '--model_path',
        dest='model_path',
        help='The path of model for prediction',
        type=str,
        default=None)
    parser.add_argument(
        '--image_path',
        dest='image_path',
        help=
        'The path of image, it can be a file or a directory including images',
        type=str,
        default=None)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the predicted results',
        type=str,
        default='./output/result')

    # augment for prediction
    parser.add_argument(
        '--aug_pred',
        dest='aug_pred',
        help='Whether to use mulit-scales and flip augment for prediction',
        action='store_true')
    parser.add_argument(
        '--scales',
        dest='scales',
        nargs='+',
        help='Scales for augment',
        type=float,
        default=1.0)
    parser.add_argument(
        '--flip_horizontal',
        dest='flip_horizontal',
        help='Whether to use flip horizontally augment',
        action='store_true')
    parser.add_argument(
        '--flip_vertical',
        dest='flip_vertical',
        help='Whether to use flip vertically augment',
        action='store_true')

    # sliding window prediction
    parser.add_argument(
        '--is_slide',
        dest='is_slide',
        help='Whether to prediction by sliding window',
        action='store_true')
    parser.add_argument(
        '--crop_size',
        dest='crop_size',
        nargs=2,
        help=
        'The crop size of sliding window, the first is width and the second is height.',
        type=int,
        default=None)
    parser.add_argument(
        '--stride',
        dest='stride',
        nargs=2,
        help=
        'The stride of sliding window, the first is width and the second is height.',
        type=int,
        default=None)

    return parser.parse_args()


def get_image_list(image_path):
    """Get image list"""
    valid_suffix = [
        '.JPEG', '.jpeg', '.JPG', '.jpg', '.BMP', '.bmp', '.PNG', '.png', '.tif'
    ]
    image_list = []
    image_dir = None
    if os.path.isfile(image_path):
        if os.path.splitext(image_path)[-1] in valid_suffix:
            image_list.append(image_path)
    elif os.path.isdir(image_path):
        image_dir = image_path
        for root, dirs, files in os.walk(image_path):  # root=image_path
            for f in files:
                if os.path.splitext(f)[-1] in valid_suffix:
                    image_list.append(os.path.join(root, f))
    else:
        raise FileNotFoundError(
            '`--image_path` is not found. it should be an image file or a directory including images'
        )

    if len(image_list) == 0:
        raise RuntimeError('There are not image file in `--image_path`')

    return image_list, image_dir  # 返回测试文件列表


def CropBigImage(ImagePath, CropScale):
    ImagePath = ImagePath
    CropScale = CropScale
    seg_list = []  # 存储分割的图块
    ori_image = cv2.imread(ImagePath)  ##
    h_step = ori_image.shape[0] // CropScale
    w_step = ori_image.shape[1] // CropScale

    h_rest = -(ori_image.shape[0] - CropScale * h_step)
    w_rest = -(ori_image.shape[1] - CropScale * w_step)

    # 循环切图
    for h in range(h_step):
        for w in range(w_step):
            # 划窗采样
            image_sample = ori_image[(h * CropScale):(h * CropScale + CropScale),
                           (w * CropScale):(w * CropScale + CropScale), :]
            seg_list.append(image_sample)
        seg_list.append(ori_image[(h * CropScale):(h * CropScale + CropScale), -CropScale:, :])
    for w in range(w_step - 1):
        seg_list.append(ori_image[-CropScale:, (w * CropScale):(w * CropScale + CropScale), :])
    seg_list.append(ori_image[-CropScale:, -CropScale:, :])

    return seg_list, ori_image, h_step, w_step, h_rest, w_rest


def PinJie(predict_list, CropScale, ori_image, h_step, w_step, h_rest, w_rest):
    # 将预测后的图像块再拼接起来
    count_temp = 0
    tmp = np.ones([ori_image.shape[0], ori_image.shape[1]])
    for h in range(h_step):
        for w in range(w_step):
            tmp[
            h * CropScale:(h + 1) * CropScale,
            w * CropScale:(w + 1) * CropScale
            ] = predict_list[count_temp]
            count_temp += 1
        tmp[h * CropScale:(h + 1) * CropScale, w_rest:] = predict_list[count_temp][:, w_rest:]
        count_temp += 1
    for w in range(w_step - 1):
        tmp[h_rest:, (w * CropScale):(w * CropScale + CropScale)] = predict_list[count_temp][h_rest:, :]
        count_temp += 1
    tmp[-(CropScale + 1):-1, -(CropScale + 1):-1] = predict_list[count_temp][:, :]
    return tmp.astype('uint8')


def main(args):
    env_info = get_sys_env()
    place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info[
        'GPUs used'] else 'cpu'

    paddle.set_device(place)
    if not args.cfg:
        raise RuntimeError('No configuration file specified.')

    cfg = Config(args.cfg)
    val_dataset = cfg.val_dataset  # 用val_dataset？
    if not val_dataset:
        raise RuntimeError(
            'The verification dataset is not specified in the configuration file.'
        )

    msg = '\n---------------Config Information---------------\n'
    msg += str(cfg)
    msg += '------------------------------------------------'
    logger.info(msg)

    model = cfg.model
    transforms = val_dataset.transforms
    # image_list, image_dir = get_image_list('data/UAV_seg/images')
    image_list, image_dir = get_image_list(args.image_path)  # 需要传入args.image_path参数   这个参数可以是测试图片的路径，也可以是单张图片的路径

    model_path = args.model_path  # 传入训练模型的路径
    save_dir = args.save_dir
    aug_pred = args.aug_pred
    scales = args.scales
    flip_horizontal = args.flip_horizontal
    flip_vertical = args.flip_vertical
    is_slide = args.is_slide
    crop_size = args.crop_size
    stride = args.stride

    para_state_dict = paddle.load(model_path)
    model.set_dict(para_state_dict)
    model.eval()
    nranks = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()
    if nranks > 1:
        img_lists = partition_list(image_list, nranks)
    else:
        img_lists = [image_list]  # 是列表还是列表的列表，等待测试   img_lists[0]  ->列表的列表

    added_saved_dir = os.path.join(save_dir, 'added_prediction')  # 伪彩色和原图叠加
    pred_saved_dir = os.path.join(save_dir, 'pseudo_color_prediction')  # 伪彩色预测结果

    # 主要将遥感大图裁剪成固定尺寸的图块，生成图块列表
    ImagePath = img_lists[local_rank][local_rank]
    CropScale = 256
    seg_list, ori_image, h_step, w_step, h_rest, w_rest = CropBigImage(ImagePath, CropScale)

    predict_list = []
    progbar_pred = progbar.Progbar(target=len(seg_list), verbose=1)
    logger.info("Start to predict...")
    with paddle.no_grad():
        for i, im in enumerate(seg_list):
            ori_shape = im.shape[:2]  # 原始图片形状（h，w）
            im, _ = transforms(im)  # im.shape(3, 512, 512)    _为None
            im = im[np.newaxis, ...]  # im.shape(1,3,512,512)
            im = paddle.to_tensor(im)

            if aug_pred:
                pred = infer.aug_inference(
                    model,
                    im,
                    ori_shape=ori_shape,
                    transforms=transforms.transforms,
                    scales=scales,
                    flip_horizontal=flip_horizontal,
                    flip_vertical=flip_vertical,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
            else:
                pred = infer.inference(
                    model,
                    im,
                    ori_shape=ori_shape,
                    transforms=transforms.transforms,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
            pred = paddle.squeeze(pred)  # 该OP会删除输入Tensor的Shape中尺寸为1的维度。查看pred的形状 应该剩下[h,w]
            pred = pred.numpy().astype('uint8')
            predict_list.append(pred)
            progbar_pred.update(i + 1)

    # 主要将图块的预测结果拼接成大图
    tmp = PinJie(predict_list, CropScale, ori_image, h_step, w_step, h_rest, w_rest)
    #############
    # 获取需要保存的图片名称，去掉前面的路径
    # get the saved name
    if image_dir is not None:
        pass
        # im_file = im_path.replace(image_dir, '')  #例：将PaddleSeg/data/optic_disc_seg/JPEGImages/P0011.jpg替换为/P0011.jpg
    else:
        im_file = os.path.basename(img_lists[local_rank][local_rank])  # 带后缀名
    if im_file[0] == '/':  # 去掉/
        im_file = im_file[1:]

    # save added image
    added_image = utils.visualize.visualize(args.image_path, tmp, weight=0.6)
    added_image_path = os.path.join(added_saved_dir, im_file)
    mkdir(added_image_path)
    cv2.imwrite(added_image_path, added_image)

    # save pseudo color prediction
    pred_mask = utils.visualize.get_pseudo_color_map(tmp)
    pred_saved_path = os.path.join(pred_saved_dir,
                                   im_file.rsplit(".")[0] + ".png")
    mkdir(pred_saved_path)
    pred_mask.save(pred_saved_path)
    logger.info("-" * 30 + "END" + "-" * 30)


if __name__ == '__main__':
    args = parse_args()
    main(args)