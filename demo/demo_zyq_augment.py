# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os
import gzip

# fmt: off
import sys

from matplotlib import cm

from tta_handler import TTAHandler

sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '.'))

# fmt: on

import tempfile
import time
import warnings
import torch
import cv2
import albumentations as A
import numpy as np
import tqdm
from pathlib import Path
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo
from PIL import Image
import torchvision.transforms as T
import math

from detectron2.utils.file_io import PathManager
from detectron2.data.detection_utils import _apply_exif_orientation, convert_PIL_to_numpy


def visualize_mask_folder(path_to_folder, offset=0):
    (path_to_folder.parent / f"visualized_{path_to_folder.stem}").mkdir(exist_ok=True)
    loaded_path = sorted(list(path_to_folder.iterdir()))
    # loaded_path = loaded_path[0:500]
    for f in tqdm_1(loaded_path, desc='visualizing masks'):
        # visualize_mask(np.array(Image.open(f)) + offset, path_to_folder.parent / f"visualized_{path_to_folder.stem}" / f.name)
        rgbs = custom2rgb(np.array(Image.open(f)))

        Image.fromarray(rgbs).save(path_to_folder.parent / f"visualized_{path_to_folder.stem}" / f.name)




def custom2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [0, 0, 0]             # cluster       black

    mask_rgb[np.all(mask_convert == 1, axis=0)] = [128, 0, 0]           # building      red
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [192, 192, 192]         # road        grey  
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [192, 0, 192]         # car           light violet
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [0, 128, 0]           # tree          green
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [128, 128, 0]         # vegetation    dark green
    mask_rgb[np.all(mask_convert == 6, axis=0)] = [255, 255, 0]         # human         yellow
    mask_rgb[np.all(mask_convert == 7, axis=0)] = [135, 206, 250]       # sky           light blue
    mask_rgb[np.all(mask_convert == 8, axis=0)] = [0, 0, 128]           # water         blue

    mask_rgb[np.all(mask_convert == 9, axis=0)] = [252,230,201]          # ground      egg
    mask_rgb[np.all(mask_convert == 10, axis=0)] = [128, 64, 128]        # mountain     dark violet

    # mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    return mask_rgb

# constants
WINDOW_NAME = "mask2former demo"
def convert_from_mask_to_semantics_and_instances_no_remap(original_mask, segments):
    id_to_class = torch.zeros(1024).int()
    original_mask = original_mask.cpu()
    for s in segments:
        id_to_class[s['id']] = s['category_id']
    return id_to_class[original_mask.flatten().numpy().tolist()].reshape(original_mask.shape)


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.zyq_code = args.zyq_code
    cfg.zyq_mapping = args.zyq_mapping
    cfg.freeze()
    return cfg


def visualize_tensor(tensor, minval=0.000, maxval=1.00, use_global_norm=True):
    x = tensor.cpu().numpy()
    x = np.nan_to_num(x)  # change nan to 0
    if use_global_norm:
        mi = minval
        ma = maxval
    else:
        mi = np.min(x)  # get minimum depth
        ma = np.max(x)
    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x_ = Image.fromarray((cm.get_cmap('jet')(x) * 255).astype(np.uint8))
    x_ = T.ToTensor()(x_)[:3, :, :]
    return x_


def probability_to_normalized_entropy(probabilities):
    entropy = torch.zeros_like(probabilities[:, :, 0])
    for i in range(probabilities.shape[2]):
        entropy = entropy - probabilities[:, :, i] * torch.log2(probabilities[:, :, i] + 1e-8)
    entropy = entropy / math.log2(probabilities.shape[2])
    return entropy


def load_and_save_with_entropy_and_confidence(out_filename, entropy, confidences):
    from torchvision.io import read_image
    from torchvision.utils import save_image
    org_img = read_image(out_filename).float() / 255.0
    e_img = visualize_tensor(1 - entropy)
    c_img = visualize_tensor(confidences)
    save_image(torch.cat([org_img.unsqueeze(0), e_img.unsqueeze(0), c_img.unsqueeze(0)], dim=0), out_filename, value_range=(0, 1), normalize=True)


def save_panoptic(predictions, predictions_notta, _demo, out_filename):
    mask, segments, probabilities, confidences = predictions["panoptic_seg"]
    if len(predictions_notta["panoptic_seg"]) != 4:
        mask_notta, segments_notta = predictions_notta["panoptic_seg"]
        confidences_notta = torch.zeros_like(mask_notta)
    else:
        mask_notta, segments_notta, _, confidences_notta = predictions_notta["panoptic_seg"]
    # since we use cat_ids from scannet, no need for mapping
    # for segment in segments:
    #     cat_id = segment["category_id"]
    #     segment["category_name"] = demo.metadata.stuff_classes[cat_id]
    with gzip.open(out_filename, "wb") as fid:
        torch.save(
            {
                "mask": mask,
                "segments": segments,
                "mask_notta": mask_notta,
                "segments_notta": segments_notta,
                "confidences_notta": confidences_notta,
                "probabilities": probabilities,
                "confidences": confidences,
                # "feats": predictions["res3_feats"]
            }, fid
        )


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--predictions",
        help="Save raw predictions together with visualizations."
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Max procs",
    )
    parser.add_argument(
        "--p",
        type=int,
        default=0,
        help="Current proc",
    )
    parser.add_argument(
        "--zyq_code",
        help="",
        default=False,
    )
    parser.add_argument(
        "--zyq_mapping",
        help="",
        default=False,
    )
    parser.add_argument(
        "--resize",
        help="",
        default=False,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)

    if args.input:
        #zyq
        used_files = []
        for ext in ('*.JPG', '*.jpg', '*.png'):
            used_files.extend(glob.glob(os.path.join(args.input[0], ext)))
        
        used_files.sort()
        print(len(used_files))
        # used_files = used_files[:330]

        for path in tqdm.tqdm(used_files, disable=not args.output):
            path = str(path)
            #### use PIL, to be consistent with evaluation
            #### zyq: 把read_image写在这，并resize
            with PathManager.open(path, "rb") as f:
                image = Image.open(f)
                # resize
                image_width, image_height = image.size
                if args.resize:
                    image = image.resize((int(image_width/4), int(image_height/4))) 
                else:
                    image = image.resize((int(image_width), int(image_height))) 

            ##### work around this bug: https://github.com/python-pillow/Pillow/issues/3973
            image = _apply_exif_orientation(image)
            img = convert_PIL_to_numpy(image, "BGR")
            # img = read_image(path, format="BGR")



            start_time = time.time()
            augmentations = [A.HorizontalFlip(always_apply=True), A.RGBShift(always_apply=True), A.CLAHE(always_apply=True), A.RandomGamma(always_apply=True, gamma_limit=(80, 120)), A.RandomBrightnessContrast(always_apply=True),
                             A.MedianBlur(blur_limit=7, always_apply=True), A.Sharpen(alpha=(0.2, 0.4), lightness=(0.5, 1.0), always_apply=True)]
            augmentations.extend([A.Compose([augmentations[1], augmentations[2]]),
                                  A.Compose([augmentations[2], augmentations[3]]),
                                  A.Compose([augmentations[1], augmentations[3]]),
                                  A.Compose([augmentations[2], augmentations[4]]),
                                  A.Compose([augmentations[5], augmentations[6]])])
            use_old_augmentation_method = False
            
            # NOTE 以下得到的semantic， instance， probability, confidences等全部是针对ADE20K的
            predictions_0, _ = demo.run_on_image(img, visualize=False)
            
            # averaged_feats = predictions_0['res3_feats'].cpu()
            list_aug_probs, list_aug_confs = [x.cpu() for x in predictions_0["panoptic_seg"][0]], [x.cpu() for x in predictions_0["panoptic_seg"][1]]
            for aud_idx, augmentation in enumerate(augmentations):
                transformed_image = augmentation(image=img)["image"]
                aug_pred, _ = demo.run_on_image(transformed_image, visualize=False)
                if not aud_idx == 0:
                    aug_probs, aug_conf = aug_pred["panoptic_seg"][0], aug_pred["panoptic_seg"][1]
                    # aug_feat = aug_pred['res3_feats']
                else:
                    aug_probs, aug_conf = aug_pred["panoptic_seg"][0], torch.fliplr(aug_pred["panoptic_seg"][1].permute((1, 2, 0))).permute((2, 0, 1))
                    # aug_feat = torch.fliplr(aug_pred['res3_feats'])
                aug_probs = aug_probs.cpu()
                aug_conf = aug_conf.cpu()
                list_aug_probs.extend([x for x in aug_probs])
                list_aug_confs.extend([x for x in aug_conf])
                # averaged_feats += aug_feat.cpu()
            # averaged_feats /= (len(augmentations) + 1)
            tta_handler_start_time = time.time()
            tta_handler = TTAHandler(list_aug_probs, list_aug_confs)
            probabilities, confidences = tta_handler.find_tta_probabilities_and_masks()
            print(f'TTA Handler time: {time.time() - tta_handler_start_time:.2f}s')
            del tta_handler
            # todo: deleted visualizations for now, turn on if needed
            predictions, visualized_output = demo.run_post_augmentation(img, probabilities, confidences, visualize=True)
            # predictions, visualized_output = demo.run_post_augmentation(img, probabilities, confidences, visualize=False)
            
            predictions_no_tta, _ = demo.run_post_augmentation(img, predictions_0["panoptic_seg"][0], predictions_0["panoptic_seg"][1], visualize=False)
            # predictions['res3_feats'] = averaged_feats
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            


            if args.output:
                if not os.path.exists(args.output):
                    os.mkdir(args.output)
                if not os.path.exists(os.path.join(args.output, 'panoptic')):
                    os.mkdir(os.path.join(args.output, 'panoptic'))
                if not os.path.exists(os.path.join(args.output, 'visualized_ptz')):
                    os.mkdir(os.path.join(args.output, 'visualized_ptz'))
                
                if visualized_output is not None:
                    out_filename = os.path.join(args.output, 'visualized_ptz',os.path.basename(path))
                    out_filename_noext, _ = os.path.splitext(out_filename)
                    visualized_output.save(out_filename_noext + ".jpg")
                    probabilities, confidences = predictions["panoptic_seg"][2], predictions["panoptic_seg"][3]
                    entropy = probability_to_normalized_entropy(probabilities)
                    load_and_save_with_entropy_and_confidence(out_filename_noext + ".jpg", entropy, confidences)

                out_filename = os.path.join(args.output, 'panoptic',os.path.basename(path))
                out_filename_noext, _ = os.path.splitext(out_filename)
                save_panoptic(predictions, predictions_no_tta, demo, out_filename_noext + ".ptz")
                
                #save semantic
                mask, segments, _, _ = predictions["panoptic_seg"]
                semantic = convert_from_mask_to_semantics_and_instances_no_remap(mask, segments)

                if not os.path.exists(os.path.join(args.output, 'labels_m2f')):
                    os.mkdir(os.path.join(args.output, 'labels_m2f'))
                out_filename = os.path.join(args.output, 'labels_m2f',os.path.basename(path))
                out_filename_noext, _ = os.path.splitext(out_filename)
                Image.fromarray(semantic.numpy().astype(np.uint16)).save(out_filename_noext+ ".png")

            else:
                print("something wrong in the code or input")       
   
        
    dest = Path(args.output)
    visualize_mask_folder(dest / "labels_m2f")        


    if args.output:
        if not os.path.exists(args.output):
            os.mkdir(args.output)
        if not os.path.exists(os.path.join(args.output, 'panoptic')):
            os.mkdir(os.path.join(args.output, 'panoptic'))
        if not os.path.exists(os.path.join(args.output, 'visualized_ptz')):
            os.mkdir(os.path.join(args.output, 'visualized_ptz'))



    label_m2f_path =  os.path.join(args.output, 'labels_m2f')
    save_path = args.output
    root_path = str(args.input[0])

    Path(save_path).mkdir(exist_ok=True)
    Path(os.path.join(save_path, 'alpha')).mkdir(exist_ok=True)
    Path(os.path.join(save_path, 'label')).mkdir(exist_ok=True)



    used_files = []
    for ext in ('*.png', '*.jpg'):
        used_files.extend(glob.glob(os.path.join(label_m2f_path, ext)))
    used_files.sort()

    for m2f_file in tqdm.tqdm(used_files):
        if os.path.exists(os.path.join(root_path, f"{Path(m2f_file).stem}.jpg")):
            img = Image.open(os.path.join(root_path, f"{Path(m2f_file).stem}.jpg"))
        elif os.path.exists(os.path.join(root_path, f"{Path(m2f_file).stem}.png")):
            img = Image.open(os.path.join(root_path, f"{Path(m2f_file).stem}.png"))
        image_width, image_height = img.size
        ####TODO  resize or not 

        if args.resize:
            img = img.resize((int(image_width/4), int(image_height/4))) 

        img = np.array(img)

        color_ori = Image.open(m2f_file)
        color_width, color_height = color_ori.size
        color_label = custom2rgb(np.array(color_ori))
        color_label = color_label.reshape((int(color_height), int(color_width),3))

        # print(img.shape)
        # print(color_label.shape)

        merge = 0.7 * img + 0.3 * color_label
        Image.fromarray(merge.astype(np.uint8)).save(os.path.join(save_path, 'alpha',f"{Path(m2f_file).stem}.jpg"))
        Image.fromarray(color_label.astype(np.uint8)).save(os.path.join(save_path, 'label',f"{Path(m2f_file).stem}.jpg"))