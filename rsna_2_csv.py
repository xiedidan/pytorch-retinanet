# -*- coding: utf-8 -*-
import sys
import os
import pickle
import argparse
import itertools
import math
import random
import shutil

import cv2
from tqdm import tqdm
import pandas as pd
import SimpleITK as sitk

# constants
RSNA_LABEL_FILE = 'stage_1_train_labels.csv'
RSNA_CLASS_FILE = 'stage_1_detailed_class_info.csv'
CLASS_MAPPING = {
    'No Lung Opacity / Not Normal': 0,
    'Lung Opacity': 1
}
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024

# helpers
def to_point(corner):
    return [
        corner[0],
        corner[1],
        corner[0] + corner[2],
        corner[1] + corner[3]
    ]

def convert_jpeg(filename, patient_id, output_path):
    output_filename = os.path.join(output_path, '{}.jpg'.format(patient_id))

    if not os.path.exists(output_path):
        os.mkdir(output_path)
        
    if not os.path.exists(output_filename):
        ds = sitk.ReadImage(filename)
        img_array = sitk.GetArrayFromImage(ds)

        cv2.imwrite(
            output_filename,
            img_array[0]
        )

def convert(filename, sample_list, anno_df, class_df, root_path):
    lines = []
    
    print('Export {}...'.format(filename))

    for patient_id in tqdm(sample_list):
        class_rows = class_df[class_df['patientId'] == patient_id]
        class_row = class_rows.iloc[0]
        class_name = class_row['class']

        anno_rows = anno_df[anno_df['patientId'] == patient_id]
        image_path = os.path.join(root_path, '{}.dcm'.format(patient_id))
        jpg_path = os.path.join('{}_jpg'.format(root_path), '{}.jpg'.format(patient_id))

        # convert to jpeg
        convert_jpeg(
            image_path,
            patient_id,
            '{}_jpg'.format(root_path)
        )

        for index, anno_row in anno_rows.iterrows():
            if class_name == 'Normal':
                # write an empty row
                line = '{},,,,,\n'.format(jpg_path)
            elif class_name == 'No Lung Opacity / Not Normal':
                # write a whole image annotation
                line = '{},{},{},{},{},{}\n'.format(
                    jpg_path,
                    0,
                    0,
                    IMAGE_WIDTH,
                    IMAGE_HEIGHT,
                    class_name
                )
            else: # Lung Opacity
                # convert coords
                a_cn = [
                    anno_row['x'],
                    anno_row['y'],
                    anno_row['width'],
                    anno_row['height']
                ]

                a_pt = to_point(a_cn)

                # write bbox row
                line = '{},{},{},{},{},{}\n'.format(
                    jpg_path,
                    int(a_pt[0]),
                    int(a_pt[1]),
                    int(a_pt[2]),
                    int(a_pt[3]),
                    class_name
                )
            
            lines.append(line)

    # write csv
    csv_path = os.path.join('./', filename)

    with open(csv_path, 'w') as file:
        for line in lines:
            file.write(line)

def pick_randomly(root, src_path, dest_path, count):
    # mk output dir
    sample_path = os.path.join(root, dest_path)
    if not os.path.exists(sample_path):
        os.mkdir(sample_path)

    # list source files
    file_path = os.path.join(root, src_path)
    file_list = os.listdir(file_path)

    # pick randomly
    sample_list = random.sample(file_list, count)

    # move
    for filename in sample_list:
        src_file = os.path.join(root, src_path, filename)
        dest_file = os.path.join(root, dest_path, filename)
        shutil.move(src_file, dest_file)

    return sample_list

# argparser
parser = argparse.ArgumentParser(description='RSNA dataset convertor (to CSV dataset)')
parser.add_argument('--root', default='./rsna-pneumonia-detection-challenge/', help='dataset root path')
parser.add_argument('--fold', default=4, type=int, help='sub-sets of k-fold for training, set 1 to disable k-fold')
parser.add_argument('--val', default=1000, type=int, help='samples for validation')
parser.add_argument('--eval', default=1000, type=int, help='samples for local evaluation')
parser.add_argument('--splitted', default=False, action='store_true', help='whether val / test are already splitted')
parser.add_argument('--classification', default=False, action='store_true', help='whether global classification is added to network')
flags = parser.parse_args()

if __name__ == '__main__':
    anno_path = os.path.join(flags.root, RSNA_LABEL_FILE)
    class_path = os.path.join(flags.root, RSNA_CLASS_FILE)

    anno_df = pd.read_csv(anno_path)
    class_df = pd.read_csv(class_path)

    if not flags.splitted:
        # split val / eval
        val_list = pick_randomly(
            flags.root,
            'train/',
            'val/',
            flags.val
        )

        eval_list = pick_randomly(
            flags.root,
            'train/',
            'eval/',
            flags.eval
        )

    # re-list train / val / eval images
    train_path = os.path.join(flags.root, 'train')
    val_path = os.path.join(flags.root, 'val')
    eval_path = os.path.join(flags.root, 'eval')

    train_list = os.listdir(train_path)
    train_list = [filename.split('.')[0] for filename in train_list]

    # k-fold split for training
    subset_size = math.ceil(len(train_list) / flags.fold)

    for i in range(flags.fold):
        sample_list = random.sample(train_list, subset_size)

        convert(
            'rsna-train-{}.csv'.format(i),
            sample_list,
            anno_df,
            class_df,
            train_path
        )

    # val
    if os.path.exists(val_path):
        val_list = os.listdir(val_path)
        val_list = [filename.split('.')[0] for filename in val_list]

        convert(
            'rsna-val.csv',
            val_list,
            anno_df,
            class_df,
            val_path
        )

    # eval
    if os.path.exists(eval_path):
        eval_list = os.listdir(eval_path)
        eval_list = [filename.split('.')[0] for filename in eval_list]

        convert(
            'rsna-eval.csv',
            eval_list,
            anno_df,
            class_df,
            eval_path
        )
        
    # class mapping
    with open('./rsna-class-mapping.csv', 'w') as file:
        for key in CLASS_MAPPING.keys():
            file.write('{},{}\n'.format(key, CLASS_MAPPING[key]))
