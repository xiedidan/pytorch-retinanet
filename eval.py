import time
import os
import copy
import argparse
import pdb
import collections
import sys
import logging
from datetime import datetime

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision

import model
from anchors import Anchors
import losses
from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

import coco_eval
import csv_eval
from log import *

assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))

LOG_SIZE = 512 * 1024 * 1024 # 512M
LOGGER_NAME = 'eval'
LOG_PATH = './log'

MAX_DETECTIONS = 3

'''
# scan from 0.3 to 0.8
IOU_MIN = 0.3
IOU_SCALE = 0.6
IOU_STEPS = 6
'''

# scan from 0.05 to 0.55
SCORE_MIN = 0.05
SCORE_SCALE = 0.55
SCORE_STEPS = 11

IOU_MIN = 0.4
IOU_SCALE = 0.0
IOU_STEPS = 1

'''
SCORE_MIN = 0.1
SCORE_SCALE = 0.0
SCORE_STEPS = 1
'''

def main(args=None):

	parser     = argparse.ArgumentParser(description='Simple training script for evaluating a RetinaNet network.')

	parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
	parser.add_argument('--coco_path', help='Path to COCO directory')
	parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
	parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

	parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
	parser.add_argument('--epochs', help='Number of epochs', type=int, default=0)
	parser.add_argument('--checkpoint', help='Path to checkpoint')


	parser.add_argument('--log_prefix', default='eval', help='log file path = "./log/{}-{}.log".format(log_prefix, now)')
	parser.add_argument('--log_level', default=logging.DEBUG, type=int, help='log level')

	parser = parser.parse_args(args)

	# setup logger
	if not os.path.isdir(LOG_PATH):
		os.mkdir(LOG_PATH)

	now = datetime.now()

	logger = setup_logger(
		LOGGER_NAME,
		os.path.join(
			LOG_PATH,
			'{}_{}.log'.format(parser.log_prefix, now.strftime('%Y-%m-%d_%H:%M:%S'))
		),
		LOG_SIZE,
		parser.log_level
	)

	# Create the data loaders
	if parser.dataset == 'coco':

		if parser.coco_path is None:
			raise ValueError('Must provide --coco_path when training on COCO,')

		dataset_train = CocoDataset(parser.coco_path, set_name='train2017', transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
		dataset_val = CocoDataset(parser.coco_path, set_name='val2017', transform=transforms.Compose([Normalizer(), Resizer()]))

	elif parser.dataset == 'csv':

		if parser.csv_classes is None:
			raise ValueError('Must provide --csv_classes when training on CSV,')

		if parser.csv_val is None:
			dataset_val = None
			print('No validation annotations provided.')
		else:
			dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))

	else:
		raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

	if dataset_val is not None:
		sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=8, drop_last=False)
		dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

	# Create the model
	if parser.depth == 18:
		retinanet = model.resnet18(num_classes=dataset_val.num_classes(), pretrained=True)
	elif parser.depth == 34:
		retinanet = model.resnet34(num_classes=dataset_val.num_classes(), pretrained=True)
	elif parser.depth == 50:
		retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)
	elif parser.depth == 101:
		retinanet = model.resnet101(num_classes=dataset_val.num_classes(), pretrained=True)
	elif parser.depth == 152:
		retinanet = model.resnet152(num_classes=dataset_val.num_classes(), pretrained=True)
	else:
		raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')		

	use_gpu = True

	if use_gpu:
		retinanet = retinanet.cuda()
	
	retinanet = torch.nn.DataParallel(retinanet).cuda()
	retinanet.training = False
    
	if parser.epochs > 0:
		for epoch in range(parser.epochs):
			logger.info('Epoch {}:'.format(epoch))

			model_file = '{}_{}.pth'.format(parser.checkpoint, epoch)
			print('Evaluating model: {}'.format(model_file))

			retinanet.module = torch.load(model_file)

			mAP = csv_eval.evaluate(dataset_val, retinanet)
			logger.info(mAP)
	else:
		retinanet.module = torch.load(parser.checkpoint)

		for i in range(IOU_STEPS):
			iou = IOU_MIN + i * (IOU_SCALE / IOU_STEPS)

			for j in range(SCORE_STEPS):
				score = SCORE_MIN + j * (SCORE_SCALE / SCORE_STEPS)

				print('iou_threshold: {}, score_threshold: {}'.format(iou, score))

				mAP = csv_eval.evaluate_rsna(
					dataset_val,
					retinanet,
					score_threshold=score,
					max_detections=MAX_DETECTIONS,
				)

				logger.info('iou_threshold: {}, score_threshold: {}'.format(iou, score))
				logger.info(mAP)

if __name__ == '__main__':
	main()
