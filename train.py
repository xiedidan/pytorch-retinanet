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
from history import *

assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))

LOG_SIZE = 512 * 1024 * 1024 # 512M
LOGGER_NAME = 'train-val'
LOG_PATH = './log'

def main(args=None):

	parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

	parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
	parser.add_argument('--coco_path', help='Path to COCO directory')
	parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
	parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
	parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

	parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
	parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)

	parser.add_argument('--log_prefix', default='train', help='log file path = "./log/{}-{}.log".format(log_prefix, now)')
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

	# setup history
	history = History()

	# Create the data loaders
	if parser.dataset == 'coco':

		if parser.coco_path is None:
			raise ValueError('Must provide --coco_path when training on COCO,')

		dataset_train = CocoDataset(parser.coco_path, set_name='train2017', transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
		dataset_val = CocoDataset(parser.coco_path, set_name='val2017', transform=transforms.Compose([Normalizer(), Resizer()]))

	elif parser.dataset == 'csv':

		if parser.csv_train is None:
			raise ValueError('Must provide --csv_train when training on CSV,')

		if parser.csv_classes is None:
			raise ValueError('Must provide --csv_classes when training on CSV,')


		dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

		if parser.csv_val is None:
			dataset_val = None
			print('No validation annotations provided.')
		else:
			dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))

	else:
		raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

	sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
	dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

	if dataset_val is not None:
		sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
		dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

	# Create the model
	if parser.depth == 18:
		retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
	elif parser.depth == 34:
		retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
	elif parser.depth == 50:
		retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
	elif parser.depth == 101:
		retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
	elif parser.depth == 152:
		retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
	else:
		raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')		

	use_gpu = True

	if use_gpu:
		retinanet = retinanet.cuda()
	
	retinanet = torch.nn.DataParallel(retinanet).cuda()

	retinanet.training = True

	optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

	loss_hist = collections.deque(maxlen=500)

	retinanet.train()
	retinanet.module.freeze_bn()

	print('Num training images: {}'.format(len(dataset_train)))

	for epoch_num in range(parser.epochs):
		print('Training dataset, epoch: {}'.format(epoch_num))

		retinanet.train()
		retinanet.module.freeze_bn()
		
		epoch_loss = []
		
		batch_size = len(dataloader_train)

		with tqdm(total=batch_size) as pbar:
			for iter_num, data in enumerate(dataloader_train):
				try:
					optimizer.zero_grad()

					classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])

					classification_loss = classification_loss.mean()
					regression_loss = regression_loss.mean()

					loss = classification_loss + regression_loss
					
					if bool(loss == 0):
						continue

					loss.backward()

					torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

					optimizer.step()

					loss_hist.append(float(loss.item()))

					epoch_loss.append(float(loss.item()))

					logger.info('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
						epoch_num,
						iter_num,
						float(classification_loss.item()),
						float(regression_loss.item()),
						np.mean(loss_hist)
					))
					
					del classification_loss
					del regression_loss
				except Exception as e:
					print(e)
					logger.error(e)
					continue
				finally:
					pbar.update(1)

			history.train_loss.append(np.mean(epoch_loss))

		if parser.dataset == 'coco':

			print('Evaluating dataset')

			coco_eval.evaluate_coco(dataset_val, retinanet)

		elif parser.dataset == 'csv' and parser.csv_val is not None:

			print('Evaluating dataset')

			mAP = csv_eval.evaluate(dataset_val, retinanet)

			# rsna specific
			history.val_accu.append(mAP[2][0])

			logger.info(mAP)
		
		scheduler.step(np.mean(epoch_loss))	

		torch.save(retinanet.module, '{}_retinanet_{}.pth'.format(parser.dataset, epoch_num))

		history.save()

	retinanet.eval()

	torch.save(retinanet, 'model_final.pth'.format(epoch_num))

if __name__ == '__main__':
 main()
