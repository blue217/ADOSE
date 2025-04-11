# -*- coding: utf-8 -*-
"""
Implements a domain adaptation network
Adapted from https://github.com/jhoffman/cycada_release
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from .models import register_model, get_model
from model.classify_model import MLPModel_TextCnn

@register_model('AdaptNet')
class AdaptNet(nn.Module):
	"Defines an Adapt Network."
	def __init__(self, num_cls=2, textcnn='roberta-non', dataset='Pheme',
				 src_weights_init=None, weights_init=None, weight_sharing='full'):
		super(AdaptNet, self).__init__()
		self.name = 'AdaptNet'

		self.num_cls = num_cls
		self.cls_criterion = nn.CrossEntropyLoss()
		self.gan_criterion = nn.CrossEntropyLoss()
		self.weight_sharing = weight_sharing
		self.setup_net(textcnn, dataset)
		if weights_init is not None:
			self.load(weights_init)
		elif src_weights_init is not None:
			self.load_src_net(src_weights_init)
		else:
			raise Exception('AdaptNet must be initialized with weights.')
	
	def custom_copy(self, src_net, weight_sharing):
		"""
		Vary degree of weight sharing between source and target CNN's
		"""
		tgt_net = copy.deepcopy(src_net)
		if weight_sharing != 'None':
			if weight_sharing == 'classifier': tgt_net.classifier = src_net.classifier
			elif weight_sharing == 'full': tgt_net = src_net
		return tgt_net
	
	def setup_net(self, textcnn_mode, dataset):
		"""Setup source, target and discriminator networks."""
		self.src_net = MLPModel_TextCnn(
            out_size=256, num_label=2, freeze_id=8,
            d_prob=0.3, kernel_sizes=[3, 4, 5], num_filters=100,
            mode=textcnn_mode, dataset_name=dataset)
		self.tgt_net = self.custom_copy(self.src_net, self.weight_sharing)

		input_dim = self.num_cls
		self.discriminator = nn.Sequential(
				nn.Linear(input_dim, 128),
				nn.ReLU(),
				nn.Linear(128, 128),
				nn.ReLU(),
				nn.Linear(128, 2),
				)

		# self.image_size = self.src_net.image_size
		# self.num_channels = self.src_net.num_channels

	def load(self, init_path):
		"Loads full src and tgt models."
		net_init_dict = torch.load(init_path, map_location=torch.device('cpu'))
		self.load_state_dict(net_init_dict, strict=False)

	def load_src_net(self, init_path):
		"""Initialize source and target with source
		weights."""
		# self.src_net.load(init_path)
		# self.tgt_net.load(init_path)
		checkpoint = torch.load(init_path, map_location='cuda')
		self.src_net.load_state_dict(checkpoint)
		self.tgt_net.load_state_dict(checkpoint)

	def save(self, out_path):
		torch.save(self.state_dict(), out_path)

	def save_tgt_net(self, out_path):
		torch.save(self.tgt_net.state_dict(), out_path)