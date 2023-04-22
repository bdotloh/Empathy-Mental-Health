import torch
import codecs
import numpy as np


import pandas as pd
import re
import csv
import numpy as np
import sys
import argparse

import time

from sklearn.metrics import f1_score

from transformers import RobertaTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split
from torch.nn.functional import log_softmax, softmax

from .models.models import BiEncoderAttentionWithRationaleClassification
from transformers import AdamW, RobertaConfig

import datetime



class EmpathyClassifier():

	def __init__(self, 
			device,
			ER_model_path, 
			IP_model_path,
			EX_model_path,
			batch_size=1):
		
		self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
		self.batch_size = batch_size
		self.device = device
		self.val2label={0:'poor',1:'decent',2:'great'} # for printing purposes

		self.model_ER = BiEncoderAttentionWithRationaleClassification()
		self.model_IP = BiEncoderAttentionWithRationaleClassification()
		self.model_EX = BiEncoderAttentionWithRationaleClassification()

		ER_weights = torch.load(ER_model_path, map_location = device)
		self.model_ER.load_state_dict(ER_weights)

		IP_weights = torch.load(IP_model_path, map_location = device)
		self.model_IP.load_state_dict(IP_weights)

		EX_weights = torch.load(EX_model_path, map_location = device)
		self.model_EX.load_state_dict(EX_weights)

		self.model_ER.to(self.device)
		self.model_IP.to(self.device)
		self.model_EX.to(self.device)

	def eval(self):
		self.model_ER.eval()
		self.model_IP.eval()
		self.model_EX.eval()
	
	def tokenize_posts(self, posts):
		input_ids = []
		attention_masks = []
		
		for sent in posts:
			encoded_dict = self.tokenizer.encode_plus(
								sent,                      # Sentence to encode.
								add_special_tokens = True, # Add '[CLS]' and '[SEP]'
								max_length = 128,           # Pad & truncate all sentences.
								padding = 'max_length',
								return_attention_mask = True,   # Construct attn. masks.
								return_tensors = 'pt',     # Return pytorch tensors.
						)
			
			input_ids.append(encoded_dict['input_ids'])
			attention_masks.append(encoded_dict['attention_mask'])
		
		input_ids = torch.cat(input_ids, dim=0)
		attention_masks = torch.cat(attention_masks, dim=0)

		return input_ids, attention_masks


	def predict_empathy(self, seeker_posts, response_posts):
		input_ids_SP, attention_masks_SP = self.tokenize_posts(seeker_posts)		
		input_ids_RP, attention_masks_RP = self.tokenize_posts(response_posts)	
	
		dataset = TensorDataset(input_ids_SP, attention_masks_SP, input_ids_RP, attention_masks_RP)

		dataloader = DataLoader(
			dataset, # The test samples.
			sampler = SequentialSampler(dataset), # Pull out batches sequentially.
			batch_size = self.batch_size # Evaluate with this batch size.
		)

		self.model_ER.eval()
		self.model_IP.eval()
		self.model_EX.eval()

		for batch in dataloader:
			b_input_ids_SP = batch[0].to(self.device)
			b_input_mask_SP = batch[1].to(self.device)
			b_input_ids_RP = batch[2].to(self.device)
			b_input_mask_RP = batch[3].to(self.device)

			with torch.no_grad():
				(logits_empathy_ER, logits_rationale_ER,) = self.model_ER(input_ids_SP = b_input_ids_SP,
														input_ids_RP = b_input_ids_RP, 
														token_type_ids_SP=None,
														token_type_ids_RP=None, 
														attention_mask_SP=b_input_mask_SP,
														attention_mask_RP=b_input_mask_RP)
				
				(logits_empathy_IP, logits_rationale_IP,) = self.model_IP(input_ids_SP = b_input_ids_SP,
														input_ids_RP = b_input_ids_RP, 
														token_type_ids_SP=None,
														token_type_ids_RP=None, 
														attention_mask_SP=b_input_mask_SP,
														attention_mask_RP=b_input_mask_RP)

				(logits_empathy_EX, logits_rationale_EX,) = self.model_EX(input_ids_SP = b_input_ids_SP,
														input_ids_RP = b_input_ids_RP, 
														token_type_ids_SP=None,
														token_type_ids_RP=None, 
														attention_mask_SP=b_input_mask_SP,
														attention_mask_RP=b_input_mask_RP)

				
			logits_empathy_ER = logits_empathy_ER.detach().cpu().numpy().tolist()
			predictions_ER = np.argmax(logits_empathy_ER, axis=1).flatten()

			logits_empathy_IP = logits_empathy_IP.detach().cpu().numpy().tolist()
			predictions_IP = np.argmax(logits_empathy_IP, axis=1).flatten()

			logits_empathy_EX = logits_empathy_EX.detach().cpu().numpy().tolist()
			predictions_EX = np.argmax(logits_empathy_EX, axis=1).flatten()


			logits_rationale_ER = logits_rationale_ER.detach().cpu().numpy()
			predictions_rationale_ER = np.argmax(logits_rationale_ER, axis=2)

			logits_rationale_IP = logits_rationale_IP.detach().cpu().numpy()
			predictions_rationale_IP = np.argmax(logits_rationale_IP, axis=2)

			logits_rationale_EX = logits_rationale_EX.detach().cpu().numpy()
			predictions_rationale_EX = np.argmax(logits_rationale_EX, axis=2)

		return (logits_empathy_ER, predictions_ER, \
		 	logits_empathy_IP, predictions_IP, \
			logits_empathy_EX, predictions_EX, \
			logits_rationale_ER, predictions_rationale_ER, \
			logits_rationale_IP, predictions_rationale_IP, \
			logits_rationale_EX,predictions_rationale_EX)
	
	def pipeline(self, conversation):
		try:
			if len(conversation) % 2 != 0:
				conversation = conversation[:-1]
    	
			empathy_scores = {'er':[],'ip':[],'ex':[]}
			empathy_rationales = {'er':[],'ip':[],'ex':[]}
			seeker_posts = conversation[::2]
			response_posts = conversation[1::2]
			input_ids_RP, attention_masks_RP = self.tokenize_posts(response_posts)	
			for t in range(len(seeker_posts)):
				_, er_pred, _, ip_pred, _, ex_pred, _, er_rtnle, _, ip_rtnle, _, ex_rtnle = self.predict_empathy([seeker_posts[t]],[response_posts[t]])
				empathy_scores['er'].extend(er_pred)
				empathy_scores['ip'].extend(ip_pred)
				empathy_scores['ex'].extend(ex_pred)

	
				empathy_rationales['er'].append(torch.masked_select(input_ids_RP[t].squeeze(0), ~torch.BoolTensor(er_rtnle)))
				empathy_rationales['ip'].append(torch.masked_select(input_ids_RP[t].squeeze(0), ~torch.BoolTensor(ip_rtnle)))
				empathy_rationales['ex'].append(torch.masked_select(input_ids_RP[t].squeeze(0), ~torch.BoolTensor(ex_rtnle)))
			return empathy_scores, empathy_rationales
		
		except IndexError:
			print('no responses to measure')
	
	def compute_empathy(self, seekerRespPostPair):
		seekerRespPostPairTokenised=self.tokenizer.batch_encode_plus(
								seekerRespPostPair,
								add_special_tokens = True, # Add '[CLS]' and '[SEP]'
								max_length = 64,           # Pad & truncate all sentences.
								padding = 'max_length',
								return_attention_mask = True,   # Construct attn. masks.
								return_tensors = 'pt',
								)

		input_ids_SP=seekerRespPostPairTokenised['input_ids'][::2]
		attention_masks_SP=seekerRespPostPairTokenised['attention_mask'][::2]

		input_ids_RP=seekerRespPostPairTokenised['input_ids'][1::2]
		attention_masks_RP=seekerRespPostPairTokenised['attention_mask'][1::2]

		dataset = TensorDataset(input_ids_SP, attention_masks_SP, input_ids_RP, attention_masks_RP)

		dataloader = DataLoader(
			dataset, # The test samples.
			sampler = SequentialSampler(dataset), # Pull out batches sequentially.
			batch_size = len(input_ids_SP)# Evaluate with this batch size.
		)

		#b_input_ids_SP, b_input_mask_SP, b_input_ids_RP, b_input_mask_RP=next(iter(dataloader))
		batch=next(iter(dataloader))
		b_input_ids_SP=batch[0].to(self.device)
		b_input_mask_SP=batch[1].to(self.device)
		b_input_ids_RP=batch[2].to(self.device)
		b_input_mask_RP=batch[3].to(self.device)

		with torch.no_grad():     
			(logits_empathy_ER, logits_rationale_ER)=self.model_ER(
				input_ids_SP = b_input_ids_SP,
				input_ids_RP = b_input_ids_RP, 
				token_type_ids_SP=None,
				token_type_ids_RP=None, 
				attention_mask_SP=b_input_mask_SP,
				attention_mask_RP=b_input_mask_RP
			)
			(logits_empathy_IP, logits_rationale_IP)=self.model_IP(
				input_ids_SP = b_input_ids_SP,
				input_ids_RP = b_input_ids_RP, 
				token_type_ids_SP=None,
				token_type_ids_RP=None, 
				attention_mask_SP=b_input_mask_SP,
				attention_mask_RP=b_input_mask_RP
			)

			(logits_empathy_EX, logits_rationale_EX)=self.model_EX(
				input_ids_SP = b_input_ids_SP,
				input_ids_RP = b_input_ids_RP, 
				token_type_ids_SP=None,
				token_type_ids_RP=None, 
				attention_mask_SP=b_input_mask_SP,
				attention_mask_RP=b_input_mask_RP
			)

		topER=torch.topk(torch.softmax(logits_empathy_ER,dim=1),k=1)
		topERVal=topER.values.view(-1)
		topERInd=topER.indices.view(-1)

		topIp=torch.topk(torch.softmax(logits_empathy_IP,dim=1),k=1)
		topIpVal=topIp.values.view(-1)
		topIpInd=topIp.indices.view(-1)

		topEx=torch.topk(torch.softmax(logits_empathy_EX,dim=1),k=1)
		topExVal=topEx.values.view(-1)
		topExInd=topEx.indices.view(-1)
		
		results_str=f"This response is a...\n\
		{self.val2label[topERInd.item()]} attempt at catering to the help-seeker's emotions ({round(topERVal.item()*100,2)}% prob.)\n\
		{self.val2label[topExInd.item()]} attempt at better understanding the implicit meanings behind help-seeker's post ({round(topExVal.item()*100,2)} % prob.)\n\
		{self.val2label[topIpInd.item()]} attempt at communicating an understanding of the help-seeker's problems ({round(topIpVal.item()*100,2)} % prob.)"
		
		return results_str,softmax(logits_empathy_ER,dim=1).view(-1), softmax(logits_empathy_EX, dim=1).view(-1), softmax(logits_empathy_IP, dim=1).view(-1)
	
	def compute_empathy_batch(self, seeker_posts_tokenised, supporter_posts_tokenised):
		# prepare data
		dataset=TensorDataset(
			seeker_posts_tokenised['input_ids'], 
			seeker_posts_tokenised['attention_mask'], 
			supporter_posts_tokenised['input_ids'], 
			supporter_posts_tokenised['attention_mask'])
    
		dataloader = DataLoader(
					dataset, # The test samples.
					sampler = SequentialSampler(dataset), # Pull out batches sequentially.
					batch_size = len(dataset)# Evaluate with this batch size.
				)

		#b_input_ids_SP, b_input_mask_SP, b_input_ids_RP, b_input_mask_RP=next(iter(dataloader))
		batch=next(iter(dataloader))
		b_input_ids_SP=batch[0].to(self.device)
		b_input_mask_SP=batch[1].to(self.device)
		b_input_ids_RP=batch[2].to(self.device)
		b_input_mask_RP=batch[3].to(self.device)

		with torch.no_grad():    
			logits_empathy_ER, _ = self.model_ER(
				input_ids_SP = b_input_ids_SP,
				input_ids_RP = b_input_ids_RP, 
				token_type_ids_SP=None,
				token_type_ids_RP=None, 
				attention_mask_SP=b_input_mask_SP,
				attention_mask_RP=b_input_mask_RP
			)
			logits_empathy_IP, _ = self.model_IP(
				input_ids_SP = b_input_ids_SP,
				input_ids_RP = b_input_ids_RP, 
				token_type_ids_SP=None,
				token_type_ids_RP=None, 
				attention_mask_SP=b_input_mask_SP,
				attention_mask_RP=b_input_mask_RP
			)
			logits_empathy_EX, _=self.model_EX(
				input_ids_SP = b_input_ids_SP,
				input_ids_RP = b_input_ids_RP, 
				token_type_ids_SP=None,
				token_type_ids_RP=None, 
				attention_mask_SP=b_input_mask_SP,
				attention_mask_RP=b_input_mask_RP
			)

		return softmax(logits_empathy_ER,dim=1).view(-1,3), softmax(logits_empathy_EX, dim=1).view(-1,3), softmax(logits_empathy_IP, dim=1).view(-1,3)