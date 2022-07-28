import torch
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer, get_linear_schedule_with_warmup
from transformers.modeling_outputs import SequenceClassifierOutput

from typing import Any, Union, Tuple, List, Dict


def get_formated_label(label: int, num_labels: int) -> List[int]:
	'''Getting formated label for NN'''
	formated_label = [0]*num_labels
	formated_label[label] = 1
	return formated_label


class Dataset(torch.utils.data.Dataset):
	
	def __init__(self, seqs: list, labels: list, tokenizer: object, num_labels: int):
		self.tokenized = [tokenizer(seq, max_length=512, padding='max_length') for seq in seqs] # max_length=512 'cause max_position_embeddings=512 in bert
		self.formated_labels = [get_formated_label(label, num_labels) for label in labels]

	def __getitem__(self, i) -> Tuple[Dict[str, torch.tensor], torch.tensor]:
		tokenized = {k: torch.tensor(v) for k, v in self.tokenized[i].items()}
		formated_label = torch.tensor(self.formated_labels[i]).float()
		return tokenized, formated_label
	
	def __len__(self) -> int:
		return len(self.tokenized)


class IntentClassifier(torch.nn.Module):

	def __init__(self, pretrained_bert_model: str, num_labels: int, load_bert_model_state_dict: bool = True):
		super(IntentClassifier, self).__init__()
		self.pretrained_bert_model = pretrained_bert_model
		self.num_labels = num_labels

		self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_bert_model)

		# layers
		if load_bert_model_state_dict:
			self.bertfsc = BertForSequenceClassification.from_pretrained(self.pretrained_bert_model, num_labels=self.num_labels)
		else:
			self.bertfsc = BertForSequenceClassification(BertConfig.from_pretrained(self.pretrained_bert_model), num_labels=self.num_labels)

		if torch.cuda.is_available():
			self.device = torch.device('cuda:0')
		else:
			self.device = torch.device('cpu')
	
	
	def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor, token_type_ids: torch.tensor, labels: torch.tensor = None) -> SequenceClassifierOutput:
		bertfsc_output = self.bertfsc(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
		return bertfsc_output

	
	def train(self, dataloader: torch.utils.data.DataLoader(Dataset), optim: Any, lr: float, epochs: int) -> List[float]:
		'''Training layers (without untrained) with dataloader of Dataset
		with optional optim func with optional learning rate (lr)'''
		# lr=2e-5 or 3e-5 or 5e-5 recommended for sequence classification by bert researchers (https://arxiv.org/pdf/1810.04805.pdf)
		optim = optim(self.parameters(), lr=lr)

		num_training_steps = len(train_dataloader) * epochs
		scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=num_training_steps)

		# for statistics
		history_train_loss = list()
		ebfps = len(train_dataloader) / 10 # ebfps - every batch for print statistics

		for epoch in range(epochs):
			for i, batch in enumerate(train_dataloader):
				input_ids = batch[0]['input_ids'].to(self.device)
				attention_mask = batch[0]['attention_mask'].to(self.device)
				token_type_ids = batch[0]['token_type_ids'].to(self.device)
				formated_labels = batch[1].to(self.device)

				self.zero_grad()

				forward_output = self.forward(input_ids, attention_mask, token_type_ids, formated_labels)
				loss = forward_output[0]
				loss.backward()

				# for statistics
				train_loss = loss.item()
				if i % ebfps == ebfps - 1:
						print(f"[Epoch: {epoch + 1}, batch: {i + 1}]. Loss: {train_loss}")
				history_train_loss.append(train_loss)
				
				torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

				optim.step()
				scheduler.step()
		
		print("Training is finish.")
		return history_train_loss
	

	def load(self, fp: str):
		'''Loading state dict (weights and biases) from pytorch checkpoint'''
		self.load_state_dict(torch.load(fp))

	
	def predict(self, text: str) -> Dict[str, Union[int, float]]:
		
		tokenized = self.tokenizer(text, return_tensors='pt').to(self.device)
		forward_output = self.forward(tokenized['input_ids'], tokenized['attention_mask'], tokenized['token_type_ids'])
		logits = forward_output[0]

		max_prob_label, max_prob = -1, -1
		for label, prob in enumerate(logits[0], 0):
			prob = prob.item()
			if prob > max_prob:
				max_prob_label, max_prob = label, prob
		
		if max_prob < 0.51:
			max_prob_label, max_prob = 6, 1.0 - torch.mean(logits[0]).item()

		print(logits) # for debug
		return {'label': max_prob_label, 'prob': round(max_prob, 3)}