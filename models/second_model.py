import torch
from transformers import BertModel, BertConfig, BertTokenizer

from typing import Any, Union, Tuple, Dict


def get_formal_label(label: int, num_labels: int, dtype: Any = int) -> torch.tensor:
	formal_label = [0]*num_labels
	formal_label[label] = 1
	return torch.tensor([formal_label], dtype=dtype)


class Dataset(torch.utils.data.Dataset):
	def __init__(self, seqs: list, labels: list, tokenizer: Any, num_labels: int):
		self.tokenized = [tokenizer.tokenize(seq, return_tensors='pt') for seq in seqs]
		self.formal_labels = [get_formal_label(label, num_labels=num_labels) for label in labels]

	def __getitem__(self, i) -> Tuple[Dict[str, torch.tensor], torch.tensor]:
		return self.tokenized[i], self.formal_labels[i]
	
	def __len__(self) -> int:
		return len(self.tokenized)


class IntentClassifier(torch.nn.Module):
	def __init__(self, pretrained_bert_model: str, num_labels: int, load_bert_model_state_dict: bool = True):
		super(IntentClassifier, self).__init__()
		self.pretrained_bert_model = pretrained_bert_model
		self.num_labels = num_labels

		# layers
		if load_bert_model_state_dict:
			self.bert = BertModel.from_pretrained(self.pretrained_bert_model)
		else:
			self.bert = BertModel(BertConfig.from_pretrained(pretrained_bert_model))
		#classification layers need define
		self.dropout = torch.nn.Dropout(p=0.1, inplace=False)
		self.classifier = torch.nn.Linear(in_features=768, out_features=self.num_labels, bias=True)

		self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_bert_model)

		if torch.cuda.is_available():
			self.device = torch.device('cuda:0')
		else:
			self.device = torch.device('cpu')
	
	
	def forward(self, input_ids: torch.tensor, token_type_ids: torch.tensor, attention_mask: torch.tensor) -> torch.tensor:
		pooled_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
		pooled_output = self.dropout(pooled_output[1])
		logits = self.classifier(pooled_output)
		return logits

	
	def train(self, dataset: Dataset, epochs: int):
		criterion = torch.nn.CrossEntropyLoss().to(self.device) #need define
		optimizer = torch.optim.AdamW(self.parameters()) #need define

		for epoch in range(epochs):

			for i, data in enumerate(dataset, 0):
				tokenized, formal_label = data
				
				optimizer.zero_grad()

				logits = self.forward(tokenized['input_ids'], tokenized['token_type_ids'], tokenized['attention_mask'])
				loss = criterion(logits, formal_label.float().to(self.device))
				loss.backward()
				optimizer.step()
			
		print("Training if finish")
	

	def load(self, fp: str):
		self.load_state_dict(torch.load(fp))

	
	def predict(self, text: str) -> Dict[str, Union[int, float]]:
		tokenized = self.tokenizer(text, return_tensors='pt')
		logits = self.forward(tokenized['input_ids'], tokenized['token_type_ids'], tokenized['attention_mask'])

		max_prob_label, max_prob = -1, -1
		for label, prob in enumerate(logits[0], 0):
			prob = prob.item()
			if prob > max_prob:
				max_prob_label, max_prob = label, prob
		
		if max_prob < 0.51:
			max_prob_label, max_prob = 6, 1.0 - torch.mean(logits[0]).item()

		return {'label': max_prob_label, 'prob': round(max_prob, 2)}