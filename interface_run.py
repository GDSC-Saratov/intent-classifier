from flask import Flask, request

# local modules
import secret
from models import second_model

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False # for return JSON by Flask as UTF-8

model = second_model.IntentClassifier('bert-base-multilingual-cased', num_labels=6, load_bert_model_state_dict=False)
model.to(model.device)
model.load(secret.CHECKPOINTS_PATH + 'second_model.pt')


@app.route('/predict', methods=['POST'])
def predict():
	data = request.get_json()
	predict = model.predict(data['text'])
	return {'predict': {'label': predict['label'],
						'prob': predict['prob']}}


if __name__ == '__main__':
	app.run(debug=True)