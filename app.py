from flask import Flask, request, render_template
import jsonify
import pandas as pd
import numpy as np
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_movie(review):
  review = '[CLS] '+review+' [SEP]'
  pred = []
  pred.append(review)
  tokenized_texts = [tokenizer.tokenize(sent) for sent in pred]
  MAX_LEN = 128
  # Pad our input tokens
  input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                            maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
  # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
  input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
  input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
  # Create attention masks
  attention_masks = []
  for seq in input_ids:
    seq_mask = [float(i>0) for i in seq]
    attention_masks.append(seq_mask) 
  prediction_inputs = torch.tensor(input_ids)
  prediction_masks = torch.tensor(attention_masks)
  #prediction_labels = torch.tensor(labels)
    
  batch_size = 32  

  prediction_data = TensorDataset(prediction_inputs, prediction_masks)
  prediction_sampler = SequentialSampler(prediction_data)
  prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

  model.eval()

  # Tracking variables 
  predictions = []
  # Predict 
  for batch in prediction_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device).long() for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask = batch
    # Telling the model not to compute or store gradients, saving memory and speeding up prediction
    with torch.no_grad():
      # Forward pass, calculate logit predictions
      logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    
    # Store predictions and true labels
    predictions.append(logits)
    return 'Positive' if predictions[0][0][0]<predictions[0][0][1] else 'Negative'


app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
	if request.method=='POST':
		review = request.form['sentiment']
		sentiment = predict_movie(review)
		return render_template('index.html', prediction_text="The Movie review is "+sentiment)

if __name__ == "__main__":
    #clApp = ClientApp()
    app.run(host='0.0.0.0', port=8000)
    app.run(debug=True)
