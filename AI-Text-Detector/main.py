# from flask import Flask, render_template, request, jsonify
# import torch
# from transformers import BertForSequenceClassification, BertTokenizer
# import os
#
# app = Flask(__name__)
#
# # Define the path where the saved model is located
# model_save_path = "bert_model"
#
# # Loading the saved model and necessary components
# loaded_model = BertForSequenceClassification.from_pretrained(model_save_path)
# loaded_tokenizer = BertTokenizer.from_pretrained(model_save_path)
# loaded_label_encoder = torch.load(os.path.join(model_save_path, 'trainedModel.pkl'))
#
# # Define the device for inference (CPU or GPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return jsonify(message="No file part")
#
#         file = request.files['file']
#
#         if file.filename == '':
#             return jsonify(message="No selected file")
#
#         if file:
#             # Read essays from the uploaded file
#             test_essays = []
#             for line in file:
#                 line = line.decode("utf-8").strip()  # Remove leading/trailing whitespace
#                 if line:  # Skip empty lines
#                     test_essays.append(line)
#
#             # Tokenize the test essays
#             test_inputs = loaded_tokenizer(test_essays, padding=True, truncation=True, return_tensors='pt')
#
#             # Move input tensor to the same device as the model
#             test_inputs = {key: value.to(device) for key, value in test_inputs.items()}
#
#             # Generate predictions using the loaded model
#             with torch.no_grad():
#                 outputs = loaded_model(**test_inputs)
#                 logits = outputs.logits
#
#             # If binary classification, handle single output class
#             if logits.dim() == 1:
#                 predictions = torch.sigmoid(logits).cpu().numpy()
#             else:
#                 # If logits have only one column, expand dimensions to simulate two classes
#                 if logits.size(1) == 1:
#                     logits = torch.cat((logits, 1 - logits), dim=1)
#                 # Assuming the second column corresponds to the positive class (AI-generated)
#                 predictions = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
#
#             # Display the predictions
#             results = []
#             for i, pred in enumerate(predictions):
#                 results.append(f"Test {i + 1}: AI-generated probability: {pred:.4f}")
#
#             return jsonify(results=results)
#
#     return render_template('index.html')
#
#
# if __name__ == '__main__':
#     app.run(debug=True)



# from flask import Flask, render_template, request, jsonify
# import torch
# from transformers import BertForSequenceClassification, BertTokenizer
# import os
#
# app = Flask(__name__)
#
# # Define the path where the saved model is located
# model_save_path = "bert_model"
#
# # Loading the saved model and necessary components
# loaded_model = BertForSequenceClassification.from_pretrained(model_save_path)
# loaded_tokenizer = BertTokenizer.from_pretrained(model_save_path)
# loaded_label_encoder = torch.load(os.path.join(model_save_path, 'trainedModel.pkl'))
#
# # Define the device for inference (CPU or GPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return jsonify(message="No file part")
#
#         file = request.files['file']
#
#         if file.filename == '':
#             return jsonify(message="No selected file")
#
#         if file:
#             # Read essays from the uploaded file
#             test_essays = []
#             for line in file:
#                 line = line.decode("utf-8").strip()  # Remove leading/trailing whitespace
#                 if line:  # Skip empty lines
#                     test_essays.append(line)
#
#             # Tokenize the test essays
#             test_inputs = loaded_tokenizer(test_essays, padding=True, truncation=True, return_tensors='pt')
#
#             # Move input tensor to the same device as the model
#             test_inputs = {key: value.to(device) for key, value in test_inputs.items()}
#
#             # Generate predictions using the loaded model
#             with torch.no_grad():
#                 outputs = loaded_model(**test_inputs)
#                 logits = outputs.logits
#
#             # If binary classification, handle single output class
#             if logits.dim() == 1:
#                 predictions = torch.sigmoid(logits).cpu().numpy()
#             else:
#                 # If logits have only one column, expand dimensions to simulate two classes
#                 if logits.size(1) == 1:
#                     logits = torch.cat((logits, 1 - logits), dim=1)
#                 # Assuming the second column corresponds to the positive class (AI-generated)
#                 predictions = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
#
#             # Prepare results for display
#             results = []
#             for i, (essay, pred) in enumerate(zip(test_essays, predictions)):
#                 results.append(f"Essay {i + 1}: AI-generated probability: {pred:.4f}<br>{essay}<br>")
#
#             return jsonify(results=results)
#
#     return render_template('index.html')
#
#
# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, request, jsonify
import torch
from transformers import BertForSequenceClassification, BertTokenizer
import os

app = Flask(__name__)

# Define the path where the saved model is located
model_save_path = "bert-base-uncased"

# Loading the saved model and necessary components
loaded_model = BertForSequenceClassification.from_pretrained(model_save_path)
loaded_tokenizer = BertTokenizer.from_pretrained(model_save_path)

# Load label encoder if it exists
label_encoder_path = os.path.join(model_save_path, 'trainedModel.pkl')
if os.path.exists(label_encoder_path):
    loaded_label_encoder = torch.load(label_encoder_path)
else:
    loaded_label_encoder = None  # Handle this case appropriately in your code

# Define the device for inference (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model = loaded_model.to(device)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify(message="No file part"), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify(message="No selected file"), 400

        if file:
            # Read essays from the uploaded file
            test_essays = []
            for line in file:
                line = line.decode("utf-8").strip()  # Remove leading/trailing whitespace
                if line:  # Skip empty lines
                    test_essays.append(line)

            # Tokenize the test essays
            test_inputs = loaded_tokenizer(test_essays, padding=True, truncation=True, return_tensors='pt')

            # Move input tensor to the same device as the model
            test_inputs = {key: value.to(device) for key, value in test_inputs.items()}

            # Generate predictions using the loaded model
            with torch.no_grad():
                outputs = loaded_model(**test_inputs)
                logits = outputs.logits

            # Process logits to get predictions
            if logits.dim() == 1:
                predictions = torch.sigmoid(logits).cpu().numpy()
            else:
                if logits.size(1) == 1:
                    logits = torch.cat((logits, 1 - logits), dim=1)
                predictions = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

            # Prepare results for display and saving
            results = []
            saved_paragraphs = []
            for i, (essay, pred) in enumerate(zip(test_essays, predictions)):
                if pred > 0.50:
                    saved_paragraphs.append(essay)
                results.append(f"Paragraph {i + 1}: AI-generated probability: {pred:.4f}<br>{essay}<br>")

            # Save paragraphs with AI probability greater than 0.50 into a text file
            with open('ai_paragraphs.txt', 'w') as f:
                f.write("\n\n".join(saved_paragraphs))

            return jsonify(results=results)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)