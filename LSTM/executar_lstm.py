import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn

# Baixar as stopwords em português
nltk.download('stopwords')
stop_words = set(stopwords.words('portuguese'))

# Carregar o arquivo de teste
test_data = pd.read_csv('Test.csv', delimiter=';')

# Pré-processamento do texto de teste (remover links e menções)
test_data['tweet_text'] = test_data['tweet_text'].apply(lambda x: ' '.join(
    [word for word in str(x).split() if word.lower() not in stop_words and not word.startswith('@') and not word.startswith('http')]
))

# Carregar o CountVectorizer salvo
vectorizer = joblib.load('4lstm_vectorizer.pkl')
X_test = vectorizer.transform(test_data['tweet_text']).toarray()

# Converter rótulos de sentimento para inteiros
y_test = test_data['sentiment'].astype(int)

# Converter para tensores
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Definindo a classe do modelo LSTM (igual ao usado no treinamento)
class SentimentLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.3):
        super(SentimentLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Usando apenas a última saída da sequência
        return out

# Configurações do modelo (use os mesmos parâmetros do treinamento)
input_size = 10000  # Número de características do CountVectorizer
hidden_size = 512
num_layers = 3
num_classes = 2

# Carregar o modelo treinado
model = SentimentLSTM(input_size, hidden_size, num_layers, num_classes)
model.load_state_dict(torch.load('4lstm_sentiment_model.pth'))
model.eval()

# Realizar previsões
y_pred = []
with torch.no_grad():
    outputs = model(X_test_tensor.unsqueeze(1))  # Adiciona dimensão para sequência
    _, predicted = torch.max(outputs, 1)
    y_pred = predicted.numpy()

# Exibir Relatório de Classificação
print("\nRelatório de Classificação LSTM (Conjunto de Teste):")
report = classification_report(y_test, y_pred, target_names=['Negativo', 'Positivo'])
print(report)

# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negativo', 'Positivo'], yticklabels=['Negativo', 'Positivo'])
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão LSTM - Conjunto de Teste')
plt.show()
