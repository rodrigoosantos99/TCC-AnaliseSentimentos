import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords

# Baixando as stopwords em português
nltk.download('stopwords')
stop_words = set(stopwords.words('portuguese'))

# Definindo a classe SentimentMLP
class SentimentMLP(nn.Module):
    def __init__(self):
        super(SentimentMLP, self).__init__()
        self.fc1 = nn.Linear(5000, 1024)  # Aumentando o número de neurônios
        self.fc2 = nn.Linear(1024, 512)   # Aumentando o número de neurônios
        self.fc3 = nn.Linear(512, 256)    # Aumentando o número de neurônios
        self.fc4 = nn.Linear(256, 2)      # Camada de saída
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

# Carregar o modelo e o CountVectorizer salvos
model = SentimentMLP()
model.load_state_dict(torch.load('4sentiment_mlp_model.pth'))
model.eval()

vectorizer = joblib.load('4vectorizer.pkl')

# Carregar o novo conjunto de dados para teste
test_data = pd.read_csv('Test.csv', delimiter=';')
test_data['tweet_text'] = test_data['tweet_text'].apply(lambda x: ' '.join(
    [word for word in x.split() if word.lower() not in stop_words]))

# Transformar os tweets em vetores usando o CountVectorizer
X_test = vectorizer.transform(test_data['tweet_text']).toarray()
y_test = test_data['sentiment'].values

# Converter os dados de entrada e saída para tensores
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Realizar as previsões
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, y_pred = torch.max(outputs, 1)

# Relatório de Classificação
print("\nRelatório de Classificação:")
report = classification_report(y_test, y_pred.numpy(), target_names=['Negativo', 'Positivo'])
print(report)

# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred.numpy())
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negativo', 'Positivo'], yticklabels=['Negativo', 'Positivo'])
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()

# Gráfico de Barras para as Métricas
report_dict = classification_report(y_test, y_pred.numpy(), output_dict=True, target_names=['Negativo', 'Positivo'])
metrics_df = pd.DataFrame(report_dict).transpose().iloc[:2, :3]

metrics_df.plot(kind='bar', figsize=(10, 5))
plt.title('Métricas de Precisão, Recall e F1-Score')
plt.xlabel('Classes')
plt.ylabel('Valores')
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.show()
