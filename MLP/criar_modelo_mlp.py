import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import joblib
import datetime

# Baixando as stopwords em português
nltk.download('stopwords')
stop_words = set(stopwords.words('portuguese'))

# Registrar o horário inicial
start_time = datetime.datetime.now()
print(f"Horário de início: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

# Carregar os dados
data = pd.read_csv('Train200.csv', delimiter=';')
data['tweet_text'] = data['tweet_text'].apply(lambda x: ' '.join(
    [word for word in x.split() if word.lower() not in stop_words]))

# Vetorização
vectorizer = CountVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['tweet_text']).toarray()
y = LabelEncoder().fit_transform(data['sentiment'])

# Salvar o CountVectorizer para uso posterior
joblib.dump(vectorizer, '4vectorizer.pkl')

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Converter para tensores
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Definindo o modelo MLP com mais camadas e neurônios
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

model = SentimentMLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Treinamento do modelo (aumentando as épocas para 20)
epochs = 20  # Aumentando o número de épocas para 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}')

# Avaliação do modelo
model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.numpy())
        y_true.extend(y_batch.numpy())

# Relatório de Classificação
print("\nRelatório de Classificação:")
report = classification_report(y_true, y_pred, target_names=['Negativo', 'Positivo'])
print(report)

# Matriz de Confusão
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negativo', 'Positivo'], yticklabels=['Negativo', 'Positivo'])
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()

# Gráfico de Barras para as Métricas
report_dict = classification_report(y_true, y_pred, output_dict=True, target_names=['Negativo', 'Positivo'])
metrics_df = pd.DataFrame(report_dict).transpose().iloc[:2, :3]

metrics_df.plot(kind='bar', figsize=(10, 5))
plt.title('Métricas de Precisão, Recall e F1-Score')
plt.xlabel('Classes')
plt.ylabel('Valores')
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.show()

# Salvando o modelo
torch.save(model.state_dict(), '4sentiment_mlp_model.pth')
print("Modelo salvo em 'sentiment_mlp_model.pth'")

# Registrar o horário final
end_time = datetime.datetime.now()
print(f"Horário de término: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
