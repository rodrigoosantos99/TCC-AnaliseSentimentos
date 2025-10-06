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
import joblib
import datetime
import psutil
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Baixando as stopwords em português
nltk.download('stopwords')
stop_words = set(stopwords.words('portuguese'))

# Registrar o horário inicial
start_time = datetime.datetime.now()
print(f"Horário de início: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

# Carregar os dados
data = pd.read_csv('Train200.csv', delimiter=';')

# Pré-processamento aprimorado (remover links e menções)
data['tweet_text'] = data['tweet_text'].apply(lambda x: ' '.join(
    [word for word in str(x).split() if word.lower() not in stop_words and not word.startswith('@') and not word.startswith('http')]
))

# Vetorização com 10.000 características
vectorizer = CountVectorizer(max_features=10000)  # Aumentado para 10.000 características
X = vectorizer.fit_transform(data['tweet_text']).toarray()
y = data['sentiment'].astype(int)  # Convertendo o tipo do rótulo de sentimento para int

# Salvar o CountVectorizer para uso posterior
joblib.dump(vectorizer, '4lstm_vectorizer.pkl')

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Converter para tensores
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Definindo o modelo LSTM
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
        out = self.fc(out[:, -1, :])  # Usando apenas a saída final da sequência
        return out

# Definir hiperparâmetros
input_size = 10000  # Aumentado para 10.000 características
hidden_size = 512
num_layers = 3
num_classes = 2
num_epochs = 10  # Reduzido para 10 épocas
learning_rate = 0.001

# Criar e treinar o modelo LSTM
model = SentimentLSTM(input_size, hidden_size, num_layers, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Inicializando o scheduler para redução de taxa de aprendizado
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# Monitorar o uso de recursos
cpu_util = []
gpu_util = []
mem_util = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        
        # Expande a dimensão de X_batch para (batch_size, sequence_length, input_size)
        outputs = model(X_batch.unsqueeze(1))  # Adiciona dimensão para sequência de comprimento 1
        
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Monitorar uso de recursos
        cpu_util.append(psutil.cpu_percent())
        gpu_util.append(torch.cuda.utilization() * 100 if torch.cuda.is_available() else 0)
        mem_util.append(psutil.virtual_memory().percent)

    # Atualizando a taxa de aprendizado com o scheduler
    scheduler.step(total_loss)

    # Acessando a taxa de aprendizado atual e imprimindo
    current_lr = scheduler.get_last_lr()[0]
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}, Current LR: {current_lr:.6f}')

# Avaliação do modelo LSTM
model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch.unsqueeze(1))
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.numpy())
        y_true.extend(y_batch.numpy())

# Relatório de Classificação
print("\nRelatório de Classificação LSTM:")
report = classification_report(y_true, y_pred, target_names=['Negativo', 'Positivo'])
print(report)

# Matriz de Confusão
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negativo', 'Positivo'], yticklabels=['Negativo', 'Positivo'])
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão LSTM')
plt.show()

# Gráfico de Barras para as Métricas
report_dict = classification_report(y_true, y_pred, output_dict=True, target_names=['Negativo', 'Positivo'])
metrics_df = pd.DataFrame(report_dict).transpose().iloc[:2, :3]

metrics_df.plot(kind='bar', figsize=(10, 5))
plt.title('Métricas de Precisão, Recall e F1-Score LSTM')
plt.xlabel('Classes')
plt.ylabel('Valores')
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.show()

# Registrar o horário final
end_time = datetime.datetime.now()
print(f"Horário de término: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

# Exibir pico de uso de recursos
print(f"Pico de uso da CPU: {max(cpu_util)}%")
print(f"Pico de uso da GPU: {max(gpu_util)}%")
print(f"Pico de uso de memória RAM: {max(mem_util)}%")

# Salvando o modelo
torch.save(model.state_dict(), '4lstm_sentiment_model.pth')
print("Modelo LSTM salvo em 'lstm_sentiment_model.pth'")
