# LSTM

Esta pasta contém os scripts para criar, treinar e avaliar o modelo LSTM para análise de sentimentos em tweets.

## Scripts
- `criar_modelo_lstm.py`: Define a arquitetura do modelo LSTM e salva para uso posterior.
- `executar_lstm.py`: Usa o modelo com os dados e gera métricas de avaliação (acurácia, recall, f1-score).

## Como executar
1. Certifique-se de que os datasets estão no local correto
2. Execute `criar_modelo_lstm.py` para gerar o modelo inicial.
3. Execute `executar_lstm.py` para treinar o modelo e visualizar os resultados.

## Dependências
- numpy, pandas, torch, scikit-learn, matplotlib, seaborn, nltk, psutil

