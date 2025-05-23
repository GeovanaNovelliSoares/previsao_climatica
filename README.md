# Previsão de Temperatura com LSTM

Este projeto utiliza Redes Neurais Recorrentes (LSTM) para prever a temperatura média diária de Delhi com base no conjunto de dados **Daily Climate Time Series Data**, disponível no Kaggle ([link](https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data?resource=download)).

## 📌 **Tecnologias Utilizadas**
- Python
- TensorFlow/Keras
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn


### Execute o script
Se estiver rodando em um Jupyter Notebook, basta executar célula por célula.
Caso prefira rodar diretamente um script Python:
```bash
python previsao_tempo.py
```

## 📊 **Etapas do Projeto**

1. **Carregamento e pré-processamento dos dados**
   - Leitura do dataset
   - Normalização da temperatura média (`meantemp`) usando `MinMaxScaler`
   - Transformação dos dados em sequências temporais de 30 dias

2. **Criação do modelo LSTM**
   - Utilização de duas camadas LSTM para modelagem de séries temporais
   - Camada densa para prever a temperatura do próximo dia
   - Compilação e treinamento do modelo

3. **Validação e visualização dos resultados**
   - Divisão dos dados em treino e teste
   - Avaliação do desempenho do modelo
   - Geração de um gráfico comparando valores reais e previstos

## 📈 **Resultados**
O modelo consegue capturar tendências na temperatura ao longo do tempo. A previsão é exibida em um gráfico onde:
- 🔵 **Linha azul** representa os valores reais da temperatura
- 🔴 **Linha vermelha tracejada** representa as previsões feitas pelo modelo



