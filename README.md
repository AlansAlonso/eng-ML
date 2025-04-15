# 🏀 Preditor de Arremessos de Kobe Bryant 🏀

**Alan Alonso**
---

## Estrutura do Projeto (TDSP)

```bash
.
├── data/
│   ├── 01_raw/
│   ├── 02_intermediate/
│   ├── 03_primary/
│   ├── 05_model_input/
│   ├── 06_models/
│   ├── 07_model_output/
│   └── 08_reporting/
├── notebooks/
├── docs/source
├── src/
├── mlruns/
├── streamlit/
├── requirements.txt
├── Dockerfile / docker-compose
├── README.md
└── aplicacao.py
```

---

## Introdução

Este projeto busca criar um preditor de acertos de arremessos de Black Mamba, também conhecido como Kobe Bryant! Baseando-se em dados de contexto da jogada. Com técnicas de engenharia de machine learning para processar, treinar, comparar e implantar modelos, visando apoiar melhorar a qualidade desta previsão.

---

## Diagrama de Pipeline

![Diagrama](/docs/source/diagrama.png)
---

## Ferramentas Utilizadas

| Ferramenta       | Papel no Projeto                                                                 |
| ---------------- | -------------------------------------------------------------------------------- |
| **MLflow**       | Registro e acompanhamento dos experimentos, métricas e modelos; versionamento    |
| **PyCaret**      | Criação automatizada de pipelines de machine learning (setup, tuning, avaliação) |
| **Scikit-learn** | Execução dos algoritmos (regressão logística, árvore), métricas personalizadas   |
| **Streamlit**    | Interface visual para simular arremessos e monitorar o modelo em tempo real      |


---

## Artefatos do Projeto

| Artefato                                 | Descrição                           |
| ---------------------------------------- | ----------------------------------- |
| `aplicacao.ipynb`                        | Notebook que contêm os pipelines    |
| `dataset_kobe_dev.parquet`               | Dados originais de desenvolvimento  |
| `dataset_kobe_prod.parquet`              | Dados originais de produção         |
| `base_filtered.parquet`                  | Dados tratados para treino / teste  |
| `base_train.parquet`                     | Dados processados para treino       |
| `base_test.parquet`                      | Dados processados para teste        |
| `model_regression_logistic_dev.pkl`      | Modelo de treinamento de reg-log    |
| `model_decision_tree_dev.pkl`            | Modelo de treinamento de dt         |
| `robust_scaler.pkl`                      | Modelo de normalização dos dados    |
| `predictions_prod.parquet`               | Previsões geradas na produção       |
| `streamlit/app.py`                       | Interface interativa                |

---

## Processamento de Dados

- Dados de entrada: `/data/raw/dataset_kobe_dev.parquet`
    - Base de dados de desenvolvimento, enquanto a `/data/raw/dataset_kobe_prod.parquet` é utilizada para a etapa de aplicação do modelo treinado escolhido
- Colunas utilizadas: `lat`, `lon`, `minutes_remaining`, `period`, `playoffs`, `shot_distance`
    - `lat` e `lon`: Latitude e longitude da quadra, ambos do tipo float.
    - `minutes_remaining`, `period`, `playoffs`: Tempo que falta no período, qual período estamos, e se é ou não play-off, todos do tipo inteiro.
    - `shot_distance`: Distância para a cêsta, também um número inteiro.
- Remoção de linhas com nulos
- Normalização robusta dos dados, para lidar melhor com outliers e ter dados adequados para ambos os modelos de teste
- Dados salvos em `/data/03_primary/` com dimensão (20285, 7)
- Split estratificado 80/20 gerando arquivos e salvando em `/data/05_model_input/`
    - Train shape: (16228, 6)
    - Test shape: (4057, 6)


> **Como o split afeta o modelo?**\
> O split busca evitar que haja overfitting e também garante generalização, impedindo que um modelo funcione bem apenas para um grupo específico de dados dentro do universo de amostragem. A estratificação preserva a distribuição das classes. Para minimizar viés, é importante normalização, balanceamento e validação cruzada.

---

## Treinamento e Seleção de Modelos

- **Modelos testados (metricas: F1 Score e Log Loss):**

  - Regressão Logística (PyCaret)
    - 📈 F1 Score: 0.5732
    - 📉 Log Loss: 0.7131
  - Árvore de Decisão (PyCaret)
    - 📈 F1 Score: 0.5478
    - 📉 Log Loss: 15.3255

- Registro completo com `mlflow`

> **Modelo escolhido:** *[Regressão Logística]*\
> **Justificativa:** *[Melhor valor de F1 score e de Log loss]*
> Nota-se que o valor de F1 score, apesar de não ser muito alto, indica que o modelo escolhido é melhor do que uma escolha aleatória. No contexto de um esporte com tantos arremessos (um universo grande de dados), mesmo valores longe do ideal podem ser suficiente para garantir a utilidade do modelo. Em especial, nota-se que a árvore de decisão é menos recomendada por um valor extremamente alto de Log loss, o que condiz com um modelo que tem tendência a gerar probabilidades muito extremas, em comparação com um modelo probabilistico como a regressão logistica. Especialmente em modelos que tenham alto risco em caso de erro (não é o caso deste), é recomendado evitar.

---

## Aplicação em Produção

- Dados de produção: `/data/raw/dataset_kobe_prod.parquet`
- Previsões realizadas com modelo final
- Resultados salvos em `/data/07_model_output/`
- Nova run registrada como `PipelineAplicacao`

**Métricas em produção:**

- F1 Score: *[0]*
- Log Loss: *[24.1700]*

> **Mudanças detectadas:** *[Distribuição de uma das variáveis]*\
> **Modelo manteve performance?** *[Não]*
> Apesar de estranho, era esperado este tipo de comportamento tendo em vista que estes arremessos aparentam ser exclusivamentes para uma "cesta de 3", dessa forma um modelo treinado para uma distribuição diferente não pode ser aplicado para estes dados.

![Arremessos do treinamento do modelo](/docs/source/arremessos_dev.png)
![Arremessos da aplicação do modelo](/docs/source/arremessos_prod.png)

---

## Monitoramento e Retreinamento

- **Com resposta:** Em um cenário com a variável resposta podemos utilizar os mesmos parâmetros como F1 Score e Log Loss para avaliar a sua contínua aplicabilidade. Além disso, curva ROC, e outras figuras de mérito como acurácia e recall podem ser parâmetros auxiliares
- **Sem resposta:** Sem as respostas é interessante observar a variação da distribuição das variáveis de entrada, comparando com a média e desvio padrão dos dados com resposta conhecida. Caso haja um desbalanceamento, um novo treinamento pode ser necessário.

**Estratégias:**

- **Reativa:** Ao detectar uma queda de performance o modelo deve ser retreinado. Seja por uma queda nos parâmetros case tenhamos a resposta, seja com uma mudança drastica na destribuição das variáveis de entrada. É um modelo de correção mais simples de aplicar mas pode ocorrer com uma certa defasagem após o momendo de necessidade de mudança do modelo.
- **Preditiva:** Possui uma periodicidade com base na análise histórica de quando o modelo pode se deteriorar. Apesar de ser menor eficiente computacionalmente, pode melhorar a estabilidade do modelo .

---

## Dashboard Streamlit

- Local: `streamlit/app.py`
- Permite simulação de arremessos com feedback instantâneo de probabilidade

![Streamlit](/docs/source/streamlit.png)

---

## Mlflow

- Registro dos parâmetros das runs

![mlflow](/docs/source/mlflow.png)

---

## Como Executar

### Instalar dependências:

```bash
pip install -r requirements.txt
```

### Rodar aplicação de inferência:

```bash
python aplicacao.py
```

### Iniciar o painel de monitoramento:

```bash
cd /workspaces/kobe-pd/streamlit
streamlit run streamlit/streamlit_app.py
```

### Iniciar o MLflow (opcional):

```bash
mlflow ui --backend-store-uri file:../mlruns
```

---

## Link do Repositório

[https://github.com/AlansAlonso/eng-ML](https://github.com/AlansAlonso/eng-ML)

## Link da Aplicacao

[https://github.com/AlansAlonso/eng-ML/blob/main/aplicacao.ipynb](https://github.com/AlansAlonso/eng-ML/blob/main/aplicacao.ipynb)

---

## Aviso ao professor
> Inicialmente este projeto foi feito com o dev container providenciado, utilizando kedro, os requirements_antigo são os originais. Contudo, devido a problemas no versionamento de kedro e mlflow, precisei realizá-lo de uma forma que eu estava mais familiar, através de notebooks, buscando emular e evidênciar em cada trecho as "pipelines" que estariam ocorrendo. No mais, aproveitei as pastas criadas pelo kedro dentro das minhas habilidades para organizar o trabalho da melhor forma possível.
