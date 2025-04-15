import streamlit as st
import pandas as pd
import pickle
from pycaret.classification import load_model, predict_model

# Título da aplicação
st.title("Simulador de Arremessos - Kobe Bryant")
st.markdown("Este simulador usa um modelo de regressão logística treinado com dados reais de arremessos.")

# Carregar modelo treinado
model = load_model("../data/06_models/model_regression_logistic_dev")

# Carregar scaler treinado
with open("../data/06_models/robust_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Interface para entrada de dados do arremesso
st.sidebar.header("Parâmetros do Arremesso")

lat = st.sidebar.slider("Latitude", min_value=33.5, max_value=34.1, value=33.9, step=0.01)
lon = st.sidebar.slider("Longitude", min_value=-118.6, max_value=-117.9, value=-118.2, step=0.01)
minutes_remaining = st.sidebar.slider("Minutos Restantes no Período", 0, 11, 5)
period = st.sidebar.selectbox("Período do Jogo", [1, 2, 3, 4, 5, 6, 7])
playoffs = st.sidebar.selectbox("Playoffs?", [0, 1])
shot_distance = st.sidebar.slider("Distância do Arremesso (ft)", 0, 80, 15)

# Organizar os dados em DataFrame
input_data = pd.DataFrame([{
    "lat": lat,
    "lon": lon,
    "minutes_remaining": minutes_remaining,
    "period": period,
    "playoffs": playoffs,
    "shot_distance": shot_distance
}])

# Normalizar os dados com o scaler treinado
input_scaled = pd.DataFrame(scaler.transform(input_data), columns=input_data.columns)

# Realizar predição
prediction = predict_model(model, data=input_scaled)

# Resultado
st.subheader("Resultado da Previsão")
pred = int(prediction.loc[0, 'prediction_label'])
score = float(prediction.loc[0, 'prediction_score'])

if pred == 1:
    st.success(f"Arremesso Previsto como **CERTO** ✔️\nProbabilidade: {score:.2%}")
else:
    st.error(f"Arremesso Previsto como **ERRADO** ❌\nProbabilidade de Acerto: {score:.2%}")

# Mostrar os dados normalizados (debug)
st.markdown("---")
with st.expander("Ver dados normalizados usados na predição"):
    st.dataframe(input_scaled)