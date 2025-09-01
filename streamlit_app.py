import streamlit as st
import pandas as pd
import joblib
import numpy as np
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# Загрузка обученной модели и препроцессоров
@st.cache_resource
def load_model_and_processors():
    try:
        model = joblib.load('xgboost_model.pkl')
        imputer = joblib.load('imputer.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, imputer, scaler
    except FileNotFoundError:
        st.error("Ошибка: Не удалось найти файлы модели. Убедитесь, что 'xgboost_model.pkl', 'imputer.pkl' и 'scaler.pkl' находятся в той же директории.")
        return None, None, None

# Загрузка всех компонентов
model, imputer, scaler = load_model_and_processors()
if model is None:
    st.stop()

# Заголовок приложения
st.title('Классификация потребности в поливе растений с использованием методов машинного обучения')

# --- Вывод метрик обученной модели ---
st.header("Метрики обученной модели на тестовом наборе")
st.markdown("Эти метрики показывают, насколько хорошо модель работает на невидимых данных.")

# Загрузка тестовых данных (предполагается, что они были сохранены)
try:
    X_test_scaled_for_metrics = joblib.load('X_test_scaled.pkl')
    y_test_for_metrics = joblib.load('y_test.pkl')
except FileNotFoundError:
    st.warning("Файлы с тестовыми данными не найдены. Метрики не будут отображены.")
    X_test_scaled_for_metrics = None
    y_test_for_metrics = None

if X_test_scaled_for_metrics is not None:
    y_pred_proba_test = model.predict_proba(X_test_scaled_for_metrics)[:, 1]
    y_pred_test = (y_pred_proba_test > 0.5).astype(int)

    accuracy = accuracy_score(y_test_for_metrics, y_pred_test)
    precision = precision_score(y_test_for_metrics, y_pred_test, zero_division=0)
    recall = recall_score(y_test_for_metrics, y_pred_test, zero_division=0)
    f1 = f1_score(y_test_for_metrics, y_pred_test, zero_division=0)
    roc_auc = roc_auc_score(y_test_for_metrics, y_pred_proba_test)
    conf_matrix = confusion_matrix(y_test_for_metrics, y_pred_test)

    # Отображение метрик
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Accuracy", f"{accuracy:.4f}")
    with col2:
        st.metric("Precision", f"{precision:.4f}")
    with col3:
        st.metric("Recall", f"{recall:.4f}")
    with col4:
        st.metric("F1 Score", f"{f1:.4f}")
    with col5:
        st.metric("ROC AUC", f"{roc_auc:.4f}")

    st.write("Матрица ошибок на тестовом наборе:")
    st.dataframe(pd.DataFrame(conf_matrix,
                              index=['True Negative', 'True Positive'],
                              columns=['Predicted Negative', 'Predicted Positive']))
else:
    st.info("Чтобы отобразить метрики, запустите скрипт обучения и сохраните 'X_test_scaled.pkl' и 'y_test.pkl'.")

# --- Боковая панель для ввода данных ---
st.sidebar.header("Ввод данных для прогноза")

feature_ranges = {
    'Temperature': (0, 50),
    ' Soil Humidity': (0, 100),
    'Time': (0, 24),
    'Air temperature (C)': (-10, 50),
    'Wind speed (Km/h)': (0, 150),
    'Air humidity (%)': (0, 100),
    'Wind gust (Km/h)': (0, 200),
    'Pressure (KPa)': (90, 110),
    'ph': (0, 14),
    'rainfall': (0, 1000),
    'N': (0, 1000),
    'P': (0, 1000),
    'K': (0, 1000),
}

input_data = {}
for feature, (min_val, max_val) in feature_ranges.items():
    input_data[feature] = st.sidebar.slider(
        f'Значение для "{feature}"',
        min_value=float(min_val),
        max_value=float(max_val),
        value=float((min_val + max_val) / 2)
    )

# Кнопка для запуска прогноза
if st.sidebar.button('Получить прогноз'):
    try:
        # Преобразование введенных данных в DataFrame
        user_input_df = pd.DataFrame([input_data])

        # Инженерные признаки
        user_input_df['Temp_Humidity_Interaction'] = user_input_df['Temperature'] * user_input_df[' Soil Humidity']
        user_input_df['NPK_Total'] = user_input_df['N'] + user_input_df['P'] + user_input_df['K']
        user_input_df['Pressure_Humidity_Ratio'] = user_input_df['Pressure (KPa)'] / user_input_df['Air humidity (%)']
        user_input_df = user_input_df.replace([np.inf, -np.inf], np.nan)

        # Применение импьютера и скейлера
        user_input_imp = imputer.transform(user_input_df)
        user_input_scaled = scaler.transform(user_input_imp)

        # Прогноз
        prediction = model.predict(user_input_scaled)
        prediction_proba = model.predict_proba(user_input_scaled)[:, 1]

        # Отображение результата
        st.subheader("Прогноз модели")
        if prediction[0] == 1:
            st.success(f"Модель предсказывает, что **полив необходим.** 💧")
        else:
            st.info(f"Модель предсказывает, что **полив не требуется.** 🌱")
        st.write(f"Вероятность положительного класса (полив необходим): {prediction_proba[0]:.2f}")

    except Exception as e:
        st.error(f"Произошла ошибка при обработке данных: {e}")
