import streamlit as st
import pandas as pd
import joblib
import numpy as np
import warnings

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
st.title('Классификация полива растений с помощью алгоритмов ML')
st.markdown("### Введите значения признаков для прогнозирования необходимости полива")

# Список признаков для ввода пользователем (обязательно в том же порядке, что и в X_train)
feature_names = [
    'Temperature', ' Soil Humidity', 'Time', 'Air temperature (C)', 
    'Wind speed (Km/h)', 'Air humidity (%)', 'Wind gust (Km/h)', 
    'Pressure (KPa)', 'ph', 'rainfall', 'N', 'P', 'K', 
    'Temp_Humidity_Interaction', 'NPK_Total', 'Pressure_Humidity_Ratio'
]

# Создание полей ввода для каждого признака
input_data = {}
st.header("Входные данные")
for feature in feature_names:
    input_data[feature] = st.number_input(f'Введите значение для "{feature}"', value=0.0)

# Кнопка для запуска прогноза
if st.button('Получить прогноз'):
    try:
        # Преобразование введенных данных в DataFrame
        user_input_df = pd.DataFrame([input_data])
        
        # Инженерные признаки (должны быть рассчитаны так же, как и при обучении)
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
        st.subheader("Результат прогнозирования")
        if prediction[0] == 1:
            st.success(f"Модель предсказывает, что **полив необходим.** 💧")
        else:
            st.info(f"Модель предсказывает, что **полив не требуется.** 🌱")
        st.write(f"Вероятность положительного класса (полив необходим): {prediction_proba[0]:.2f}")
    except Exception as e:
        st.error(f"Произошла ошибка при обработке данных: {e}")
