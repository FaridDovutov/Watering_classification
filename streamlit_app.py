import warnings
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
from xgboost import XGBClassifier

# Игнорирование предупреждений
warnings.filterwarnings("ignore")

# Название приложения Streamlit
st.title('XGBoost Model Performance Analyzer')
st.markdown("### Анализ производительности модели XGBoost с тюнингом гиперпараметров и инженерными признаками")

# Загрузка данных
@st.cache_data
def load_data():
    df = pd.read_csv('.streamlit/TARP.csv')
    return df

df = load_data()

# Обработка данных
@st.cache_data
def preprocess_data(data):
    df_processed = data.copy()

    # Удаление признака и кодирование целевой переменной
    df_processed = df_processed.drop('Soil Moisture', axis=1)
    encoder = LabelBinarizer()
    df_processed['Status'] = encoder.fit_transform(df_processed['Status'])

    # Создание новых признаков
    try:
        df_processed['Temp_Humidity_Interaction'] = df_processed['Temperature'] * df_processed[' Soil Humidity']
        df_processed['NPK_Total'] = df_processed['N'] + df_processed['P'] + df_processed['K']
        df_processed['Pressure_Humidity_Ratio'] = df_processed['Pressure (KPa)'] / df_processed['Air humidity (%)']
        df_processed = df_processed.replace([np.inf, -np.inf], np.nan)
    except KeyError as e:
        st.error(f"Ошибка: Не найден признак для создания инженерных признаков: {e}")
        return None, None

    # Разделение данных
    X = df_processed.drop('Status', axis=1)
    y = df_processed['Status']
    return X, y

X, y = preprocess_data(df)
if X is None:
    st.stop()

# Разделение данных до любых преобразований
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Импьютация пропущенных значений (обучение только на тренировочных данных)
imputer = IterativeImputer(random_state=42)
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

# Масштабирование данных (обучение только на тренировочных данных)
scaler = MinMaxScaler((-1,1))
X_train_scl = scaler.fit_transform(X_train_imp)
X_test_scl = scaler.transform(X_test_imp)

# Преобразование в DataFrame
X_train = pd.DataFrame(X_train_scl, columns=X_train.columns)
X_test = pd.DataFrame(X_test_scl, columns=X_test.columns)

# Настройка и обучение модели
def train_model(X_train, y_train, X_test, y_test):
    # Определение сетки параметров для GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.1, 0.05],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    # Инициализация модели
    xgb_model = XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42)

    # Настройка K-Fold кросс-валидации
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Настройка GridSearchCV
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=kfold,
        scoring='f1',
        verbose=1,
        n_jobs=-1
    )

    st.text("Запуск GridSearchCV...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    
    return best_model, grid_search.best_params_, grid_search.best_score_

best_model, best_params, best_score = train_model(X_train, y_train, X_test, y_test)

# Отображение результатов
st.write("---")
st.header("Результаты GridSearchCV")
st.write("Лучшие параметры:", best_params)
st.metric("Лучший F1-score на кросс-валидации", f"{best_score:.4f}")

# Функция для отображения метрик
def display_metrics(y_true, y_pred_proba, data_name):
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    conf_matrix = confusion_matrix(y_true, y_pred)

    st.subheader(f"Метрики для набора данных: {data_name}")
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
    
    st.write("Матрица ошибок:")
    st.dataframe(pd.DataFrame(conf_matrix, 
                              index=['True Negative', 'True Positive'], 
                              columns=['Predicted Negative', 'Predicted Positive']))

# Оценка лучшей модели на тестовой выборке
y_pred_proba_test = best_model.predict_proba(X_test)[:, 1]
display_metrics(y_test, y_pred_proba_test, 'Test')

# Оценка лучшей модели на тренировочной выборке
y_pred_proba_train = best_model.predict_proba(X_train)[:, 1]
display_metrics(y_train, y_pred_proba_train, 'Train')
