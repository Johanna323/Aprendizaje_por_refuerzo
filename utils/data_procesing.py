import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def cargar_y_preparar_datos(path):
    """
    Carga y prepara los datos del archivo CSV para entrenamiento.

    - Convierte TotalCharges a numérico
    - Elimina filas con valores nulos
    - Elimina customerID
    - Convierte 'Churn' en variable binaria
    - Codifica variables categóricas
    - Normaliza todos los valores

    Returns:
        DataFrame listo para entrenamiento
    """
    df = pd.read_csv(path)

    # Limpieza de TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)

    # Eliminar columnas innecesarias
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
    
    # Variable objetivo
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0}).astype(int)

    # Codificación de variables categóricas
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    
    # Normalización
    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])

    return df