from pathlib import Path
import pandas as pd
import streamlit as st
from joblib import load
from sklearn.pipeline import Pipeline


def get_user_data() -> pd.DataFrame:
    user_data = {}

    # dividir la pantall en dos columnas
    col_a, col_b = st.columns(2)
    with col_a:
        user_data['age'] = st.number_input(
            label='Edad', min_value=0, max_value=100, value=20, step=1
        )
        user_data['sibsp'] = st.slider(
            label="Número de hermanos y cónyuges a bordo:",
            min_value=0, max_value=15, value=3, step=1,
        )
    with col_b:
        user_data['fare'] = st.number_input(
            label='Tarifa pagada', min_value=0.0, max_value=3000.0, value=80.0, step=1.0
        )
        user_data['parch'] = st.slider(
            label = "Número de padres y niños a bordo:",
            min_value=0, max_value=10, value=1, step=1,
        )

    # Divir en tres columnas para las variables categóricas
    col1, col2, col3 = st.columns(3)
    with col1:
        user_data['sex'] = st.radio(
            label='Sexo', options=['Woman','Man'], horizontal=False
        )
    with col2:
        user_data['pclass'] = st.radio(
            label='Clase del boleto',
            options=['1st','2nd', '3rd'], horizontal = False
        )
    with col3:
        user_data['embarked'] = st.radio(
            label='Puerto de embarque',
            options=['Cherbourg', 'Queenstown', 'Southampton'],
            index=1,
        )

    # Convertir el diccionario a Dataframe
    df = pd.DataFrame.from_dict(user_data, orient='index').T

    # Preprocesamiento: mapear los valores de texto a los formatos que necesita el modelo
    df['sex'] = df['sex'].map({'Man':'male', 'Woman': 'female'})
    df['pclass'] = df['pclass'].map({'1st':1, '2nd':2, '3rd':3})
    df['embarked'] = df['embarked'].map({'Cherbourg':'C', 'Queenstown':'Q', 'Southampton':'S'})

    return df
 
@st.cache_resource
def load_model(model_file_path: Path) -> Pipeline:
    with st.spinner('cargando el modelo...'):
        model = load(model_file_path)
    return model

def main() -> None:

    model_name = 'randomforest_best_model.joblib'

    # Obtener el directorio raíz del proyecto (un nivel arriba de notebooks/)
    PROJECT_ROOT = Path(__file__).parent.parent

    MODEL_DIR = PROJECT_ROOT / "modelos"
    model_path = MODEL_DIR / model_name

    # Verificar si el archivo del modelo existe
    if not model_path.exists():
        st.error(f"El archivo del modelo no se encontró en la ruta: {model_path}")
        st.stop()

    # Mostrar una imagen de Titanic (si existe)
    IMAGES_DIR = PROJECT_ROOT / "imagenes"
    image_path = IMAGES_DIR / "titanic.jpg"
    if image_path.exists():
        st.image(str(image_path), caption="Acá morimos todos...")

    # Título de la aplicación
    st.header("¿SOBREVIVIRÍAS AL NAUFRAGIO DEL TITANIC?")

    # Recoger datos de usuario
    df_user_data = get_user_data()

    # Cargar el modelo
    model = load_model(model_path)

    state = model.predict(df_user_data)[0]

    # usar emojios para visualización
    emojis = ['💀','🛟']

    st.write("")
    st.title(f"Tu destino sería: {emojis[state]}")

    if state == 0:
        st.error('¡Lo siento, serás comida para orcas!')
    else:
        st.success('¡Felicidades, sobrevivirás al naufragio!')

if __name__ == "__main__":
    main()
