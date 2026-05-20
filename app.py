
import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# --- 1. Cargar el modelo entrenado, el vectorizador y el DataFrame procesado ---
try:
    kmeans_model = joblib.load('kmeans_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    all_themes_raw = joblib.load('all_themes_raw.pkl') # Lista de todos los temas originales
    df_app = joblib.load('processed_df.pkl') # Cargar el DataFrame procesado directamente
    
    # Asegurarse de que 'Cantidad de vistas' sea numérico para la recomendación
    if 'Cantidad de vistas' in df_app.columns and not pd.api.types.is_numeric_dtype(df_app['Cantidad de vistas']):
        df_app['Cantidad de vistas'] = pd.to_numeric(df_app['Cantidad de vistas'], errors='coerce')
    
except FileNotFoundError:
    st.error("Error: Asegúrate de que 'kmeans_model.pkl', 'tfidf_vectorizer.pkl', 'all_themes_raw.pkl' y 'processed_df.pkl' estén en el mismo directorio.")
    st.stop()

# --- Función de Recomendación adaptada para Streamlit ---
# Esta función es una adaptación de la que tienes en Colab, pero ahora recibe `df_data` como argumento.
def recommend_songs_streamlit(input_theme, df_data, n_recommendations=5):
    clean_input_theme = str(input_theme).replace('/', ' ').replace(',', ' ').replace(';', ' ').strip()
    input_vector = tfidf_vectorizer.transform([clean_input_theme])
    cluster_id = kmeans_model.predict(input_vector)[0]

    cluster_songs = df_data[df_data['cluster'] == cluster_id].copy()

    if 'Cantidad de vistas' in cluster_songs.columns:
        cluster_songs['Cantidad de vistas_numeric'] = pd.to_numeric(cluster_songs['Cantidad de vistas'], errors='coerce')
        cluster_songs = cluster_songs.dropna(subset=['Cantidad de vistas_numeric'])

        if not cluster_songs.empty:
            recommendations = cluster_songs.sort_values(by='Cantidad de vistas_numeric', ascending=False)['Canción'].head(n_recommendations).tolist()
        else:
            recommendations = []
    else: # Si 'Cantidad de vistas' no está, simplemente devuelve canciones del cluster
        if not cluster_songs.empty:
            recommendations = cluster_songs['Canción'].head(n_recommendations).tolist()
        else:
            recommendations = []

    return recommendations, cluster_id

# --- Interfaz de Usuario de Streamlit ---
st.title('Recomendador de Canciones de Taylor Swift por Tema')
st.write('¡Encuentra tu próxima canción favorita de Taylor Swift basada en el tema que te apetezca!')

# Opción para elegir tema de una lista o escribir uno nuevo
selected_theme = st.selectbox(
    'Selecciona un tema para empezar:',
    [''] + sorted(list(set(all_themes_raw))), # Agrega una opción vacía y ordena los temas
    index=0 # Por defecto, no selecciona nada
)

custom_theme = st.text_input('O escribe un tema personalizado aquí:')

# Usar el tema personalizado si se ha escrito algo, de lo contrario, usar el seleccionado
final_theme = custom_theme if custom_theme else selected_theme

n_recommendations = st.slider('¿Cuántas canciones quieres recomendar?', 1, 10, 5)

# --- Modificación aquí: Añadir `key` a los botones --- 
if st.button('Recomendar Canciones', key='recommend_button_main') and final_theme:
    with st.spinner('Buscando recomendaciones...'):
        if not df_app.empty: # Asegúrate de que df_app esté disponible y no vacío
            recommended_songs, cluster = recommend_songs_streamlit(final_theme, df_app, n_recommendations)
            if recommended_songs:
                st.success(f"Para el tema '{final_theme}' (cluster {cluster}), te recomiendo las siguientes canciones:")
                for i, song in enumerate(recommended_songs):
                    st.write(f"{i+1}. {song}")
            else:
                st.warning(f"No se encontraron canciones para el tema '{final_theme}' en el cluster {cluster}.")
        else:
            st.error("Error: El DataFrame de canciones no se ha cargado correctamente o está vacío en la aplicación.")
elif st.button('Recomendar Canciones', key='recommend_button_no_theme') and not final_theme:
    st.warning('Por favor, selecciona o escribe un tema para obtener recomendaciones.')

st.markdown("""
### Cómo usar:
1. Selecciona un tema de la lista desplegable.
2. Opcionalmente, escribe un tema personalizado en el campo de texto.
3. Ajusta el número de canciones que deseas.
4. Haz clic en 'Recomendar Canciones'.
""")
