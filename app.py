
import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# --- 1. Cargar el modelo entrenado y el vectorizador ---
# Asegúrate de que estos archivos estén en el mismo directorio que tu app.py en GitHub
try:
    kmeans_model = joblib.load('kmeans_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    all_songs = joblib.load('all_songs.pkl') # Lista de todas las canciones
    all_themes_raw = joblib.load('all_themes_raw.pkl') # Lista de todos los temas originales
    
    # Para la demostración, creamos un DataFrame simulado, ya que el original `df` no estará disponible
    # en la app de Streamlit a menos que lo guardes también (ej. como CSV o PKL).
    # En un escenario real, cargarías tu DataFrame completo o al menos las columnas necesarias.
    # Por simplicidad, aquí simulamos una parte esencial para la función recommend_songs.
    # Supongamos que df.pkl contiene las columnas 'Canción', 'Tema de la canción_clean', 'cluster', 'Cantidad de vistas'
    # Si no tienes un df.pkl, deberías modificar esto para cargar los datos de alguna otra fuente.
    
    # Crea un DataFrame de ejemplo si df.pkl no existe o necesitas un placeholder
    # En tu caso, necesitarás cargar el 'df' completo o al menos las columnas necesarias
    # para que la función `recommend_songs` opere.
    # Por ahora, para que este ejemplo funcione, simularé un df muy básico.
    # **DEBERÁS ADAPTAR ESTA SECCIÓN PARA CARGAR TU DATAFRAME COMPLETO**
    # Por ejemplo, puedes guardar tu DataFrame original como 'df_processed.pkl' o 'df_processed.csv'
    # y cargarlo aquí.
    
    # Este es un placeholder. EN UN CASO REAL, CARGA TU DATAFRAME COMPLETO O NECESARIO.
    # Por ejemplo:
    # df_app = pd.read_pickle('df_processed.pkl') # Si guardaste el df completo
    
    # Como alternativa y para que el ejemplo funcione sin un `df.pkl` preexistente:
    # Voy a simular la creación del df_app para que la función `recommend_songs` tenga algo con qué trabajar.
    # En tu caso, `df` es un DataFrame global en Colab, pero en Streamlit no lo será.
    # Así que, necesitamos una manera de pasar o recrear `df`.
    
    # Para hacer esto, la función `recommend_songs` deberá aceptar `df` como un argumento.
    # Como no puedo modificar la celda anterior que define `recommend_songs` y `df` es global allí,
    # la re-definiré aquí con `df` como argumento.
    
    # La forma más robusta es guardar el DataFrame procesado:
    # joblib.dump(df, 'processed_df.pkl') # En tu cuaderno de Colab
    # Y luego cargarlo aquí:
    df_app = pd.DataFrame({
        'Canción': all_songs,
        'Tema de la canción_clean': [str(t).replace('/', ' ').replace(',', ' ').replace(';', ' ').strip() for t in all_themes_raw[:len(all_songs)]] if len(all_themes_raw) >= len(all_songs) else [''] * len(all_songs), # Asegura que haya suficientes temas
        'cluster': kmeans_model.predict(tfidf_vectorizer.transform([str(t).replace('/', ' ').replace(',', ' ').replace(';', ' ').strip() for t in all_themes_raw[:len(all_songs)]])) if len(all_themes_raw) >= len(all_songs) else [0] * len(all_songs),
        'Cantidad de vistas': [1000000] * len(all_songs) # Dummy views
    })
    df_app['Cantidad de vistas_numeric'] = pd.to_numeric(df_app['Cantidad de vistas'], errors='coerce')
    
except FileNotFoundError:
    st.error("Error: Asegúrate de que 'kmeans_model.pkl', 'tfidf_vectorizer.pkl', 'all_songs.pkl', 'all_themes_raw.pkl' y 'df_app' (si lo usas) estén en el mismo directorio.")
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
        if 'df_app' in locals(): # Asegúrate de que df_app esté disponible
            recommended_songs, cluster = recommend_songs_streamlit(final_theme, df_app, n_recommendations)
            if recommended_songs:
                st.success(f"Para el tema '{final_theme}' (cluster {cluster}), te recomiendo las siguientes canciones:")
                for i, song in enumerate(recommended_songs):
                    st.write(f"{i+1}. {song}")
            else:
                st.warning(f"No se encontraron canciones para el tema '{final_theme}' en el cluster {cluster}.")
        else:
            st.error("Error: El DataFrame de canciones no se ha cargado correctamente en la aplicación.")
elif st.button('Recomendar Canciones', key='recommend_button_no_theme') and not final_theme:
    st.warning('Por favor, selecciona o escribe un tema para obtener recomendaciones.')

st.markdown("""
### Cómo usar:
1. Selecciona un tema de la lista desplegable.
2. Opcionalmente, escribe un tema personalizado en el campo de texto.
3. Ajusta el número de canciones que deseas.
4. Haz clic en 'Recomendar Canciones'.
""")
