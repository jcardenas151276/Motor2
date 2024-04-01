import streamlit as st
from PIL import Image
import pandas as pd
from itertools import product  # Para generar combinaciones de 'Register' y fechas.
from scipy.stats import linregress
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.api import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
from pmdarima.arima import auto_arima
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tqdm import tqdm
import tempfile
import shutil
import os
from zipfile import ZipFile

# Asume que tus funciones están correctamente importadas
import MOTOR_VTAxMES_FORECAST_SIMPLE_STREAMLIT_FUNCIONES as funciones
import usuarios as validacion

from usuarios import usuarios
# Diccionario de tipos de datos para asegurar el formato al cargar
dtype_dict = {
    'Unidades': float, 'Valor_Venta': float, 'Costo_Venta': float,
    'Cod_Item': str, 'Canal': str, 'Region': str,
    'Categoria': str, 'Subcategoria': str
}


def validar_usuario(usuario, contrasena):
    """Valida si el usuario y la contraseña son correctos."""
    return usuario in usuarios and usuarios[usuario] == contrasena

def save_df_as_excel(df, directory, filename):
                file_path = os.path.join(directory, filename)
                df.to_excel(file_path, index=False)

def read_zip_file(filepath):
    with open(filepath, 'rb') as f:
        return f.read()


def aplicacion_principal(authenticated):
    """Función que contiene la lógica principal de la aplicación."""

    if authenticated:
        
        # Siempre muestra el cargador de archivos si 'uploaded_file' es None en st.session_state
        if 'uploaded_file' not in st.session_state or st.session_state['uploaded_file'] is None:
            uploaded_file = st.file_uploader("Elige un archivo Excel (.xlsx)", type=["xlsx"])
            # Si se carga un archivo, actualiza 'uploaded_file' en st.session_state
            if uploaded_file is not None:
                st.session_state['uploaded_file'] = uploaded_file
        else:
            # Si ya hay un archivo cargado en st.session_state, pregunta si se desea cargar uno nuevo
            uploaded_file = st.session_state['uploaded_file']  # Recupera el archivo actual de la sesión
            if st.button("Cargar un archivo nuevo"):
                st.session_state['uploaded_file'] = None  # Esto limpiará el archivo cargado
                st.experimental_rerun()  # Reinicia la app para mostrar el file_uploader

        # Continúa solo si uploaded_file no es None
        if uploaded_file is not None:
            if st.button("Iniciar Procesamiento"):
                # Aquí iría la lógica de procesamiento después de cargar el archivo
                
                # Verifica el tipo de archivo para asegurarse de que sea un Excel válido
                if uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                    # Después de cargar el archivo Excel
                    df = pd.read_excel(uploaded_file, dtype=dtype_dict)
                    st.write("Datos Cargados:", df.head())  # Muestra las primeras filas de los datos cargados

                    # Crea una nueva columna 'Register' que combine las columnas 'Cod_Item', 'Canal' y 'Region'
                    df['Register'] = df['Cod_Item'].astype(str) + df['Canal'] + df['Region']

                    # Cuenta el número de valores únicos en la columna 'Register'
                    num_registros_unicos = df['Register'].nunique()
                    st.write(f"Número de Registros Únicos Cargados: {num_registros_unicos}")
                    
                    # Ejecuta las funciones de procesamiento de datos en secuencia
                    data_clean = funciones.preparar_limpiar_datos(df)
                    st.write(f"Limpieza de datos - Terminada")

                    data_filtered, data_counts = funciones.filtrar_registros(data_clean)
                    st.write(f"Filtrado de datos debido a registros insuficientes - Terminado")

                    data_completa = funciones.rellenar_huecos(data_filtered)
                    st.write(f"Relleno de registros con faltantes en data - Terminado")

                    # Después de rellenar datos faltantes
                    data_filled = funciones.rellenar_datos_faltantes(data_clean, data_completa)

                    # Si 'Register' no se creó previamente en data_filled, crea la columna 'Register' de nuevo
                    # Esto es solo necesario si la columna 'Register' no se mantiene en las funciones de procesamiento
                    if 'Register' not in data_filled.columns:
                        data_filled['Register'] = data_filled['Cod_Item'].astype(str) + data_filled['Canal'] + data_filled['Region']

                    # Cuenta el número de valores únicos en la columna 'Register'
                    num_registros_unicos_final = data_filled['Register'].nunique()
                    st.write(f"Número de Registros Únicos resultantes: {num_registros_unicos_final}")
                    
                    if num_registros_unicos_final > 800:
                        st.error("Máximo 100 registros por corrida. Cargar un archivo con menos registros.")
                        # No es necesario un botón de reinicio aquí, ya que el usuario puede usar "Cargar un archivo nuevo"
                        st.stop()
                                        
                    ### CALCULO DE PENDIENTE ###

                    # Eliminamos las columnas 'Canal' y 'Region'
                    Data_Temp = data_filled.drop(columns=['Canal', 'Region'])

                    #Extraemos el mes de la fecha y lo llevamos a una columna aparte llamada Month.
                    Data_Temp['Month'] = Data_Temp['Fecha'].dt.to_period('M')

                    # Agrupamos por las columnas restantes y sumamos los valores numéricos
                    DatosHistoria = Data_Temp.groupby(['Fecha', 'Cod_Item', 'Categoria', 'Subcategoria', 'Month']).sum().reset_index()
                    
                    DatosHistoria_SKUS12Meses_Grouped = funciones.filter_and_group(DatosHistoria, 'Cod_Item',min_months = 12 )
                    DatosHistoria_Categ12Meses_Grouped = funciones.filter_and_group(DatosHistoria, 'Categoria', min_months = 12)
                    DatosHistoria_SubCateg12Meses_Grouped = funciones.filter_and_group(DatosHistoria, 'Subcategoria', min_months = 12)
                        
                        # Diccionario para almacenar los DataFrames de cada informe de pendientes
                    all_slopes_reports = {}

                    for grouped_data, name in [
                        (DatosHistoria_SKUS12Meses_Grouped, 'SKU'), 
                        (DatosHistoria_Categ12Meses_Grouped, 'Categ'), 
                        (DatosHistoria_SubCateg12Meses_Grouped, 'SubCateg')
                    ]:
                        rows = []
                        
                        # Iteramos sobre cada grupo único
                        for group, df_group in grouped_data.groupby(grouped_data.columns[0]):
                            unidades_slope = funciones.calculate_slope(df_group, 'Unidades')
                            pvp_slope = funciones.calculate_slope(df_group, 'PVP')
                            rows.append({
                                grouped_data.columns[0]: group,
                                'Unidades_Slope': unidades_slope,
                                'PVP_Slope': pvp_slope
                            })

                        # Convertimos la lista de diccionarios a un DataFrame
                        Slopes_Report = pd.DataFrame(rows)

                        # Almacenamos este DataFrame en el diccionario utilizando 'name' como clave
                        all_slopes_reports[name] = Slopes_Report

                    # Mostrando un preview de cada informe de pendientes en Streamlit
                    for name, slopes_df in all_slopes_reports.items():
                        st.write(f"Preview de pendientes para {name}:")
                        st.dataframe(slopes_df.head())  # Muestra las primeras filas de cada DataFrame

                    
                    ### CALCULO DE ESTACIONALIDAD ###
                    
                    DatosHistoria_SKUS24Meses_Grouped = funciones.filter_and_group(DatosHistoria, 'Cod_Item',min_months = 24 )
                    DatosHistoria_Categ24Meses_Grouped = funciones.filter_and_group(DatosHistoria, 'Categoria', min_months = 24)
                    DatosHistoria_SubCateg24Meses_Grouped = funciones.filter_and_group(DatosHistoria, 'Subcategoria', min_months = 24)

                    seasonality_reports = {}
                    for grouped_data, name in [(DatosHistoria_SKUS24Meses_Grouped, 'Cod_Item'), (DatosHistoria_Categ24Meses_Grouped, 'Categoria'), (DatosHistoria_SubCateg24Meses_Grouped, 'Subcategoria')]:
                        seasonality_report = funciones.calculate_seasonality_por_ano(grouped_data, name)  # Calcula la estacionalidad
                        seasonality_reports[name] = seasonality_report  # Guarda el informe en un diccionario
                    
                    # Diccionario para almacenar los DataFrames de cada reporte
                    all_seasonality_reports = {}

                    for name, df in seasonality_reports.items():
                        report = []
                        unique_identifiers = df[name].unique()
                        for identifier in unique_identifiers:
                            data_filtered = df[df[name] == identifier]
                        # Desempaqueta los cuatro diccionarios devueltos
                            sig_months_high_un, sig_months_low_un, sig_months_high_pvp, sig_months_low_pvp = funciones.find_significant_months(data_filtered, umbral=0.3, periodos=2)
                            for month in range(1, 13):
                                report.append({
                                    'Identificador': identifier,  # Asegúrate de incluir esto si necesitas identificar cada reporte
                                    'Mes': month,
                                    'Estacionalidad_Unidades': sig_months_high_un.get(month, sig_months_low_un.get(month, 'Sin_estacionalidad')),
                                    'Estacionalidad_PVP': sig_months_high_pvp.get(month, sig_months_low_pvp.get(month, 'Sin_estacionalidad'))
                                    })
            
                        # Convertimos el reporte de esta iteración a un DataFrame
                        report_df = pd.DataFrame(report)
            
                        # Usamos un identificador único (puedes ajustar esto según tus necesidades) para almacenar este DataFrame en el diccionario
                        all_seasonality_reports[name] = report_df

                        # Mostrando un preview de cada informe en Streamlit
                    for name, report_df in all_seasonality_reports.items():
                        st.write(f"Preview de estacionalidades para {name}:")
                        st.dataframe(report_df.head())  # Muestra las primeras filas de cada DataFrame

                    ### CORRIDA DE PRONOSTICOS ###
                    # # Selecciona combinaciones únicas de 'Cod_Item', 'Canal' y 'Region'
                    unique_combinations = data_filled[['Cod_Item', 'Canal', 'Region']].drop_duplicates()

                    # # %%
                    # # Parametros de pronostico
                    TS1 = 6  # Periodo de revision del modelo (Partición de train/test set)
                    DF1 = 'MS'  # Frecuencia de la data
                    SP1 = 12  # Periodos de estacionalidad
                    FP1 = 6  # Periodos a pronosticar
                    
                    # # %%
                    # # Se divide en Df en partes para que podamos ir trabajando los pronosticos por etapas.
                    # # Definir el número de particiones
                    num_particiones = 20
                    
                    # # Aplicar la función para dividir 'data_filled_w_vbles' en partes
                    partes1, unique_series_total = funciones.dividir_df_en_partes(data_filled, ['Cod_Item', 'Canal', 'Region'], num_particiones)
                    
                    from tqdm import tqdm
                    test_KPIS, fcst_test, optimal_params = funciones.Entrenamiento(partes1, 
                                                                                        TS1, 
                                                                                        DF1, 
                                                                                        SP1, 
                                                                                        FP1)
                    
                    st.dataframe(test_KPIS.head(30))  # Muestra las primeras filas de cada DataFrame
                    
                    ### Reporte de mejores modelos ###
                    best_models_report = funciones.select_best_model(test_KPIS)
                    st.dataframe(best_models_report.head(30))  # Muestra las primeras filas de cada DataFrame
                    
                    future_fcst = funciones.Forecast (best_models_report, optimal_params, partes1, TS1, DF1, SP1, FP1)
                    st.dataframe(future_fcst .head(30))  # Muestra las primeras filas de cada DataFrame
                    
                    temp_dir = tempfile.mkdtemp()

                    # Guarda cada DataFrame como un archivo Excel en el directorio temporal
                    save_df_as_excel(data_filled, temp_dir, 'data_filtrada.xlsx')

                    for name, df in all_slopes_reports.items():
                        save_df_as_excel(df, temp_dir, f'{name}_slopes_report.xlsx')

                    for name, df in all_seasonality_reports.items():
                        save_df_as_excel(df, temp_dir, f'{name}_seasonality_report.xlsx')

                    save_df_as_excel(best_models_report, temp_dir, 'Mejores_modelos_x_registro.xlsx')
                    save_df_as_excel(future_fcst, temp_dir, 'Pronostico.xlsx')
                    
                    # 2. Comprimir el directorio en un archivo ZIP
                    zip_filename = 'Resultados.zip'
                    zip_filepath = os.path.join(temp_dir, zip_filename)

                    with ZipFile(zip_filepath, 'w') as zipf:
                        for root, dirs, files in os.walk(temp_dir):
                            for file in files:
                                if file.endswith('.xlsx'):
                                    zipf.write(os.path.join(root, file), arcname=file)

                    with open(zip_filepath, 'rb') as f:
                        zip_data = f.read()

                    st.download_button(label='Descargar Todos los Archivos',
                                    data=zip_data,
                                    file_name=zip_filename,
                                    mime='application/zip')
                    
                    
                    # Opcional: Limpia el directorio temporal después de la descarga
                    shutil.rmtree(temp_dir)
                
                else:
                    st.error('Error: El archivo no es un .xlsx. Por favor, carga un archivo Excel válido.')
                

#streamlit run main.py

def main():
    # Configura la página para utilizar todo el ancho disponible y establece el título de la página en la pestaña del navegador
    st.set_page_config(page_title="Motor de Pronósticos", layout="wide")

    # Carga la imagen desde un archivo
    imagen = Image.open("Motor_Strmlt_2024-04-01_11-58-50.jpg")

    # Contenedor para imagen y título
    col1, _ = st.columns([1, 50])
    
    with col1:
        st.image(imagen, width=200)  # Ajusta el ancho según necesites
    
    # Opción para descargar la plantilla de Excel siempre visible
    st.markdown("### Descarga la plantilla de Excel para cargar tus datos")
    with open("In_Formato_Ventas_MesAno_Maderkit_Marzo.xlsx", "rb") as file:
        st.download_button(
            label="Descargar Plantilla de Excel",
            data=file,
            file_name="Plantilla_Excel.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # A continuación, procedemos con la lógica de inicio de sesión
    st.title('Motor de pronósticos')
        
    # Lógica de inicio de sesión
    if not st.session_state.get('authenticated', False):
        usuario = st.text_input("Correo electrónico")
        contrasena = st.text_input("Contraseña", type="password")

        if st.button("Iniciar sesión"):
            if validar_usuario(usuario, contrasena):
                st.session_state['authenticated'] = True  # Marcar como autenticado
                st.success("Has iniciado sesión exitosamente.")
            else:
                st.error("El correo electrónico o la contraseña son incorrectos.")

    if st.session_state.get('authenticated', False):
        aplicacion_principal(True)  # Continúa con la aplicación si el usuario está autenticado

if __name__ == "__main__":
    main()
