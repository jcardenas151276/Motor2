# %%
# %%
#Importacion de librerias

import pandas as pd
import numpy as np
from itertools import product
from scipy.stats import linregress

def preparar_limpiar_datos(data):
    # Convierte la columna 'Fecha' a formato datetime, manejando errores con 'coerce'.
    data['Fecha'] = pd.to_datetime(data['Fecha'], errors='coerce')
    
    # Elimina filas duplicadas para mantener la unicidad en los datos.
    data_limpia = data.drop_duplicates()
    
    # Elimina filas con valores NaN en columnas clave para análisis posterior.
    data_limpia = data_limpia.dropna(subset=['Categoria', 'Subcategoria', 'Canal', 'Region', 'Unidades', 'Valor_Venta', 'Costo_Venta'])
    
    # Ahora, elimina filas con valores negativos en 'Unidades', 'Valor_Venta', o 'Costo_Venta'.
    # Utiliza una condición que chequea por valores negativos en estas columnas específicas.
    columnas_verificar = ['Unidades', 'Valor_Venta', 'Costo_Venta']
    for columna in columnas_verificar:
        data_limpia = data_limpia[data_limpia[columna] >= 0]
    
    
    # Define las columnas no numéricas por las cuales agrupar para sumar Unidades, Valor_Venta y Costo_Venta para registros iguales, en la misma fecha.
    columnas_para_agrupar = ['Fecha','Cod_Item', 'Categoria', 'Subcategoria', 'Canal', 'Region']

    # Agrupa por las columnas no numéricas y suma las numéricas
    data_limpia2 = data_limpia.groupby(columnas_para_agrupar, as_index=False).agg({
        #'Fecha': 'first',  # Mantiene la primera fecha encontrada (ajustar según sea necesario)
        'Unidades': 'sum',
        'Valor_Venta': 'sum',
        'Costo_Venta': 'sum'
        })
    
    # Reajusta los índices del DataFrame para que sean secuenciales, comenzando en 0.
    data_limpia2.reset_index(drop=True, inplace=True)
    
    data_limpia2['Register'] = data_limpia2['Cod_Item'].astype(str) + data_limpia2['Canal'] + data_limpia2['Region']
    
    num_valores_unicos = data_limpia2['Register'].nunique()
    print('Número de Registros inciales:', num_valores_unicos)
    
    return data_limpia2



# %% [markdown]
# Funcion filtrar registros busca limpiar la data de los registros con muy pocas entradas. En este caso elimina registros con menos de 6 entradas en los ultimos 12 meses.

# %%
#Funcion filtrar_registros

def filtrar_registros(data, meses=12):
    
    # Obtiene la última fecha en los datos.
    ultima_fecha = data['Fecha'].max()

    # Calcula la fecha límite retrocediendo "meses" desde la última fecha, y luego le sumas 1 mes para ajustar.
    n_meses_antes_de_ultima_fecha = ultima_fecha - pd.DateOffset(months=meses) + pd.DateOffset(months=1)
    
    # Combina 'Cod_Item', 'Canal', y 'Region' en una nueva columna 'Register'.
    data['Register'] = data['Cod_Item'].astype(str) + data['Canal'] + data['Region']
    
    # Cuenta el total de registros por 'Register'.
    conteo_total = data.groupby('Register').size().reset_index(name='total_counts')
    
    # Cuenta registros recientes por 'Register' desde la fecha límite calculada.
    conteo_recientes = data[data['Fecha'] >= n_meses_antes_de_ultima_fecha].groupby('Register').size().reset_index(name='recent_counts')
    
    # Fusiona conteos totales y recientes con los datos originales.
    Data_with_counts = pd.merge(data, conteo_total, on='Register', how='left')
    Data_with_counts = pd.merge(Data_with_counts, conteo_recientes, on='Register', how='left')
    
    # Rellena con 0s donde 'recent_counts' es NaN.
    Data_with_counts['recent_counts'] = Data_with_counts['recent_counts'].fillna(0)
    
    # Filtra registros con 6 o más entradas en el último año.
    Data_filtered = Data_with_counts[Data_with_counts['recent_counts'] >= 6]
    
    # Elimina columnas de conteo y reajusta índices en el DataFrame filtrado.
    Data_filtered = Data_filtered.drop(columns=['total_counts', 'recent_counts']).reset_index(drop=True)
    
    num_valores_unicos = Data_filtered['Register'].nunique()
    print('Número de Registros finales (con depuracion):', num_valores_unicos)
    
    return Data_filtered, Data_with_counts



# %% [markdown]
# Funcion rellenar huecos lo que busca es que todos los registros que hayan quedado en data, tengan la misma cantidad de entradas. Busca entonces fecha min y fecha max 
# en la data, crea un listado de fechas entre esos dos limites y se trae toda la informacion original de cada registro rellenando con cero unidades, ventas$ y Costos de 
# registros sin dato.

# %%
#Funcion rellenar_huecos

from itertools import product  # Para generar combinaciones de 'Register' y fechas.

def rellenar_huecos(data):
    # Genera un rango de fechas mensuales entre la fecha mínima y máxima.
    rango_meses = pd.date_range(start=data['Fecha'].min(), end=data['Fecha'].max(), freq='MS')
    
    # Crea combinaciones de 'Register' y fechas para asegurar cobertura completa.
    combinaciones = pd.DataFrame(list(product(data['Register'].unique(), rango_meses)), columns=['Register', 'Fecha'])
    
    # Combina las combinaciones con los datos originales, manteniendo todas las fechas para cada 'Register'.
    data_rellenada = pd.merge(combinaciones, data, on=['Register', 'Fecha'], how='left')
    
    # Rellena NaNs en 'Unidades', 'Valor_Venta', y 'Costo_Venta' con 0s para consistencia.
    data_rellenada = data_rellenada.fillna({'Unidades': 0, 'Valor_Venta': 0, 'Costo_Venta': 0})
    
    # Reajusta índices para que sean secuenciales después de la fusión y el relleno.
    data_rellenada.reset_index(drop=True, inplace=True)
    
    return data_rellenada


# %% [markdown]
# Funcion rellenar datos faltantes busca que todos los registros ya con las mismas fechas, tengan todos la informacion de caracterisiticas completa (Categoria, Subcategoria, 
# region, etc...)

# %%
#Funcion rellenar_datos_faltantes

def rellenar_datos_faltantes(data_original, data_procesada):
    # Extrae registros sin NaN en 'Subcategoria' y 'Categoria', y elimina duplicados por 'Register'.
    valores_unicos = data_original.dropna(subset=['Subcategoria', 'Categoria']).drop_duplicates(
        subset='Register')[['Register', 'Subcategoria', 'Categoria', 'Cod_Item', 'Canal', 'Region']]
    
    # Prepara 'data_procesada' eliminando columnas específicas y luego rellena datos faltantes desde 'valores_unicos'.
    data_rellenada = pd.merge(data_procesada.drop(columns=['Subcategoria', 'Categoria', 'Cod_Item', 'Canal', 
                                                           'Region']), valores_unicos, on='Register', how='left')
    
    # Reajusta índices para secuencia desde 0 después de la fusión.
    data_rellenada.reset_index(drop=True, inplace=True)
    
    return data_rellenada

# # Definición de la función filter_and_group
def filter_and_group(data, group_col, min_months):
    # Filtra los datos para incluir solo aquellos con más de 'min_months' meses de datos
    filtered_data = data[data.groupby(group_col)['Month'].transform('nunique') > min_months]

    # Agrupa los datos filtrados por la columna especificada, fecha y mes, y calcula la suma de unidades, valor de venta y costo de venta
    grouped_data = filtered_data.groupby([group_col, 'Fecha', 'Month']).agg({'Unidades': 'sum', 'Valor_Venta': 'sum', 
                                                                             'Costo_Venta': 'sum'}).reset_index()

    # Calcula el PVP (precio de venta promedio) para cada grupo
    grouped_data['PVP'] = grouped_data['Valor_Venta'] / grouped_data['Unidades']
    return grouped_data  # Retorna los datos agrupados y con PVP calculado

# # Definición de la función calculate_slope
def calculate_slope(df, y_column):
    # Crea un rango numérico como variable independiente para la regresión lineal
    x_numeric = np.arange(len(df))
    # Realiza una regresión lineal y retorna solo la pendiente
    return linregress(x_numeric, df[y_column])[0]


# # %% [markdown]
# # Este bloque de código es la funcion para calcular el Seasonal Index para diferentes categorías de datos, como Cod_Item, Categoría y Subcategoría. La estacionalidad se calcula como La division entre las ventas de un mes con el promedio de las  ventas por mes de un ano.el Seasonal index para cada mes y cada ano que exista en data. El objetivo con esta estrategia es identificar que siempre en el mismo mes de cada ano, exista un patron de venta por encima del promedio o venta por debajo del promedio.

# # %%
# #Funcion calculate_seasonality_por_ano

def calculate_seasonality_por_ano(data, group_col):
    # Extrae el año y el mes de la columna 'Fecha'
    data['Year'] = data['Fecha'].dt.year
    data['Month'] = data['Fecha'].dt.month
    
    # Agrupa los datos por año, mes y la columna especificada, y luego calcula la suma de unidades, valor de venta y costo de venta
    grouped_data = data.groupby([group_col, 'Year', 'Month']).agg({'Unidades': 'sum', 'Valor_Venta': 'sum', 'Costo_Venta': 'sum'}).reset_index()
    
    # Calcula el PVP para cada grupo
    grouped_data['PVP'] = grouped_data['Valor_Venta'] / grouped_data['Unidades']
    
    # Calcula los valores medios de unidades y PVP para cada categoría y año
    mean_values = grouped_data.groupby([group_col, 'Year'])[['Unidades', 'PVP']].mean().rename(columns={'Unidades': 'Mean_Unidades', 'PVP': 'Mean_PVP'})
    
    # Fusiona los valores medios calculados con los datos agrupados
    seasonality_index = grouped_data.merge(mean_values, on=[group_col, 'Year'], how='left')
    
    # Calcula los índices de estacionalidad para 'Unidades' y 'PVP'
    seasonality_index['Indice_Estac_Un'] = seasonality_index['Unidades'] / seasonality_index['Mean_Unidades']
    seasonality_index['Indice_Estac_PVP'] = seasonality_index['PVP'] / seasonality_index['Mean_PVP']
    
    return seasonality_index  # Devuelve el DataFrame con los índices de estacionalidad calculados


# # %% [markdown]
# # Funcion de busqueda de estacionalidades (Picos y valles) comunes en los mismos meses de cada ano. La funcion busca comportamientos que se repitan en un mismo registro en el mismo mes un x periodos.
# # Umbral es el tamano de la caida en ventas o el pico de ventas frente al promedio mensual.
# # Periodos es la cantidad de anos en que el registro debe presentar la caida o el pico en el mismo mes.
# # La estacionalidad se calcula tanto para unidades como pvps.

# # %%
# #Funcion find_significant_months para encontrar las estacionalidades significativas por registro / periodo

def find_significant_months(data, umbral, periodos):
    significant_months_high_un = {}
    significant_months_low_un = {}
    significant_months_high_pvp = {}
    significant_months_low_pvp = {}
    
    for month in range(1, 13):
        month_data = data[data['Month'] == month]
        
        high_count_un = month_data[month_data['Indice_Estac_Un'] >= 1+umbral].shape[0]
        low_count_un = month_data[month_data['Indice_Estac_Un'] <= 1-umbral].shape[0]
        high_count_pvp = month_data[month_data['Indice_Estac_PVP'] >= 1+umbral].shape[0]
        low_count_pvp = month_data[month_data['Indice_Estac_PVP'] <= 1-umbral].shape[0]
        
        if high_count_un >= periodos:
            significant_months_high_un[month] = 'Estacionalidad_Positiva_>=_1.30'
        if low_count_un >= periodos:
            significant_months_low_un[month] = 'Estacionalidad_Negativa_<=_0.70'
        
        if high_count_pvp >= periodos:
            significant_months_high_pvp[month] = 'Estacionalidad_Positiva_>=_1.30'
        if low_count_pvp >= periodos:
            significant_months_low_pvp[month] = 'Estacionalidad_Negativa_<=_0.70'

    return significant_months_high_un, significant_months_low_un, significant_months_high_pvp, significant_months_low_pvp

# # %%
# # Definición de función para calcular errores de pronóstico
def ForecastErrors(V, F):
    # Verifica que V y F sean Series de pandas y tengan el mismo índice
    if not (isinstance(V, pd.Series) and isinstance(F, pd.Series)):
        raise ValueError("V and F must be pandas Series.")
    if not V.index.equals(F.index):
        raise ValueError("V and F must have the same index.")

    # Inicializa listas para almacenar valores individuales de errores
    mfas, biases, maes, rmses = [], [], [], []

    # Calcula MFA, BIAS, MAE y RMSE para cada periodo
    for actual, forecast in zip(V, F):
        # Manejo de casos especiales cuando actual y forecast son cero
        if actual == 0 and forecast == 0:
            mfa = bias = 0
        elif actual == 0 and forecast != 0:
            mfa = 100
            bias = -100
        else:
            # Calculo de MFA y BIAS para casos generales
            mfa = np.abs((actual - forecast) / np.maximum(np.abs(actual), np.abs(forecast))) * 100
            bias = (actual - forecast) / actual * 100
        mae = np.abs(actual - forecast)
        rmse = np.sqrt(np.mean((actual - forecast) ** 2))
        
        # Almacena los cálculos en las listas correspondientes
        mfas.append(mfa)
        biases.append(bias)
        maes.append(mae)
        rmses.append(rmse)

    # Calcula los valores promedio de cada métrica de error
    average_mfa = np.mean(mfas)
    average_bias = np.mean(biases)
    average_mae = np.mean(maes)
    average_rmse = np.mean(rmses)

    # Crea un DataFrame para los resultados por periodo
    results_df = pd.DataFrame({'MFA': mfas, 'BIAS': biases, 'MAE': maes,'RMSE': rmses })

    # Diccionario para los resultados promedio
    average_results = {'Average MFA': average_mfa, 'Average BIAS': average_bias, 'Average MAE': average_mae,
                       'Average RMSE': average_rmse}

    return results_df, average_results

# # %%
# #-----#
# #-----#
# # MODELOS DE PRONOSTICO

from dateutil.relativedelta import relativedelta
def NAIVE(V, TS, DF, SP, FP, opt_parameters=None):
    # Asegurarse de que el índice de V sea de tipo datetime y esté ordenado
    if not pd.api.types.is_datetime64_any_dtype(V.index):
        V['Fecha'] = pd.to_datetime(V['Fecha'])
        V.set_index('Fecha', inplace=True)
    V.sort_index(inplace=True)

    # Entrenamiento y Seteo
    TrainingSet = V.iloc[:-TS, :]  # Todo menos los últimos TS para entrenar el modelo
    TestSet = V.iloc[-TS:, :]  # Los últimos TS para setear el modelo

    # Inicializar lista para almacenar predicciones
    Predictions = []

    for i in range(TS):
        if len(TrainingSet) >= 3:
            # Mes anterior (i-1)
            previous_month_value = TrainingSet['y'].iloc[-1] * 0.3

            # Mes anterior al anterior (i-2)
            two_months_ago_value = TrainingSet['y'].iloc[-2] * 0.2

            # Tres meses atrás (i-3)
            three_months_ago_value = TrainingSet['y'].iloc[-3] * 0.1
        else:
            previous_month_value = two_months_ago_value = three_months_ago_value = 0

        # Mismo mes del año anterior
        same_month_last_year = TrainingSet.index[-1] - pd.DateOffset(months=12) + relativedelta(months=i)
        same_month_last_year_value = V['y'].get(same_month_last_year, 0) * 0.4

        # Predicción para el mes actual
        prediction = previous_month_value + two_months_ago_value + three_months_ago_value + same_month_last_year_value
        Predictions.append(prediction)

    # Convertir predicciones a una Serie de pandas y ajustar el índice
    Predictions = pd.Series(Predictions, index=TestSet.index).rename('NAIVE')

    # Ajustar pronósticos negativos a cero
    Predictions[Predictions < 0] = 0

    # Errores de pronóstico
    results_df, average_results = ForecastErrors(TestSet['y'], Predictions)

    return Predictions, average_results, None

#-----#
#-----#
from statsmodels.tsa.api import SimpleExpSmoothing
def Simple_Exp(V, TS, DF, SP, FP, opt_parameters = None):
    
    TrainingSet = V.iloc[:-TS, :] # Todo menos los ultimos TestDays para entrenar el modelo
    TestSet = V.iloc[-TS:, :] # Los ultimos TestDays para setear el modelo

    # Modelo de Forecast (Ante problemas de convergencia a una solucion optima, 
    # se puede cambiar initialization_method='heuristic')
    model = SimpleExpSmoothing(endog = TrainingSet['y'],initialization_method='estimated').fit()

    # Predicciones
    Predictions = pd.Series(model.forecast(steps=len(TestSet))).rename('Holt')
    Predictions.index = TestSet.index
    
    # Ajustar pronósticos negativos a cero
    Predictions[Predictions < 0] = 0
    
    # Errores de pronóstico    
    results_df, average_results = ForecastErrors(TestSet['y'], Predictions)

    return Predictions, average_results , None

#-----#
#-----#
from statsmodels.tsa.api import Holt
def Holt_Double(V, TS, DF, SP, FP, opt_parameters=None):

    # Entrenamieto y Seteo
    
    TrainingSet = V.iloc[:-TS, :] # Todo menos los ultimos TestDays para entrenar el modelo
    TestSet = V.iloc[-TS:, :] # Los ultimos TestDays para setear el modelo

    # Modelo de Forecast (Ante problemas de convergencia a una solucion optima, 
    # se puede cambiar initialization_method='heuristic')
    model = Holt(endog = TrainingSet['y'],initialization_method='estimated').fit()

    # Predicciones
    Predictions = pd.Series(model.forecast(steps=len(TestSet))).rename('Holt')
    Predictions.index = TestSet.index
    
    # Ajustar pronósticos negativos a cero
    Predictions[Predictions < 0] = 0
    
    # Errores de pronóstico    
    results_df, average_results = ForecastErrors(TestSet['y'], Predictions)

    return Predictions, average_results , None


#-----#
#-----#
from statsmodels.tsa.holtwinters import ExponentialSmoothing
def HoltWinters(V, TS, DF, SP, FP, opt_parameters = None):
    
    # Entrenamiento y Seteo
    TrainingSet = V.iloc[:-TS]
    TestSet = V.iloc[-TS:]
    
    # Modelo de Forecast (Ante problemas de convergencia a una solucion optima, 
    # se puede cambiar initialization_method='heuristic')
    model = ExponentialSmoothing(endog=TrainingSet['y'], trend='add', seasonal='add', seasonal_periods=SP, 
                                    initialization_method='estimated').fit()
    # Predicciones
    Predictions = model.forecast(steps=len(TestSet)).rename('HW')
    Predictions.index = TestSet.index
    
    # Ajustar pronósticos negativos a cero
    Predictions[Predictions < 0] = 0
    
    # Errores de pronóstico
    results_df, average_results = ForecastErrors(TestSet['y'], Predictions)
    return Predictions, average_results, None

#-----#
#-----#
#ARIMA Y SARIMA quedan sin cross validation ni con la posibilidad de uso de parametros optimos. 
#Imposible que el modelo tome los parametros otpimos de un Df
from pmdarima import auto_arima
def ARIMA(V,TS,DF,SP, FP, opt_parameters = None):
    
     # Validación de parámetros
    if TS <= 0 or TS >= len(V):
        raise ValueError("El tamaño del conjunto de prueba TS debe ser mayor que 0 y menor que la longitud de V.")
    
    # Entrenamieto y Seteo
    TrainingSet=V.iloc[:-TS,:] #Todo menos los ultimos TestDays para entrenar el modelo
    TestSet=V.iloc[-TS:,:] #Los ultimos TestDays para setear el modelo
    
    # Modelo de Forecast (En caso de que se este tomando mucho tiempo, se puede llevar stepwise=True, para una busqueda
    # mas eficiente de la configuracion del modelo AR=# , I=#, MA=#)
    model=auto_arima(y=TrainingSet['y'], seasonal=False,stepwise=True,maxlag= 24, error_action='warn', missing='mean')
    
    # Predicciones
    Predictions=pd.Series(model.predict(n_periods=TS)).rename('Arima')
    Predictions.index=TestSet.index
    
    # Ajustar pronósticos negativos a cero, si es necesario
    Predictions[Predictions < 0] = 0
    
    #Errores de pronóstico
    results_df, average_results = ForecastErrors(TestSet['y'],Predictions)
    return Predictions, average_results, None

#-----#
#-----#
#ARIMA Y SARIMA quedan sin cross validation ni con la posibilidad de uso de parametros optimos. 
#Imposible que el modelo tome los parametros otpimos de un Df
from pmdarima.arima import auto_arima
def SARIMA(V,TS,DF,SP, FP, opt_parameters = None):
    
        # Validación de parámetros
    if TS <= 0 or TS >= len(V):
        raise ValueError("El tamaño del conjunto de prueba TS debe ser mayor que 0 y menor que la longitud de V.")

    if SP <= 0:
        raise ValueError("La periodicidad estacional SP debe ser mayor que 0.")
    
    # Entrenamieto y Seteo
    TrainingSet=V.iloc[:-TS,:] #Todo menos los ultimos TestDays para entrenar el modelo
    TestSet=V.iloc[-TS:,:] #Los ultimos TestDays para setear el modelo
    
    # Modelo de Forecast (En caso de que se este tomando mucho tiempo, se puede llevar stepwise=True, para una busqueda
    # mas eficiente de la configuracion del modelo AR=# , I=#, MA=#)
    model=auto_arima(y=TrainingSet['y'],m=SP,seasonal=True,stepwise=True,maxlag= 24, error_action='warn', missing='mean')
    
    # Predicciones
    Predictions=pd.Series(model.predict(n_periods=TS)).rename('Sarima')
    Predictions.index=TestSet.index
    
    # Ajustar pronósticos negativos a cero, si es necesario
    Predictions[Predictions < 0] = 0
    
    #Errores de pronóstico
    results_df, average_results = ForecastErrors(TestSet['y'],Predictions)
    return Predictions, average_results, None


#-----#
#-----#
from statsmodels.tsa.forecasting.theta import ThetaModel
def AUTO_THETA(V, TS, DF, SP, FP, opt_parameters=None):
    # Entrenamiento y Seteo
    TrainingSet = V.iloc[:-TS, :]  # Todo menos los últimos TS para entrenar el modelo
    TestSet = V.iloc[-TS:, :]      # Los últimos TS para testear el modelo

    # Modelo de Forecast
    model = ThetaModel(TrainingSet['y'], deseasonalize=True, period=SP, method='auto')
    model_fit = model.fit()

    # Predicciones
    Predictions = pd.Series(model_fit.forecast(steps=len(TestSet))).rename('AutoTheta')
    Predictions.index = TestSet.index

    # Ajustar pronósticos negativos a cero, si es necesario
    Predictions[Predictions < 0] = 0
    # Errores de pronóstico
    results_df, average_results = ForecastErrors(TestSet['y'], Predictions)

    # Retorno de información
    return Predictions, average_results, None

#-----#
#-----#
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
def AUTO_ETS(V, TS, DF, SP, FP, opt_parameters=None):
    # Entrenamiento y Seteo
    TrainingSet = V.iloc[:-TS, :]  # Todo menos los últimos TS para entrenar el modelo
    TestSet = V.iloc[-TS:, :]      # Los últimos TS para testear el modelo

    # Modelo de Forecast
    model = ETSModel(TrainingSet['y'], error='add', trend='add', seasonal='add', damped_trend=True, seasonal_periods=SP)
    model_fit = model.fit()

    # Predicciones
    Predictions = pd.Series(model_fit.forecast(steps=len(TestSet))).rename('AutoETS')
    Predictions.index = TestSet.index

    # Ajustar pronósticos negativos a cero, si es necesario
    Predictions[Predictions < 0] = 0

    # Errores de pronóstico
    results_df, average_results = ForecastErrors(TestSet['y'], Predictions)

    # Retorno de información
    return Predictions, average_results, None

#-----#
#-----#
from statsmodels.tsa.holtwinters import ExponentialSmoothing
def AUTO_CES(V, TS, DF, SP, FP, opt_parameters = None):
    # Entrenamiento y Seteo
    TrainingSet = V.iloc[:-TS, :]  # Todo menos los últimos TS para entrenar el modelo
    TestSet = V.iloc[-TS:, :]      # Los últimos TS para testear el modelo

    # Modelo de Forecast
    # Nota: No existe una implementación directa de AutoCES, así que usaremos ExponentialSmoothing con ajuste automático
    model = ExponentialSmoothing(TrainingSet['y'], trend='add', seasonal='add', seasonal_periods=SP)
    model_fit = model.fit(optimized=True)

    # Predicciones
    Predictions = pd.Series(model_fit.forecast(steps=len(TestSet))).rename('AutoCES')
    Predictions.index = TestSet.index

    # Errores de pronóstico
    results_df, average_results = ForecastErrors(TestSet['y'], Predictions)

    # Retorno de información
    return Predictions, average_results, None


# Función para dividir el DataFrame en partes
def dividir_df_en_partes(df, columnas_unicas, num_partes):
    # Obtener las filas únicas basadas en las columnas especificadas
    unique_series = df[columnas_unicas].drop_duplicates().reset_index(drop=True)
    unique_series['Parte'] = unique_series.index % num_partes
    
    # Combinar el DataFrame original con la información de las partes
    df_con_partes = df.merge(unique_series, on=columnas_unicas)
    
    # Crear un diccionario para almacenar las partes divididas
    partes = {}
    
    # Dividir el DataFrame en partes y almacenarlas en el diccionario
    for i in range(num_partes):
        partes[i] = df_con_partes[df_con_partes['Parte'] == i].drop(columns='Parte')
    
    return partes, unique_series



# %%
from tqdm import tqdm

def Entrenamiento(partes, TS, DF, SP, FP):
    # Variables adicionales para modelos multivariable (machine learning)
    Variable_objetivo = 'Unidades'
    
    Modelos_Univariables = [NAIVE, Simple_Exp, Holt_Double, HoltWinters, ARIMA, 
                                    SARIMA, AUTO_THETA, AUTO_ETS, AUTO_CES]
    
    # DataFrame para almacenar los resultados detallados
    detailed_results = pd.DataFrame(columns=['Fecha', 'Cod_Item', 'Canal', 'Region','Modelo', 'V', 'F'])

    # DataFrame para almacenar los resultados de todos los modelos
    results_df_all_series = pd.DataFrame(columns=['Modelo', 'Cod_Item', 'Canal', 'Region', 'Average MFA', 'Average BIAS', 'Average MAE', 'Average RMSE'])

    # DataFrame para almacenar los parametros optimos de cada modelo
    opt_params_df = pd.DataFrame(columns=['Cod_Item', 'Canal', 'Region', 'Modelo', 'Opt_parameters'])

    consolidated_results_df_all_series = pd.DataFrame()
    consolidated_detailed_results = pd.DataFrame()
    # Iterar sobre cada serie temporal en la muestra
    # Luego, iterar sobre cada parte y realizar los pronósticos

    for num_parte, df_parte in partes.items():
        
        #if num_parte not in [4]: # Descomentar si estoy haciendo pruebas
            #continue
        
        # Restablece estos DataFrames para cada parte
        results_df_all_series = pd.DataFrame()
        detailed_results = pd.DataFrame()
        unique_series_per_df_parte = df_parte[['Cod_Item', 'Canal', 'Region']].drop_duplicates()
        unique_series_per_df_parte.reset_index(drop=True, inplace=True)
        
        print('Particion#: ',num_parte)

        for index, series in tqdm(unique_series_per_df_parte.iterrows(), desc=f'Procesando series en la parte {num_parte}', total=len(unique_series_per_df_parte)):
            
            print(f"Cod_Item: {series['Cod_Item']}, Canal: {series['Canal']}, Region: {series['Region']}")
            # Preparar los datos para cada combinación de Cod_Item, Canal y Region
            filtered_series = df_parte[(df_parte['Cod_Item'] == series['Cod_Item']) & 
                                                (df_parte['Canal'] == series['Canal']) & 
                                                (df_parte['Region'] == series['Region'])]
            
            filtered_series = filtered_series.copy()
            filtered_series['Fecha'] = pd.to_datetime(filtered_series['Fecha'])
            filtered_series = filtered_series.set_index('Fecha')
            
            # Verificar duplicados y sumar las unidades en caso de fechas duplicadas
            filtered_series = filtered_series.groupby(filtered_series.index).sum()
            
            filtered_series = filtered_series.asfreq('MS')
            
            for model_func in Modelos_Univariables:
                try:
                    
                    V = filtered_series[[Variable_objetivo]].rename(columns={Variable_objetivo: 'y'})
                    predictions, average_results, opt_parameters = model_func(V, TS, DF, SP, FP)
                            
                    # Añadir los parámetros óptimos al DataFrame de parámetros
                    new_opt_params_row = pd.DataFrame({'Cod_Item': [series['Cod_Item']],
                                                    'Canal': [series['Canal']],
                                                    'Region': [series['Region']],
                                                    'Modelo': [model_func.__name__],
                                                    'Opt_parameters': [opt_parameters]
                                                    })

                    opt_params_df = pd.concat([opt_params_df, new_opt_params_row], ignore_index=True)
                    
                    # Almacenar los resultados en el DataFrame
                    new_row = pd.DataFrame({
                        'Modelo': [model_func.__name__],
                        'Cod_Item': [series['Cod_Item']],
                        'Canal': [series['Canal']],
                        'Region': [series['Region']],
                        'Average MFA': [average_results['Average MFA']],
                        'Average BIAS': [average_results['Average BIAS']],
                        'Average MAE': [average_results['Average MAE']],
                        'Average RMSE': [average_results['Average RMSE']]
                        })
                    
                    results_df_all_series = pd.concat([results_df_all_series, new_row], ignore_index=True)

                    # Crear y almacenar resultados detallados
                    if predictions is not None:
                        # Aquí, se incluyen todos los valores de 'Unidades', tanto de entrenamiento como de testeo
                        detailed_df = pd.DataFrame({
                            'Fecha': filtered_series.index,
                            'Cod_Item': series['Cod_Item'],
                            'Canal': series['Canal'],
                            'Region': series['Region'],
                            'Modelo': [model_func.__name__] * len(filtered_series),
                            'V': filtered_series[Variable_objetivo],  # Todos los valores reales
                            'F': None  # Inicialmente, no hay valores pronosticados
                        })

                        # Rellenar los valores pronosticados en el período de testeo
                        detailed_df.loc[detailed_df.index[-len(predictions):], 'F'] = predictions.values
                        
                        detailed_results = pd.concat([detailed_results, detailed_df], ignore_index=True)
            
                except Exception as e:
                    print(f"Error al ejecutar el modelo {model_func.__name__} en la serie {series['Cod_Item']}, {series['Canal']}, {series['Region']}: {e}")
                    
        # Muestra los resultados
        consolidated_results_df_all_series = pd.concat([consolidated_results_df_all_series, results_df_all_series])
        consolidated_detailed_results = pd.concat([consolidated_detailed_results, detailed_results])

    return consolidated_results_df_all_series, consolidated_detailed_results, opt_params_df

# %%
#Select best Model

def select_best_model(Results_All_Models):
    # Inicializar DataFrame para almacenar los mejores modelos
    best_models = pd.DataFrame(columns=['Mejor_Modelo', 'Puntuacion_Mejor', 'Segundo_Mejor_Modelo', 'Puntuacion_Segundo', 'Cod_Item', 'Canal', 'Region'])

    # Agrupar por Cod_Item, Canal, Region
    grouped = Results_All_Models.groupby(['Cod_Item', 'Canal', 'Region'])

    for name, group in grouped:
        group = group.copy()

        # Asignar puntos por Average MFA
        group['MFA_Score'] = group['Average MFA'].rank(method='min', ascending=True)
        group['MFA_Points'] = group['MFA_Score'].apply(lambda x: 3 if x == 1 else (2 if x == 2 else 0))

        # Asignar puntos por Average BIAS
        group['BIAS_Score'] = group['Average BIAS'].abs().rank(method='min', ascending=True)
        group['BIAS_Points'] = group['BIAS_Score'].apply(lambda x: 3 if x == 1 else (2 if x == 2 else 0))

        # Asignar puntos por Average MAE
        group['MAE_Score'] = group['Average MAE'].rank(method='min', ascending=True)
        group['MAE_Points'] = group['MAE_Score'].apply(lambda x: 2 if x == 1 else (1 if x == 2 else 0))

        # Asignar puntos por Average RMSE
        group['RMSE_Score'] = group['Average RMSE'].rank(method='min', ascending=True)
        group['RMSE_Points'] = group['RMSE_Score'].apply(lambda x: 2 if x == 1 else (1 if x == 2 else 0))

        # Calcular puntuación total
        group['Total_Score'] = group[['MFA_Points', 'BIAS_Points', 'MAE_Points', 'RMSE_Points']].sum(axis=1)

        # Encontrar el mejor y segundo mejor modelo
        best_models_group = group.sort_values(by='Total_Score', ascending=False).head(2)
        best_model = best_models_group.iloc[0]
        second_best_model = best_models_group.iloc[1] if len(best_models_group) > 1 else best_model

        # Añadir al DataFrame de resultados
        new_row = pd.DataFrame({
            'Mejor_Modelo': [best_model['Modelo']],
            'Puntuacion_Mejor': [best_model['Total_Score']],
            'Average MFA Mejor': [best_model['Average MFA']],
            'Average BIAS Mejor': [best_model['Average BIAS']],
            'Average MAE Mejor': [best_model['Average MAE']],
            'Average RMSE Mejor': [best_model['Average RMSE']],
            'Segundo_Mejor_Modelo': [second_best_model['Modelo']],
            'Puntuacion_Segundo': [second_best_model['Total_Score']],
            'Average MFA Segundo': [second_best_model['Average MFA']],
            'Average BIAS Segundo': [second_best_model['Average BIAS']],
            'Average MAE Segundo': [second_best_model['Average MAE']],
            'Average RMSE Segundo': [second_best_model['Average RMSE']],
            'Cod_Item': [name[0]],
            'Canal': [name[1]],
            'Region': [name[2]]
        })
        best_models = pd.concat([best_models, new_row], ignore_index=True)

    return best_models

def Forecast (best_models_report, opt_params_df, partes, TS, DF, SP, FP):
    # %%
    # DataFrame para almacenar los resultados detallados de las predicciones futuras
    future_detailed_results = pd.DataFrame()

    # se define un diccionario donde se relacionan los nombres de los modelos con las funciones de los modelos.
    model_name_to_function = {
        'NAIVE' : NAIVE ,
        'Simple_Exp':Simple_Exp ,
        'Holt_Double' : Holt_Double,
        'HoltWinters' : HoltWinters,
        'ARIMA' : ARIMA,
        'SARIMA' : SARIMA,
        'AUTO_THETA' : AUTO_THETA,
        'AUTO_ETS' : AUTO_ETS,
        'AUTO_CES' : AUTO_CES}

    for num_parte, df_parte in partes.items():
        
        #if num_parte not in [4]: # Descomentar si estoy haciendo pruebas
            #continue

        # Restablece estos DataFrames para cada parte
        results_df_all_series_fcst = pd.DataFrame()
        detailed_results_fcst = pd.DataFrame()
        unique_series_per_df_parte = df_parte[['Cod_Item', 'Canal', 'Region']].drop_duplicates()
        unique_series_per_df_parte.reset_index(drop=True, inplace=True)

        print('Particion#: ',num_parte)

        for index, series in tqdm(unique_series_per_df_parte.iterrows(), desc=f'Procesando series en la parte {num_parte}', 
                                total=len(unique_series_per_df_parte)):
            
            print(f"Cod_Item: {series['Cod_Item']}, Canal: {series['Canal']}, Region: {series['Region']}")
            
            # Preparar los datos para cada combinación de Cod_Item, Canal y Region
            filtered_series = df_parte[(df_parte['Cod_Item'] == series['Cod_Item']) & 
                                                (df_parte['Canal'] == series['Canal']) & 
                                                (df_parte['Region'] == series['Region'])]
            
            filtered_series = filtered_series.copy()
            filtered_series['Fecha'] = pd.to_datetime(filtered_series['Fecha'])
            filtered_series = filtered_series.set_index('Fecha')
            filtered_series = filtered_series.groupby(filtered_series.index).sum()
            filtered_series = filtered_series.asfreq('MS')
            
            #Obtener el mejor y segundo mejor modelo para esta serie
            best_models = best_models_report[(best_models_report['Cod_Item'] == series['Cod_Item']) & 
                                            (best_models_report['Canal'] == series['Canal']) & 
                                            (best_models_report['Region'] == series['Region'])]

            # Lista de modelos a ejecutar (mejor y segundo mejor)
            models_to_run = [best_models.iloc[0]['Mejor_Modelo'], best_models.iloc[0]['Segundo_Mejor_Modelo']]
            
            for model_name in models_to_run:
                model_func = model_name_to_function.get(model_name)
                
                try:
                    print(f"Procesando modelo: {model_func.__name__}")
                    #Obtener parámetros óptimos para este modelo y serie
                    opt_params_row = opt_params_df[(opt_params_df['Cod_Item'] == series['Cod_Item']) & 
                                            (opt_params_df['Canal'] == series['Canal']) & 
                                            (opt_params_df['Region'] == series['Region']) & 
                                            (opt_params_df['Modelo'] == model_name)]
                    
                    opt_params = opt_params_row.iloc[0]['Opt_parameters'] if not opt_params_row.empty else None
                    
                    print('Parametros', opt_params)
                    
                    V = filtered_series[['Unidades']].rename(columns={'Unidades': 'y'})
                    predictions, _, _ = eval(model_name)(
                        V, TS, DF, SP, FP, opt_parameters=opt_params) if opt_params else eval(model_name)(
                        V, TS, DF, SP, FP)
                    
                    # Crear y almacenar resultados detallados
                    if predictions is not None:
                        # Aquí, se incluyen todos los valores de 'Unidades', tanto de entrenamiento como de testeo
                        
                        # Calcular la cantidad de periodos a predecir
                        num_periods_to_predict = min(TS, len(predictions))

                        # Generar fechas futuras en frecuencia mensual
                        future_dates = pd.date_range(start=filtered_series.index[-1] + pd.DateOffset(months=1),
                                                periods=num_periods_to_predict, freq=DF)
                        
                        # Convertir opt_params a una representación adecuada para incluir en el DataFrame
                        opt_params_str = str(opt_params) if opt_params else "Default"
                        
                        future_df = pd.DataFrame({
                        'Fecha': future_dates,
                        'Cod_Item': series['Cod_Item'],
                        'Canal': series['Canal'],
                        'Region': series['Region'],
                        'Modelo': model_name,
                        'F': predictions.values[:num_periods_to_predict],
                        'Opt_parameters': [opt_params_str] * len(future_dates)
                        })

                        # Concatenar con los resultados existentes
                        future_detailed_results = pd.concat([future_detailed_results, future_df], ignore_index=True)
                
                except Exception as e:
                    print(f"Error al ejecutar el modelo {model_func.__name__} en la serie {series['Cod_Item']}, {series['Canal']},{series['Region']}: {e}")
    
    return future_detailed_results




