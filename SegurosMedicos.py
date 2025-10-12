# ANÁLISIS DE FACTORES QUE AFECTAN COSTOS DE SEGUROS MÉDICOS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats # Importar stats de scipy para análisis estadístico avanzado
import warnings  # Importar warnings para suprimir advertencias no críticas
warnings.filterwarnings('ignore')  # Ignorar advertencias para mantener la salida limpia

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Imprimir título del proyecto
print(" ANÁLISIS DE SEGUROS MÉDICOS - FACTORES DE COSTO")
print("=" * 60)

# Cargar el dataset desde el archivo CSV
# pd.read_csv() convierte un archivo CSV en un DataFrame de pandas
df = pd.read_csv('insurance.csv')

# Confirmar que el dataset se cargó correctamente
print(" DATASET CARGADO EXITOSAMENTE")
# Mostrar dimensiones del dataset (filas, columnas)
print(f" Dimensiones: {df.shape[0]} filas, {df.shape[1]} columnas")
# Mostrar nombres de todas las columnas disponibles
print(f" Variables: {list(df.columns)}")


#%% ANÁLISIS EXPLORATORIO INICIAL

print("\n" + "="*50)
print("EJERCICIO 1: ANÁLISIS EXPLORATORIO INICIAL")
print("="*50)

# .info() muestra información general del DataFrame:
# - Tipos de datos de cada columna
# - Número de valores no nulos
# - Uso de memoria
print(" INFORMACIÓN DEL DATASET:")
print(df.info())

# .describe() genera estadísticas descriptivas para columnas numéricas:
# - count: número de valores no nulos
# - mean: promedio
# - std: desviación estándar
# - min, max: valores mínimo y máximo
# - percentiles 25%, 50%(mediana), 75%
print("\n ESTADÍSTICAS DESCRIPTIVAS:")
print(df.describe())

# .isnull() detecta valores faltantes, .sum() los cuenta por columna
print("\n VALORES FALTANTES:")
print(df.isnull().sum())

# .head(8) muestra las primeras 8 filas del DataFrame
print("\n MUESTRA DE DATOS:")
print(df.head(8))


#%% EJERCICIO 2: ANÁLISIS DE VARIABLES CATEGÓRICAS


print("\n" + "="*50)
print("EJERCICIO 2: ANÁLISIS DE VARIABLES CATEGÓRICAS")
print("="*50)

# .value_counts() cuenta la frecuencia de cada valor único en una columna
print(" DISTRIBUCIÓN POR GÉNERO:")
dist_sex = df['sex'].value_counts()
print(dist_sex)

print("\n DISTRIBUCIÓN POR REGIÓN:")
dist_region = df['region'].value_counts()
print(dist_region)

print("\n DISTRIBUCIÓN FUMADORES:")
dist_smoker = df['smoker'].value_counts()
print(dist_smoker)

# .sort_index() ordena los resultados por el índice (número de hijos)
print("\n DISTRIBUCIÓN POR NÚMERO DE HIJOS:")
dist_children = df['children'].value_counts().sort_index()
print(dist_children)

#%% ANÁLISIS DE COSTOS (CHARGES)

print("\n" + "="*50)
print("EJERCICIO 3: ANÁLISIS DE COSTOS MÉDICOS")
print("="*50)

# .describe() aplicado solo a la columna 'charges' para estadísticas de costos
costos_stats = df['charges'].describe()
print(" ESTADÍSTICAS DE COSTOS MÉDICOS:")
# :,.2f formatea números con separadores de miles y 2 decimales
print(f"• Promedio: ${costos_stats['mean']:,.2f}")
print(f"• Mediana: ${costos_stats['50%']:,.2f}")
print(f"• Mínimo: ${costos_stats['min']:,.2f}")
print(f"• Máximo: ${costos_stats['max']:,.2f}")
print(f"• Desviación estándar: ${costos_stats['std']:,.2f}")

# Cálculo de outliers usando el método del rango intercuartílico (IQR)
# Q1: percentil 25%, Q3: percentil 75%
Q1 = df['charges'].quantile(0.25)
Q3 = df['charges'].quantile(0.75)
# IQR = diferencia entre Q3 y Q1
IQR = Q3 - Q1
# Outliers superiores: valores mayores a Q3 + 1.5*IQR
outliers_superiores = df[df['charges'] > (Q3 + 1.5 * IQR)]
print(f"\n Pacientes con costos extremadamente altos: {len(outliers_superiores)}")

#%% ANÁLISIS DE FACTORES DE RIESGO

print("\n" + "="*50)
print("EJERCICIO 4: ANÁLISIS DE FACTORES DE RIESGO")
print("="*50)

# 1. Impacto de FUMAR en costos
print(" IMPACTO DE FUMAR EN COSTOS MÉDICOS:")

# groupby('smoker') agrupa los datos por valores únicos de 'smoker' (yes/no)
# .agg() aplica diferentes funciones de agregación a cada columna
analisis_fumadores = df.groupby('smoker').agg({
    'charges': ['mean', 'median', 'count'],  # Para charges: promedio, mediana y conteo
    'age': 'mean',      # Para age: solo promedio
    'bmi': 'mean'       # Para bmi: solo promedio
}).round(2)  # Redondea todos los resultados a 2 decimales

# Renombrar columnas para mayor claridad
analisis_fumadores.columns = ['costo_promedio', 'costo_mediano', 'total_personas', 'edad_promedio', 'bmi_promedio']
print(analisis_fumadores)

# Calcular diferencia entre costos de fumadores y no fumadores
# .loc[] accede a valores específicos por etiqueta en el DataFrame
costo_fumadores = analisis_fumadores.loc['yes', 'costo_promedio']
costo_no_fumadores = analisis_fumadores.loc['no', 'costo_promedio']
diferencia = costo_fumadores - costo_no_fumadores
print(f"\n Los fumadores pagan ${diferencia:,.2f} MÁS en promedio")

# 2. Análisis de BMI (Índice de Masa Corporal)
print("\n ANÁLISIS DE BMI (Índice de Masa Corporal):")

# Definir función para clasificar BMI en categorías
def clasificar_bmi(bmi):
    """
    Clasifica el BMI en categorías según estándares médicos
    """
    if bmi < 18.5:
        return 'Bajo peso'
    elif 18.5 <= bmi < 25:
        return 'Normal'
    elif 25 <= bmi < 30:
        return 'Sobrepeso'
    else:
        return 'Obeso'

# .apply() aplica la función clasificar_bmi a cada valor de la columna 'bmi'
df['bmi_category'] = df['bmi'].apply(clasificar_bmi)

# Análisis por categoría de BMI
analisis_bmi = df.groupby('bmi_category').agg({
    'charges': ['mean', 'median', 'count'],
    'age': 'mean',
    # lambda function: calcula el porcentaje de fumadores en cada categoría
    'smoker': lambda x: (x == 'yes').mean() * 100
}).round(2)

analisis_bmi.columns = ['costo_promedio', 'costo_mediano', 'total_personas', 'edad_promedio', 'porcentaje_fumadores']
print(analisis_bmi)

#%% ANÁLISIS DEMOGRÁFICO AVANZADO

print("\n" + "="*50)
print("EJERCICIO 5: ANÁLISIS DEMOGRÁFICO AVANZADO")
print("="*50)

# pd.cut() divide la columna 'age' en intervalos y asigna etiquetas
df['age_group'] = pd.cut(df['age'], 
                        bins=[18, 30, 40, 50, 65],  # Límites de los intervalos
                        labels=['18-29', '30-39', '40-49', '50-65'])  # Etiquetas

# Análisis por grupo de edad y género (agrupación por dos columnas)
analisis_edad_genero = df.groupby(['age_group', 'sex']).agg({
    'charges': ['mean', 'median'],
    'bmi': 'mean',
    'smoker': lambda x: (x == 'yes').mean() * 100,
    'children': 'mean'
}).round(2)

# Renombrar las columnas multi-nivel resultantes
analisis_edad_genero.columns = ['costo_promedio', 'costo_mediano', 'bmi_promedio', 'porcentaje_fumadores', 'hijos_promedio']
print(" ANÁLISIS POR EDAD Y GÉNERO:")
print(analisis_edad_genero)

# Análisis por región
print("\n ANÁLISIS POR REGIÓN:")
analisis_region = df.groupby('region').agg({
    'charges': ['mean', 'median'],
    'bmi': 'mean',
    'smoker': lambda x: (x == 'yes').mean() * 100,
    'age': 'mean'
}).round(2)

analisis_region.columns = ['costo_promedio', 'costo_mediano', 'bmi_promedio', 'porcentaje_fumadores', 'edad_promedio']
print(analisis_region)


#%% EJERCICIO 6: VISUALIZACIONES PROFESIONALES

print("\n" + "="*50)
print("EJERCICIO 6: CREANDO VISUALIZACIONES PROFESIONALES")
print("="*50)

# Crear figura con subplots: 2 filas, 3 columnas, tamaño 20x12 pulgadas
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
# Añadir título general a toda la figura
fig.suptitle(' ANÁLISIS COMPLETO DE SEGUROS MÉDICOS', fontsize=16, fontweight='bold')

# --- GRÁFICA 1: Distribución de costos médicos (Histograma) ---
# axes[0,0] accede al subplot en fila 0, columna 0
axes[0, 0].hist(df['charges'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Distribución de Costos Médicos')
axes[0, 0].set_xlabel('Cargos ($)')
axes[0, 0].set_ylabel('Frecuencia')
axes[0, 0].grid(True, alpha=0.3)  # Añadir grid semitransparente

# --- GRÁFICA 2: Costos: Fumadores vs No Fumadores (Boxplot) ---
# sns.boxplot de seaborn para boxplot más atractivo
sns.boxplot(data=df, x='smoker', y='charges', ax=axes[0, 1], palette=['lightgreen', 'salmon'])
axes[0, 1].set_title('Costos Médicos: Fumadores vs No Fumadores')
axes[0, 1].set_xlabel('¿Fumador?')
axes[0, 1].set_ylabel('Cargos ($)')

# --- GRÁFICA 3: Costos por categoría de BMI (Boxplot ordenado) ---
bmi_orden = ['Bajo peso', 'Normal', 'Sobrepeso', 'Obeso']  # Orden personalizado
sns.boxplot(data=df, x='bmi_category', y='charges', order=bmi_orden, ax=axes[0, 2], palette='viridis')
axes[0, 2].set_title('Costos por Categoría de BMI')
axes[0, 2].set_xlabel('Categoría de BMI')
axes[0, 2].set_ylabel('Cargos ($)')
axes[0, 2].tick_params(axis='x', rotation=45)  # Rotar etiquetas del eje X

# --- GRÁFICA 4: Costos vs Edad (Scatter plot con colores) ---
# Crear diccionario de colores para fumadores
colors = {'yes': 'red', 'no': 'blue'}
# Scatter plot donde el color depende de si es fumador o no
axes[1, 0].scatter(df['age'], df['charges'], c=df['smoker'].map(colors), alpha=0.6)
axes[1, 0].set_title('Costos vs Edad (Rojo=Fumadores, Azul=No fumadores)')
axes[1, 0].set_xlabel('Edad')
axes[1, 0].set_ylabel('Cargos ($)')
# Crear leyenda personalizada
axes[1, 0].legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Fumador'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='No fumador')
])

# --- GRÁFICA 5: Distribución por región (Gráfico de pastel) ---
region_counts = df['region'].value_counts()
axes[1, 1].pie(region_counts.values, labels=region_counts.index, autopct='%1.1f%%', startangle=90)
axes[1, 1].set_title('Distribución de Pacientes por Región')

# --- GRÁFICA 6: Costos por número de hijos (Boxplot) ---
sns.boxplot(data=df, x='children', y='charges', ax=axes[1, 2], palette='coolwarm')
axes[1, 2].set_title('Costos por Número de Hijos')
axes[1, 2].set_xlabel('Número de Hijos')
axes[1, 2].set_ylabel('Cargos ($)')

# Ajustar el espaciado entre subplots
plt.tight_layout()
# Mostrar todas las gráficas
plt.show()

#%% ANÁLISIS ESTADÍSTICO

print("\n" + "="*50)
print("EJERCICIO 7: ANÁLISIS ESTADÍSTICO")
print("="*50)

# 1. Correlación entre variables numéricas
print(" MATRIZ DE CORRELACIÓN:")
# .corr() calcula coeficientes de correlación de Pearson entre columnas numéricas
correlation_matrix = df[['age', 'bmi', 'children', 'charges']].corr()
print(correlation_matrix.round(3))  # Redondear a 3 decimales

# 2. Test t para comparar medias entre dos grupos independientes
# Filtrar datos: costos de fumadores
fumadores = df[df['smoker'] == 'yes']['charges']
# Filtrar datos: costos de no fumadores
no_fumadores = df[df['smoker'] == 'no']['charges']

# stats.ttest_ind: test t para muestras independientes
# equal_var=False asume varianzas diferentes (test de Welch)
t_stat, p_value = stats.ttest_ind(fumadores, no_fumadores, equal_var=False)

print(f"\n TEST ESTADÍSTICO - FUMADORES VS NO FUMADORES:")
print(f"   t-statistic: {t_stat:.4f}")  # Estadístico t
print(f"   p-value: {p_value:.4f}")     # Valor p
# Interpretación: p-value < 0.05 indica diferencia estadísticamente significativa
print(f"   Conclusión: {'Diferencia SIGNIFICATIVA' if p_value < 0.05 else 'No hay diferencia significativa'}")
print(f"   Los fumadores pagan {fumadores.mean()/no_fumadores.mean():.1f}x más")

# 3. Análisis de varianza (ANOVA) para comparar múltiples grupos
# Crear lista de grupos: costos para cada región
grupos_region = [df[df['region'] == region]['charges'] for region in df['region'].unique()]
# stats.f_oneway: ANOVA de una vía
f_stat, p_value_anova = stats.f_oneway(*grupos_region)

print(f"\n ANOVA - DIFERENCIAS POR REGIÓN:")
print(f"   F-statistic: {f_stat:.4f}")      # Estadístico F
print(f"   p-value: {p_value_anova:.4f}")   # Valor p
print(f"   Conclusión: {'Hay diferencias SIGNIFICATIVAS entre regiones' if p_value_anova < 0.05 else 'No hay diferencias significativas entre regiones'}")

#%% CONCLUSIONES Y REPORTE EJECUTIVO

print("\n" + "="*50)
print("EJERCICIO 8: REPORTE EJECUTIVO Y CONCLUSIONES")
print("="*50)

# Calcular métricas clave para el reporte ejecutivo
costo_promedio = df['charges'].mean()
porcentaje_fumadores = (df['smoker'] == 'yes').mean() * 100
costo_fumadores = df[df['smoker'] == 'yes']['charges'].mean()
costo_no_fumadores = df[df['smoker'] == 'no']['charges'].mean()
# .idxmax() encuentra la región con el costo promedio más alto
region_mas_cara = df.groupby('region')['charges'].mean().idxmax()

print(" REPORTE EJECUTIVO - HALLAZGOS PRINCIPALES")
print("=" * 40)
print(f" COSTO PROMEDIO: ${costo_promedio:,.2f}")
print(f" PORCENTAJE DE FUMADORES: {porcentaje_fumadores:.1f}%")
print(f" FUMADORES PAGAN: ${costo_fumadores - costo_no_fumadores:,.2f} MÁS")
print(f" REGIÓN MÁS COSTOSA: {region_mas_cara}")
print(f" BMI PROMEDIO: {df['bmi'].mean():.1f}")
print(f" EDAD PROMEDIO: {df['age'].mean():.1f} años")

print("\n INSIGHTS CLAVE:")
print("1.  Fumar es el factor MÁS influyente en costos médicos")
print("2.  La edad muestra fuerte correlación positiva con costos")
print("3.  El BMI afecta costos, especialmente en categoría 'Obeso'")
print("4.  Diferencias regionales existen pero son menores")
print("5.  Número de hijos tiene impacto moderado en costos")

print("\n RECOMENDACIONES:")
print("•  Enfocar programas anti-tabaco para reducir costos")
print("•  Monitorear BMI de clientes para prevención")
print("•  Desarrollar programas wellness por región")
print("•  Considerar edad como factor clave en primas")

# Exportar el dataset analizado (con nuevas columnas) a CSV
df.to_csv('dataset_analizado.csv', index=False)
print("\n Análisis exportado a 'dataset_analizado.csv'")

print("\n ¡ANÁLISIS COMPLETADO EXITOSAMENTE!")