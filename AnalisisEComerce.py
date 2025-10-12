# Objetivos:
#Análisis de comportamiento de clientes
#Segmentación de productos
#Análisis temporal de ventas
#Métricas de negocio

import pandas as pd  #Para manipulación de datos
import numpy as np   #Para operaciones matemáticas y números aleatorios
import matplotlib.pyplot as plt   #Para graficas y visualizaciones
import seaborn as sns    #Para graficas más atractivas
from datetime import datetime, timedelta   #Para trabajar con fechas

# Configurar estilo de las gráficas
plt.style.use('seaborn-v0_8') 
sns.set_palette("husl")   #Paleta de colores

#%% CREAR DATASET DE E-COMMERCE

np.random.seed(42) #Establece una semilla para números aleatorios
n_registros = 1000 #Definir el numero de registros

# Crear un diccionario con los datos simulados de e-commerce
data = {
        #Crear Ids de pedido únicos del 1001 al 2000
        'order_id' : range(1001 , 1001 + n_registros),
        #Generar IDs de clientes aleatorios entre 100 y 499
        'customer_id' : np.random.randint(100, 500, n_registros),
        # Asignar categorías de productos aleatoriamente
        'product_category' : np.random.choice(['Electrónicos', 'Hogar', 'Ropa', 'Deportes', 'Libros'], n_registros),
        # Generar precios aleatorios entre $10 y $500 con dos decimales
        'product_price' : np.random.uniform(10, 500, n_registros).round(2) ,
        # Generar cantidades compradas entre 1 y 4 unidades
        'quantity' : np.random.randint(1 , 5 , n_registros)  ,
        # Crear fechas de pedido distribuidas entre enero y marzo de 2024
        'order_date' : pd.date_range('2024-01-01', '2024-03-31', periods= n_registros) ,
        # Generar edades de clientes entre 18 y 69 años
        'customer_age' : np.random.randint(18, 70 , n_registros),
        # Asignar regiones geográficas aleatoriamente 
        'customer_region' : np.random.choice(['Norte', 'Sur', 'Este', 'Oeste'], n_registros),
        #Generar ratings de satisfacción entre 1 y 5 estrellas
        'rating' : np.random.randint(1, 5 , n_registros),
        }

# Convertir el diccionario en un DataFrame de pandas
df = pd.DataFrame(data)

#%% Calcular el monto toal de cada pedido
df['total_amount'] = df['product_price'] * df['quantity']

print("===DATASET E-COMERCE CREADO ===")

# Mostrar las dimensiones del DataFrame (filas, columnas)
print(f"Dimensiones: {df.shape}" ),
print("\nPrimeras 5 filas:")
print(df.head(5))

#%% ANÁLISIS DE VENTAS POR CATEGORÍA

print("\n" + "="*45)

# Agrupar los datos por categoría de producto y calcular múltiples métricas
ventas_por_categoria = df.groupby('product_category').agg({
    # Sumar todos los montos totales por categoría
    'total_amount' : 'sum',
    # Sumar todas las cantidades vendidas por categoría
    'quantity' : 'sum',
    # Contar el número de pedidos únicos por categoría
    'order_id' : 'count',
    }).round(2)   

# Renombrar las columnas
ventas_por_categoria.columns = ['ingresos_totales', 'unidades_vendidas', 'pedidos_totales']

# Calcular el ticket promedio (ingresos totales / numero de pedidos)
ventas_por_categoria['ticket_promedio'] = (ventas_por_categoria['ingresos_totales'] / ventas_por_categoria['pedidos_totales']).round(2)

print("📊 Resumen por categoría:")
print(ventas_por_categoria)

# Encontrar la categoría con mayores ingresos usando idxmax()
categoria_top_ingresos = ventas_por_categoria['ingresos_totales'].idxmax()
print(f"\n Categoría más rentable :{categoria_top_ingresos} ")



#%%  ANÁLISIS TEMPORAL DE VENTAS
print("\n" + "=" *45)

# Extraer el mes de la fecha para análisis mensual
df['mes'] = df['order_date'].dt.month
# Extraer el nombre del día de la semana 
df['dia_semana'] = df['order_date'].dt.day_name()
# Extraer el número de la semana del año
df['semana'] = df['order_date'].dt.isocalendar().week

# Agrupar por mes y calcular métricas de ventas
ventas_mensuales = df.groupby('mes').agg({
    # Suma total de ventas y promedio por mes
    'total_amount' : ['sum' , 'mean'],
    # Contar número total de pedidos
    'order_id': 'count'
    
    }).round(2)

# Renombrar las columnas para mayor claridad
ventas_mensuales.columns = ['ventas_totales', 'ticket_promedio', 'total_pedidos']
print("📅 Ventas mensuales:")
print(ventas_mensuales)

# Calcular ventas totales por día de la semana
ventas_diarias = df.groupby('dia_semana')['total_amount'].sum()
print(f"\n📊 Ventas por día de la semana:\n{ventas_diarias}")

#%% ANÁLISIS DE CLIENTES
print("\n" + "="*45)


# Crear rangos de edad para segmentar a los clientes
df['rango_edad'] = pd.cut(df['customer_age'], 
                         # Definir los límites de los rangos de edad
                         bins=[18, 25, 35, 50, 70], 
                         # Etiquetas para cada rango
                         labels=['18-25', '26-35', '36-50', '51-70'])

# Analizar el comportamiento de compra por rango de edad
comportamiento_edad = df.groupby('rango_edad').agg({
    # Gasto total, promedio y número de compras
    'total_amount': ['sum', 'mean', 'count'],
    # Rating promedio de satisfacción
    'rating': 'mean'
}).round(2)

# Renombrar las columnas
comportamiento_edad.columns = ['gasto_total', 'gasto_promedio', 'compras_totales', 'rating_promedio']
print("👥 Comportamiento por edad:")
print(comportamiento_edad)

# Analizar clientes individualmente agrupando por customer_id
clientes_top = df.groupby('customer_id').agg({
    # Gasto total de cada cliente
    'total_amount': 'sum',
    # Número de compras de cada cliente
    'order_id': 'count',
    # Satisfacción promedio del cliente
    'rating': 'mean'
}).round(2)

# Renombrar columnas
clientes_top.columns = ['gasto_total', 'compras_totales', 'satisfaccion_promedio']
# Ordenar clientes por gasto total (de mayor a menor)
clientes_top = clientes_top.sort_values('gasto_total', ascending=False)

print(f"\n💰 Top 5 clientes más valiosos:")
# Mostrar solo los 5 clientes que más han gastado
print(clientes_top.head())

#%% EJERCICIO 4: VISUALIZACIONES AVANZADAS
print("\n" + "="*45)


# Crear una figura con 2 filas y 2 columnas de subgráficas
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# --- GRÁFICA 1: Ventas por categoría (Barras verticales) ---
# Ordenar categorías por ingresos (de mayor a menor)
categorias_orden = ventas_por_categoria.sort_values('ingresos_totales', ascending=False)
# Crear gráfica de barras en la posición [0,0]
axes[0, 0].bar(categorias_orden.index, categorias_orden['ingresos_totales'])
axes[0, 0].set_title('Ingresos por Categoría de Producto')
axes[0, 0].set_ylabel('Ingresos ($)')
# Rotar etiquetas del eje X para mejor legibilidad
axes[0, 0].tick_params(axis='x', rotation=45)

# --- GRÁFICA 2: Evolución mensual (Línea) ---
# Crear gráfica de línea con puntos marcados
axes[0, 1].plot(ventas_mensuales.index, ventas_mensuales['ventas_totales'], marker='o', linewidth=2)
axes[0, 1].set_title('Evolución de Ventas Mensuales')
axes[0, 1].set_xlabel('Mes')
axes[0, 1].set_ylabel('Ventas Totales ($)')
# Agregar grid para mejor lectura
axes[0, 1].grid(True, alpha=0.3)

# --- GRÁFICA 3: Gasto por edad (Barras horizontales) ---
# Gráfica de barras horizontales
axes[1, 0].barh(comportamiento_edad.index, comportamiento_edad['gasto_promedio'])
axes[1, 0].set_title('Gasto Promedio por Grupo de Edad')
axes[1, 0].set_xlabel('Gasto Promedio ($)')

# --- GRÁFICA 4: Distribución de ratings (Pastel) ---
# Contar cuántos ratings hay de cada tipo (1-5 estrellas)
distribucion_ratings = df['rating'].value_counts().sort_index()
# Crear gráfica de pastel
axes[1, 1].pie(distribucion_ratings.values, 
               labels=distribucion_ratings.index, 
               autopct='%1.1f%%')  # Mostrar porcentajes
axes[1, 1].set_title('Distribución de Ratings de Clientes')

# Ajustar el espaciado entre subgráficas
plt.tight_layout()
# Mostrar todas las gráficas
plt.show()


#%% EJERCICIO 5: SEGMENTACIÓN DE CLIENTES


print("\n" + "="*50)
print("EJERCICIO 5: SEGMENTACIÓN DE CLIENTES")
print("="*50)

# Analizar cada cliente individualmente
segmentos_clientes = df.groupby('customer_id').agg({
    # Gasto total del cliente
    'total_amount': 'sum',
    # Número de compras del cliente
    'order_id': 'count',
    # Días entre primera y última compra (antigüedad)
    'order_date': lambda x: (x.max() - x.min()).days
}).round(2)

# Renombrar columnas
segmentos_clientes.columns = ['gasto_total', 'frecuencia_compras', 'dias_activo']

# Definir función para segmentar clientes basada en sus comportamientos
def segmentar_cliente(fila):
    """
    Clasifica a los clientes en segmentos según su gasto y frecuencia de compra
    """
    if fila['gasto_total'] > 1000 and fila['frecuencia_compras'] > 3:
        return 'VIP'           # Clientes de muy alto valor
    elif fila['gasto_total'] > 500:
        return 'Leal'          # Clientes que gastan bastante
    elif fila['frecuencia_compras'] > 2:
        return 'Frecuente'     # Clientes que compran seguido
    else:
        return 'Ocasional'     # Clientes esporádicos

# Aplicar la función de segmentación a cada cliente
segmentos_clientes['segmento'] = segmentos_clientes.apply(segmentar_cliente, axis=1)

print("📈 Distribución de segmentos de clientes:")
# Contar cuántos clientes hay en cada segmento
print(segmentos_clientes['segmento'].value_counts())

# Analizar métricas por segmento
analisis_segmentos = segmentos_clientes.groupby('segmento').agg({
    'gasto_total': ['mean', 'sum'],           # Gasto promedio y total
    'frecuencia_compras': 'mean',             # Frecuencia promedio de compra
    'dias_activo': 'mean'                     # Antigüedad promedio
}).round(2)

print(f"\n📊 Métricas por segmento:")
print(analisis_segmentos)


#%% EJERCICIO 6: EXPORTAR REPORTES


print("\n" + "="*50)
print("EJERCICIO 6: EXPORTAR REPORTES")
print("="*50)

# Crear un archivo Excel con múltiples hojas
with pd.ExcelWriter('reporte_ecommerce_completo.xlsx') as writer:
    # Hoja 1: Datos completos
    df.to_excel(writer, sheet_name='Datos_Completos', index=False)
    # Hoja 2: Ventas por categoría
    ventas_por_categoria.to_excel(writer, sheet_name='Ventas_Categoria')
    # Hoja 3: Ventas mensuales
    ventas_mensuales.to_excel(writer, sheet_name='Ventas_Mensuales')
    # Hoja 4: Análisis por edad
    comportamiento_edad.to_excel(writer, sheet_name='Analisis_Edad')
    # Hoja 5: Segmentos de clientes
    segmentos_clientes.to_excel(writer, sheet_name='Segmentos_Clientes')

# Crear un resumen ejecutivo con las métricas más importantes
resumen_ejecutivo = pd.DataFrame({
    'Metrica': [
        'Total Ingresos',
        'Total Pedidos', 
        'Ticket Promedio',
        'Categoría Top',
        'Cliente Más Valioso',
        'Rating Promedio',
        'Total Clientes Únicos'
    ],
    'Valor': [
        # Formatear números con separadores de miles
        f"${df['total_amount'].sum():,.2f}",
        # Contar pedidos únicos
        df['order_id'].nunique(),
        # Calcular ticket promedio
        f"${df['total_amount'].mean():.2f}",
        # Categoría más rentable
        categoria_top_ingresos,
        # ID del cliente que más gastó
        f"Cliente #{clientes_top.index[0]}",
        # Rating promedio formateado
        f"{df['rating'].mean():.2f} estrellas",
        # Contar clientes únicos
        df['customer_id'].nunique()
    ]
})

# Exportar el resumen ejecutivo a CSV
resumen_ejecutivo.to_csv('resumen_ejecutivo_ecommerce.csv', index=False)
print("✅ Reportes exportados:")
print("   - reporte_ecommerce_completo.xlsx")
print("   - resumen_ejecutivo_ecommerce.csv")

print("\n📋 RESUMEN EJECUTIVO:")
print(resumen_ejecutivo)

print("\n🎉 ¡ANÁLISIS COMPLETADO EXITOSAMENTE!")
