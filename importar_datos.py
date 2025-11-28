import pandas as pd
import json
import os
import csv
from datetime import datetime

# --- 1. CONFIGURACIÓN (¡IMPORTANTE!) ---

# Define la fecha desde la que quieres empezar a importar.
# Formato: "AÑO-MES-DÍA"
FECHA_INICIO_IMPORTAR = "2025-09-01"  # <-- CAMBIA ESTA FECHA

# El nombre de tu archivo de exportación de Hevy
HEVY_EXPORT_CSV = "workouts.csv" # <-- CAMBIA ESTE NOMBRE

# El nombre del CSV que usa tu bot (donde se guardarán los datos)
BOT_CSV = "entrenos.csv"

# La base de datos de músculos que creamos
DB_JSON_FILE = "ejercicios_db.json"

# --- 2. Cargar Base de Datos de Músculos ---
print(f"Cargando base de datos de músculos '{DB_JSON_FILE}'...")
try:
    with open(DB_JSON_FILE, 'r', encoding='utf-8') as f:
        ejercicios_db = json.load(f)
    print(f"Cargada con {len(ejercicios_db)} ejercicios.")
except FileNotFoundError:
    print(f"Error: No se encontró '{DB_JSON_FILE}'.")
    print("Asegúrate de que esté en la misma carpeta.")
    exit()

# --- 3. Cargar y Filtrar Historial de Hevy ---
print(f"Cargando historial de Hevy desde '{HEVY_EXPORT_CSV}'...")
try:
    df_historial = pd.read_csv(HEVY_EXPORT_CSV)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo '{HEVY_EXPORT_CSV}'.")
    exit()

print(f"Se encontraron {len(df_historial)} registros en total.")

# Convertir la columna 'start_time' de Hevy a un objeto de fecha
# Formato de Hevy: "13 Nov 2025, 14:12"
df_historial['fecha_dt'] = pd.to_datetime(df_historial['start_time'], format='%d %b %Y, %H:%M')

# Filtrar por la fecha de inicio
fecha_inicio = pd.to_datetime(FECHA_INICIO_IMPORTAR)
df_filtrado = df_historial[df_historial['fecha_dt'] >= fecha_inicio].copy()

print(f"Se procesarán {len(df_filtrado)} registros a partir del {FECHA_INICIO_IMPORTAR}.")

# --- 4. Preparar y Guardar los Datos ---

# Estas son las columnas EXACTAS que usa tu bot
columnas_bot_csv = [
    'fecha', 'titulo_rutina', 'nombre_ejercicio', 
    'grupo_principal', 'grupos_secundarios', 'cabezas_regiones',
    'set_index', 'peso_kg', 'reps', 'tipo_set'
]

# Revisar si el archivo del bot ya existe para no duplicar encabezados
archivo_existe = os.path.exists(BOT_CSV)

# Usamos 'a+' (append) para AÑADIR al final del archivo
with open(BOT_CSV, 'a+', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    
    # Si el archivo es nuevo o está vacío, escribe los encabezados
    if not archivo_existe or f.tell() == 0:
        writer.writerow(columnas_bot_csv)
    
    print(f"Añadiendo {len(df_filtrado)} registros a '{BOT_CSV}'...")
    
    # Iteramos sobre cada fila del historial filtrado
    for _, row in df_filtrado.iterrows():
        
        # 1. Buscar info muscular en nuestra BD
        nombre_ejercicio = row['exercise_title']
        info_muscular = ejercicios_db.get(nombre_ejercicio, {})
        
        grupo_principal = info_muscular.get('grupo_principal', 'N/A')
        grupos_secundarios = ", ".join(info_muscular.get('grupos_secundarios', []))
        cabezas_regiones = ", ".join(info_muscular.get('cabezas_o_regiones', []))

        # 2. ¡CRÍTICO! Reformatear la fecha
        # 'analisis.py' espera el formato: "Thursday, Nov 13, 2025 at 2:12pm"
        # Así que convertimos el 'fecha_dt' de pandas a ese string
        fecha_formateada = row['fecha_dt'].strftime('%A, %b %d, %Y at %I:%M%p')
        
        # 3. Rellenar 'NaN' (vacíos) con 0
        peso_kg = row.get('weight_kg', 0)
        reps = row.get('reps', 0)
        
        # 4. Crear la fila con el orden correcto
        fila_para_guardar = [
            fecha_formateada,
            row['title'],
            nombre_ejercicio,
            grupo_principal,
            grupos_secundarios,
            cabezas_regiones,
            row['set_index'],
            peso_kg if pd.notna(peso_kg) else 0, # Asegurar que no sea NaN
            reps if pd.notna(reps) else 0,       # Asegurar que no sea NaN
            row['set_type']
        ]
        
        # 5. Escribir la fila en el CSV
        writer.writerow(fila_para_guardar)

print("\n-------------------------")
print("¡Importación completada!")
print(f"Se añadieron {len(df_filtrado)} nuevos registros a '{BOT_CSV}'.")