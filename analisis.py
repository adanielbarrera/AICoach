import pandas as pd
import matplotlib
matplotlib.use('Agg')  # <-- ESTA ES LA LÍNEA MÁGICA
import matplotlib.pyplot as plt
import numpy as np
import os
from dotenv import load_dotenv
import json
import google.generativeai as genai
from datetime import datetime

# --- 1. CONFIGURACIÓN ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    print("Error: No se encontró la GOOGLE_API_KEY en el archivo .env")
    exit()

genai.configure(api_key=GEMINI_API_KEY)

# Archivos y Benchmarks
CSV_FILE = "entrenos.csv"
BENCHMARK_SERIES_SEMANALES = 20.0
MEDIDAS_CSV_FILE = "medidas.csv"
ALTURA_CM = 189.0

# Gráficas de salida
GRAFICA_RADAR_FILE = "static/balance_radar.png"
GRAFICA_FUERZA_FILE = "static/progreso_fuerza.png"
GRAFICA_MEDIDAS_FILE = "static/progreso_corporal.png"
GRAFICA_COMPOSICION_FILE = "static/progreso_composicion.png"

HISTORIAL_JSON_FILE = "historial_reportes.json"

# (¡EDITA ESTA LISTA!)
EJERCICIOS_CLAVE = [
    'Bench Press (Barbell)',
    'Incline Bench Press (Barbell)',
    'Seated Overhead Press (Barbell)',
    'Squat (Smith Machine)',
    'Deadlift (Barbell)'
    ''
]
os.makedirs('static', exist_ok=True)
# --- 2. FUNCIONES DE CÁLCULO (Ayudantes) ---

def calcular_e1rm(peso, reps):
    """
    Calcula el 1RM Estimado usando la fórmula de Epley.
    """
    if reps == 1:
        return peso
    if reps > 1 and reps < 15:
        # Fórmula de Epley
        return peso * (1 + (reps / 30))
    return None # Retorna None si son muchas reps (ej. 25) o 0

# --- (NUEVA) FUNCIÓN DE INTERPRETACIÓN DE IA ---

def generar_interpretacion_llm(reporte_actual_str, historial_str):
    """
    Toma el JSON con TODO el resumen y pide a Gemini un análisis holístico.
    """
    print("\nGenerando interpretación holística con IA...")
    try:
        model = genai.GenerativeModel('models/gemini-flash-latest')
        
        prompt = f"""
            Eres "Hevy Coach", un entrenador personal especializado en análisis de datos,
            tendencias de rendimiento y recomposición corporal.

            Tu objetivo: evaluar el progreso de ESTA SEMANA en contexto con las semanas anteriores,
            identificando patrones reales, no solo números aislados.

        --- DATOS ACTUALES (ESTA SEMANA) ---
        {reporte_actual_str}

        --- HISTORIAL RECIENTE (ÚLTIMAS SEMANAS) ---
        {historial_str}

        Por favor, genera un REPORTE EJECUTIVO con la siguiente estructura, usando exclusivamente
    FORMATO COMPATIBLE CON WHATSAPP:
    - Solo texto plano.
    - Nada de Markdown.
    - No uses asteriscos, guiones de markdown, ni símbolos que se interpreten como formato.
    - Usa numeración simple (1., 2., 3.) y saltos de línea claros.

    Estructura:

    1. Estado actual vs tendencia:
    Evalúa el e1RM comparándolo con las semanas del historial e indica si cada lift va hacia arriba, abajo o estancado (usa emojis de tendencia como 📈, 📉, ➖).
    Analiza si el puntaje RPG mejoró, se mantuvo o retrocedió respecto a la semana pasada.

    2. Análisis de composición corporal:
    Explica cómo ha cambiado mi porcentaje de grasa y masa magra en el tiempo.
    Indica si hay una tendencia positiva de recomposición o no.

    3. Veredicto y ajustes:
    Da recomendaciones accionables basadas en patrones de fuerza, volumen, estancamiento, recuperación y consistencia.
    Si el volumen ha bajado varias semanas, repórtalo.
    Si la fuerza lleva estancada tres semanas o más, sugiere descarga o cambio de rango de repeticiones.
    Si las tendencias son buenas, señala qué debería mantener.

    Estilo:
    - Sé directo y técnico.
    - Usa emojis solo para marcar tendencias.
    - Responde únicamente en español.
    - Máximo 4096 caracteres.

    Entrega tu análisis:
"""

        
        response = model.generate_content(prompt)
        return response.text
    
    except Exception as e:
        print(f"Error al generar interpretación con Gemini: {e}")
        return "No se pudo generar la interpretación de la IA."

# --- 3. FUNCIONES DE PROCESAMIENTO (Las nuevas funciones) ---

def cargar_y_limpiar_datos(csv_file):
    """
    Carga el CSV, lo limpia, maneja errores y filtra series efectivas.
    Devuelve un DataFrame limpio y el total de semanas únicas.
    """
    print(f"Cargando y limpiando datos de '{csv_file}'...")
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{csv_file}'.")
        return None, None
    
    if df.empty:
        print("El archivo CSV está vacío. No hay nada que analizar.")
        return None, None

    # Limpieza de Fecha (Crítico)
    df['fecha_dt'] = pd.to_datetime(df['fecha'], format='%A, %b %d, %Y at %I:%M%p', errors='coerce')
    df['fecha_dia'] = df['fecha_dt'].dt.date
    
    # Filtrar series efectivas (¡IMPORTANTE!)
    df_efectivo = df[df['tipo_set'] != 'warmup'].copy()
    
    if df_efectivo.empty:
        print("No se encontraron series efectivas (no-warmup).")
        return None, None

    # Calcular semanas únicas de entrenamiento
    df_efectivo['semana_del_anio'] = df_efectivo['fecha_dt'].dt.isocalendar().week
    df_efectivo['anio'] = df_efectivo['fecha_dt'].dt.isocalendar().year
    df_efectivo['semana_id_unica'] = df_efectivo['anio'].astype(str) + '-' + df_efectivo['semana_del_anio'].astype(str)
    
    total_semanas_unicas = df_efectivo['semana_id_unica'].nunique()
    if total_semanas_unicas == 0:
        total_semanas_unicas = 1 # Evitar división por cero

    print(f"Análisis basado en {total_semanas_unicas} semanas únicas de entrenamiento.")
    return df_efectivo, total_semanas_unicas

def analizar_balance(df_efectivo, total_semanas_unicas):
    """
    Calcula el puntaje RPG de balance muscular.
    Devuelve un DataFrame con las estadísticas.
    """
    print("\nCalculando estadísticas de balance (Puntaje RPG)...")
    
    series_totales_por_grupo = df_efectivo.groupby('grupo_principal').size()
    avg_series_semanal = (series_totales_por_grupo / total_semanas_unicas)
    puntaje_rpg = (avg_series_semanal / BENCHMARK_SERIES_SEMANALES) * 100

    stats_df = pd.DataFrame({
        'series_totales': series_totales_por_grupo,
        'avg_series_semanal': avg_series_semanal,
        'puntaje_rpg': puntaje_rpg
    }).sort_values(by='puntaje_rpg', ascending=False)
    
    print("\n--- Estadísticas de Balance (vs Benchmark de 15 series/sem) ---")
    print(stats_df)
    return stats_df

def generar_grafica_radar(stats_df, output_file):
    """
    Genera la gráfica de radar con estética BLUEPRINT.
    """
    print(f"\nGenerando gráfica de radar (Blueprint) en '{output_file}'...")

    labels = stats_df.index.values
    scores = stats_df['puntaje_rpg'].values
    N = len(labels)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    scores = np.concatenate((scores, [scores[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # --- INICIO DE ESTILO BLUEPRINT ---
    BLUEPRINT_BG = '#0A417A'  # Un azul oscuro "plano"
    LINE_COLOR = '#FFFFFF'    # Blanco brillante
    GRID_COLOR = '#AAAAAA'    # Gris claro
    TEXT_COLOR = '#EFEFEF'    # Blanco suave

    # Aplicar color de fondo
    fig.patch.set_facecolor(BLUEPRINT_BG)
    ax.set_facecolor(BLUEPRINT_BG)
    
    # Ejes y etiquetas (los nombres de los músculos)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color=TEXT_COLOR, size=10)
    
    # Rejilla de fondo (la telaraña)
    ax.set_rlabel_position(0)
    ax.tick_params(colors=GRID_COLOR) # Números (0, 25, 50...)
    ax.grid(color=GRID_COLOR, alpha=0.3)

    # Línea del Benchmark (100)
    ax.plot(np.linspace(0, 2*np.pi, 100), [100] * 100, color=GRID_COLOR, linestyle='--', linewidth=1, label='Benchmark (100)')
    ax.set_ylim(0)

    # Línea de Datos
    ax.plot(angles, scores, color=LINE_COLOR, linewidth=2, linestyle='solid')
    ax.fill(angles, scores, color=LINE_COLOR, alpha=0.2)
    
    # Título y Leyenda
    plt.title('Balance Muscular (Puntaje RPG)', size=15, y=1.1, color=TEXT_COLOR)
    legend = plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
    # --- FIN DE ESTILO BLUEPRINT ---
    
    # Guardar la gráfica (IMPORTANTE: guardar con el facecolor)
    plt.savefig(output_file, facecolor=fig.get_facecolor(), transparent=False)
    plt.close(fig)
    print("¡Gráfica de radar guardada!")

def analizar_fuerza(df_efectivo, ejercicios_clave):
    """
    Calcula el e1RM para los ejercicios clave.
    Devuelve un DataFrame con los top sets diarios.
    """
    print("\nCalculando progreso de fuerza (e1RM)...")
    
    df_clave = df_efectivo[df_efectivo['nombre_ejercicio'].isin(ejercicios_clave)].copy()
    
    if df_clave.empty:
        print(f"No se encontraron datos para los ejercicios clave definidos.")
        return None

    df_clave['e1rm'] = df_clave.apply(lambda row: calcular_e1rm(row['peso_kg'], row['reps']), axis=1)
    df_clave = df_clave.dropna(subset=['e1rm'])

    if df_clave.empty:
        print(f"No se pudieron calcular e1RM (quizás las reps están fuera de rango 1-15).")
        return None

    top_sets_diarios = df_clave.loc[df_clave.groupby(['fecha_dia', 'nombre_ejercicio'])['e1rm'].idxmax()]
    top_sets_diarios = top_sets_diarios.sort_values(by='fecha_dia')

    print(f"Se analizarán {len(top_sets_diarios)} 'top sets' de tus ejercicios clave.")
    return top_sets_diarios

def generar_grafica_fuerza(top_sets_diarios, ejercicios_clave, output_file):
    """
    Genera la gráfica de líneas de progreso de fuerza (BLUEPRINT).
    """
    print(f"\nGenerando gráfica de fuerza (Blueprint) en '{output_file}'...")
    
    # Colores claros que contrastan con el azul
    BLUEPRINT_COLORS = ['#FFFFFF', '#AFEEEE', '#FFFFE0', '#F0E68C', '#98FB98']
    # (Blanco, Turquesa Pálido, Amarillo Claro, Khaki, Verde Pálido)
    
    BLUEPRINT_BG = '#0A417A'
    GRID_COLOR = '#AAAAAA'
    TEXT_COLOR = '#EFEFEF'
    
    fig, ax = plt.subplots(figsize=(12, 7))

    # --- INICIO DE ESTILO BLUEPRINT ---
    
    # Aplicar color de fondo
    fig.patch.set_facecolor(BLUEPRINT_BG)
    ax.set_facecolor(BLUEPRINT_BG)

    # Dibujar las líneas de datos
    for i, ejercicio_nombre in enumerate(ejercicios_clave):
        df_ejercicio = top_sets_diarios[top_sets_diarios['nombre_ejercicio'] == ejercicio_nombre]
        if not df_ejercicio.empty:
            color = BLUEPRINT_COLORS[i % len(BLUEPRINT_COLORS)]
            ax.plot(df_ejercicio['fecha_dia'], df_ejercicio['e1rm'], 
                    marker='o', linestyle='-', label=ejercicio_nombre,
                    color=color, mec=color, mfc=color, ms=4)

    # Título y etiquetas de ejes
    ax.set_title('Progreso de Fuerza (e1RM en Top Sets)', color=TEXT_COLOR)
    ax.set_ylabel('e1RM (Kg Estimado)', color=TEXT_COLOR)
    ax.set_xlabel('Fecha', color=TEXT_COLOR)
    
    # Color de los números en los ejes
    ax.tick_params(axis='x', colors=TEXT_COLOR)
    ax.tick_params(axis='y', colors=TEXT_COLOR)
    
    # Rejilla de fondo
    ax.grid(True, linestyle='--', alpha=0.2, color=GRID_COLOR)
    
    # Bordes (spines)
    ax.spines['top'].set_color(GRID_COLOR)
    ax.spines['bottom'].set_color(GRID_COLOR)
    ax.spines['left'].set_color(GRID_COLOR)
    ax.spines['right'].set_color(GRID_COLOR)

    # Leyenda
    legend = ax.legend()
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
    # --- FIN DE ESTILO BLUEPRINT ---
    
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(output_file, facecolor=fig.get_facecolor(), transparent=False)
    plt.close(fig)
    print("¡Gráfica de fuerza guardada!")

def analizar_composicion_corporal(medidas_file, altura_cm):
    """
        Lee medidas.csv, calcula Densidad, % Grasa y LBM.
    """
    print(f"\nCalculando composición corporal desde '{medidas_file}'...")
    
    try:
        df_medidas = pd.read_csv(medidas_file)
    except FileNotFoundError:
        print(f"No se encontró '{medidas_file}'. Omitiendo análisis de composición.")
        return None, None
    
    if df_medidas.empty:
        print("medidas.csv está vacío. Omitiendo.")
        return None, None
        
    # Limpieza de Fecha
    df_medidas['fecha_dt'] = pd.to_datetime(df_medidas['fecha'])
    df_medidas = df_medidas.sort_values(by='fecha_dt')
    
    # --- Cálculos (US Navy) ---
    # Solo podemos calcular si tenemos las 3 medidas clave
    cols_necesarias = ['peso', 'cintura', 'cuello']
    df_calculos = df_medidas.dropna(subset=cols_necesarias).copy()
    
    if df_calculos.empty:
        print("No hay filas con 'peso', 'cintura' y 'cuello'. Omitiendo cálculos de grasa.")
        return df_medidas, None # Devolvemos las medidas básicas

    # ¡Importante! El log10 falla si (cintura - cuello) es 0 or negativo.
    df_calculos['cintura_cuello_diff'] = df_calculos['cintura'] - df_calculos['cuello']
    
    # Filtrar solo filas donde la resta es positiva
    df_calculos = df_calculos[df_calculos['cintura_cuello_diff'] > 0]
    
    if df_calculos.empty:
        print("Cintura es menor o igual al cuello en todos los registros. No se puede calcular % grasa.")
        return df_medidas, None

    # Ahora sí, calculamos
    log_diff = np.log10(df_calculos['cintura_cuello_diff'])
    log_altura = np.log10(altura_cm)
    
    # 1. Densidad Corporal
    df_calculos['densidad_corporal'] = 1.0324 - (0.19077 * log_diff) + (0.15456 * log_altura)
    
    # 2. % Grasa
    df_calculos['grasa_corporal_pct'] = (495 / df_calculos['densidad_corporal']) - 450
    
    # 3. LBM (Masa Magra)
    df_calculos['lbm_kg'] = df_calculos['peso'] * (1 - (df_calculos['grasa_corporal_pct'] / 100))

    print("\n--- Últimos Cálculos de Composición ---")
    print(df_calculos[['fecha_dt', 'grasa_corporal_pct', 'lbm_kg']].tail(5))
    
    # Devolvemos ambos dataframes
    return df_medidas, df_calculos


def generar_grafica_medidas_basicas(df_medidas, output_file):
    """
    Grafica las medidas básicas (peso, cintura, brazo) en estilo Blueprint.
    """
    print(f"\nGenerando gráfica de medidas en '{output_file}'...")
    
    # Columnas a graficar
    cols_a_graficar = ['peso', 'cintura', 'brazo', 'pierna']
    
    # --- Estilo Blueprint ---
    BLUEPRINT_BG = '#0A417A'
    GRID_COLOR = '#AAAAAA'
    TEXT_COLOR = '#EFEFEF'
    BLUEPRINT_COLORS = ['#FFFFFF', '#AFEEEE', '#FFFFE0', '#F0E68C']

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(BLUEPRINT_BG)
    ax.set_facecolor(BLUEPRINT_BG)

    for i, col in enumerate(cols_a_graficar):
        if col in df_medidas.columns and not df_medidas[col].isnull().all():
            color = BLUEPRINT_COLORS[i % len(BLUEPRINT_COLORS)]
            ax.plot(df_medidas['fecha_dt'], df_medidas[col], marker='o', linestyle='-', label=col.capitalize(), color=color, ms=4)

            for x, y in zip(df_medidas['fecha_dt'], df_medidas[col]):
                if pd.notnull(y):
                    ax.annotate(
                        f"{y}",
                        (x, y),
                        textcoords="offset points",
                        xytext=(0, 6),
                        ha='center',
                        color=color,
                        fontsize=8
                    )

    ax.set_title('Progreso Corporal (Medidas Básicas)', color=TEXT_COLOR)
    ax.set_ylabel('Medida (cm o Kg)', color=TEXT_COLOR)
# Esta línea solo se encarga del COLOR de las etiquetas
    ax.tick_params(axis='x', colors=TEXT_COLOR)
# Esta línea (que ya usamos en la otra gráfica) se encarga de la ROTACIÓN
    plt.xticks(rotation=30, ha='right')
    ax.tick_params(axis='y', colors=TEXT_COLOR)
    ax.grid(True, linestyle='--', alpha=0.2, color=GRID_COLOR)
    ax.spines['top'].set_color(GRID_COLOR)
    ax.spines['bottom'].set_color(GRID_COLOR)
    ax.spines['left'].set_color(GRID_COLOR)
    ax.spines['right'].set_color(GRID_COLOR)
    legend = ax.legend()
    for text in legend.get_texts(): text.set_color(TEXT_COLOR)
    
    plt.tight_layout()
    plt.savefig(output_file, facecolor=fig.get_facecolor(), transparent=False)
    plt.close(fig)
    print("¡Gráfica de medidas básicas guardada!")


def generar_grafica_composicion_calculada(df_calculos, output_file):
    """
    Grafica el % Grasa y LBM en ejes Y separados (Blueprint).
    """
    print(f"\nGenerando gráfica de composición en '{output_file}'...")

    # --- Estilo Blueprint ---
    BLUEPRINT_BG = '#0A417A'
    GRID_COLOR = '#AAAAAA'
    TEXT_COLOR = '#EFEFEF'
    COLOR_GRASA = '#FF9900' # Naranja
    COLOR_LBM = '#00FFFF' # Cian

    fig, ax1 = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(BLUEPRINT_BG)
    
    # Eje 1 (% Grasa)
    ax1.set_facecolor(BLUEPRINT_BG)
    ax1.plot(df_calculos['fecha_dt'], df_calculos['grasa_corporal_pct'], marker='o', linestyle='-', label='% Grasa Corporal', color=COLOR_GRASA, ms=4)
    ax1.set_ylabel('% Grasa Corporal (Calculado)', color=COLOR_GRASA)
    ax1.tick_params(axis='y', colors=COLOR_GRASA)
    ax1.spines['left'].set_color(COLOR_GRASA)

    # Eje 2 (LBM) - Eje Y compartido
    ax2 = ax1.twinx()
    ax2.plot(df_calculos['fecha_dt'], df_calculos['lbm_kg'], marker='o', linestyle='-', label='Masa Magra (LBM)', color=COLOR_LBM, ms=4)
    ax2.set_ylabel('Masa Magra (LBM en Kg)', color=COLOR_LBM)
    ax2.tick_params(axis='y', colors=COLOR_LBM)
    ax2.spines['right'].set_color(COLOR_LBM)

    # Estilo común
    ax1.set_title('Progreso de Composición Corporal', color=TEXT_COLOR)
    
    # --- LA CORRECCIÓN ESTÁ AQUÍ ---
    # Línea INCORRECTA (la que da el error):
    # ax1.tick_params(axis='x', colors=TEXT_COLOR, rotation=30, ha='right')
    
    # Líneas CORRECTAS (separando las tareas):
    ax1.tick_params(axis='x', colors=TEXT_COLOR)  # Esta línea SÍ está bien
    plt.xticks(rotation=30, ha='right')           # Esta línea hace la rotación
    # --- FIN DE LA CORRECCIÓN ---
    
    ax1.grid(True, linestyle='--', alpha=0.2, color=GRID_COLOR)
    ax1.spines['top'].set_color(GRID_COLOR)
    ax1.spines['bottom'].set_color(GRID_COLOR)
    
    # Leyenda
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    for text in legend.get_texts(): text.set_color(TEXT_COLOR)

    plt.tight_layout()
    plt.savefig(output_file, facecolor=fig.get_facecolor(), transparent=False)
    plt.close(fig)
    print("¡Gráfica de composición guardada!")
    """
    Grafica el % Grasa y LBM en ejes Y separados (Blueprint).
    """
    print(f"\nGenerando gráfica de composición en '{output_file}'...")

# --- FUNCIONES DE HISTORIAL  ---

def guardar_reporte_en_historial(reporte_actual):
    """
    Guarda el reporte actual en un JSON histórico.
    Si ya existe una entrada para la semana actual, la actualiza.
    """
    print(f"Guardando reporte en '{HISTORIAL_JSON_FILE}'...")
    
    # 1. Identificar la semana actual (ID único: "2025-46")
    hoy = datetime.now()
    semana_id = f"{hoy.year}-{hoy.isocalendar().week}"
    
    # Añadimos la fecha y el ID al reporte antes de guardar
    reporte_actual['meta_data'] = {
        'fecha_generacion': hoy.isoformat(),
        'semana_id': semana_id
    }

    # 2. Cargar historial existente
    historial = []
    if os.path.exists(HISTORIAL_JSON_FILE):
        try:
            with open(HISTORIAL_JSON_FILE, 'r', encoding='utf-8') as f:
                historial = json.load(f)
        except json.JSONDecodeError:
            print("Error leyendo historial JSON (estaba corrupto), se creará uno nuevo.")
            historial = []

    # 3. Buscar si ya existe esta semana y actualizar, o añadir nueva
    indice_existente = next((index for (index, d) in enumerate(historial) if d.get('meta_data', {}).get('semana_id') == semana_id), None)
    
    if indice_existente is not None:
        print(f"Actualizando reporte existente para la semana {semana_id}...")
        historial[indice_existente] = reporte_actual
    else:
        print(f"Añadiendo nuevo reporte para la semana {semana_id}...")
        historial.append(reporte_actual)
    
    # 4. Guardar de vuelta al archivo
    with open(HISTORIAL_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(historial, f, indent=2)
    
    print("¡Historial actualizado!")

def obtener_contexto_historico(semanas_atras=8):
    """
    Lee el historial y devuelve un string con los últimos N reportes
    listo para pasárselo a la IA.
    """
    if not os.path.exists(HISTORIAL_JSON_FILE):
        return "No hay historial disponible (es la primera semana)."
    
    with open(HISTORIAL_JSON_FILE, 'r', encoding='utf-8') as f:
        historial = json.load(f)
    
    # Ordenar por semana_id para asegurar cronología
    # (Asumimos formato "YYYY-WW")
    historial.sort(key=lambda x: x.get('meta_data', {}).get('semana_id', '0000-00'))
    
    # Tomar los últimos N (excluyendo el actual que acabamos de generar si ya se guardó)
    # Para efectos del prompt, mandamos TODO el historial reciente relevante
    historial_reciente = historial[-semanas_atras:]
    
    return json.dumps(historial_reciente, indent=2)
# --- 4. FUNCIÓN PRINCIPAL (El "Director de Orquesta") ---

def analizar_progreso():
    """
    Función principal que dirige el flujo del análisis y recopila los datos.
    AHORA DEVUELVE EL TEXTO DE LA IA.
    """
    print(f"--- Iniciando Análisis de Progreso ---")
    
    reporte_final = {}

    # --- Cargar Datos de Entreno ---
    df_efectivo, total_semanas_unicas = cargar_y_limpiar_datos(CSV_FILE)
    
    if df_efectivo is None:
        print("--- Omitiendo Análisis 1 y 2 (Sin datos de entreno) ---")
    else:
        # --- Análisis 1: Balance Muscular (Radar) ---
        print("\n--- Análisis 1: Balance Muscular (Radar) ---")
        stats_df = analizar_balance(df_efectivo, total_semanas_unicas)
        generar_grafica_radar(stats_df, GRAFICA_RADAR_FILE)
        reporte_final['balance_muscular_stats'] = stats_df.to_dict()

        # --- Análisis 2: Progreso de Fuerza (e1RM) ---
        print("\n--- Análisis 2: Progreso de Fuerza (e1RM) ---")
        top_sets_diarios = analizar_fuerza(df_efectivo, EJERCICIOS_CLAVE)
        if top_sets_diarios is not None:
            generar_grafica_fuerza(top_sets_diarios, EJERCICIOS_CLAVE, GRAFICA_FUERZA_FILE)
            ultimos_e1rm = top_sets_diarios.groupby('nombre_ejercicio').last()['e1rm']
            reporte_final['fuerza_e1rm_reciente'] = ultimos_e1rm.to_dict()

    # --- Análisis 3: Composición Corporal ---
    print("\n--- Análisis 3: Composición Corporal ---")
    df_medidas_basicas, df_medidas_calculadas = analizar_composicion_corporal(MEDIDAS_CSV_FILE, ALTURA_CM)
    
    if df_medidas_basicas is not None:
        generar_grafica_medidas_basicas(df_medidas_basicas, GRAFICA_MEDIDAS_FILE)
        medidas_recientes = df_medidas_basicas.iloc[-1][['peso', 'pecho', 'cintura', 'cadera', 'brazo', 'pierna', 'cuello']]
        reporte_final['medidas_corporales_recientes'] = medidas_recientes.to_dict()
        
    if df_medidas_calculadas is not None:
        generar_grafica_composicion_calculada(df_medidas_calculadas, GRAFICA_COMPOSICION_FILE)
        calculos_recientes = df_medidas_calculadas.iloc[-1][['grasa_corporal_pct', 'lbm_kg']]
        reporte_final['composicion_corporal_reciente'] = calculos_recientes.to_dict()
    
    # 2. Guardar el reporte actual en el historial JSON
    guardar_reporte_en_historial(reporte_final)
    
    # 3. Obtener el historial para el contexto
    historial_str = obtener_contexto_historico(semanas_atras=8)
    reporte_actual_str = json.dumps(reporte_final, indent=2)
    
    # 4. Generar interpretación CON historial
    interpretacion_final = generar_interpretacion_llm(reporte_actual_str, historial_str)
    print("\n--- Análisis Completado ---")
    
    print("\nInterpretación recibida de la IA:\n", interpretacion_final)
    
    # --- (NUEVO) DEVOLVER EL RESULTADO ---
    return interpretacion_final

# --- 5. EJECUTAR EL SCRIPT ---
if __name__ == "__main__":
    # Esto es para que puedas seguir probando 'analisis.py' por sí solo
    print("Ejecutando análisis en modo 'standalone'...")
    analisis_texto = analizar_progreso()
    print("\n==============================================")
    print("  INTERPRETACIÓN DEL COACH DE IA (STANDALONE)")
    print("==============================================")
    print(analisis_texto)
    print("==============================================")