from flask import Flask, request
import json
import requests  # <-- Lo necesitamos de vuelta
import os        # <-- NUEVO: Para leer las variables de entorno
from dotenv import load_dotenv  # <-- NUEVO: Para cargar el .env
import google.generativeai as genai # <-- NUEVO: Para Gemini
import re  # <-- NUEVO: Para Expresiones Regulares
import csv # <-- NUEVO: Para escribir el archivo CSV
from datetime import datetime
from analisis import analizar_progreso
import pandas as pd # <-- Asegúrate de tener esto al inicio


# --- CONFIGURACIÓN INICIAL ---
# 1. Carga todas las variables de tu archivo .env
load_dotenv()

#Variables
MEDIDAS_CSV_FILE = "medidas.csv"

TODAS_LAS_MEDIDAS = [
    'fecha', 'peso', 'pecho', 'cintura', 
    'cadera', 'brazo', 'pierna', 'cuello', 'hombro'
]
# Patrón RegEx para encontrar "palabra: numero"
MEDIDAS_PATTERN = re.compile(r"([\wáéíóúñ]+)\s*:\s*([\d\.]+)")

# 2. Configura la API de Gemini
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# 3. Configura las variables de WhatsApp
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID")
# -----------------------------

app = Flask(__name__)
VERIFY_TOKEN = "TU_TOKEN_SECRETO" # Asegúrate que este sea el mismo

# Usamos un 'set' porque buscar en él es mucho más rápido que en una lista
MENSAJES_PROCESADOS = set()

# ... (Tu función verify_webhook GET sigue igual) ...
# --- RUTA DE VERIFICACIÓN (MÉTODO GET) ---
@app.route("/webhook", methods=['GET'])
def verify_webhook():
    """
    Verifica el webhook con Facebook.
    """
    mode = request.args.get('hub.mode')
    token = request.args.get('hub.verify_token')
    challenge = request.args.get('hub.challenge')

    if mode and token:
        if mode == 'subscribe' and token == VERIFY_TOKEN:
            print("WEBHOOK_VERIFIED")
            return challenge, 200
        else:
            return 'Forbidden', 403
    
    return 'Hola! El servidor está corriendo correctamente.', 200

# --- (SOBREESCRIBIR) FUNCIÓN DE PARSEO DE TEXTO (v2.0) ---

# Compilamos las expresiones regulares una sola vez para eficiencia
# Patrón para: "Set 1: 40 kg x 15 [Warm-up]"  o "Set 2: 80 kg x 8"
SET_PATTERN = re.compile(
    r"Set \d+:\s*"                 # Coincide "Set 1: "
    r"([\d\.]+)\s*(kg|lbs)\s*x\s*" # Grupo 1: Peso (ej. 40), Grupo 2: Unidad (kg/lbs)
    r"([\d\.]+)"                   # Grupo 3: Reps (ej. 15)
    r"\s*(\[.*\])?"                # Grupo 4: Tipo (ej. [Warm-up]) (Opcional)
)
# --- REEMPLAZA TU FUNCIÓN ANTERIOR POR ESTA ---

def parsear_y_guardar_medidas(texto_medidas):
    """
    Parsea "peso: 80, brazo: 38", valida contra las columnas fijas
    y guarda en 'medidas.csv'.
    """
    print("Parseando registro de medidas (v2 - Columnas Fijas)...")
    
    # 1. Encontrar todos los pares (ej. ["peso", "80.5"], ["brazo", "38"])
    partes = MEDIDAS_PATTERN.findall(texto_medidas.lower())
    
    datos_medidas = {}
    medidas_reconocidas = []
    medidas_ignoradas = []

    # 2. Validar las medidas contra nuestra lista fija
    for key, value in partes:
        key_limpia = key.strip()
        # Si la palabra SÍ está en nuestra lista...
        if key_limpia in TODAS_LAS_MEDIDAS:
            datos_medidas[key_limpia] = float(value)
            medidas_reconocidas.append(f"- {key_limpia.capitalize()}: {value}")
        else:
            # Si no, la ignoramos
            medidas_ignoradas.append(key_limpia)

    if not medidas_reconocidas:
        print("No se encontraron medidas válidas.")
        return "No pude reconocer ninguna medida válida (ej. 'peso', 'brazo', 'cintura').\nUsa el formato: `medidas peso: 80, brazo: 38`"
    
    # 3. Añadir la fecha actual
    datos_medidas['fecha'] = datetime.now().isoformat()
    
    # 4. Guardar en el CSV
    archivo_existe = os.path.exists(MEDIDAS_CSV_FILE)
    
    try:
        # Usamos DictWriter con nuestra lista FIJA de columnas
        with open(MEDIDAS_CSV_FILE, 'a+', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=TODAS_LAS_MEDIDAS)
            
            if not archivo_existe or f.tell() == 0:
                writer.writeheader() # Escribe los encabezados fijos
            
            # DictWriter guardará los datos que tiene (ej. 'peso')
            # y dejará en blanco las columnas que no (ej. 'cadera')
            writer.writerow(datos_medidas)
            
        print(f"Medidas guardadas en '{MEDIDAS_CSV_FILE}': {datos_medidas}")
        
        # 5. Crear mensaje de confirmación
        confirmacion = "✅ Medidas Registradas:\n"
        confirmacion += "\n".join(medidas_reconocidas)
        
        if medidas_ignoradas:
            confirmacion += f"\n\n(Se ignoraron: {', '.join(medidas_ignoradas)})"
            
        return confirmacion

    except Exception as e:
        print(f"Error al guardar medidas: {e}")
        return "Hubo un error al guardar tus medidas."

# --- REEMPLAZA ESTA FUNCIÓN EN APP.PY ---

def buscar_sesion_anterior(ejercicios_hoy_json):
    """
    Identifica los músculos de hoy, busca la última vez que se entrenaron
    (IGNORANDO coincidencias menores como Core si hay músculos grandes).
    """
    print("🔍 Buscando sesión anterior relevante (Lógica Mejorada)...")
    
    # --- 1. Identificar qué músculos estamos entrenando hoy
    musculos_hoy = set()
    
    if os.path.exists(DB_JSON_FILE):
        with open(DB_JSON_FILE, 'r', encoding='utf-8') as f:
            ejercicios_db = json.load(f)
    else:
        return "No hay base de datos de ejercicios para comparar."

    for ex in ejercicios_hoy_json['ejercicios']:
        nombre = ex['nombre']
        info = ejercicios_db.get(nombre, {})
        grupo = info.get('grupo_principal')
        if grupo and grupo != "N/A":
            musculos_hoy.add(grupo)
            
    if not musculos_hoy:
        return "No se pudieron identificar grupos musculares hoy."

    print(f"  -> Músculos detectados hoy: {musculos_hoy}")

    # --- 2. Cargar el historial CSV
    if not os.path.exists(CSV_FILE_NAME):
        return "No hay historial previo."
    
    try:
        df = pd.read_csv(CSV_FILE_NAME)
    except:
        return "Error leyendo el historial."
        
    if df.empty: return "Historial vacío."

    # Especificamos el formato exacto de Hevy para eliminar el warning
    df['fecha_dt'] = pd.to_datetime(df['fecha'], format='%A, %b %d, %Y at %I:%M%p', errors='coerce')
    df = df.sort_values('fecha_dt', ascending=False) 

    # --- 3. Lógica de Búsqueda Inteligente ---
    fecha_hoy = pd.to_datetime(datetime.now()).date()
    ultima_fecha_encontrada = None
    
    # Lista de músculos "secundarios" que NO deben disparar una coincidencia 
    # si estamos buscando músculos grandes.
    GRUPOS_MENORES = {'Core', 'Abdomen', 'Cardio', 'Antebrazo', 'Otro', 'N/A'}
    
    # Identificamos si hoy es un día "Importante" (tiene músculos que no son menores)
    musculos_principales_hoy = musculos_hoy - GRUPOS_MENORES
    es_dia_importante = len(musculos_principales_hoy) > 0

    for fecha in df['fecha_dt'].dt.date.unique():
        if fecha == fecha_hoy:
            continue 
            
        df_dia = df[df['fecha_dt'].dt.date == fecha]
        musculos_ese_dia = set(df_dia['grupo_principal'].unique())
        
        # Calculamos la intersección (qué músculos se repiten)
        coincidencias = musculos_hoy.intersection(musculos_ese_dia)
        
        if coincidencias:
            # --- FILTRO DE CALIDAD ---
            if es_dia_importante:
                # Si hoy hice Espalda, y la coincidencia es SOLO Core...
                coincidencias_relevantes = coincidencias - GRUPOS_MENORES
                
                if not coincidencias_relevantes:
                    # Significa que solo coincidimos en Core/Cardio.
                    # ¡Esta NO es la sesión que buscamos! Sigue buscando atrás.
                    print(f"  -> Saltando fecha {fecha} (Solo coincidió en {coincidencias})")
                    continue
            
            # Si pasamos el filtro, encontramos la fecha correcta
            ultima_fecha_encontrada = fecha
            break 
    
    if not ultima_fecha_encontrada:
        return "No encontré sesiones previas relevantes de estos grupos musculares."

    # --- 4. Construir el resumen ---
    print(f"  -> MATCH ENCONTRADO: {ultima_fecha_encontrada}")
    
    df_anterior = df[df['fecha_dt'].dt.date == ultima_fecha_encontrada]
    resumen_anterior = f"--- SESIÓN PREVIA ({ultima_fecha_encontrada}) ---\n"
    
    for ejercicio in df_anterior['nombre_ejercicio'].unique():
        # Solo incluimos en el resumen los ejercicios que COINCIDEN con la intención de hoy
        # O si quieres ver todo lo que hiciste ese día, quita este 'if'
        sets = df_anterior[df_anterior['nombre_ejercicio'] == ejercicio]
        sets_efectivos = sets[sets['tipo_set'] != 'warmup']
        
        if not sets_efectivos.empty:
            # Buscar el mejor set (Volumen de carga)
            sets_efectivos = sets_efectivos.copy() # Evitar warning de pandas
            sets_efectivos['volumen'] = sets_efectivos['peso_kg'] * sets_efectivos['reps']
            best_set = sets_efectivos.loc[sets_efectivos['volumen'].idxmax()]
            
            resumen_anterior += f"- {ejercicio}: Mejor set {best_set['peso_kg']}kg x {best_set['reps']}\n"
    
    return resumen_anterior

# --- (SOBREESCRIBIR) FUNCIÓN DE PARSEO DE TEXTO (v3.0 - Robustez) ---

# 1. Regex para sets normales (con peso)
# Ej: "Set 1: 40 kg x 15 [Warm-up]"
SET_PATTERN_WEIGHT = re.compile(
    r"Set \d+:\s*([\d\.]+)\s*(kg|lbs)\s*x\s*([\d\.]+)\s*(\[.*\])?"
)

# 2. Regex para sets de peso corporal o solo reps (SIN peso)
# Ej: "Set 1: 15 reps", "Set 2: 12"
SET_PATTERN_BODYWEIGHT = re.compile(
    r"Set \d+:\s*([\d\.]+)\s*(reps)?\s*(\[.*\])?"
)

def parsear_texto_hevy(texto_plano):
    print("Iniciando parseo de texto v3.0 (Bodyweight + Cleaner)...")
    
    json_del_entreno = {
        "titulo": "",
        "fecha": "",
        "ejercicios": []
    }
    
    ejercicio_actual = None
    lineas = texto_plano.strip().split('\n')
    
    try:
        # Intentar sacar titulo y fecha, si falla ponemos default
        if len(lineas) >= 2:
            json_del_entreno["titulo"] = lineas[0].strip()
            json_del_entreno["fecha"] = lineas[1].strip()
        
        for linea in lineas[2:]: # Empezamos desde la línea 2 para saltar titulo/fecha
            linea = linea.strip()
            
            # --- 1. FILTRO DE BASURA (IGNORAR) ---
            # Si la línea está vacía o es basura de footer, la saltamos
            if not linea or linea.startswith(('http', '@', 'Workout created')):
                continue
            
            # --- 2. DETECCIÓN DE SETS ---
            
            # A) ¿Es un set con peso? (Prioridad 1)
            match_weight = SET_PATTERN_WEIGHT.search(linea)
            
            # B) ¿Es un set de solo reps? (Prioridad 2)
            match_bw = SET_PATTERN_BODYWEIGHT.search(linea)
            
            if match_weight:
                if ejercicio_actual:
                    peso = float(match_weight.group(1))
                    unidad = match_weight.group(2)
                    reps = int(match_weight.group(3))
                    tipo_raw = match_weight.group(4)
                    
                    if unidad == "lbs": peso = peso * 0.453592
                    
                    tipo_set = tipo_raw.strip("[]").lower() if tipo_raw else "normal"
                    
                    ejercicio_actual["sets"].append({
                        "texto_original": linea,
                        "peso_kg": peso,
                        "reps": reps,
                        "tipo_set": tipo_set
                    })
                    
            elif match_bw:
                if ejercicio_actual:
                    # En bodyweight, el grupo 1 son las reps
                    reps = int(match_bw.group(1))
                    tipo_raw = match_bw.group(3) # El grupo 2 es la palabra "reps" (opcional)
                    
                    tipo_set = tipo_raw.strip("[]").lower() if tipo_raw else "normal"
                    
                    ejercicio_actual["sets"].append({
                        "texto_original": linea,
                        "peso_kg": 0, # Peso corporal = 0 extra
                        "reps": reps,
                        "tipo_set": tipo_set
                    })
                    
            # --- 3. DETECCIÓN DE NUEVO EJERCICIO ---
            else:
                # Si NO es un set y NO es basura, TIENE que ser un nombre
                ejercicio_actual = {
                    "nombre": linea,
                    "sets": []
                }
                json_del_entreno["ejercicios"].append(ejercicio_actual)
                
        print("Parseo v3.0 exitoso.")
        return json_del_entreno
        
    except Exception as e:
        print(f"Error durante el parseo del texto v3.0: {e}")
        return None
    """
    Toma el bloque de texto y lo convierte en un JSON detallado
    extrayendo peso, reps y tipo de set.
    """
    print("Iniciando parseo de texto v2.0...")
    
    json_del_entreno = {
        "titulo": "",
        "fecha": "",
        "ejercicios": []
    }
    
    ejercicio_actual = None
    lineas = texto_plano.strip().split('\n')
    
    try:
        json_del_entreno["titulo"] = lineas[0].strip()
        json_del_entreno["fecha"] = lineas[1].strip() # (Necesitaremos parsear esto para el CSV)
        
        for linea in lineas[3:]:
            linea = linea.strip()
            
            if not linea:
                ejercicio_actual = None
                continue
            
            # Revisa si la línea es un SET
            match = SET_PATTERN.search(linea)
            
            if match:
                # Si es un set, extraemos los datos
                if ejercicio_actual:
                    peso = float(match.group(1))
                    unidad = match.group(2)
                    reps = int(match.group(3))
                    
                    # Convierte todo a KG (si es necesario)
                    if unidad == "lbs":
                        peso = peso * 0.453592
                    
                    # Limpia el tipo de set (ej. "[Warm-up]" -> "warmup")
                    tipo_set_raw = match.group(4)
                    if tipo_set_raw:
                        tipo_set = tipo_set_raw.strip("[]").lower()
                    else:
                        tipo_set = "normal"
                        
                    # Guardamos los datos estructurados
                    ejercicio_actual["sets"].append({
                        "texto_original": linea,
                        "peso_kg": peso,
                        "reps": reps,
                        "tipo_set": tipo_set
                    })
            else:
                # Si no es un set, es un nombre de ejercicio
                ejercicio_actual = {
                    "nombre": linea,
                    "sets": []
                }
                json_del_entreno["ejercicios"].append(ejercicio_actual)
                
        print("Parseo de texto v2.0 exitoso.")
        return json_del_entreno
        
    except Exception as e:
        print(f"Error durante el parseo del texto v2.0: {e}")
        return None

# --- NUEVA FUNCIÓN: Generar Feedback con LLM ---
def generar_feedback_llm(datos_entreno_json, contexto_anterior_str=""):
    """
    Toma el JSON del entreno y pide feedback a la API de Gemini.
    """
    print("Generando feedback con LLM...")
    
    # Convierte el JSON de Python a un string de textocontexto_anterior_str=""
    datos_str = json.dumps(datos_entreno_json, indent=2)
    
    # Elige el modelo (gemini-1.5-flash es rápido y potente)
    model = genai.GenerativeModel('models/gemini-flash-latest')
    
    # Este es el "prompt" o la instrucción. ¡Aquí está la magia!
    prompt = f"""
    Eres un entrenador de clase mundial especializado en desarrollo muscular estético.
    Tu enfoque combina principios de hipertrofia, sobrecarga progresiva, buena técnica y manejo óptimo del volumen.
    No priorizas competir en fisicoculturismo ni en powerlifting: el objetivo es construir un físico marcado,
    proporcionado y atlético.

    A continuación recibirás un entrenamiento en formato JSON:

    Datos del entrenamiento:
    {datos_str}
    --- CONTEXTO HISTÓRICO (Última vez que entrené estos músculos) ---
    {contexto_anterior_str}
    
    
    Tu tarea:
    -Compara mi desempeño de HOY con la SESIÓN PREVIA (si existe). ¿Subí pesos? ¿Mismas reps?
    - Evalúa la sesión de forma clara y específica.
    - Da un feedback motivador pero basado en evidencia.
    -**IMPORTANTE: Menciona brevemente el desempeño de TODOS los grupos musculares trabajados hoy, incluyendo accesorios y core.**
    - Ofrece 2 a 3 recomendaciones accionables enfocadas en:
    progresión semanal,
    volumen efectivo,
    distribución de series,
    tempo o técnica inferida,
    ajustes para optimizar el estímulo del músculo trabajado.
    - Mantén el feedback práctico y aplicable en la siguiente sesión.

    IMPORTANTE: Formatea tu respuesta en texto plano compatible con WhatsApp.
    Esto significa:
    - No usar Markdown.
    - No usar asteriscos, guiones especiales o símbolos que generen formato.
    - Usar numeración simple (1., 2., 3.) y saltos de línea claros.
    - Nada de tablas ni encabezados tipo Markdown.

    Ejemplo de estilo de respuesta:
    Gran sesión de {datos_entreno_json['titulo']}!
    1. El volumen en Press de Banca estuvo sólido, buen trabajo.
    2. La próxima semana intenta subir 2.5 kg en la primera serie o añadir 1 a 2 reps.
    3. Cuida los descansos entre series para mantener el estímulo.

    Límite máximo: 4096 caracteres.
    Tu feedback en español:
"""

    
    try:
        response = model.generate_content(prompt)
        print("Feedback generado por IA exitosamente.")
        return response.text
    except Exception as e:
        print(f"Error al generar feedback con Gemini: {e}")
        return "Hubo un error al analizar tu entreno, pero ¡sigue así!"

# --- (SOBREESCRIBIR) FUNCIÓN DE CSV (v2.0) ---

DB_JSON_FILE = "ejercicios_db.json"
CSV_FILE_NAME = "entrenos.csv"

# Cargar la base de datos de ejercicios en memoria
try:
    with open(DB_JSON_FILE, 'r', encoding='utf-8') as f:
        ejercicios_db = json.load(f)
    print(f"Base de datos '{DB_JSON_FILE}' cargada con {len(ejercicios_db)} ejercicios.")
except FileNotFoundError:
    print(f"ADVERTENCIA: No se encontró '{DB_JSON_FILE}'. Los grupos musculares estarán vacíos.")
    ejercicios_db = {}

def obtener_info_muscular_ia(nombre_ejercicio):
    """
    Consulta a Gemini para clasificar un ejercicio nuevo al vuelo.
    """
    print(f"⚠️ Ejercicio nuevo detectado: '{nombre_ejercicio}'. Consultando a IA...")
    try:
        model = genai.GenerativeModel('models/gemini-flash-latest')
        prompt = f"""
        Analiza el ejercicio: "{nombre_ejercicio}".
        Responde SOLAMENTE con un JSON válido:
        {{
        "grupo_principal": "Grupo muscular primario (ej. Pecho, Espalda)",
        "grupos_secundarios": ["Lista de secundarios"],
        "cabezas_o_regiones": ["Lista de detalles"]
        }}
        """
        response = model.generate_content(prompt)
        texto = response.text.strip()
        
        # Limpieza básica de markdown
        if texto.startswith("```json"): texto = texto[7:-3].strip()
        elif texto.startswith("```"): texto = texto[3:-3].strip()
        
        return json.loads(texto)
    except Exception as e:
        print(f"Error clasificando ejercicio: {e}")
        # Valor por defecto para no romper nada
        return {
            "grupo_principal": "Otro", 
            "grupos_secundarios": [], 
            "cabezas_o_regiones": []
        }

def generar_y_guardar_csv(datos_entreno_json):
    """
    Toma el JSON del entreno, lo enriquece con la BD de músculos
    y lo guarda en 'entrenos.csv'.
    """
    print("Generando y guardando datos en CSV...")
    hubo_cambios_db = False

    # 1. Define los encabezados del CSV
    columnas_csv = [
        'fecha', 'titulo_rutina', 'nombre_ejercicio', 
        'grupo_principal', 'grupos_secundarios', 'cabezas_regiones',
        'set_index', 'peso_kg', 'reps', 'tipo_set'
    ]
    
    # 2. Comprueba si el archivo CSV ya existe
    archivo_existe = os.path.exists(CSV_FILE_NAME)
    
    try:
        # 3. Abre el archivo en modo 'append' (a+) para añadir al final
        with open(CSV_FILE_NAME, 'a+', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 4. Si el archivo es nuevo, escribe los encabezados
            if not archivo_existe or f.tell() == 0:
                writer.writerow(columnas_csv)
            
            # 5. Itera sobre los datos del entreno
            fecha_entreno = datos_entreno_json['fecha'] # (Faltaría parsear la fecha)
            titulo_rutina = datos_entreno_json['titulo']
            
            for ejercicio in datos_entreno_json['ejercicios']:
                nombre_ejercicio = ejercicio['nombre']

                
                if nombre_ejercicio not in ejercicios_db:
                    # 1. Preguntar a la IA
                    info_nueva = obtener_info_muscular_ia(nombre_ejercicio)
                    # 2. Actualizar memoria RAM
                    ejercicios_db[nombre_ejercicio] = info_nueva
                    # 3. Marcar para guardar
                    hubo_cambios_db = True

                # 6. Busca el ejercicio en nuestra BD (el diccionario inteligente)
                info_muscular = ejercicios_db.get(nombre_ejercicio, {})
                grupo_principal = info_muscular.get('grupo_principal', 'N/A')
                # (Convertimos listas a strings para el CSV)
                grupos_secundarios = ", ".join(info_muscular.get('grupos_secundarios', []))
                cabezas_regiones = ", ".join(info_muscular.get('cabezas_o_regiones', []))
                
                # 7. Itera sobre cada set y escribe la fila
                for i, set_data in enumerate(ejercicio['sets']):
                    fila = [
                        fecha_entreno,
                        titulo_rutina,
                        nombre_ejercicio,
                        grupo_principal,
                        grupos_secundarios,
                        cabezas_regiones,
                        i + 1, # set_index (1, 2, 3...)
                        set_data['peso_kg'],
                        set_data['reps'],
                        set_data['tipo_set']
                    ]
                    writer.writerow(fila)
                
                if hubo_cambios_db:
                    print("🧠 Aprendí nuevos ejercicios. Actualizando ejercicios_db.json...")
                    with open(DB_JSON_FILE, 'w', encoding='utf-8') as f:
                        json.dump(ejercicios_db, f, indent=2, ensure_ascii=False)
                    
        print(f"Datos guardados exitosamente en '{CSV_FILE_NAME}'.")
        
    except Exception as e:
        print(f"Error al escribir en el CSV: {e}")

# --- NUEVA FUNCIÓN: Enviar Mensaje de WhatsApp ---
def enviar_mensaje_whatsapp(numero_destino, texto_mensaje):
    """
    Envía un mensaje de texto al usuario usando la API de Meta.
    """
    print(f"Enviando mensaje a {numero_destino}...")
    
    url = f"https://graph.facebook.com/v19.0/{WHATSAPP_PHONE_ID}/messages"
    
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    
    data = {
        "messaging_product": "whatsapp",
        "to": numero_destino,
        "type": "text",
        "text": {
            "body": texto_mensaje
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response_json = response.json()
        
        if response.status_code == 200:
            print("Mensaje de WhatsApp enviado exitosamente.")
        else:
            print(f"Error al enviar WhatsApp. Status: {response.status_code}")
            print(f"Respuesta de Meta: {response_json}")
            
    except Exception as e:
        print(f"Excepción al enviar WhatsApp: {e}")

# --- REEMPLAZA TU FUNCIÓN get_ngrok_url POR ESTA ---

# Pon aquí tu URL exacta de Azure (sin la barra al final)
AZURE_URL = "https://coach-ai.azurewebsites.net/t" 

def get_ngrok_url():
    """
    En producción (Azure), devolvemos la URL fija.
    En desarrollo local, podríamos intentar buscar ngrok, 
    pero para simplificar, usaremos la de Azure o devolvemos None.
    """
    # Si estamos en la nube, regresamos la URL directa
    return AZURE_URL

    # (Nota: Ya no intentamos conectar a localhost:4040 porque en Azure no existe)
def enviar_imagen_whatsapp(numero_destino, url_imagen, caption=""):
    """
    Envía un mensaje de IMAGEN al usuario usando una URL pública.
    """
    print(f"Enviando imagen {url_imagen} a {numero_destino}...")

    url = f"https://graph.facebook.com/v19.0/{WHATSAPP_PHONE_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {
        "messaging_product": "whatsapp",
        "to": numero_destino,
        "type": "image",
        "image": {
            "link": url_imagen
        },
        "caption": caption # El texto que va debajo de la imagen
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response_json = response.json()

        if response.status_code == 200:
            print("Imagen de WhatsApp enviada exitosamente.")
        else:
            print(f"Error al enviar imagen. Status: {response.status_code}")
            print(f"Respuesta de Meta: {response_json}")

    except Exception as e:
        print(f"Excepción al enviar imagen: {e}")


# --- RUTA PRINCIPAL (ACTUALIZADA) ---
# --- RUTA PRINCIPAL (ACTUALIZADA PARA IGNORAR 'STATUS') ---
# --- RUTA PRINCIPAL (ACTUALIZADA CON 'medidas') ---
@app.route("/webhook", methods=['POST'])
# --- RUTA PRINCIPAL (ACTUALIZADA CON 'reporte') ---
@app.route("/webhook", methods=['POST'])
def receive_message():
    """
    Recibe todos los mensajes y decide qué hacer.
    """
    body = request.get_json()
    #print(json.dumps(body, indent=2)) 

    try:
        if 'messages' in body['entry'][0]['changes'][0]['value']:
            message_info = body['entry'][0]['changes'][0]['value']['messages'][0]
            message_id = message_info['id']
            
            #verificar si ya lo estamos procesando
            if message_id in MENSAJES_PROCESADOS:
                print(f"🚫 Mensaje repetido detectado ({message_id}). Ignorando.")
                return "OK", 200 # Le decimos a Meta "Ya lo tengo, gracias"
            
            # --- 3. SI ES NUEVO, LO GUARDAMOS ---
            MENSAJES_PROCESADOS.add(message_id)
            
            # (Opcional: Limpieza básica para que la lista no crezca infinito)
            if len(MENSAJES_PROCESADOS) > 1000:
                MENSAJES_PROCESADOS.clear()
                
            if 'text' not in message_info:
                print("No es un mensaje de texto. Ignorando.")
                return "OK", 200

            phone_number = message_info['from']
            message_text = message_info['text']['body'].strip()

            print(f"RECIBIDO: '{message_text}' de {phone_number}")

            # --- NUEVO SISTEMA DE COMANDOS ---

            mensaje_normalizado = message_text.lower()

            # 1. Comando de Medidas
            if mensaje_normalizado.startswith("medidas"):
                print("Detectado comando 'medidas'.")
                respuesta_medidas = parsear_y_guardar_medidas(message_text)
                enviar_mensaje_whatsapp(phone_number, respuesta_medidas)

            # 2. (NUEVO) Comando de Reporte
            elif mensaje_normalizado == "reporte":
                print("Detectado comando 'reporte'.")

                # 1. Envía un mensaje de "espera"
                enviar_mensaje_whatsapp(phone_number, "¡Recibido! 🤖\nGenerando tu reporte completo...\nEsto puede tardar hasta un minuto...")

                # 2. (NUEVO) Obtener nuestra URL pública actual
                public_url = get_ngrok_url()

                if not public_url:
                    enviar_mensaje_whatsapp(phone_number, "Error: No pude encontrar la URL de ngrok. Revisa la consola del servidor.")
                    return "OK", 200 # Salir si ngrok no corre

                try:
                    # 3. Llama a tu script pesado
                    print("Ejecutando analizar_progreso()...")
                    interpretacion_ia = analizar_progreso() # Esto crea las gráficas

                    # 4. (NUEVO) Enviar las 4 gráficas
                    print("Análisis completado. Enviando gráficas...")

                    # (Asegúrate de que estos nombres coincidan con tus archivos en /static/)
                    enviar_imagen_whatsapp(phone_number, f"{public_url}/static/balance_radar.png", "1/4: Tu Balance Muscular (RPG)")
                    enviar_imagen_whatsapp(phone_number, f"{public_url}/static/progreso_fuerza.png", "2/4: Tu Progreso de Fuerza (e1RM)")
                    enviar_imagen_whatsapp(phone_number, f"{public_url}/static/progreso_corporal.png", "3/4: Tus Medidas Corporales")
                    enviar_imagen_whatsapp(phone_number, f"{public_url}/static/progreso_composicion.png", "4/4: Tu Composición (Grasa/LBM)")

                    # 5. Envía el resultado final
                    print("Enviando interpretación de IA...")
                    enviar_mensaje_whatsapp(phone_number, interpretacion_ia)

                except Exception as e:
                    print(f"ERROR al ejecutar analisis.py: {e}")
                    enviar_mensaje_whatsapp(phone_number, "Ups, algo salió mal al generar tu reporte. Revisa la consola del servidor.")

            # 3. Comando de Entreno
            elif mensaje_normalizado.startswith("push") or mensaje_normalizado.startswith("pull") or mensaje_normalizado.startswith("leg"):
                print("Detectado registro de entreno.")
                datos_entreno = parsear_texto_hevy(message_text)

                if datos_entreno:
                    print("Parseo exitoso.")
                    generar_y_guardar_csv(datos_entreno)
                    contexto_previo = buscar_sesion_anterior(datos_entreno)
                    feedback_ia = generar_feedback_llm(datos_entreno,contexto_previo)
                    enviar_mensaje_whatsapp(phone_number, feedback_ia)
                else:
                    print("El parseo del texto falló.")
                    enviar_mensaje_whatsapp(phone_number, "No pude entender el formato de tu entreno.")

            # 4. Comando no reconocido
            else:
                print("Comando no reconocido.")
                enviar_mensaje_whatsapp(phone_number, "No reconocí ese comando. Prueba con:\n- `medidas ...`\n- `reporte`\n- (o pega tu entreno)")

        else:
            print("Recibido un update de 'status'. Ignorando.")

        return "OK", 200

    except Exception as e:
        print(f"Error GRAVE en 'receive_message': {e}")
        return "Error", 500
# ... (Tu código para iniciar el servidor sigue igual) ...
if __name__ == '__main__':
    app.run(port=3000, debug=True)