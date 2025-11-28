import pandas as pd
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv
import time

# --- CONFIGURACIÓN ---

# 1. (MUY IMPORTANTE) Cambia esto por el nombre exacto de tu archivo CSV
NOMBRE_DEL_CSV = "workouts.csv" 

# 2. Nombre de la columna que tiene los nombres de ejercicios
COLUMNA_EJERCICIOS = "exercise_title"

# 3. El archivo JSON que usaremos como base de datos
DB_JSON_FILE = "ejercicios_db.json"

# 4. Cargar la API de Gemini (usa la misma del .env)
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# ----------------------

def llamar_gemini_para_grupo(nombre_ejercicio):
    """
    Llama a Gemini para obtener un desglose detallado de los grupos musculares.
    Espera una respuesta en formato JSON.
    """
    try:
        # Usamos flash-latest, es rápido y más que suficiente
        model = genai.GenerativeModel('models/gemini-flash-latest') 
        
        # --- ESTE ES EL NUEVO PROMPT DETALLADO ---
        prompt = f"""
        Analiza el ejercicio: "{nombre_ejercicio}".
        Proporcióname los grupos musculares que trabaja.
        
        Responde SOLAMENTE con un objeto JSON válido que siga esta estructura exacta:
        {{
          "grupo_principal": "El grupo muscular primario (ej. Pecho, Espalda, Pierna)",
          "grupos_secundarios": ["Lista de músculos sinérgicos (ej. Tríceps, Hombro, Bíceps)"],
          "cabezas_o_regiones": ["Lista de cabezas o regiones específicas (ej. Pectoral clavicular, Vasto lateral, Cabeza larga del tríceps, Deltoides anterior)"]
        }}
        
        Ejemplo para "Bench Press (Barbell)":
        {{
          "grupo_principal": "Pecho",
          "grupos_secundarios": ["Tríceps", "Hombro"],
          "cabezas_o_regiones": ["Pectoral mayor (cabeza esternal)", "Pectoral mayor (cabeza clavicular)", "Tríceps (todas las cabezas)", "Deltoides anterior"]
        }}
        
        Ejemplo para "Bicep Curl (Dumbbell)":
        {{
          "grupo_principal": "Bíceps",
          "grupos_secundarios": ["Antebrazo"],
          "cabezas_o_regiones": ["Bíceps braquial (cabeza larga)", "Bíceps braquial (cabeza corta)", "Braquial"]
        }}
        
        Si el ejercicio es "Cardio" o no se puede clasificar, usa:
        {{
          "grupo_principal": "Cardio",
          "grupos_secundarios": [],
          "cabezas_o_regiones": []
        }}
        
        Tu respuesta (solo el JSON):
        """
        
        response = model.generate_content(prompt)
        
        # Limpiar la respuesta de Gemini (quitar markdown, etc.)
        texto_respuesta = response.text.strip()
        
        # Quitar ```json ... ``` si es que lo añade
        if texto_respuesta.startswith("```json"):
            texto_respuesta = texto_respuesta[7:-3].strip()
        elif texto_respuesta.startswith("```"):
             texto_respuesta = texto_respuesta[3:-3].strip()
        
        # Convertir el string de JSON a un objeto Python
        datos_musculares = json.loads(texto_respuesta)
        
        return datos_musculares
        
    except Exception as e:
        print(f"  -> Error de API o parseo JSON: {e}")
        return None # Devolvemos None para indicar fallo

def actualizar_diccionario():
    print(f"Iniciando actualización de '{DB_JSON_FILE}'...")
    print(f"Leyendo CSV: '{NOMBRE_DEL_CSV}'...")
    
    # Cargar la base de datos JSON existente (o crear una nueva)
    if os.path.exists(DB_JSON_FILE):
        with open(DB_JSON_FILE, 'r', encoding='utf-8') as f:
            ejercicios_db = json.load(f)
        print(f"'{DB_JSON_FILE}' cargado. Contiene {len(ejercicios_db)} ejercicios.")
    else:
        ejercicios_db = {}
        print(f"No se encontró '{DB_JSON_FILE}'. Se creará uno nuevo.")
        
    # Cargar el CSV con pandas
    try:
        df = pd.read_csv(NOMBRE_DEL_CSV)
    except FileNotFoundError:
        print(f"--- ERROR ---")
        print(f"No se encontró el archivo '{NOMBRE_DEL_CSV}'.")
        print(f"Asegúrate de que el nombre es correcto y está en la misma carpeta.")
        return
    except Exception as e:
        print(f"Error leyendo el CSV: {e}")
        return

    # Obtener la lista de ejercicios únicos del CSV
    ejercicios_unicos_csv = df[COLUMNA_EJERCICIOS].unique()
    print(f"Se encontraron {len(ejercicios_unicos_csv)} ejercicios únicos en tu historial.")
    
    nuevos_ejercicios_count = 0
    
    # Iterar y actualizar
    for ejercicio_nombre in ejercicios_unicos_csv:
        if pd.isna(ejercicio_nombre):
            continue
            
        # 1. Comprobar si ya lo tenemos en nuestra DB
        if ejercicio_nombre not in ejercicios_db:
            
            # 2. Si es nuevo, llamar a Gemini
            print(f"\n[NUEVO] Ejercicio encontrado: '{ejercicio_nombre}'")
            print("  -> Preguntando a Gemini por el desglose muscular...")
            nuevos_ejercicios_count += 1
            
            datos_musculares = llamar_gemini_para_grupo(ejercicio_nombre)
            
            # 3. Si Gemini respondió exitosamente (no es None)
            if datos_musculares:
                print(f"  -> Respuesta de IA: Grupo Principal '{datos_musculares.get('grupo_principal')}'")
                
                # 4. Guardar el objeto JSON completo
                ejercicios_db[ejercicio_nombre] = datos_musculares
                
                # 5. Guardar el JSON en cada paso (para no perder progreso)
                with open(DB_JSON_FILE, 'w', encoding='utf-8') as f:
                    json.dump(ejercicios_db, f, indent=2, ensure_ascii=False)
            
            else:
                print(f"  -> Fallo al obtener grupo para '{ejercicio_nombre}'. Se reintentará la próxima vez.")
                
            # Pausa para no saturar la API
            time.sleep(1) # 1 segundo de pausa entre cada llamada a la API
            
    print("\n---------------------------------")
    print(f"¡Proceso completado!")
    print(f"Se añadieron {nuevos_ejercicios_count} nuevos ejercicios.")
    print(f"Tu base de datos '{DB_JSON_FILE}' ahora tiene {len(ejercicios_db)} ejercicios.")

# --- Ejecutar el script ---
if __name__ == "__main__":
    actualizar_diccionario()