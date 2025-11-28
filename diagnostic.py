import google.generativeai as genai
import os
from dotenv import load_dotenv

# Carga tu archivo .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    print("--- ERROR ---")
    print("No pude encontrar tu GOOGLE_API_KEY en el archivo .env.")
    print("Asegúrate de que el archivo .env esté en la misma carpeta.")
else:
    try:
        print("Conectando con Google AI Studio...")
        # Configura la API key
        genai.configure(api_key=GEMINI_API_KEY)
        
        print("\n--- MODELOS DISPONIBLES PARA TU API KEY ---")
        
        # Pide la lista de modelos
        model_list = list(genai.list_models())
        
        if not model_list:
            print("¡No se encontró ningún modelo!")
            print("Esto es un problema. Revisa que tu API key esté correcta y que la API esté HABILITADA en Google Cloud.")
        
        # Itera y muéstralos
        for m in model_list:
            # Imprime el nombre exacto que debemos usar
            print(f"Nombre del modelo: {m.name}")
            
            # Revisa si el modelo sirve para lo que queremos (generar texto)
            if 'generateContent' in m.supported_generation_methods:
                print("  -> ¡Este modelo SÍ sirve! (soporta 'generateContent')\n")
            else:
                print("  -> (Este modelo NO sirve para generar contenido)\n")

        print("--- FIN DE LA LISTA ---")

    except Exception as e:
        print("\n--- ¡ERROR AL CONECTAR! ---")
        print(f"Ocurrió un error: {e}")