# Organizador Inteligente de Eventos

## Autores
- Melani Forsythe Matos 312 
- Olivia Ibañez Mustelier 311
- Orlando De La Torre Leal 312

## Descripción del Problema
La gestión eficiente y personalizada de eventos es un desafío creciente ante la cantidad masiva de información disponible en múltiples fuentes. Este proyecto aborda la creación de un organizador inteligente capaz de recolectar, procesar y recomendar eventos relevantes mediante técnicas avanzadas de procesamiento de lenguaje natural, representación del conocimiento y optimización.

## Requerimientos
Para el correcto desempeño del sistema se requieren:

- Python 3.8 o superior.
- Entorno virtual configurado con las librerías definidas en `requirements.txt`.
- Acceso a Internet para consumo de APIs externas.
- Capacidad para ejecutar aplicaciones Streamlit para la interfaz de usuario.
- Espacio en disco para almacenar eventos y embeddings generados.

## APIs Utilizadas
El sistema integra datos de eventos provenientes de las siguientes APIs públicas:

- **SeatGeek API**: Para extracción de eventos en tiempo real y con amplio catálogo.
- **PredictHQ API**: Para eventos especializados y de alta calidad.
- **Ticketmaster API**: Para acceso a grandes eventos y conciertos.

Cada fuente es accedida mediante claves de API configuradas en los módulos correspondientes.

## Uso y Ejecución
1. Clonar el repositorio.
2. Configurar un entorno virtual e instalar dependencias:

   ```bash
   python -m venv venv
   source venv/bin/activate  # o venv\Scripts\activate en Windows
   pip install -r requirements.txt
   luego pararse en la raiz del proyecto y ejecutar ./startup.sh
