# Asistente Comercial WhatsApp — FourPlayers / Motorola

Motor base de procesamiento de lenguaje natural y validación estructurada (Pre-Entrega 2 — Forward Deployed Engineer).

## Descripción
Interpreta mensajes operativos de campo (asesores de PDV, supervisores de país y perfiles estratégicos de FP/Motorola en Paraguay, Uruguay y ROLA), extrayendo intención, punto de venta, modelo de equipo y alertas preventivas de inventario (<= 2 unidades o sobrestock > 30 unidades) mediante Pydantic y LLMs.

## Estructura del proyecto
- `src/motor.py`: Contrato de datos (`Entrada`/`Salida`), plantilla de prompt, llamada al modelo (`openai/gpt-oss-120b` vía Groq), cálculo de tokens/costo/latencia y manejo de errores con reintento.
- `.env.example`: Variables de entorno requeridas (`GROQ_API_KEY`, `GEMINI_API_KEY`).
- `requirements.txt` / `pyproject.toml`: Dependencias del proyecto.

## Ejecución
```bash
uv run src/motor.py
```