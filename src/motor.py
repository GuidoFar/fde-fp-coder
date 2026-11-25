import os
import time
from enum import Enum
from typing import Optional
from dotenv import load_dotenv
from groq import Groq, APIError
from pydantic import BaseModel, Field, ValidationError

# Cargamos las variables del archivo .env (GROQ_API_KEY)
load_dotenv()

# ==============================================================================
# 1. CONTRATO DE DATOS (Pydantic con Enums, rangos, largos mínimos y regex)
# ==============================================================================
class RolUsuario(str, Enum):
    asesor_pdv = "asesor_pdv"
    supervisor_pais = "supervisor_pais"
    estrategico_fp_moto = "estrategico_fp_moto"

class Pais(str, Enum):
    paraguay = "paraguay"
    uruguay = "uruguay"
    rola_reg = "rola"

class IntencionConsulta(str, Enum):
    consulta_stock = "consulta_stock"
    consulta_ventas = "consulta_ventas"
    consulta_floorshare = "consulta_floorshare"
    reporte_alerta = "reporte_alerta"

class TipoAlerta(str, Enum):
    ninguna = "ninguna"
    stock_critico_2 = "stock_critico_2"   # Umbral <= 2 unidades
    sobrestock_30 = "sobrestock_30"       # Umbral > 30 unidades

class Entrada(BaseModel):
    consulta: str = Field(min_length=5, description="Mensaje de WhatsApp del usuario")
    rol: RolUsuario
    pais: Pais
    id_pdv_asignado: Optional[str] = Field(
        default=None,
        pattern=r"^PDV-\d{6}$",
        description="Código oficial del PDV según itinerario (ej. PDV-014587)"
    )

class Salida(BaseModel):
    intencion: IntencionConsulta
    pdv_mencionado: str = Field(min_length=2, description="Nombre de fantasía del local")
    modelo_equipo: str = Field(min_length=2, description="Modelo Motorola consultado")
    tipo_alerta: TipoAlerta
    requiere_aprobacion_humana: bool
    confianza: float = Field(ge=0.0, le=1.0)

# ==============================================================================
# 2. PROMPT COMO PLANTILLA SEPARADA
# ==============================================================================
PROMPT = """Sos el motor de análisis semántico del asistente comercial de FourPlayers y Motorola (Paraguay, Uruguay y ROLA).
Tu tarea es interpretar el mensaje de WhatsApp de un usuario de campo y extraer los parámetros estructurados para consultar las bases de Mototalk y Base de Cluster.

Reglas de negocio:
- intencion: elegí entre "consulta_stock", "consulta_ventas", "consulta_floorshare" o "reporte_alerta".
- pdv_mencionado: extraé el nombre de fantasía del punto de venta (si no se menciona, poné "No especificado").
- modelo_equipo: extraé el modelo de celular Motorola mencionado (ej. "Moto G85", "Edge 50").
- tipo_alerta: si el mensaje reporta que quedan 2 unidades o menos, usá "stock_critico_2". Si reporta más de 30 unidades, usá "sobrestock_30". Caso contrario, "ninguna".
- requiere_aprobacion_humana: debe ser true si tipo_alerta es distinto de "ninguna" o si el rol es "asesor_pdv" pidiendo datos fuera de su alcance; caso contrario false.
- confianza: número decimal entre 0.0 y 1.0 indicando qué tan claro es el mensaje.

Respondé ÚNICAMENTE con un objeto JSON válido con las claves exactas:
intencion, pdv_mencionado, modelo_equipo, tipo_alerta, requiere_aprobacion_humana, confianza.

Datos de la consulta:
- Rol del usuario: {rol}
- País/Región: {pais}
- PDV asignado por itinerario: {id_pdv_asignado}
- Mensaje de WhatsApp: "{consulta}"
"""

# ==============================================================================
# 3. LLAMADA REAL AL MODELO (Groq - openai/gpt-oss-120b)
# ==============================================================================
MODELO_LLM = "openai/gpt-oss-120b"

def llamar_modelo(prompt_texto: str) -> tuple[str, dict]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Falta configurar GROQ_API_KEY en el archivo .env")

    client = Groq(api_key=api_key)
    respuesta = client.chat.completions.create(
        model=MODELO_LLM,
        messages=[{"role": "user", "content": prompt_texto}],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    uso_meta = respuesta.usage
    uso = {
        "tokens_entrada": uso_meta.prompt_tokens if uso_meta else 0,
        "tokens_salida": uso_meta.completion_tokens if uso_meta else 0,
    }
    return respuesta.choices[0].message.content, uso

# ==============================================================================
# 4. MEDICIÓN DE COSTOS (Precios públicos openai/gpt-oss-120b en Groq USD por 1M tokens)
# ==============================================================================
PRECIO_ENTRADA = 0.15
PRECIO_SALIDA = 0.75

def calcular_costo(uso: dict) -> float:
    return (
        (uso["tokens_entrada"] / 1_000_000) * PRECIO_ENTRADA
        + (uso["tokens_salida"] / 1_000_000) * PRECIO_SALIDA
    )

# ==============================================================================
# 5. MOTOR PRINCIPAL CON MANEJO DE ERRORES Y REINTENTO
# ==============================================================================
def motor(entrada: Entrada) -> dict:
    inicio = time.perf_counter()
    prompt_formateado = PROMPT.format(**entrada.model_dump(mode="json"))

    # Primer intento de llamada a la API
    try:
        texto, uso_total = llamar_modelo(prompt_formateado)
    except (APIError, ValueError, Exception) as e:
        latencia_ms = (time.perf_counter() - inicio) * 1000
        return {
            "error": "Error de comunicación con la API del LLM",
            "detalle": str(e),
            "latencia_ms": round(latencia_ms, 1),
        }

    # Validación con Pydantic + 1 reintento si la salida del LLM es inválida
    try:
        salida = Salida.model_validate_json(texto)
    except ValidationError as primer_error:
        campos_fallidos = [err["loc"] for err in primer_error.errors()]
        prompt_reintento = (
            f"{prompt_formateado}\n\n"
            f"ATENCIÓN: Tu respuesta anterior falló la validación en los campos {campos_fallidos}. "
            f"Detalle del error: {primer_error}. Corregilo y devolvé solo el JSON válido."
        )
        try:
            texto_reintento, uso_reintento = llamar_modelo(prompt_reintento)
            uso_total["tokens_entrada"] += uso_reintento["tokens_entrada"]
            uso_total["tokens_salida"] += uso_reintento["tokens_salida"]
            salida = Salida.model_validate_json(texto_reintento)
        except ValidationError as segundo_error:
            latencia_ms = (time.perf_counter() - inicio) * 1000
            return {
                "error": "Validación estructural fallida tras reintento",
                "campos_invalidos": [err["loc"] for err in segundo_error.errors()],
                "tokens": uso_total,
                "costo_usd": round(calcular_costo(uso_total), 6),
                "latencia_ms": round(latencia_ms, 1),
            }
        except Exception as e:
            latencia_ms = (time.perf_counter() - inicio) * 1000
            return {
                "error": "Error de API en el reintento",
                "detalle": str(e),
                "latencia_ms": round(latencia_ms, 1),
            }

    latencia_ms = (time.perf_counter() - inicio) * 1000
    costo = calcular_costo(uso_total)

    return {
        "salida": salida.model_dump(mode="json"),
        "tokens": uso_total,
        "costo_usd": round(costo, 6),
        "latencia_ms": round(latencia_ms, 1),
    }

# ==============================================================================
# PRUEBAS LOCALES DE EJECUCIÓN
# ==============================================================================
if __name__ == "__main__":
    print("--- PRUEBA 1: Consulta normal de stock de un asesor ---")
    caso_1 = Entrada(
        consulta="Hola, fijate si quedó stock del Moto G85 en Nissei Asunción",
        rol=RolUsuario.asesor_pdv,
        pais=Pais.paraguay,
        id_pdv_asignado="PDV-014587",
    )
    print(motor(caso_1))

    print("\n--- PRUEBA 2: Alerta preventiva (<= 2 unidades) ---")
    caso_2 = Entrada(
        consulta="Che en Shopping del Sol nos quedaron solo 2 unidades del Edge 50 Fusion",
        rol=RolUsuario.supervisor_pais,
        pais=Pais.paraguay,
    )
    print(motor(caso_2))