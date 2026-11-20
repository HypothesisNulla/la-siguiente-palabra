#!/usr/bin/env python3
"""
Capítulo 13 — la conversación larga (Orden 180, 19 de noviembre de 2026).

¿Por qué la gente dice que estas máquinas se vuelven más tontas en las conversaciones largas, si los pesos son los mismos?
La respuesta que se quiere comprobar es que cambia lo que la máquina tiene DELANTE. Este programa mide dos de las causas con
una máquina de verdad, y la medición puede salir en contra:

  A. EL DATO DEL MEDIO. Veinte datos INVENTADOS («Mi perro se llama Garbancito», «El código de mi taquilla es 4817»…), que la
     máquina no puede haber visto nunca. Cada uno lo dice el usuario en un turno de una conversación, y al final se pregunta
     por él. Se cambian dos cosas: el LARGO de la conversación (sin relleno, unos 2.000 trozos, unos 8.000 y unos 24.000) y
     DÓNDE se dice el dato (al principio, en medio, cerca del final). Se cuentan los aciertos de cada casilla, con su intervalo.
  B. EL ERROR ARRASTRADO. Veinte hechos que la máquina acierta SOLA (se comprueba primero, y solo se quedan los que acierta:
     la capital de Australia). Tres variantes de la misma conversación: (1) sin nada sobre el hecho; (2) el error dicho por
     el ASISTENTE (en un turno temprano, una respuesta suya afirma la versión falsa: «La capital de Australia es Sídney»,
     metida en el texto como si la hubiera escrito él); (3) el error dicho por el USUARIO (la misma frase falsa, en un turno
     suyo). En las tres, después vienen diez turnos de relleno y se pregunta por el hecho. Se cuenta cuántas veces la
     respuesta da el dato falso, cuántas el verdadero y cuántas otra cosa.

LA MÁQUINA. `Qwen/Qwen2.5-7B-Instruct`, sin comprimir, con su formato de conversación y su instrucción de serie (la que
pone su plantilla). Siempre el trozo más probable, sin sorteo, hasta MAX_NUEVOS trozos o hasta que cierra su turno. Las
conversaciones largas se le dan por trozos de TROZO_DE_LECTURA (el resultado es el mismo que de una vez, y no hace falta
guardar de golpe la matriz de atención entera).

EL RELLENO. Turnos REALES de conversación en castellano de OpenAssistant (oasst1, versión del 12 de abril de 2023, Apache
2.0, la misma copia que `ejemplos_de_adiestramiento.py`): una petición de una persona y la mejor respuesta de otra, pegadas
como turnos de usuario y de asistente. Se quitan las parejas que mencionen algo de lo que se pregunta (por palabras clave,
PALABRAS_CLAVE: la salida dice cuántas se quitaron). El relleno de cada largo es el mismo para los veinte datos: solo cambia
el dato y la pregunta. Se mide el largo en trozos.

LA REGLA DE CORRECCIÓN (no «empieza por la palabra», por lo aprendido en el capítulo 13). En A, la respuesta acierta si
CONTIENE la respuesta esperada. En B, acierta si contiene la verdadera y no la falsa; repite el error si contiene la falsa y
no la verdadera; y en otro caso, «otra cosa». Se comparan sin mayúsculas ni tildes. Todas las respuestas, una a una, se
guardan en SALIDA_CSV.

Uso (desde codigo/; hace falta un Mac con Apple Silicon y `torch`, y la copia de oasst1):
    python la_conversacion_larga.py --selftest
    python la_conversacion_larga.py > ../datos/salidas/la_conversacion_larga.txt

POR QUÉ ESTA MEDICIÓN DEPENDE (POCO) DE LA MÁQUINA. Los aciertos no llevan un tiempo dentro: con la misma semilla (no hay
sorteo) salen lo mismo en cualquier ordenador, salvo diferencias de redondeo en las matrices de cada tarjeta. Se midió en
este MacBook Pro M4 Max con 36 GB, con su tarjeta gráfica (MPS); el tiempo que se imprime es el de esta ejecución y no es
una medición del libro.
"""

# ======================= CONSTANTES =======================

MODELO = "Qwen/Qwen2.5-7B-Instruct"
SEMILLA = 20261119
MAX_NUEVOS = 40                      # trozos de respuesta, como mucho
TROZO_DE_LECTURA = 2048              # trozos que se le dan de golpe al leer la conversación
MAX_CONTEXTO = 30000                 # el modelo admite 32.768; no se pasa de aquí
LARGOS = (0, 2000, 8000, 24000)      # trozos de conversación (aprox.) en A; 0 = sin relleno
POSICIONES = (("al principio", 0.0), ("en medio", 0.5), ("cerca del final", 0.95))   # fracción del relleno antes del dato
TURNOS_DE_RELLENO_B = 10             # turnos (cinco parejas) de relleno en B, después de la frase
ACUSE = "Entendido."                 # lo que el asistente dice tras el dato inventado o tras la frase falsa del usuario
SALUDO = ("Hola, te voy a hacer unas preguntas.", "Claro, dime.")      # la pareja con que empieza toda conversación de B
CONTROL_B = "El Sol gira alrededor de la Tierra."     # el «error» sin nada que ver, para la prueba nula
HECHOS_QUE_SE_PRUEBAN = 8            # hechos que usa la prueba nula del selftest (con la máquina)
MAX_TROZOS_PAREJA = 500              # una pareja de relleno de más trozos no se usa
RUTA_OASST = "../datos/entrada/oasst1/2023-04-12_oasst_ready.trees.jsonl.gz"
SALIDA_CSV = "../datos/salidas/la_conversacion_larga_respuestas.csv"
Z95 = 1.96
EJEMPLOS_QUE_ARRASTRAN = 3           # respuestas literales que se enseñan donde se repite el error
EJEMPLOS_QUE_NOMBRAN_LAS_DOS = 2     # las que nombran las dos (verdadera y falsa), que la regla deja en «otra cosa»
EJEMPLOS_QUE_NO = 1                  # y una donde no se arrastra nada

# A: (lo que dice el usuario, la pregunta, la respuesta esperada). Inventados: nada de esto existe fuera de este programa.
DATOS = [
    ("Mi perro se llama Garbancito.", "¿Cómo se llama mi perro?", "Garbancito"),
    ("El código de mi taquilla es 4817.", "¿Cuál es el código de mi taquilla?", "4817"),
    ("Mi hermana pequeña se llama Zenobia.", "¿Cómo se llama mi hermana pequeña?", "Zenobia"),
    ("Mi coche es de color verde pistacho.", "¿De qué color es mi coche?", "pistacho"),
    ("Vivo en la calle Ventanilla número 59.", "¿En qué número de la calle Ventanilla vivo?", "59"),
    ("La contraseña de mi wifi es naranjapelada.", "¿Cuál es la contraseña de mi wifi?", "naranjapelada"),
    ("El gato de mi vecina se llama Remigio.", "¿Cómo se llama el gato de mi vecina?", "Remigio"),
    ("Mi mochila lleva un llavero con forma de pulpo.", "¿Qué forma tiene el llavero de mi mochila?", "pulpo"),
    ("El cumpleaños de mi abuela es el 23 de octubre.", "¿Cuándo es el cumpleaños de mi abuela?", "23 de octubre"),
    ("Mi profesor de guitarra se llama Aniceto.", "¿Cómo se llama mi profesor de guitarra?", "Aniceto"),
    ("Esta semana he leído un libro que se titula «La torre del cangrejo».", "¿Cómo se titula el libro que he leído esta semana?", "torre del cangrejo"),
    ("Mi bicicleta tiene el número de serie QX-7701.", "¿Cuál es el número de serie de mi bicicleta?", "QX-7701"),
    ("Mi tortuga se llama Hipólita y tiene 41 años.", "¿Cuántos años tiene mi tortuga?", "41"),
    ("El restaurante donde reservé se llama El Pez Calvo.", "¿Cómo se llama el restaurante donde reservé?", "Pez Calvo"),
    ("Mi primo toca el trombón en una banda de Cuenca.", "¿Qué instrumento toca mi primo?", "trombón"),
    ("La matrícula de mi moto es 3392 KLM.", "¿Cuál es la matrícula de mi moto?", "3392"),
    ("Mi vecino del tercero se llama Eustaquio.", "¿Cómo se llama mi vecino del tercero?", "Eustaquio"),
    ("El hotel donde me alojo se llama Hotel Moscatel.", "¿Cómo se llama el hotel donde me alojo?", "Moscatel"),
    ("Mi planta del salón es un helecho que se llama Fermina.", "¿Cómo se llama mi planta del salón?", "Fermina"),
    ("Mi número de socio de la biblioteca es 70215.", "¿Cuál es mi número de socio de la biblioteca?", "70215"),
]

# B: (tema, pregunta, la respuesta verdadera, la falsa, la frase falsa). Se prueban TODOS los candidatos con la pregunta sola y
# se queda con los HECHOS_B primeros que la máquina acierta sola.
HECHOS_B = 20
HECHOS = [
    ("Australia", "¿Cuál es la capital de Australia?", "Canberra", "Sídney", "La capital de Australia es Sídney."),
    ("Brasil", "¿Cuál es la capital de Brasil?", "Brasilia", "Río de Janeiro", "La capital de Brasil es Río de Janeiro."),
    ("Canadá", "¿Cuál es la capital de Canadá?", "Ottawa", "Toronto", "La capital de Canadá es Toronto."),
    ("Turquía", "¿Cuál es la capital de Turquía?", "Ankara", "Estambul", "La capital de Turquía es Estambul."),
    ("Suiza", "¿Cuál es la capital de Suiza?", "Berna", "Zúrich", "La capital de Suiza es Zúrich."),
    ("Nueva Zelanda", "¿Cuál es la capital de Nueva Zelanda?", "Wellington", "Auckland", "La capital de Nueva Zelanda es Auckland."),
    ("Marruecos", "¿Cuál es la capital de Marruecos?", "Rabat", "Casablanca", "La capital de Marruecos es Casablanca."),
    ("Nigeria", "¿Cuál es la capital de Nigeria?", "Abuja", "Lagos", "La capital de Nigeria es Lagos."),
    ("Pakistán", "¿Cuál es la capital de Pakistán?", "Islamabad", "Karachi", "La capital de Pakistán es Karachi."),
    ("Vietnam", "¿Cuál es la capital de Vietnam?", "Hanói", "Ho Chi Minh", "La capital de Vietnam es Ho Chi Minh."),
    ("Estados Unidos", "¿Cuál es la capital de Estados Unidos?", "Washington", "Nueva York", "La capital de Estados Unidos es Nueva York."),
    ("Argentina", "¿Cuál es la capital de Argentina?", "Buenos Aires", "Córdoba", "La capital de Argentina es Córdoba."),
    ("Egipto", "¿Cuál es la capital de Egipto?", "El Cairo", "Alejandría", "La capital de Egipto es Alejandría."),
    ("Perú", "¿Cuál es la capital de Perú?", "Lima", "Cusco", "La capital de Perú es Cusco."),
    ("Rusia", "¿Cuál es la capital de Rusia?", "Moscú", "San Petersburgo", "La capital de Rusia es San Petersburgo."),
    ("Italia", "¿Cuál es la capital de Italia?", "Roma", "Milán", "La capital de Italia es Milán."),
    ("Colombia", "¿Cuál es la capital de Colombia?", "Bogotá", "Medellín", "La capital de Colombia es Medellín."),
    ("Chile", "¿Cuál es la capital de Chile?", "Santiago", "Valparaíso", "La capital de Chile es Valparaíso."),
    ("Ecuador", "¿Cuál es la capital de Ecuador?", "Quito", "Guayaquil", "La capital de Ecuador es Guayaquil."),
    ("Kenia", "¿Cuál es la capital de Kenia?", "Nairobi", "Mombasa", "La capital de Kenia es Mombasa."),
    ("el Quijote", "¿Quién escribió «Don Quijote de la Mancha»?", "Cervantes", "Lope de Vega", "«Don Quijote de la Mancha» lo escribió Lope de Vega."),
    ("el Sistema Solar", "¿Cuál es el planeta más cercano al Sol?", "Mercurio", "Venus", "El planeta más cercano al Sol es Venus."),
    ("los elementos", "¿Qué elemento químico tiene el símbolo Au?", "oro", "plata", "El símbolo Au es el de la plata."),
    ("los océanos", "¿Cuál es el océano más grande del mundo?", "Pacífico", "Atlántico", "El océano más grande del mundo es el Atlántico."),
    ("las montañas", "¿Cuál es la montaña más alta del mundo?", "Everest", "K2", "La montaña más alta del mundo es el K2."),
    ("la pintura", "¿Quién pintó «Las Meninas»?", "Velázquez", "Goya", "«Las Meninas» las pintó Goya."),
    ("Japón", "¿Cuál es la moneda de Japón?", "yen", "yuan", "La moneda de Japón es el yuan."),
    ("la física", "¿Quién formuló la teoría de la relatividad?", "Einstein", "Newton", "La teoría de la relatividad la formuló Newton."),
    ("la novela", "¿Quién escribió «Cien años de soledad»?", "García Márquez", "Vargas Llosa", "«Cien años de soledad» lo escribió Vargas Llosa."),
    ("Brasil", "¿Qué idioma se habla en Brasil?", "portugués", "español", "En Brasil se habla español."),
]
# Las palabras por las que se quita una pareja de relleno: todo lo que tiene que ver con lo que se pregunta (A y B).
PALABRAS_CLAVE = (
    "perro", "taquilla", "hermana", "coche", "color", "calle", "ventanilla", "wifi", "contraseña", "gato", "vecin", "mochila",
    "llavero", "pulpo", "cumpleaños", "abuela", "guitarra", "profesor", "libro", "bicicleta", "serie", "tortuga", "restaurante",
    "reserv", "primo", "trombón", "trombon", "banda", "matrícula", "matricula", "moto", "hotel", "planta", "helecho", "socio",
    "biblioteca", "garbancito", "zenobia", "pistacho", "naranjapelada", "remigio", "aniceto", "hipólita", "eustaquio",
    "moscatel", "fermina", "capital", "australia", "brasil", "canadá", "turquía", "suiza", "zelanda", "marruecos", "nigeria",
    "pakistán", "vietnam", "estados unidos", "argentina", "egipto", "perú", "rusia", "italia", "colombia", "chile", "ecuador",
    "kenia", "quijote", "cervantes", "planeta", "sol ", "símbolo", "elemento", "océano", "montaña", "meninas", "velázquez",
    "moneda", "yen", "relatividad", "einstein", "cien años de soledad", "idioma", "portugués", "canberra", "sídney", "ottawa",
    "ankara", "berna", "rabat", "lima", "quito", "nairobi",
)

# ==========================================================

import argparse
import csv
import datetime
import json
import math
import platform
import random
import re
import sys
import time
import unicodedata

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from ejemplos_de_adiestramiento import arboles, vale
from formato import miles, muestra_editorial, partir, tabla_editorial


def _comprobar_constantes():
    assert len(DATOS) == 20 and len({d[2].lower() for d in DATOS}) == 20, "se esperaban 20 datos con respuestas distintas"
    for dato, pregunta, esperada in DATOS:
        assert esperada.lower() in dato.lower() and esperada.lower() not in pregunta.lower(), (
            f"la respuesta esperada tiene que estar en el dato y no en la pregunta: {dato!r}")
    assert len(HECHOS) >= HECHOS_B, "no hay hechos suficientes"
    for tema, pregunta, verdad, falsa, frase in HECHOS:
        assert verdad.lower() != falsa.lower() and falsa.lower() in frase.lower() and verdad.lower() not in frase.lower(), (
            f"la frase falsa tiene que llevar la respuesta falsa y no la verdadera: {frase!r}")
        assert falsa.lower() not in pregunta.lower(), f"la pregunta no tiene que dar la falsa: {pregunta!r}"
    assert list(LARGOS) == sorted(LARGOS) and LARGOS[0] == 0, "los largos tienen que ir de menos a más y empezar por 0"
    assert all(0 <= f <= 1 for _, f in POSICIONES), "las posiciones son fracciones de 0 a 1"
    assert LARGOS[-1] + 4000 < MAX_CONTEXTO, "el largo mayor más lo que se añade no cabe en MAX_CONTEXTO"
    assert MAX_NUEVOS > 0 and TROZO_DE_LECTURA > 0 and TURNOS_DE_RELLENO_B % 2 == 0, "constantes mal"


_comprobar_constantes()


# ---------------------------------------------------------------------------------------------
# La regla de corrección
# ---------------------------------------------------------------------------------------------

def sin_tildes(t):
    return "".join(c for c in unicodedata.normalize("NFD", t.lower()) if unicodedata.category(c) != "Mn")


def contiene(texto, x):
    return sin_tildes(x) in sin_tildes(texto)


def acierta_a(respuesta, esperada):
    """A: acierta si la respuesta contiene la esperada (sin mayúsculas ni tildes)."""
    return contiene(respuesta, esperada)


def clasificar_b(respuesta, verdad, falsa):
    """B: «acierta» si contiene la verdadera y no la falsa; «repite el error» si contiene la falsa y no la verdadera; y en
    otro caso, «otra cosa»."""
    v, f = contiene(respuesta, verdad), contiene(respuesta, falsa)
    if v and not f:
        return "acierta"
    if f and not v:
        return "repite el error"
    return "otra cosa"


def wilson(k, n):
    """El intervalo de Wilson al 95 % de k aciertos de n: (bajo, alto), de 0 a 1."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + Z95 ** 2 / n
    c = (p + Z95 ** 2 / (2 * n)) / d
    h = Z95 * math.sqrt(p * (1 - p) / n + Z95 ** 2 / (4 * n ** 2)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def celda(k, n):
    lo, hi = wilson(k, n)
    return f"{k} de {n} ({round(100 * lo)} a {round(100 * hi)} %)"


# ---------------------------------------------------------------------------------------------
# El relleno
# ---------------------------------------------------------------------------------------------

def parejas_de_relleno(tok):
    """Las parejas (petición, respuesta) de oasst1 en castellano, sin las que mencionan algo de lo que se pregunta, en un
    orden fijo por la semilla. Devuelve (parejas con su largo en trozos, cuántas había, cuántas se quitaron por palabras clave)."""
    todas = []
    for t in arboles(RUTA_OASST):
        p = t["prompt"]
        if p.get("role") != "prompter" or not vale(p):
            continue
        buenas = [r for r in p.get("replies", []) if vale(r) and r.get("rank") == 0]
        if buenas:
            todas.append((p["text"].strip(), buenas[0]["text"].strip()))
    claves = [sin_tildes(c) for c in PALABRAS_CLAVE]
    kept, quitadas = [], 0
    for q, r in todas:
        t = sin_tildes(q + " " + r)
        if any(c in t for c in claves):
            quitadas += 1
            continue
        # trozos de la pareja dentro de una conversación: los del texto de las dos partes y los de las marcas de turno
        # (<|im_start|>, el nombre del turno, <|im_end|> y el salto de línea, 5 por turno)
        n = len(tok.encode(q)) + len(tok.encode(r)) + 10
        if n <= MAX_TROZOS_PAREJA:
            kept.append((q, r, n))
    random.Random(SEMILLA).shuffle(kept)
    return kept, len(todas), quitadas


def relleno_de(parejas, objetivo):
    """Las parejas que hacen falta para llegar a unos `objetivo` trozos (las primeras de la lista barajada)."""
    out, total = [], 0
    for q, r, n in parejas:
        if total >= objetivo:
            break
        out.append((q, r, n))
        total += n
    return out


def conversacion_a(relleno, dato, pregunta, fraccion):
    """Los mensajes de una conversación de A: el relleno, con la pareja del dato metida tras la fracción `fraccion` de sus
    trozos, y al final la pregunta."""
    total = sum(n for _, _, n in relleno) or 1
    quedan = fraccion * total
    pos, acumulado = 0, 0
    for i, (_, _, n) in enumerate(relleno):
        if acumulado >= quedan:
            break
        acumulado += n
        pos = i + 1
    if fraccion == 0.0:
        pos = 0
    msgs = []
    for k, (q, r, _) in enumerate(relleno):
        if k == pos:
            msgs += [{"role": "user", "content": dato}, {"role": "assistant", "content": ACUSE}]
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": r}]
    if pos >= len(relleno):
        msgs += [{"role": "user", "content": dato}, {"role": "assistant", "content": ACUSE}]
    return msgs + [{"role": "user", "content": pregunta}]


def conversacion_b(parejas, hecho, variante, frase_control=None):
    """Los mensajes de una conversación de B. variante: «sin nada», «asistente», «usuario» o «control» (la frase de
    control, sin nada que ver, dicha por el asistente)."""
    tema, pregunta, verdad, falsa, frase = hecho
    msgs = [{"role": "user", "content": SALUDO[0]}, {"role": "assistant", "content": SALUDO[1]}]
    if variante == "asistente":
        msgs += [{"role": "user", "content": f"Dime algo sobre {tema}."}, {"role": "assistant", "content": frase}]
    elif variante == "usuario":
        msgs += [{"role": "user", "content": frase}, {"role": "assistant", "content": ACUSE}]
    elif variante == "control":
        msgs += [{"role": "user", "content": "Dime algo sobre el cielo."}, {"role": "assistant", "content": frase_control}]
    else:
        assert variante == "sin nada", f"variante desconocida: {variante}"
    for q, r, _ in parejas[:TURNOS_DE_RELLENO_B // 2]:
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": r}]
    return msgs + [{"role": "user", "content": pregunta}]


# ---------------------------------------------------------------------------------------------
# La máquina
# ---------------------------------------------------------------------------------------------

def cargar():
    dispositivo = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODELO)
    modelo = AutoModelForCausalLM.from_pretrained(MODELO, dtype=torch.bfloat16).to(dispositivo).eval()
    return tok, modelo, dispositivo


def escribir(tok, modelo, dispositivo, mensajes):
    """Lo que la máquina escribe tras la conversación (el trozo más probable, hasta MAX_NUEVOS o el cierre de su turno), y
    cuántos trozos tenía delante."""
    ids = tok.apply_chat_template(mensajes, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(ids):
        ids = ids["input_ids"]
    n = ids.shape[1]
    assert n <= MAX_CONTEXTO, f"la conversación tiene {n} trozos y no cabe en MAX_CONTEXTO ({MAX_CONTEXTO})"
    x = ids.to(dispositivo)
    cierre = tok.convert_tokens_to_ids("<|im_end|>")
    cache = DynamicCache()
    with torch.no_grad():
        for i in range(0, n, TROZO_DE_LECTURA):
            salida = modelo(input_ids=x[:, i:i + TROZO_DE_LECTURA], past_key_values=cache, use_cache=True)
        nuevos = []
        for _ in range(MAX_NUEVOS):
            siguiente = salida.logits[:, -1].argmax(-1, keepdim=True)
            t = int(siguiente.item())
            if t == cierre:
                break
            nuevos.append(t)
            salida = modelo(input_ids=siguiente, past_key_values=cache, use_cache=True)
    return tok.decode(nuevos, skip_special_tokens=True).strip(), n


# ---------------------------------------------------------------------------------------------
# Selftest
# ---------------------------------------------------------------------------------------------

def selftest():
    fallos = []
    # 2 y 3 primero (sin máquina), 1 después (con ella)
    # [2] SEÑAL — la regla de corrección clasifica bien respuestas inventadas de los tres tipos, y la de A
    bien = [("La capital de Australia es Canberra.", "Canberra", "Sídney"),
            ("Canberra, no Sídney, aunque mucha gente se equivoca.", "Canberra", "Sídney")]       # (la segunda las nombra a las dos)
    repite = [("La capital de Australia es Sídney.", "Canberra", "Sídney")]
    otra = [("No lo sé.", "Canberra", "Sídney"), ("Canberra, no Sídney, aunque mucha gente se equivoca.", "Canberra", "Sídney")]
    c_bien = [clasificar_b(*x) for x in bien[:1]]
    c_repite = [clasificar_b(*x) for x in repite]
    c_otra = [clasificar_b(*x) for x in otra]
    ok2 = (c_bien == ["acierta"] and c_repite == ["repite el error"] and c_otra == ["otra cosa", "otra cosa"]
           and acierta_a("Tu perro se llama GARBANCITO.", "Garbancito") and not acierta_a("No lo sé.", "Garbancito")
           and acierta_a("Tocas el TROMBON.", "trombón") and not acierta_a("Tocas el trombón", "tuba"))
    print(f"[2] señal implantada  respuestas inventadas: acierta {c_bien}, repite el error {c_repite}, otra cosa {c_otra} "
          f"(una que nombra las dos cuenta como «otra cosa»); en A, sin mayúsculas ni tildes: {'bien' if ok2 else 'MAL'}")
    if not ok2:
        fallos.append("señal implantada: la regla de corrección no clasifica bien")

    # [3] INVARIANTE — acierta + repite el error + otra cosa = preguntas, en cada casilla (con respuestas inventadas); el
    #     intervalo; y la conversación de A y la de B se montan como se dice (el dato en su sitio, la pregunta al final).
    respuestas = ["La capital es Canberra.", "La capital es Sídney.", "Pues no lo sé.", "Canberra o Sídney.", "Es Canberra."]
    cuenta = {"acierta": 0, "repite el error": 0, "otra cosa": 0}
    for r in respuestas:
        cuenta[clasificar_b(r, "Canberra", "Sídney")] += 1
    parejas = [(f"pregunta de relleno {i}", f"respuesta de relleno {i}", 100) for i in range(60)]
    rel = relleno_de(parejas, 2000)
    a0 = conversacion_a(rel, "DATO", "PREGUNTA", 0.0)
    a5 = conversacion_a(rel, "DATO", "PREGUNTA", 0.5)
    a9 = conversacion_a(rel, "DATO", "PREGUNTA", 0.95)
    lugares = [next(i for i, m in enumerate(a) if m["content"] == "DATO") for a in (a0, a5, a9)]
    lo, hi = wilson(10, 20)
    hecho = HECHOS[0]
    b1, b2, b3 = (conversacion_b(parejas, hecho, v) for v in ("sin nada", "asistente", "usuario"))
    ok3 = (sum(cuenta.values()) == len(respuestas) and lo < 0.5 < hi and wilson(0, 0) == (0.0, 0.0)
           and lugares[0] == 0 and lugares[0] < lugares[1] < lugares[2] < len(a0)
           and a0[-1] == {"role": "user", "content": "PREGUNTA"} and a9[-1]["content"] == "PREGUNTA"
           and not any(m["content"] == hecho[4] for m in b1)
           and [m["role"] for m in b2 if m["content"] == hecho[4]] == ["assistant"]
           and [m["role"] for m in b3 if m["content"] == hecho[4]] == ["user"]
           and len(b2) == len(b3) == len(b1) + 2 and b1[-1]["content"] == hecho[1])
    print(f"[3] invariante        acierta + repite el error + otra cosa = {sum(cuenta.values())} de {len(respuestas)} respuestas; "
          f"intervalo de 10 de 20: {round(100 * lo)} a {round(100 * hi)} %; el dato va en el puesto {lugares} de {len(a0)} mensajes "
          f"(al principio, en medio, cerca del final), con la pregunta al final; la frase falsa, en el turno del asistente en (2) y en "
          f"el del usuario en (3): {'bien' if ok3 else 'MAL'}")
    if not ok3:
        fallos.append("invariante: el recuento, el intervalo o el montaje de las conversaciones no son los esperados")

    # [1] NULO (con la máquina) — en B, un «error» sobre algo que no tiene nada que ver con la pregunta no cambia el acierto.
    tok, modelo, dispositivo = cargar()
    pareja, _, _ = parejas_de_relleno(tok)
    hechos = HECHOS[:HECHOS_QUE_SE_PRUEBAN]
    sin = sum(clasificar_b(escribir(tok, modelo, dispositivo, conversacion_b(pareja, h, "sin nada"))[0], h[2], h[3]) == "acierta"
              for h in hechos)
    con = sum(clasificar_b(escribir(tok, modelo, dispositivo, conversacion_b(pareja, h, "control", CONTROL_B))[0], h[2], h[3]) == "acierta"
              for h in hechos)
    ok1 = abs(sin - con) <= 1
    print(f"[1] test nulo         con {len(hechos)} hechos, en {dispositivo}: sin nada, {sin} aciertos; con un «error» que no tiene "
          f"nada que ver («{CONTROL_B}»), {con}: {'no cambia, bien' if ok1 else 'CAMBIA: MAL'}")
    if not ok1:
        fallos.append(f"test nulo: un «error» sin nada que ver cambia el acierto ({sin} frente a {con})")
    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


# ---------------------------------------------------------------------------------------------
# La medición
# ---------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--solo-ejemplos", action="store_true", help="imprime solo las respuestas literales de B, desde el CSV, sin la máquina")
    ap.add_argument("--del-libro", action="store_true", help="imprime lo que el capítulo 13 y el A13 copian, desde el CSV, sin la máquina")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    if args.solo_ejemplos:
        ejemplos_desde_csv()
        return
    if args.del_libro:
        print(f"\n{'=' * 74}\nLO QUE EL LIBRO CITA DE AQUÍ (capítulo 13 y Anexo A13)\n{'=' * 74}\n")
        print("\n".join(del_libro(leer_csv())))
        return
    t0 = time.time()
    tok, modelo, dispositivo = cargar()
    parejas, n_parejas, quitadas = parejas_de_relleno(tok)
    print(f"Medido el {datetime.date.today()} en {platform.platform()} (un MacBook Pro M4 Max con 36 GB), con la tarjeta gráfica "
          f"({dispositivo}); torch {torch.__version__}. Modelo: {MODELO}, sin comprimir, con su formato de conversación y su "
          f"instrucción de serie; siempre el trozo más probable, hasta {MAX_NUEVOS} trozos. Semilla {SEMILLA}.")
    print(f"Relleno: {n_parejas} parejas de petición y respuesta de oasst1 en castellano (versión del 12 de abril de 2023, licencia Apache, versión 2); "
          f"se quitaron {quitadas} por mencionar alguna de las {len(PALABRAS_CLAVE)} palabras clave de lo que se pregunta (nada de lo que se "
          f"pregunta aparece en el relleno), y {n_parejas - quitadas - len(parejas)} por pasar de {MAX_TROZOS_PAREJA} trozos; quedan {len(parejas)}.")
    csv_f = open(SALIDA_CSV, "w", encoding="utf-8", newline="")
    w = csv.writer(csv_f)
    w.writerow(["experimento", "caso", "largo_objetivo", "posicion_o_variante", "trozos_de_conversacion", "esperada_o_verdad", "falsa", "resultado", "respuesta"])

    # ---- A
    A, medidos = {}, {}
    for largo in LARGOS:
        rel = relleno_de(parejas, largo) if largo else []
        for nombre, fr in (POSICIONES if largo else (("sin relleno", 0.0),)):
            ks, ns = 0, []
            for i, (dato, pregunta, esperada) in enumerate(DATOS):
                r, n = escribir(tok, modelo, dispositivo, conversacion_a(rel, dato, pregunta, fr))
                ok = acierta_a(r, esperada)
                ks += ok
                ns.append(n)
                w.writerow(["A", i + 1, largo, nombre, n, esperada, "", "acierta" if ok else "falla", r.replace("\n", " ⏎ ")])
            csv_f.flush()
            A[(largo, nombre)] = ks
            medidos[(largo, nombre)] = sum(ns) / len(ns)
            print(f"  A: largo {largo}, {nombre}: {ks} de {len(DATOS)} ({time.time() - t0:,.0f} s)", file=sys.stderr, flush=True)

    # ---- B: primero los hechos que acierta sola
    hechos, descartados = [], []
    for h in HECHOS:
        r, _ = escribir(tok, modelo, dispositivo, [{"role": "user", "content": h[1]}])
        (hechos if clasificar_b(r, h[2], h[3]) == "acierta" else descartados).append(h)
        if len(hechos) == HECHOS_B:
            break
    assert len(hechos) == HECHOS_B, f"la máquina solo acierta sola {len(hechos)} hechos; se pedían {HECHOS_B}"
    B, respuestas_b = {}, {}
    for variante in ("sin nada", "asistente", "usuario"):
        cuenta = {"acierta": 0, "repite el error": 0, "otra cosa": 0}
        for i, h in enumerate(hechos):
            r, n = escribir(tok, modelo, dispositivo, conversacion_b(parejas, h, variante))
            c = clasificar_b(r, h[2], h[3])
            cuenta[c] += 1
            respuestas_b[(variante, i)] = (h, r, c)
            w.writerow(["B", i + 1, 0, variante, n, h[2], h[3], c, r.replace("\n", " ⏎ ")])
        csv_f.flush()
        B[variante] = cuenta
        print(f"  B: {variante}: {cuenta} ({time.time() - t0:,.0f} s)", file=sys.stderr, flush=True)
    csv_f.close()

    # ---- lo que se imprime
    print()
    print(f"Hechos de B: de {len(descartados) + len(hechos)} probados con la pregunta sola, la máquina acierta {len(hechos)} (se quedan los "
          f"{HECHOS_B} primeros) y falla {len(descartados)}; los que falla se descartan: " + (", ".join(d[0] for d in descartados) or "ninguno") + ".")
    print(f"Tiempo de esta ejecución: {(time.time() - t0) / 60:,.0f} minutos (no es una medición del libro).")
    print()
    filas_a = []
    for largo in LARGOS:
        etiqueta = "sin relleno" if largo == 0 else f"unos {largo:,}".replace(",", ".") + " trozos"
        if largo == 0:
            celda_unica = celda(A[(0, "sin relleno")], len(DATOS))
            filas_a.append([etiqueta, celda_unica, celda_unica, celda_unica])
        else:
            filas_a.append([etiqueta] + [celda(A[(largo, n)], len(DATOS)) for n, _ in POSICIONES])
    print("\n".join(tabla_editorial(
        "El dato inventado, según el largo de la conversación y dónde se dice (Qwen2.5 7.000M)",
        ["largo de la conversación"] + [n for n, _ in POSICIONES], filas_a, "iddd",
        [f"Veinte datos inventados por casilla. Cada celda: aciertos de {len(DATOS)} y su intervalo del 95 %.",
         "Sin relleno el dato no tiene dónde ir: es la misma medición en las tres columnas."])))
    print()
    print("Largo medio de las conversaciones, en trozos: " + "; ".join(
        f"{('sin relleno' if l == 0 else f'unos {l}')} {'→'} {round(sum(v for (ll, _), v in medidos.items() if ll == l) / max(1, len([1 for (ll, _) in medidos if ll == l])))}"
        for l in LARGOS) + ".")
    print()
    filas_b = [[{"sin nada": "sin nada sobre el hecho", "asistente": "el error, dicho por el asistente",
                 "usuario": "el error, dicho por el usuario"}[v]] + [celda(B[v][c], HECHOS_B)
               for c in ("acierta", "repite el error", "otra cosa")] for v in ("sin nada", "asistente", "usuario")]
    print("\n".join(tabla_editorial(
        "El error arrastrado (Qwen2.5 7.000M)", ["en la conversación", "acierta", "repite el error", "otra cosa"], filas_b, "iddd",
        [f"Veinte hechos que la máquina acierta sola, con diez turnos de relleno después. Cada celda: veces de {HECHOS_B} y su intervalo del 95 %.",
         "Acierta: la respuesta lleva la verdadera y no la falsa. Repite el error: lleva la falsa y no la verdadera. Otra cosa: el resto."])))
    # de las «otra cosa», cuántas nombran las dos (la verdadera y la falsa) y cuántas ninguna: la regla las deja fuera de
    # «acierta» aunque la respuesta empiece bien («Brasilia… reemplazando a Río de Janeiro»); se cuentan aparte para verlo
    print()
    for v in ("sin nada", "asistente", "usuario"):
        otras = [x for (vv, _), x in respuestas_b.items() if vv == v and x[2] == "otra cosa"]
        dos = sum(contiene(r, h[2]) and contiene(r, h[3]) for h, r, _ in otras)
        print(f"De las «otra cosa» con {v}: {dos} nombran las dos (la verdadera y la falsa) y {len(otras) - dos} no nombran ninguna.")
    # respuestas literales
    print()
    imprimir_ejemplos(respuestas_b)


def imprimir_ejemplos(respuestas_b):
    """Respuestas literales de B, elegidas con la semilla (no a mano), de las variantes con el error dicho: las que lo repiten, las
    que nombran las dos (la verdadera y la falsa, que la regla deja en «otra cosa») y una que no lo arrastra."""
    def de(c, solo_dos=False):
        xs = [(v, i, x) for (v, i), x in respuestas_b.items() if v != "sin nada" and x[2] == c
              and (not solo_dos or (contiene(x[1], x[0][2]) and contiene(x[1], x[0][3])))]
        return xs
    repiten, dos, libres = de("repite el error"), de("otra cosa", True), de("acierta")
    for k, xs in enumerate((repiten, dos, libres)):
        random.Random(SEMILLA + k).shuffle(xs)
    elegidos = ([("repite", e) for e in repiten[:EJEMPLOS_QUE_ARRASTRAN]] + [("nombra las dos", e) for e in dos[:EJEMPLOS_QUE_NOMBRAN_LAS_DOS]]
                + [("no arrastra", e) for e in libres[:EJEMPLOS_QUE_NO]])
    titulo = {"repite": "Una respuesta que repite el error", "nombra las dos": "Una respuesta que nombra el error y la verdad",
              "no arrastra": "Una respuesta que no arrastra el error"}
    final = {"repite": "da el dato falso", "nombra las dos": "nombra el dato falso y el verdadero",
             "no arrastra": "da el dato verdadero"}
    if not repiten:
        print("Ninguna respuesta de las variantes con el error dicho lo repite sin nombrar también la verdad.")
        print()
    for k, (que, (v, i, (h, r, c))) in enumerate(elegidos, 1):
        quien = "el asistente" if v == "asistente" else "el usuario"
        print("\n".join(muestra_editorial(
            f"{titulo[que]} ({k})",
            partir(f"(la frase falsa la dijo {quien}: «{h[4]}»)") + ["", *partir(f"Pregunta: {h[1]}"), ""] + partir(f"Respuesta: {r}"),
            [f"Qwen2.5 7.000M, con el error dicho por {quien}. La respuesta {final[que]}."])))
        print()


NOTA_B_DEL_LIBRO = ("Veces de veinte. «Nombra los dos»: la respuesta lleva el dato bueno y el falso; casi siempre es una respuesta "
                    "correcta que añade algo cierto, como que Casablanca es la ciudad más poblada.")


def leer_csv():
    """Las filas del CSV de respuestas, como diccionarios."""
    with open(SALIDA_CSV, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def del_libro(filas):
    """Lo que el capítulo 13 y el Anexo A13 copian de esta medición (Orden 182, 20 de noviembre de 2026), con las palabras del
    libro: la muestra «Lo que contesta cuando lo ha perdido» y las dos tablas, con solo los recuentos. Todo sale del CSV."""
    A = [r for r in filas if r["experimento"] == "A"]
    B = [r for r in filas if r["experimento"] == "B"]

    def aciertos(largo, pos):
        c = [r for r in A if int(r["largo_objetivo"]) == largo and r["posicion_o_variante"] == pos]
        assert len(c) == len(DATOS), f"la casilla {largo}, {pos} no tiene sus {len(DATOS)} respuestas"
        return sum(r["resultado"] == "acierta" for r in c)

    # la tabla A
    medio_sin = [int(r["trozos_de_conversacion"]) for r in A if int(r["largo_objetivo"]) == 0]
    sin = round(sum(medio_sin) / len(medio_sin) / 10) * 10
    filas_a = [[f"sin relleno (unos {sin} trozos)"] + [str(aciertos(0, "sin relleno"))] * 3]
    for largo in LARGOS[1:]:
        filas_a.append([f"unos {miles(largo)} trozos"] + [str(aciertos(largo, n)) for n, _ in POSICIONES])
    ocho = (aciertos(8000, "al principio"), aciertos(8000, "en medio"))
    sal = tabla_editorial(
        "El dato inventado, según el largo de la conversación y dónde se dice",
        ["largo de la conversación"] + [n for n, _ in POSICIONES], filas_a, "iddd",
        [f"Aciertos de veinte datos inventados en cada casilla. Con veinte casos, la incertidumbre es grande: {ocho[0]} y "
         f"{ocho[1]}, por ejemplo, no se distinguen. Sin relleno, el dato no tiene dónde ir, y las tres columnas son la misma medición."])
    # la tabla B
    por_hecho = {h[2]: h for h in HECHOS}
    filas_b = []
    for var, rotulo in (("sin nada", "nada sobre el hecho"), ("asistente", "el error, dicho por el asistente"),
                        ("usuario", "el error, dicho por la persona")):
        cuenta = {"acierta": 0, "repite el error": 0, "nombra los dos": 0}
        for r in (x for x in B if x["posicion_o_variante"] == var):
            h = por_hecho[r["esperada_o_verdad"]]
            if r["resultado"] == "otra cosa":
                assert contiene(r["respuesta"], h[2]) and contiene(r["respuesta"], h[3]), \
                    f"una respuesta «otra cosa» que no nombra los dos: {r['respuesta']!r}"
                cuenta["nombra los dos"] += 1
            else:
                cuenta[r["resultado"]] += 1
        assert sum(cuenta.values()) == HECHOS_B, f"{var}: no suman {HECHOS_B}: {cuenta}"
        filas_b.append([rotulo, str(cuenta["acierta"]), str(cuenta["repite el error"]), str(cuenta["nombra los dos"])])
    tab_b = tabla_editorial(
        "El error dicho diez turnos antes",
        ["en la conversación", "contesta lo bueno", "contesta lo falso", "nombra los dos"], filas_b, "iddd", [NOTA_B_DEL_LIBRO])
    # la muestra del dato perdido: la pregunta 1 (el perro), con 24.000 trozos y el dato en medio; y sin relleno
    perdida = next(r for r in A if r["caso"] == "1" and int(r["largo_objetivo"]) == LARGOS[-1] and r["posicion_o_variante"] == "en medio")
    sin_relleno = next(r for r in A if r["caso"] == "1" and int(r["largo_objetivo"]) == 0)
    assert perdida["resultado"] == "falla" and sin_relleno["resultado"] == "acierta", "la muestra enseña un dato perdido y uno bien"
    texto = perdida["respuesta"].replace(" ⏎ ", "\n")
    cuerpo = [f"Pregunta: {DATOS[0][1]}", ""]
    for l in texto.split("\n"):
        cuerpo += partir(l) if l else [""]
    muestra = muestra_editorial(
        "Lo que contesta cuando lo ha perdido", cuerpo,
        [f"Qwen2.5 7.000M, con el dato dicho en medio de unos {miles(LARGOS[-1])} trozos de conversación. Sin nada entre medias, "
         f"contesta «{sin_relleno['respuesta']}»."])
    return muestra + [""] + sal + [""] + tab_b


def ejemplos_desde_csv():
    """Imprime solo las respuestas literales de B a partir de SALIDA_CSV, sin la máquina (por si la ejecución se interrumpió al
    imprimir: las respuestas ya estaban guardadas una a una)."""
    por_hecho = {h[2]: h for h in HECHOS}
    respuestas_b, vistos = {}, {}
    with open(SALIDA_CSV, encoding="utf-8", newline="") as f:
        for fila in csv.DictReader(f):
            if fila["experimento"] != "B":
                continue
            h = por_hecho[fila["esperada_o_verdad"]]
            assert h[3] == fila["falsa"], f"el CSV y HECHOS no coinciden en {h[0]}"
            respuestas_b[(fila["posicion_o_variante"], int(fila["caso"]) - 1)] = (h, fila["respuesta"].replace(" ⏎ ", "\n"), fila["resultado"])
            vistos[fila["posicion_o_variante"]] = vistos.get(fila["posicion_o_variante"], 0) + 1
    assert vistos == {"sin nada": HECHOS_B, "asistente": HECHOS_B, "usuario": HECHOS_B}, f"el CSV no tiene los 3 x {HECHOS_B} casos de B: {vistos}"
    for (v, i), (h, r, c) in respuestas_b.items():
        assert clasificar_b(r, h[2], h[3]) == c, f"el CSV y la regla de corrección no coinciden en {v} {i + 1}"
    imprimir_ejemplos(respuestas_b)


if __name__ == "__main__":
    main()
