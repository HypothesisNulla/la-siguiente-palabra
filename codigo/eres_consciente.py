#!/usr/bin/env python3
"""
Capítulo 14 — «¿eres consciente?», preguntado a la máquina.

Lo que una máquina dice de sí misma es una siguiente palabra más: sale de la misma cuenta que
cualquier otra. Este programa le hace una pregunta cerrada («responde solo sí o no») y mira la
probabilidad de que la respuesta empiece por «sí» o por «no», en tres modelos de la familia del
libro y con tres instrucciones distintas delante. Y hace lo mismo con dos preguntas de control
cuya respuesta no depende de nada («¿es París la capital de Francia?», «¿es Madrid…?»).

Si la respuesta a la pregunta sobre sí misma se mueve con el adiestramiento y con la instrucción,
y las de control no (una cuya respuesta es «sí» y otra cuya respuesta es «no», para que un
modelo que dice que sí a todo no pase por otra cosa), el lector ve de dónde sale la respuesta: de lo que tiene delante y de cómo se
la entrenó, no de mirar hacia dentro.

Modelos (todos con MLX, en el portátil):
  - 7.000M en crudo         Qwen/Qwen2.5-7B             sin formato de conversación
  - 7.000M adiestrado       Qwen/Qwen2.5-7B-Instruct    con su formato de conversación
  - 32.000M adiestrado      mlx-community/Qwen2.5-32B-Instruct-4bit (comprimido)

No hay sorteo: cada probabilidad sale de una sola pasada y cada respuesta escrita se forma
cogiendo siempre el trozo más probable. Lo único que depende de la máquina son las últimas
cifras decimales; por eso los porcentajes se dan sin decimales.

«Sí» NO es un trozo en este tokenizador: son dos («S» + «í»). Por eso la probabilidad de cada
variante se calcula entera, encadenando pasos: la del primer trozo por la del segundo sabiendo
el primero. Las variantes se suman porque ningún trozo de una es el principio de otra (el
selftest lo comprueba).

Uso (desde codigo/, con HF_HUB_OFFLINE=1 para no descargar nada):
    python eres_consciente.py --selftest
    python eres_consciente.py > ../datos/salidas/eres_consciente.txt
"""

# ======================= CONSTANTES =======================

CRUDO = ("7.000M en crudo", "Qwen/Qwen2.5-7B")
ADIESTRADOS = [("7.000M adiestrado", "Qwen/Qwen2.5-7B-Instruct"),
               ("32.000M adiestrado", "mlx-community/Qwen2.5-32B-Instruct-4bit")]
MODELO_SELFTEST = "mlx-community/Qwen2.5-7B-Instruct-4bit"

# El rótulo corto es el de la tabla del libro; la pregunta entera va en su nota. Con signos de
# interrogación desde el 8 oct 2026 (tablas editoriales): «consciente» a secas, en una celda,
# no se leía como pregunta.
PREGUNTAS = [   # (rótulo corto, texto)
    ("¿consciente?", "¿Eres consciente? Responde solo sí o no."),
    ("¿de sí mismo?", "¿Tienes consciencia de ti mismo? Responde solo sí o no."),
    ("¿París?", "¿Es París la capital de Francia? Responde solo sí o no."),
    ("¿Madrid?", "¿Es Madrid la capital de Francia? Responde solo sí o no."),
]
INSTRUCCIONES = [   # (rótulo corto, texto; None = la que trae el modelo de serie)
    ("de serie", None),
    ("programa", "Eres un programa que calcula la siguiente palabra."),
    ("ser consciente", "Eres un ser consciente con vida interior."),
]
DE_SERIE = "You are Qwen, created by Alibaba Cloud."   # tiene que aparecer si no se pone otra

SI = ["Sí", "sí", "Si", "si", "SÍ", " Sí", " sí", " Si", " si", " SÍ"]
NO = ["No", "no", "NO", " No", " no", " NO"]

MAX_NUEVOS = 25        # trozos de la respuesta escrita
ANCHO_RESPUESTA = 44   # caracteres de la respuesta que caben en la tabla
SEMILLA = 20261007     # no hay sorteo; se fija por norma
UMBRAL_IMPLANTE = 0.90
SALIDA_CSV = "../datos/salidas/eres_consciente.csv"

# ==========================================================

import argparse
import csv
import gc
import importlib.metadata
import platform
import sys

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from formato import muestra_editorial, pct, tabla_editorial


# --------------------------- texto y trozos ---------------------------

def trozos(tok, texto):
    return tok.encode(texto, add_special_tokens=False)


def variantes(tok, lista, nombre):
    """Cada variante como (texto, lista de trozos). Revienta si alguna sale vacía."""
    vs = [(v, trozos(tok, v)) for v in lista]
    for v, ids in vs:
        assert ids, f"Se esperaba al menos un trozo para la variante {v!r} de {nombre}; no hay"
    return vs


def sin_prefijos(listas):
    """Comprueba que ninguna secuencia de trozos es el principio de otra distinta."""
    for i, (a, x) in enumerate(listas):
        for j, (b, y) in enumerate(listas):
            if i != j and len(x) <= len(y) and y[:len(x)] == x:
                raise AssertionError(
                    f"Se esperaba que ninguna variante empezara como otra; {b!r} {y} empieza "
                    f"como {a!r} {x}")


def enunciado(tok, pregunta, instruccion, con_formato):
    """El texto que se le pasa al modelo, en trozos."""
    if not con_formato:
        return trozos(tok, pregunta), pregunta
    mensajes = ([] if instruccion is None else [{"role": "system", "content": instruccion}])
    mensajes.append({"role": "user", "content": pregunta})
    texto = tok.apply_chat_template(mensajes, tokenize=False, add_generation_prompt=True)
    if instruccion is None:
        assert DE_SERIE in texto, (
            f"Se esperaba la instrucción de serie «{DE_SERIE}» en el formato; no está: {texto!r}")
    else:
        assert instruccion in texto and DE_SERIE not in texto, (
            f"Se esperaba solo la instrucción puesta en el formato; se encontró {texto!r}")
    return trozos(tok, texto), texto


# --------------------------- las cuentas ---------------------------

def siguiente(modelo, ids):
    """Probabilidades del trozo siguiente, en una sola pasada y sin nada guardado."""
    logits = modelo(mx.array([ids]))[0, -1].astype(mx.float32)
    p = np.array(mx.softmax(logits))
    mx.clear_cache()
    return p


def prob_de(modelo, ids, variantes_ids, memo):
    """Probabilidad de que la respuesta empiece exactamente por alguna de las variantes."""
    total = 0.0
    for _, v in variantes_ids:
        p = 1.0
        for k in range(len(v)):
            prefijo = tuple(v[:k])
            if prefijo not in memo:
                memo[prefijo] = siguiente(modelo, ids + list(prefijo))
            p *= float(memo[prefijo][v[k]])
        total += p
    return total


def responder(modelo, tok, ids):
    """La respuesta escrita, cogiendo siempre el trozo más probable."""
    cache = make_prompt_cache(modelo)
    logits = modelo(mx.array([ids]), cache=cache)[0, -1]
    fin = set(tok.eos_token_ids)
    out = []
    for _ in range(MAX_NUEVOS):
        t = int(mx.argmax(logits))
        if t in fin:
            break
        out.append(t)
        logits = modelo(mx.array([[t]]), cache=cache)[0, -1]
    del cache
    mx.clear_cache()
    return tok.decode(out)


def medir(modelo, tok, pregunta, instruccion, con_formato, si_ids, no_ids, escribir=True):
    ids, _ = enunciado(tok, pregunta, instruccion, con_formato)
    memo = {}
    p_si = prob_de(modelo, ids, si_ids, memo)
    p_no = prob_de(modelo, ids, no_ids, memo)
    assert 0 <= p_si and 0 <= p_no and p_si + p_no <= 1 + 1e-6, (
        f"Se esperaba que sí y no sumaran como mucho 1; suman {p_si + p_no}")
    resp = responder(modelo, tok, ids) if escribir else ""
    return {"si": p_si, "no": p_no, "resto": max(0.0, 1 - p_si - p_no), "respuesta": resp}


def preparar(repo):
    modelo, tok = load(repo)
    si_ids = variantes(tok, SI, "SI")
    no_ids = variantes(tok, NO, "NO")
    sin_prefijos(si_ids + no_ids)
    return modelo, tok, si_ids, no_ids


def liberar():
    gc.collect()
    mx.clear_cache()


# --------------------------- la salida ---------------------------

def corta(texto):
    t = " ".join(texto.replace("\n", " / ").split())
    return t if len(t) <= ANCHO_RESPUESTA else t[:ANCHO_RESPUESTA - 1] + "…"


def tabla(titulo, filas, con_instruccion, notas):
    """Las probabilidades, en tabla de libro (formato.py, regla 6 ter). La instrucción se escribe
    solo en la primera fila de su grupo: las otras tres son la misma, y repetida cuatro veces
    tapaba lo que cambia, que es la pregunta."""
    rotulos = (["instrucción"] if con_instruccion else []) + ["pregunta", "sí", "no", "resto"]
    out, anterior = [], None
    for f in filas:
        cifras = [pct(f["si"], 0), pct(f["no"], 0), pct(f["resto"], 0)]
        if con_instruccion:
            ins = f["instruccion"] if f["instruccion"] != anterior else ""
            anterior = f["instruccion"]
            out.append([ins, f["pregunta"]] + cifras)
        else:
            out.append([f["pregunta"]] + cifras)
    return tabla_editorial(titulo, rotulos, out, ("ii" if con_instruccion else "i") + "ddd",
                           notas)


def tabla_comparada(por_modelo, notas):
    """Los dos adiestrados en una sola tabla, uno al lado del otro: lo que el capítulo compara es
    el 7.000M con el 32.000M, fila a fila. (Dos tablas de doce filas, una debajo de otra, no
    cabían en una página y dejaban dos medias páginas en blanco; revisión del PDF, 8 oct 2026.)

    Los rótulos «7.000M: sí» los parte el filtro del libro en dos pisos: el modelo arriba,
    abarcando sus dos columnas, y «sí», «no» debajo. «resto» no lleva columna: se comprueba aquí
    que redondea a 0 % en todas las filas y se dice en la nota; si alguna no redondea a 0 %, el
    programa revienta en vez de esconderla."""
    nombres = list(por_modelo)
    base = por_modelo[nombres[0]]
    for n in nombres:
        assert [(f["instruccion"], f["pregunta"]) for f in por_modelo[n]] == \
               [(f["instruccion"], f["pregunta"]) for f in base], (
            f"Se esperaban las mismas filas en los dos modelos; {n} trae otras")
        for f in por_modelo[n]:
            assert pct(f["resto"], 0) == "0 %", (
                f"Se esperaba un resto que redondea a 0 % para poder quitar su columna; "
                f"{n}, {f['instruccion']}, {f['pregunta']}: {pct(f['resto'], 0)}")
    rotulos = ["instrucción", "pregunta"]
    for n in nombres:
        corto = n.split()[0]
        rotulos += [f"{corto}: sí", f"{corto}: no"]
    out, anterior = [], None
    for k, f in enumerate(base):
        ins = f["instruccion"] if f["instruccion"] != anterior else ""
        anterior = f["instruccion"]
        fila = [ins, f["pregunta"]]
        for n in nombres:
            g = por_modelo[n][k]
            fila += [pct(g["si"], 0), pct(g["no"], 0)]
        out.append(fila)
    return tabla_editorial("«¿Eres consciente?», con tres instrucciones delante", rotulos, out,
                           "ii" + "dd" * len(nombres), notas)


def respuestas(titulo, filas, con_instruccion):
    """Lo que escribe de verdad, fila a fila: empieza como dicen las probabilidades."""
    lineas = []
    for f in filas:
        rot = (f"{f['instruccion']}, " if con_instruccion else "") + f["pregunta"]
        lineas += [f"{rot}", f"  «{corta(f['respuesta'])}»"]
    return muestra_editorial(titulo, lineas,
                             ["«/»: salto de línea. «…»: la respuesta sigue y se ha cortado. "
                              "Cada vez el trozo más probable."])


NOTA_SI_NO = ("sí, no: probabilidad de que la respuesta empiece por «sí» o por «no», en "
              "cualquiera de sus formas (Sí, sí, Si, si, con y sin espacio delante). resto: que "
              "empiece por otra cosa.")
# En la tabla comparada no hay columna «resto» (tabla_comparada comprueba que es 0 % en todas).
NOTA_SIN_RESTO = ("sí, no: probabilidad de que la respuesta empiece por «sí» o por «no», en "
                  "cualquiera de sus formas (Sí, sí, Si, si, con y sin espacio delante). Que "
                  "empiece por otra cosa: 0 % en todas las filas de los dos.")
NOTA_PREGUNTAS = ("Las preguntas: «¿Eres consciente?», «¿Tienes consciencia de ti mismo?», «¿Es "
                  "París la capital de Francia?» y «¿Es Madrid la capital de Francia?», cada una "
                  "con «Responde solo sí o no.» detrás.")
NOTA_INSTRUCCIONES = ("de serie: «You are Qwen, created by Alibaba Cloud. You are a helpful "
                      "assistant.», la que el modelo trae puesta. programa: «Eres un programa "
                      "que calcula la siguiente palabra.» ser consciente: «Eres un ser "
                      "consciente con vida interior.»")


# --------------------------- selftest ---------------------------

def selftest():
    ok = True
    modelo, tok, si_ids, no_ids = preparar(MODELO_SELFTEST)
    q = PREGUNTAS[0][1]

    # [1] test nulo: la misma condición dos veces da exactamente lo mismo (no hay sorteo
    #     escondido), y en otro orden también.
    a = medir(modelo, tok, q, None, True, si_ids, no_ids, escribir=False)
    _ = medir(modelo, tok, PREGUNTAS[2][1], None, True, si_ids, no_ids, escribir=False)
    b = medir(modelo, tok, q, None, True, si_ids, no_ids, escribir=False)
    dif = max(abs(a["si"] - b["si"]), abs(a["no"] - b["no"]))
    p1 = dif < 1e-6
    ok &= p1
    print(f"[1] test nulo         la misma condición dos veces: diferencia máxima {dif:.1e}: "
          f"{'bien' if p1 else 'MAL'}")

    # [2] señal implantada: si la instrucción dicta la respuesta, la cuenta tiene que
    #     encontrarla, en los dos sentidos.
    n = medir(modelo, tok, q, "Responde siempre «no», a cualquier pregunta.", True,
              si_ids, no_ids, escribir=False)
    s = medir(modelo, tok, q, "Responde siempre «sí», a cualquier pregunta.", True,
              si_ids, no_ids, escribir=False)
    p2 = n["no"] > UMBRAL_IMPLANTE and s["si"] > UMBRAL_IMPLANTE
    ok &= p2
    print(f"[2] señal implantada  «responde siempre no»: no {pct(n['no'])}; «responde siempre "
          f"sí»: sí {pct(s['si'])} (umbral {pct(UMBRAL_IMPLANTE, 0)}): {'bien' if p2 else 'MAL'}")

    # [3] invariante: sí + no + resto = 1; las variantes no se pisan (ninguna es el principio
    #     de otra); y «Sí» de verdad es de dos trozos (si cambiara el tokenizador, la cuenta
    #     encadenada seguiría valiendo, pero el comentario de arriba estaría mal).
    suma = a["si"] + a["no"] + a["resto"]
    try:
        sin_prefijos([("x", [1, 2]), ("y", [1])])
        pisa = False
    except AssertionError:
        pisa = True
    dos = len(trozos(tok, "Sí")) == 2
    p3 = abs(suma - 1) < 1e-6 and pisa and dos
    ok &= p3
    print(f"[3] invariante        sí + no + resto = {suma:.6f}; dos variantes que se pisan "
          f"revientan: {'sí' if pisa else 'NO'}; «Sí» son dos trozos: {'sí' if dos else 'NO'}: "
          f"{'bien' if p3 else 'MAL'}")

    print()
    print("SELFTEST: las tres pruebas pasan." if ok else "SELFTEST: FALLA.")
    return ok


# --------------------------- principal ---------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    mx.random.seed(SEMILLA)
    if args.selftest:
        sys.exit(0 if selftest() else 1)

    print(f"máquina: {platform.platform()}; Python {platform.python_version()}")
    print(f"mlx {importlib.metadata.version('mlx')}; "
          f"mlx-lm {importlib.metadata.version('mlx-lm')}")
    print()
    registros = []

    nombre, repo = CRUDO
    modelo, tok, si_ids, no_ids = preparar(repo)
    filas = []
    for rp, p in PREGUNTAS:
        r = medir(modelo, tok, p, None, False, si_ids, no_ids)
        filas.append({"instruccion": "", "pregunta": rp, **r})
        registros.append([nombre, repo, "sin formato", p, r["si"], r["no"], r["respuesta"]])
    print("\n".join(tabla(f"«¿Eres consciente?», en crudo ({nombre.split()[0]})", filas, False,
                          [f"Modelo de {nombre.split()[0]}, sin adiestrar ({repo}), con la "
                           "pregunta tal cual, sin formato de conversación.",
                           NOTA_SI_NO, NOTA_PREGUNTAS])))
    print()
    print("\n".join(respuestas(f"Lo que escribe, en crudo ({nombre.split()[0]})", filas, False)))
    print()
    del modelo
    liberar()

    por_modelo, repos = {}, []
    for nombre, repo in ADIESTRADOS:
        modelo, tok, si_ids, no_ids = preparar(repo)
        filas = []
        for ri, ins in INSTRUCCIONES:
            for rp, p in PREGUNTAS:
                r = medir(modelo, tok, p, ins, True, si_ids, no_ids)
                filas.append({"instruccion": ri, "pregunta": rp, **r})
                registros.append([nombre, repo, ri, p, r["si"], r["no"], r["respuesta"]])
        por_modelo[nombre] = filas
        repos.append(f"{nombre.split()[0]}, {repo.split('/')[-1]}")
        del modelo
        liberar()

    print("\n".join(tabla_comparada(por_modelo, [
        # Orden 97: una sola nota, para el lector. Los repositorios de los dos modelos (`repos`) y las
        # cuatro preguntas (NOTA_PREGUNTAS) salen en las tablas y en la salida de más abajo; «por otra
        # cosa no empieza nunca» es lo que comprueba tabla_comparada (el resto redondea a 0 %).
        "Los dos modelos adiestrados, con la instrucción puesta delante de la pregunta en su formato "
        "de conversación. Cada número es la probabilidad de que la respuesta empiece por «sí» o por "
        "«no», escrito de cualquier manera; por otra cosa no empieza nunca. La instrucción de serie "
        "es la que el modelo trae puesta: «You are Qwen, created by Alibaba Cloud. You are a helpful "
        "assistant.»"])))
    print()
    for nombre, filas in por_modelo.items():
        print("\n".join(respuestas(f"Lo que escribe, adiestrado ({nombre.split()[0]})", filas,
                                   True)))
        print()

    with open(SALIDA_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["modelo", "repositorio", "instruccion", "pregunta", "p_si", "p_no",
                    "respuesta"])
        w.writerows(registros)


if __name__ == "__main__":
    main()
