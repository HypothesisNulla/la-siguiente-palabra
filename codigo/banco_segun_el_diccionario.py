#!/usr/bin/env python3
"""
Capítulo 5 — ¿qué sentido de «banco» se ha quedado su lista? Con listas de fuera (L24).

El capítulo 5 enseña que el método da una sola lista a «banco», que tiene dos sentidos. Para
decir cuál de los dos se ha quedado la lista hacen falta dos grupos de palabras, uno por sentido,
y el grupo no puede escribirlo quien ya ha visto las vecinas: las listas de `cuantas_veces_sale.py`
se escribieron después de ver la primera tirada (ver su aviso). Aquí los grupos salen de fuera y
de antes: **los sustantivos de las definiciones de «banco» en el Diccionario de la lengua
española** (https://dle.rae.es/banco, consultado el 28 de septiembre de 2026):

  - acepción 1: «Asiento, con respaldo o sin él, en que pueden sentarse dos o más personas.»
  - acepción 2: «Madero grueso escuadrado que se coloca horizontalmente sobre cuatro pies y sirve
    de mesa para labores de carpinteros y otros artesanos.»
  - acepción 5: «Empresa dedicada a realizar operaciones financieras con el dinero procedente de
    sus accionistas y de los depósitos de sus clientes.»

El mueble son los sustantivos de las acepciones 1 y 2; el dinero, los de la 5. Se toman tal como
salen escritos en la definición; los que el modelo no conoce (menos de diez apariciones) se dicen
y se quedan fuera.

Entrena el modelo de `palabras_numeros.py` con las cinco semillas de `cuantas_veces_sale.py` (las
cinco tiradas del capítulo) y mide, en cada una, en qué puesto sale cada palabra de los dos grupos
entre las vecinas de «banco». Cada tirada tarda más de un minuto; con `--cache DIR` guarda cada
una al terminar y no la vuelve a entrenar (el resultado es el mismo: lo guardado es lo medido).

Uso:
    python banco_segun_el_diccionario.py --selftest
    python banco_segun_el_diccionario.py [--cache DIR]
"""

# ======================= CONSTANTES =======================

PALABRA = "banco"
FUENTE = "https://dle.rae.es/banco"
CONSULTADO = "2026-09-28"
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre",
         "octubre", "noviembre", "diciembre"]
MUEBLE = ["asiento", "respaldo", "personas",                                   # acepción 1
          "madero", "pies", "mesa", "labores", "carpinteros", "artesanos"]     # acepción 2
DINERO = ["empresa", "operaciones", "dinero", "accionistas", "depósitos", "clientes"]  # acep. 5
VECINAS = 5

# ==========================================================

import argparse
import json
import os
import sys

import numpy as np

from cuantas_veces_sale import SEMILLAS
from formato import miles, tabla_editorial
from palabras_numeros import CORPUS_BIBLIOTECA, cargar_corpus, entrenar

AQUI = os.path.dirname(os.path.abspath(__file__))


def puestos(vectores, palabra, lista):
    """Puesto de cada palabra de `lista` entre las vecinas de `palabra` (1 = la más parecida).
    Las que el modelo no conoce devuelven None."""
    todas = vectores.get_normed_vectors()
    s = todas @ todas[vectores.key_to_index[palabra]]
    orden = [vectores.index_to_key[i] for i in np.argsort(-s) if vectores.index_to_key[i] != palabra]
    rango = {w: k + 1 for k, w in enumerate(orden)}
    return {w: rango.get(w) for w in lista}


def medir_tirada(semilla, frases):
    m = entrenar(frases, semilla=semilla)
    wv = m.wv
    return {"semilla": semilla, "n": len(wv.index_to_key) - 1,
            "vecinas": [w for w, _ in wv.most_similar(PALABRA, topn=VECINAS)],
            "puestos": puestos(wv, PALABRA, MUEBLE + DINERO)}


def fecha_larga(iso):
    """'2026-09-28' -> '28 de septiembre de 2026'."""
    a, m, d = (int(x) for x in iso.split("-"))
    return f"{d} de {MESES[m - 1]} de {a}"


def mediana(valores):
    v = [x for x in valores if x is not None]
    assert v, "ninguna palabra del grupo está en el modelo"
    return float(np.median(v))


def informe(tiradas):
    """Dos tablas editoriales (L24, 9 de octubre; regla 6 ter): los puestos y las vecinas."""
    n = tiradas[0]["n"]
    conocidas = lambda lista: [w for w in lista if all(t["puestos"][w] is not None for t in tiradas)]
    fuera = [w for w in MUEBLE + DINERO if w not in conocidas(MUEBLE + DINERO)]
    filas = []
    for titulo, lista in (("el mueble", MUEBLE), ("el dinero", DINERO)):
        for k, w in enumerate(conocidas(lista)):
            filas.append([titulo if k == 0 else "", w] + [miles(t["puestos"][w]) for t in tiradas])
    for titulo, lista in (("el mueble", MUEBLE), ("el dinero", DINERO)):
        filas.append(["**mediana**", f"**{titulo}**"] + [
            f"**{miles(round(mediana([t['puestos'][w] for w in conocidas(lista)])))}**"
            for t in tiradas])
    filas.append(["a media tabla", ""] + [miles((t["n"] + 1) // 2) for t in tiradas])
    puestos_ = tabla_editorial(
        f"Puesto de cada palabra entre las vecinas de «{PALABRA}», en las cinco tiradas",
        ["sentido", "palabra"] + [f"tirada: {k + 1}" for k in range(len(tiradas))], filas,
        "ii" + "d" * len(tiradas),
        [f"Cada número es el puesto de esa palabra entre las {miles(n)} vecinas de «{PALABRA}»: el 1 "
         "es la más parecida. Las dos filas de la mediana dan el puesto del medio de cada grupo, y "
         "la última, el que saca una palabra cualquiera.",
         # Orden 97: la nota ya no lleva la dirección ni la fecha de consulta del diccionario (están
         # en FUENTE y CONSULTADO, y en la bibliografía del capítulo 16), ni dice que las palabras
         # están en el modelo. Si alguna faltara, la nota lo diría, como antes.
         f"Las palabras de cada grupo son los sustantivos de las definiciones de «{PALABRA}» en el "
         "Diccionario de la lengua española: las acepciones 1 y 2, el asiento y la mesa de "
         "carpintero, para el mueble, y la 5, la empresa de operaciones financieras, para el dinero."
         + (f" Faltan en el modelo: {', '.join(fuera)}." if fuera else "")])
    vecinas = tabla_editorial(
        f"Las {VECINAS} vecinas de «{PALABRA}» en cada tirada",
        ["tirada", f"las {VECINAS} más parecidas"],
        [[str(k), ", ".join(t["vecinas"])] for k, t in enumerate(tiradas, 1)], "ci")
    return puestos_ + [""] + vecinas


def selftest():
    fallos = []
    from gensim.models import KeyedVectors
    rng = np.random.default_rng(20260928)
    nombres = [PALABRA] + MUEBLE + DINERO + [f"x{i}" for i in range(400)]

    def modelo(cerca_de_banco):
        v = rng.normal(size=(len(nombres), 50))
        for w in cerca_de_banco:
            v[nombres.index(w)] = v[0] + 0.1 * rng.normal(size=50)
        kv = KeyedVectors(50)
        kv.add_vectors(nombres, v.astype(np.float32))
        return kv

    # 1. TEST NULO — con listas al azar, ningún grupo queda arriba: las dos medianas, a media tabla.
    p = puestos(modelo([]), PALABRA, MUEBLE + DINERO)
    m1, m2 = mediana([p[w] for w in MUEBLE]), mediana([p[w] for w in DINERO])
    n = len(nombres) - 1
    print(f"[1] test nulo         medianas con todo al azar: {m1:.0f} y {m2:.0f} de {n}")
    if min(m1, m2) < n / 8:
        fallos.append("test nulo: un grupo al azar sale arriba")
    # 2. SEÑAL IMPLANTADA — si las palabras del dinero se ponen pegadas a «banco», su mediana
    #    tiene que salir en los primeros puestos y la del mueble no.
    p = puestos(modelo(DINERO), PALABRA, MUEBLE + DINERO)
    m1, m2 = mediana([p[w] for w in MUEBLE]), mediana([p[w] for w in DINERO])
    print(f"[2] señal implantada  dinero pegado a «banco»: mediana dinero {m2:.0f}, mueble {m1:.0f}")
    if not (m2 <= len(DINERO) and m1 > n / 8):
        fallos.append("señal implantada: no encuentra el grupo pegado")
    # 3. INVARIANTE — los puestos son distintos y van de 1 a n, y el primero es la primera de
    #    most_similar.
    kv = modelo([])
    p = puestos(kv, PALABRA, nombres[1:])
    primera = kv.most_similar(PALABRA, topn=1)[0][0]
    ok = sorted(p.values()) == list(range(1, n + 1)) and p[primera] == 1
    print(f"[3] invariante        puestos de 1 a {n} sin repetir, y el 1 es «{primera}»: {ok}")
    if not ok:
        fallos.append("invariante: los puestos no son una ordenación")
    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def main():
    import platform
    from datetime import date
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cache", metavar="DIR")
    args = ap.parse_args()
    codigo = selftest()
    if codigo or args.selftest:
        sys.exit(codigo)
    print(f"\nMáquina: {platform.machine()}, {platform.system()} {platform.release()}.")
    print(f"Medido el {date.today().isoformat()}. Semillas: {', '.join(map(str, SEMILLAS))}.")
    frases, total = cargar_corpus(os.path.join(AQUI, CORPUS_BIBLIOTECA))
    print(f"Biblioteca: {miles(total)} palabras.\n")
    tiradas = []
    for s in SEMILLAS:
        ruta = os.path.join(args.cache, f"{s}.json") if args.cache else None
        if ruta and os.path.exists(ruta):
            t = json.load(open(ruta, encoding="utf-8"))
        else:
            t = medir_tirada(s, frases)
            if ruta:
                os.makedirs(args.cache, exist_ok=True)
                json.dump(t, open(ruta, "w", encoding="utf-8"), ensure_ascii=False)
        tiradas.append(t)
    print("\n".join(informe(tiradas)))


if __name__ == "__main__":
    main()
