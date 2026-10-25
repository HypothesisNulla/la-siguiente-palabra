#!/usr/bin/env python3
"""
Capítulo 7 — la máquina entera: seguir una frase de punta a punta.

Enseña, sobre un modelo real y pequeño, los cuatro momentos del recorrido:
  1. El texto se parte en trozos.
  2. Cada trozo se convierte en una lista de números.
  3. Los números atraviesan una ronda de mirar y mezclar detrás de otra, y dentro de
     cada ronda la mirada se hace varias veces a la vez.
  4. Sale una lista de probabilidades sobre TODOS los trozos posibles, se elige uno,
     y se vuelve a empezar.

Uso:
    python maquina_entera.py
    python maquina_entera.py --selftest
"""

# ======================= CONSTANTES =======================

MODELO = "Qwen/Qwen2.5-0.5B"
DTYPE = "float32"          # los números del modelo; la lista final, en float64 (ver siguientes())

FRASE = "La capital de Francia es"
FRASES_TROCEADO = [
    "La capital de Francia es",
    "El murciélago hambriento vigilaba",
    "anticonstitucionalmente",
]
TOP_N = 10
PASOS_BUCLE = 8

# selftest
PROMPT_FACIL = "uno, dos, tres, cuatro,"
CONTINUACION_FACIL = " cinco"
UMBRAL_FACIL = 0.30
TOL_SUMA = 1e-4

SALIDA_CSV = "maquina_entera.csv"

# ==========================================================

from formato import barra, coma, miles, pct, tabla_editorial, trozo, unidos

import argparse
import csv
import math
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def cargar():
    tok = AutoTokenizer.from_pretrained(MODELO)
    modelo = AutoModelForCausalLM.from_pretrained(MODELO, dtype=getattr(torch, DTYPE))
    modelo.eval()
    return tok, modelo


def trozos(tok, texto):
    ids = tok(texto)["input_ids"]
    return [tok.decode([i]) for i in ids]


def siguientes(tok, modelo, texto, n=TOP_N):
    ids = tok(texto, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        logits = modelo(ids).logits[0, -1]
    # L24 (9 de octubre): la lista se saca en float64. Con torch 2.14 en el Mac (procesador ARM),
    # la de float32 suma 1,000131 y no pasa la comprobación de abajo (tolerancia 1e-4); en float64
    # suma 1 con un error de 1e-15. Medido contra la salida anterior (float32, 18 de septiembre):
    # cambian dos cifras de los ocho pasos en la segunda decimal (65,84 % → 65,83 %; 40,17 % →
    # 40,16 %); el resto, igual.
    p = torch.softmax(logits.double(), dim=-1)
    suma = float(p.sum())
    assert abs(suma - 1.0) < TOL_SUMA, \
        f"Invariante roto: se esperaba que las probabilidades sumaran 1; se encontró {suma:.6f}"
    assert float(p.min()) >= 0, \
        f"Invariante roto: se esperaban probabilidades no negativas; la menor es {float(p.min()):.2e}"
    top = torch.topk(p, n)
    return [(tok.decode([int(i)]), float(v)) for v, i in zip(top.values, top.indices)], p


def dispersion(p):
    """Entropía normalizada: 0 = seguro de una sola opción, 1 = todas igual de probables."""
    q = p[p > 0]
    return float(-(q * q.log()).sum() / math.log(len(p)))


def selftest(tok, modelo):
    fallos = []

    # 1. TEST NULO — una frase sin estructura debe dejar al modelo mucho menos decidido
    #    que una frase con una continuación evidente.
    _, p_facil = siguientes(tok, modelo, PROMPT_FACIL, 3)
    _, p_nula = siguientes(tok, modelo, "qx zr vb kk pl", 3)
    d_facil, d_nula = dispersion(p_facil), dispersion(p_nula)
    print(f"[1] test nulo         dispersión: frase con sentido {coma(d_facil, 3)}  "
          f"sin sentido {coma(d_nula, 3)}")
    if not d_nula > d_facil:
        fallos.append(f"test nulo: la frase sin sentido ({coma(d_nula, 3)}) no sale más dispersa "
                      f"que la que tiene sentido ({coma(d_facil, 3)})")

    # 2. SEÑAL IMPLANTADA — tras «uno, dos, tres, cuatro,» la continuación es obvia.
    top, _ = siguientes(tok, modelo, PROMPT_FACIL, 5)
    prob_cinco = dict(top).get(CONTINUACION_FACIL, 0.0)
    print(f"[2] señal implantada  «{PROMPT_FACIL}» -> «{top[0][0]}» con {coma(top[0][1], 3)}; "
          f"«{CONTINUACION_FACIL.strip()}» tiene {coma(prob_cinco, 3)}")
    if prob_cinco < UMBRAL_FACIL:
        fallos.append(f"señal implantada: se esperaba «cinco» con al menos "
                      f"{coma(UMBRAL_FACIL, 2)}; se obtuvo {coma(prob_cinco, 3)}")

    # 3. INVARIANTE DEL DOMINIO — la lista de probabilidades cubre TODO el vocabulario y
    #    suma uno. (Los asserts de `siguientes` ya lo comprueban en cada llamada.)
    _, p = siguientes(tok, modelo, FRASE)
    print(f"[3] invariante        {miles(len(p))} trozos posibles, "
          f"suman {coma(float(p.sum()), 6)}, mínima {float(p.min()):.2e}")
    if len(p) < 1000:
        fallos.append(f"invariante: vocabulario sospechosamente pequeño ({miles(len(p))})")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    tok, modelo = cargar()
    if args.selftest:
        sys.exit(selftest(tok, modelo))

    filas = []

    # L24 (9 de octubre): cada bloque, una tabla editorial (regla 6 ter). Las cuentas no cambian.
    print("--- 1. EL TEXTO SE PARTE EN TROZOS ---\n")
    filas_t = []
    for t in FRASES_TROCEADO:
        piezas = trozos(tok, t)
        filas_t.append([t, str(len(piezas)), unidos([p.replace(" ", "_") for p in piezas])])
    print("\n".join(tabla_editorial(
        "El texto, partido en trozos", ["el texto", "trozos", "cómo se parte"], filas_t, "idi",
        ["El guion bajo marca que ahí había un espacio."])))
    print()

    print("--- 2. EL TAMAÑO DE LA MÁQUINA ---\n")
    cfg = modelo.config
    total = sum(p.numel() for p in modelo.parameters())
    print("\n".join(tabla_editorial(
        "El tamaño de la máquina", ["", "cuánto"],
        [["modelo", MODELO],
         ["números por trozo", str(cfg.hidden_size)],
         ["rondas, una detrás de otra", str(cfg.num_hidden_layers)],
         ["miradas a la vez dentro de cada ronda", str(cfg.num_attention_heads)],
         ["miradas en total", str(cfg.num_hidden_layers * cfg.num_attention_heads)],
         ["trozos posibles en la salida", miles(cfg.vocab_size)],
         ["números ajustables en total", miles(total)]], "id")))
    print()

    print("--- 3. LA LISTA DE PROBABILIDADES ---\n")
    top, p = siguientes(tok, modelo, FRASE)
    # Con su barra a escala fija: el número dice cuánto, y la barra deja verlo sin leerlo.
    resto = 1.0 - sum(v for _, v in top)
    print("\n".join(tabla_editorial(
        f"Qué viene después de «{FRASE}»", ["trozo", "probabilidad", ""],
        [[trozo(w.replace(" ", "_")), pct(v, 2), barra(v)] for w, v in top] +
        [["(el resto)", pct(resto, 2), barra(resto)]], "idi",
        [f"Los {TOP_N} trozos más probables. El resto se reparte entre las otras "
         f"{miles(len(p) - TOP_N)} posibilidades."])))
    filas += [["probabilidad", FRASE, w, f"{v:.6f}"] for w, v in top]
    print()

    print("--- 4. Y SE VUELVE A EMPEZAR ---\n")
    texto = FRASE
    filas_b = []
    for i in range(1, PASOS_BUCLE + 1):
        top, _ = siguientes(tok, modelo, texto, 1)
        palabra, prob = top[0]
        texto += palabra
        filas_b.append([str(i), trozo(palabra.replace(" ", "_")), pct(prob, 2), texto])
        filas.append(["bucle", str(i), palabra, f"{prob:.6f}"])
    print("\n".join(tabla_editorial(
        "Y se vuelve a empezar", ["paso", "trozo", "probabilidad", "el texto, con el trozo pegado"],
        filas_b, "cidi",
        [f"Cada paso pega al texto el trozo más probable y vuelve a empezar, {PASOS_BUCLE} veces."])))

    with open(SALIDA_CSV, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([["medicion", "contexto", "palabra", "probabilidad"]] + filas)
    print(f"\nEscrito {SALIDA_CSV}")


if __name__ == "__main__":
    main()
