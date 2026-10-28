#!/usr/bin/env python3
"""
Capítulo 11 — el espejismo, con la cuenta hecha (L24, hallazgo D13).

El capítulo cuenta el argumento del trabajo de 2023 («¿son un espejismo?») con palabras: si una
respuesta tiene cinco partes y solo cuenta cuando están las cinco, lo que se mide es el acierto
de cada parte multiplicado por sí mismo cinco veces, y eso «se queda pegado al cero mucho rato y
luego se dispara». Aquí está la cuenta, para una máquina imaginaria que acierta cada parte suelta
con un tanto por ciento que sube de diez en diez. No es una medición: son multiplicaciones, y
las imprime el programa para que el libro no escriba ninguna cifra a mano.

Uso:
    python todo_o_nada.py --selftest
    python todo_o_nada.py
"""

# ======================= CONSTANTES =======================

PARTES = 5                          # partes de la respuesta, como en el capítulo
TABLA = [50, 60, 70, 80, 90, 100]   # acierto en cada parte suelta, de cada cien
CURVA = range(0, 101)               # para la figura: de uno en uno
SALIDA_CSV = "../datos/salidas/todo_o_nada.csv"

# ==========================================================

import argparse
import csv
import sys

from formato import barra, coma, tabla_editorial


def a_la_vez(por_parte, partes=PARTES):
    """Acierto, de cada cien, cuando cuentan solo las respuestas con todas las partes bien."""
    return 100 * (por_parte / 100) ** partes


def cadena(por_parte, partes=PARTES):
    """La multiplicación escrita en palabras, para la primera fila."""
    return " por ".join([coma(por_parte / 100, 1)] * partes)


def bloque():
    """Desde el 9 oct 2026, tabla de libro (formato.py; REGLAS 6 ter), con una barra por columna
    para que se vea a simple vista cuál sube despacio y cuál se queda abajo y luego se dispara."""
    filas = [[str(p), barra(p / 100), coma(a_la_vez(p), 0), barra(a_la_vez(p) / 100)]
             for p in TABLA]
    return tabla_editorial(
        "Una respuesta de cinco partes: cada parte, o las cinco a la vez",
        ["cada parte suelta", "", "las cinco a la vez", ""], filas, "didi",
        [# Orden 97: la cuenta con un 50 («0,5 por 0,5 por… da 0,031») ya no va en la nota; está en
         # `cadena` y en la primera fila de la tabla.
         "De cada cien veces. A la izquierda, cuántas sale bien cada parte por separado; a la "
         "derecha, cuántas salen bien las cinco a la vez, que es lo único que cuenta cuando se "
         "puntúa todo o nada."])


def selftest():
    fallos = []
    # 1. TEST NULO — con una sola parte, puntuar «todo o nada» no cambia nada.
    ok = all(abs(a_la_vez(p, 1) - p) < 1e-12 for p in CURVA)
    print(f"[1] test nulo         con una parte, las dos maneras dan lo mismo: {'sí' if ok else 'NO'}")
    if not ok:
        fallos.append("test nulo: con una sola parte las dos puntuaciones difieren")
    # 2. SEÑAL — la cuenta a mano del capítulo: la mitad en cada parte, cinco veces, es 1/32.
    v = a_la_vez(50)
    print(f"[2] señal             la mitad cinco veces: {v} de cada cien (1/32 = {100 / 32})")
    if abs(v - 100 / 32) > 1e-12:
        fallos.append("señal: 0,5 multiplicado cinco veces no da 1/32")
    # 3. INVARIANTE — todo o nada nunca pasa de cada parte, y los extremos coinciden.
    ok = all(a_la_vez(p) <= p + 1e-12 for p in CURVA) and a_la_vez(0) == 0 and a_la_vez(100) == 100
    print(f"[3] invariante        las cinco a la vez nunca por encima de cada parte: {'sí' if ok else 'NO'}")
    if not ok:
        fallos.append("invariante: todo o nada supera a cada parte o no coincide en los extremos")
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
    codigo = selftest()
    if codigo or args.selftest:
        sys.exit(codigo)
    print()
    print("\n".join(bloque()))
    with open(SALIDA_CSV, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([["cada_parte", "las_cinco"]] +
                                 [[p, f"{a_la_vez(p):.4f}"] for p in CURVA])
    print(f"\nEscrito {SALIDA_CSV}")


if __name__ == "__main__":
    main()
