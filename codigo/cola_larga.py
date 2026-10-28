#!/usr/bin/env python3
"""
Capítulo 15 — la cola larga, contada en los trescientos libros (L24, E28).

El capítulo 15 explica «por qué crecer funciona» con una hipótesis: lo que se aprende son muchas
regularidades pequeñas, unas poquísimas muy frecuentes y una cola larguísima de raras. Lo decía
sin un solo ejemplo. Aquí está el ejemplo que el lector ya conoce: las palabras de los trescientos
libros del repositorio, ordenadas de la más a la menos frecuente. Para los puestos 1, 10, 100,
1.000, 10.000 y 100.000: qué palabra es y cuántas veces sale.

No depende de la máquina: cuenta palabras.

Uso:
    python cola_larga.py --selftest
    python cola_larga.py > ../datos/salidas/cola_larga.txt
"""

# ======================= CONSTANTES =======================

CORPUS = "../datos/corpus_es"
PUESTOS = (1, 10, 100, 1_000, 10_000, 100_000)
PALABRA = r"[a-záéíóúüñ]+"       # una palabra: letras seguidas, en minúscula
LIBROS_ESPERADOS = 300

# ==========================================================

import argparse
import collections
import glob
import os
import re
import sys

from formato import coma, miles, tabla_editorial


def uno(x):
    """Un número con una decimal y el punto de los miles."""
    entero, dec = f"{x:.1f}".split(".")
    return miles(int(entero)) + "," + dec


def contar(textos):
    c = collections.Counter()
    for t in textos:
        c.update(re.findall(PALABRA, t.lower()))
    return c


def libros():
    fs = sorted(glob.glob(os.path.join(CORPUS, "*.txt")))
    assert len(fs) == LIBROS_ESPERADOS, f"se esperaban {LIBROS_ESPERADOS} libros; hay {len(fs)}"
    for f in fs:
        with open(f, encoding="utf-8", errors="strict") as fh:
            yield fh.read()


def en_puestos(c):
    orden = c.most_common()
    return [(p, orden[p - 1][0], orden[p - 1][1]) for p in PUESTOS if p <= len(orden)]


def informe(c):
    """Las palabras por puestos y cuánto bajan de un puesto al siguiente, en tablas editoriales
    (L24, 9 de octubre de 2026; antes, renglones sangrados que el capítulo 15 copiaba)."""
    total = sum(c.values())
    filas = en_puestos(c)
    solo_una = sum(1 for n in c.values() if n == 1)
    print("--- LAS PALABRAS DE LOS TRESCIENTOS LIBROS, POR PUESTOS ---")
    print()
    for l in tabla_editorial(
            "Las palabras de los trescientos libros, por puestos",
            ["puesto", "palabra", "veces", "de cada millón"],
            [[miles(p), w, miles(n), uno(1e6 * n / total)] for p, w, n in filas],
            "didd",
            [# Orden 97: nota para el lector, en una sola frase larga.
             "El puesto es el lugar de cada palabra en la lista, de la que más sale a la que menos; "
             "la última columna, cuántas de cada millón de palabras de los libros son ésa. En total "
             f"hay {miles(total)} palabras, y {miles(len(c))} distintas.",
             f"Palabras que salen una sola vez: {miles(solo_una)}, "
             f"{coma(100 * solo_una / len(c), 0)} de cada cien de las distintas."]):
        print(l)
    print()
    for l in tabla_editorial(
            "Cada vez que el puesto se multiplica por diez, las veces se dividen entre",
            ["del puesto", "al puesto", "las veces se dividen entre"],
            [[miles(p1), miles(p2), coma(n1 / n2, 1)]
             for (p1, _, n1), (p2, _, n2) in zip(filas, filas[1:])],
            "ddd"):
        print(l)


def selftest():
    fallos = []
    # 1. TEST NULO — un texto donde todas las palabras salen las mismas veces no tiene cola: el
    #    primer puesto y el último salen igual.
    plano = contar([" ".join(["uno dos tres cuatro cinco"] * 50)])
    v = sorted(plano.values())
    print(f"[1] test nulo         texto plano: el primero sale {v[-1]} veces y el último {v[0]}")
    if v[0] != v[-1]:
        fallos.append("test nulo: aparece una cola en un texto sin ella")

    # 2. SEÑAL IMPLANTADA — un texto fabricado en que la palabra del puesto k sale 1000/k veces:
    #    del puesto 1 al 10 las veces se dividen entre diez.
    fab = contar([" ".join(" ".join([f"p{'x' * k}"] * (1000 // k)) for k in range(1, 101))
                  .replace("p", "palabra").replace("x", "a")])
    o = fab.most_common()
    razon = o[0][1] / o[9][1]
    print(f"[2] señal implantada  texto fabricado: del puesto 1 al 10, las veces entre {razon:.1f}")
    if abs(razon - 10) > 0.5:
        fallos.append("señal: el recuento no ve la cola fabricada")

    # 3. INVARIANTE DEL DOMINIO — en los libros de verdad, las veces no suben nunca al bajar de
    #    puesto, y las de cada palabra suman el total.
    c = contar(libros())
    orden = [n for _, n in c.most_common()]
    ok = all(a >= b for a, b in zip(orden, orden[1:])) and sum(orden) == sum(c.values())
    print(f"[3] invariante        {miles(len(c))} palabras distintas; las veces bajan con el "
          f"puesto: {'sí' if ok else 'NO'}")
    if not ok:
        fallos.append("invariante: la lista ordenada no está ordenada")
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
    if ap.parse_args().selftest:
        sys.exit(selftest())
    informe(contar(libros()))


if __name__ == "__main__":
    main()
