#!/usr/bin/env python3
"""
Capítulo 14 — dónde se equivoca la cuenta escrita paso a paso.

No carga ningún modelo: lee `pasos_escritos.csv`, donde están las 20 cuentas enteras que escribió
el modelo de 32.000 millones, y mira cada una que falla. Una multiplicación de dos números de
cuatro cifras, hecha como en la escuela, son cuatro productos parciales (el primer número por
cada cifra del segundo, con sus ceros) y una suma. Para cada cuenta que falla se comprueba si
los cuatro productos parciales correctos están escritos en el texto. Si lo están, la cuenta se
torció al sumar; si falta alguno, se torció antes.

Un producto parcial que vale 0 (una cifra 0 en el segundo número) se da por escrito.

Y, en las 20, si está escrito lo que se lleva de una columna a la siguiente («llevamos 1»): al
multiplicar los parciales, o en la suma. La suma empieza donde la respuesta dice por primera vez
«suma», «sumamos» o «sumar» (auditoría del 10 de octubre de 2026: el capítulo decía que lo que se
lleva «casi nunca» está escrito, y hacía falta el recuento).

Uso (desde codigo/):
    python pasos_a_la_vista.py --selftest
    python pasos_a_la_vista.py > ../datos/salidas/pasos_a_la_vista.txt
"""

# ======================= CONSTANTES =======================

ENTRADA = "../datos/salidas/pasos_escritos.csv"
N_ESPERADAS = 20
# Orden 97: cifras que las notas escriben con letra; si cambia el recuento, falla aquí y no en la página.
EN_LETRA = {20: "veinte", 10: "diez", 9: "nueve"}

# ==========================================================

import argparse
import csv
import re
import sys

from formato import miles, tabla_editorial

ENTERO = re.compile(r"\d+")
LLEVA = re.compile(r"llev|acarre", re.I)       # «llevamos 1», «me llevo», «acarreo»
SUMA = re.compile(r"\bsum", re.I)              # donde empieza la suma


def lleva(texto):
    """¿Está escrito lo que se lleva? (al multiplicar los parciales, en la suma)."""
    m = SUMA.search(texto)
    corte = m.start() if m else len(texto)
    return bool(LLEVA.search(texto[:corte])), bool(LLEVA.search(texto[corte:]))


def parciales(a, b):
    """Los productos parciales de la cuenta de la escuela: a por cada cifra de b, con sus ceros."""
    return [a * int(c) * 10 ** i for i, c in enumerate(reversed(str(b)))]


def faltan(texto, a, b):
    """Los productos parciales correctos (distintos de 0) que no aparecen escritos en el texto."""
    escritos = {int(x) for x in ENTERO.findall(texto.replace(".", ""))}
    return [p for p in parciales(a, b) if p != 0 and p not in escritos]


def leer(ruta=ENTRADA):
    fs = list(csv.DictReader(open(ruta, encoding="utf-8")))
    assert len(fs) == N_ESPERADAS, f"Se esperaban {N_ESPERADAS} cuentas en {ruta}; hay {len(fs)}"
    return fs


def selftest():
    ok = True
    # [1] test nulo: una cuenta que no escribe ningún parcial no los tiene todos
    f1 = faltan("El producto es 32470340.", 6737, 4820)
    p1 = len(f1) == 3          # 4820 tiene un 0: quedan tres parciales distintos de 0
    p1 &= lleva("6737 por 4820: 0 + 134740 + 5389600 + 26948000. Sumamos: 32472340.") == (
        False, False)
    ok &= p1
    print(f"[1] test nulo         sin parciales escritos, faltan {len(f1)} de 3: "
          f"{'bien' if p1 else 'MAL'}")
    # [2] señal implantada: los cuatro parciales escritos y la suma mal, se reconoce como «al
    #     sumar»; con uno quitado, como «antes»
    t = "0 + 134740 + 5389600 + 26948000 = 32470340"
    t2 = "0 + 134740 + 26948000 = 27082740"
    p2 = faltan(t, 6737, 4820) == [] and faltan(t2, 6737, 4820) == [5389600]
    p2 &= lleva("4 * 4 = 16, escribimos 6 y llevamos 1. Sumamos: 6 + 9 = 15, me llevo 1") == (
        True, True)
    p2 &= lleva("4 * 4 = 16, escribimos 6 y llevamos 1. Sumamos: 16 + 40 = 56") == (True, False)
    ok &= p2
    print(f"[2] señal implantada  con los cuatro parciales: faltan {faltan(t, 6737, 4820)}; sin "
          f"uno: faltan {faltan(t2, 6737, 4820)}: {'bien' if p2 else 'MAL'}")
    # [3] invariante: los parciales suman el producto, en las 20 preguntas del CSV
    fs = leer()
    malas = sum(sum(parciales(int(f["a"]), int(f["b"]))) != int(f["producto"]) for f in fs)
    p3 = malas == 0
    ok &= p3
    print(f"[3] invariante        los parciales suman el producto en {N_ESPERADAS - malas} de "
          f"{N_ESPERADAS}: {'bien' if p3 else 'MAL'}")
    print()
    print("SELFTEST: las tres pruebas pasan." if ok else "SELFTEST: FALLA.")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        sys.exit(0 if selftest() else 1)
    fs = leer()
    mal = [f for f in fs if f["leido"] != f["producto"]]
    filas, al_sumar = [], 0
    for f in mal:
        a, b, prod, leido = int(f["a"]), int(f["b"]), int(f["producto"]), int(f["leido"])
        falta = faltan(f["respuesta"], a, b)
        al_sumar += not falta
        # Los factores, de cuatro cifras, sin punto: como los escribe la pregunta que recibió. Con
        # «por» y no con el signo: la regla 1 (cero notación) vale también para lo que imprime un
        # programa del libro, como en el capítulo 3.
        filas.append([f"{a} por {b}", miles(leido), miles(prod),
                      "bien" if not falta else f"falta {', '.join(miles(x) for x in falta)}"])
    print("\n".join(tabla_editorial(
        "Las cuentas escritas paso a paso que fallan (32.000M)",
        ["multiplicación", "escribe", "exacto", "sus productos parciales"], filas, "iddi",
        [# Orden 97: notas para el lector. Lo que se lleva «al multiplicar los parciales» (3 de 20) y el
         # criterio de «falta» (el parcial que no aparece en ninguna parte) están en `lleva` y `faltan`.
         f"Las {EN_LETRA[len(mal)]} de las {EN_LETRA[len(fs)]} cuentas que fallan.",
         "Productos parciales: el primer número por cada cifra del segundo, con sus ceros, como "
         "en la cuenta de la escuela. «Bien» quiere decir que los cuatro están escritos y son "
         f"correctos: el error llegó al sumarlos. Pasa en {EN_LETRA[al_sumar]} de {EN_LETRA[len(mal)]}.",
         f"En ninguna de las {EN_LETRA[len(fs)]} escribe lo que se lleva de una columna a la "
         "siguiente al sumar."])))
    assert sum(lleva(f["respuesta"])[1] for f in fs) == 0, (
        "la nota dice que en ninguna de las veinte escribe lo que se lleva al sumar")


if __name__ == "__main__":
    main()
