#!/usr/bin/env python3
"""
Capítulo 6 — el ejercicio de memoria, con una prueba escrita entera (L24).

El capítulo 6 mide cuánto recuerda una red que lee arrastrando un resumen: se le enseña un
símbolo, luego un montón de símbolos «de paja», y al final tiene que decir cuál era el primero.
El libro lo contaba sin enseñar ni una prueba, y daba el acierto en tantos por uno. Esto:

  1. Escribe pruebas de verdad, hechas con la misma función que usa el entrenamiento
     (`memoria_recurrente.lote_memoria`), con los símbolos puestos en letras y números para que
     se puedan leer: los ocho que hay que recordar son letras (A a H) y los de paja, números.
  2. Pone la tabla de aciertos de `datos/salidas/memoria_recurrente.txt` en «de cada 100
     pruebas». No entrena nada: lee esa salida (entrenar las veinte redes otra vez daría las
     mismas cifras, y es lo que ya hizo aquel programa, con sus pruebas).

Uso:
    python una_prueba_de_memoria.py --selftest
    python una_prueba_de_memoria.py
"""

# ======================= CONSTANTES =======================

SALIDA_MEMORIA = "../datos/salidas/memoria_recurrente.txt"
PRUEBAS_A_ESCRIBIR = [(10, 1), (40, 1)]        # (cuánta paja, cuántas pruebas)
SEMILLA_EJEMPLO = 20260928
LETRAS = "ABCDEFGH"
NUMEROS_PAJA = "1234"
# Orden 97: cifras que las notas escriben con letra; si cambia el parámetro, falla aquí y no en la página.
EN_LETRA = {8: "ocho", 2: "dos"}

# ==========================================================

import argparse
import os
import re
import sys

import numpy as np

import memoria_recurrente as mr
from formato import coma, tabla_editorial

AQUI = os.path.dirname(os.path.abspath(__file__))


def a_texto(x):
    """Un símbolo del programa, como se lee en el libro."""
    x = int(x)
    assert 0 <= x < mr.SIMBOLOS + mr.RELLENO, f"símbolo fuera de rango: {x}"
    return LETRAS[x] if x < mr.SIMBOLOS else NUMEROS_PAJA[x - mr.SIMBOLOS]


def leer_tabla(ruta):
    """distancia -> (media simple, peor, mejor, media compuertas, peor, mejor), de la salida."""
    texto = open(ruta, encoding="utf-8").read()
    assert "1. RECORDAR A DISTANCIA" in texto, f"se esperaba el bloque 1 en {ruta}"
    bloque = texto.split("1. RECORDAR A DISTANCIA", 1)[1].split("\n--- ", 1)[0]
    num = r"(\d+,\d+)"
    out = {}
    for l in bloque.splitlines():
        m = re.match(rf"^\s*(\d+)\s+{num} \({num}-{num}\)\s+{num} \({num}-{num}\)\s*$", l)
        if m:
            v = [float(x.replace(",", ".")) for x in m.groups()[1:]]
            out[int(m.group(1))] = v
    assert sorted(out) == mr.DISTANCIAS, f"se esperaban las distancias {mr.DISTANCIAS}; hay {sorted(out)}"
    return out


def de_cada_cien(x):
    return coma(100 * x, 0)


def bloque_pruebas():
    """Cada prueba, como tabla editorial (L24, 9 de octubre; regla 6 ter)."""
    rng = np.random.default_rng(SEMILLA_EJEMPLO)
    out = ["--- 1. UNA PRUEBA, ESCRITA ENTERA ---", ""]
    for distancia, n in PRUEBAS_A_ESCRIBIR:
        x, y = mr.lote_memoria(distancia, n, rng)
        for fila, objetivo in zip(x.numpy(), y.numpy()):
            simbolos = [a_texto(s) for s in fila]
            out += tabla_editorial(
                f"Una prueba, con {distancia} de paja", ["", "los símbolos"],
                [["se le enseña", simbolos[0]], ["luego, la paja", " ".join(simbolos[1:])],
                 ["pregunta", "¿cuál era el primero?"],
                 ["**respuesta buscada**", f"**{a_texto(objetivo)}**"]], "ii",
                [f"Las letras, de la {LETRAS[0]} a la {LETRAS[-1]}, son lo que se le puede pedir; la "
                 f"paja son cifras del {NUMEROS_PAJA[0]} al {NUMEROS_PAJA[-1]}, así que nunca se "
                 "confunde con ellas."])
            out.append("")
    return out[:-1]


def bloque_tabla(tabla):
    azar = 1 / mr.SIMBOLOS
    return ["--- 2. DE CADA 100 PRUEBAS, CUÁNTAS ACIERTA ---", ""] + tabla_editorial(
        "De cada 100 pruebas, cuántas acierta",
        ["paja en medio, en símbolos", "memoria simple", "con compuertas"],
        [[str(d), de_cada_cien(tabla[d][0]), de_cada_cien(tabla[d][3])] for d in mr.DISTANCIAS],
        "ddd",
        [f"Por puro azar acertaría {coma(100 * azar, 1)} de cada cien, una de cada "
         f"{EN_LETRA[mr.SIMBOLOS]}. Cada cifra es la media de {EN_LETRA[mr.SEMILLAS_MEMORIA]} "
         "entrenamientos desde cero."])


def selftest():
    fallos = []
    # 1. TEST NULO — en muchas pruebas, cada una de las ocho letras sale más o menos lo mismo
    #    de primera (una de cada ocho): si no, acertar «al azar» no sería una de cada ocho.
    rng = np.random.default_rng(1)
    _, y = mr.lote_memoria(10, 8000, rng)
    frec = np.bincount(y.numpy(), minlength=mr.SIMBOLOS) / 8000
    print(f"[1] test nulo         cada letra de primera: de {coma(100 * frec.min(), 1)} a "
          f"{coma(100 * frec.max(), 1)} de cada 100 (azar: {coma(100 / mr.SIMBOLOS, 1)})")
    if not (frec.min() > 0.11 and frec.max() < 0.14):
        fallos.append(f"test nulo: las letras no salen parejas: {frec}")
    # 2. SEÑAL IMPLANTADA — la respuesta buscada es siempre el primer símbolo, y la paja nunca
    #    es una de las letras que hay que recordar.
    x, y = mr.lote_memoria(40, 500, rng)
    x = x.numpy()
    ok = bool((x[:, 0] == y.numpy()).all() and (x[:, 1:] >= mr.SIMBOLOS).all())
    print(f"[2] señal implantada  respuesta = primer símbolo y paja fuera de las letras: {ok}")
    if not ok:
        fallos.append("señal implantada: la prueba no es la que describe el capítulo")
    # 3. INVARIANTE — la tabla que se lee tiene las cinco distancias, cada media está entre su
    #    peor y su mejor, y todo está entre 0 y 1.
    t = leer_tabla(os.path.join(AQUI, SALIDA_MEMORIA))
    malas = [d for d, v in t.items()
             if not (v[1] <= v[0] <= v[2] and v[4] <= v[3] <= v[5] and all(0 <= z <= 1 for z in v))]
    print(f"[3] invariante        distancias leídas {sorted(t)}; filas incoherentes: {malas or 'ninguna'}")
    if malas:
        fallos.append(f"invariante: filas incoherentes {malas}")
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
    if args.selftest:
        sys.exit(selftest())
    if selftest():
        sys.exit("El selftest falla.")
    print()
    print("\n".join(bloque_pruebas()) + "\n")
    print("\n".join(bloque_tabla(leer_tabla(os.path.join(AQUI, SALIDA_MEMORIA)))))


if __name__ == "__main__":
    main()
