#!/usr/bin/env python3
"""
Capítulo 3 — qué sortea la semilla (Carlos, 1 de octubre de 2026: «hemos de poner un ejemplo
de qué se trata eso de arrancar desde un sorteo distinto»).

La tabla del capítulo da, para un solo comité, «91,9 % (de 90,4 % a 93,0 %)»: la media, el peor y
el mejor de cinco entrenamientos desde cero (`que_mira_cada_una.py`). Esto abre esos cinco
entrenamientos, uno por fila, con las mismas cinco semillas, y enseña las dos cosas que sortea
cada semilla:

  1. el peso con el que arranca cada uno de los 64 puntos, al azar entre -1 y 1 (el listón
     arranca en 0). Desde ahí las 20.000 correcciones no tienen azar ninguno;
  2. qué 540 de los 1.797 dígitos se apartan para el examen y no se enseñan al entrenar.

Y cuánto acierta cada tirada al final. El selftest comprueba que la media, el peor y el mejor de
estas cinco filas son los de `datos/salidas/que_mira_cada_una.txt`.

Uso:
    python el_sorteo_del_arranque.py --selftest
    python el_sorteo_del_arranque.py > ../datos/salidas/el_sorteo_del_arranque.txt
"""

# ======================= CONSTANTES =======================

# Los tres puntos que se enseñan: los mismos tres de «la cuenta, punto a punto» del capítulo 3,
# para que el lector ya sepa dónde caen. (fila, columna), contando desde 1 arriba a la izquierda.
PUNTOS = [(6, 4), (4, 6), (2, 6)]
# Un punto en el que ningún dígito del conjunto tiene tinta: su peso no se corrige nunca.
PUNTO_SIN_TINTA = (1, 1)
SALIDA_TABLA = "../datos/salidas/que_mira_cada_una.txt"
SALIDA_CSV = "../datos/salidas/el_sorteo_del_arranque.csv"

TITULO_1 = "LO QUE PONE AL AZAR LA SEMILLA: LOS PESOS CON LOS QUE ARRANCA"
SUBTITULO_1 = ["un solo comité, cinco veces desde cero; de sus 64 pesos, los",
               "de tres puntos, puestos al azar entre -1 y 1; y cuántos acierta al",
               "final, de cada 100 dígitos del examen"]
TITULO_2 = "Y LO QUE TAMBIÉN PONE AL AZAR: QUÉ DÍGITOS VAN AL EXAMEN"
SUBTITULO_2 = "de los {total} dígitos, {examen} se apartan y no se enseñan al entrenar"

# ==========================================================

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

from formato import coma, pct, tabla_editorial
from perceptron import cargar_digitos
from retropropagacion import Red
from que_mira_cada_una import (datos, SEMILLA, REPETICIONES, LADO, TASA, PASOS, UMBRAL,
                               FRACCION_PRUEBA)

AQUI = Path(__file__).resolve().parent


def indice(fila, columna):
    assert 1 <= fila <= LADO and 1 <= columna <= LADO, \
        f"Se esperaba fila y columna entre 1 y {LADO}; se encontró ({fila}, {columna})"
    return (fila - 1) * LADO + (columna - 1)


def semillas():
    """Las mismas cinco de que_mira_cada_una.media_de_varias: SEMILLA + 1000·k."""
    return [SEMILLA + 1000 * k for k in range(REPETICIONES)]


def examen(semilla):
    """Qué dígitos (su número dentro del conjunto) van al examen con esta semilla. Repite el
    sorteo de que_mira_cada_una.datos y comprueba que sale lo mismo que allí."""
    Xs, _ = cargar_digitos()
    rng = np.random.default_rng(semilla)
    idx = rng.permutation(len(Xs))
    corte = int(len(Xs) * (1 - FRACCION_PRUEBA))
    _, _, Xte, _ = datos(semilla)
    assert np.array_equal(Xs[idx[corte:]], Xte), \
        "Se esperaba que el examen repetido aquí fuera el mismo que el de que_mira_cada_una.datos"
    return set(idx[corte:].tolist()), len(Xs)


def una_tirada(semilla, permutar=False):
    """Un solo comité, entrenado desde cero con esta semilla. Devuelve los pesos con los que
    arranca (antes de la primera corrección), los del final y cuánto acierta en el examen."""
    Xtr, ytr, Xte, yte = datos(semilla, permutar)
    assert Xtr.shape[1] == LADO * LADO, \
        f"Se esperaban {LADO * LADO} puntos por dibujo; se encontraron {Xtr.shape[1]}"
    red = Red([LADO * LADO, 1], semilla=semilla)
    arranque = red.W[0][:, 0].copy()
    liston_arranque = float(red.b[0][0])
    red.entrenar(Xtr, ytr, TASA, PASOS)
    dice = red.adelante(Xte)[-1].ravel() > UMBRAL
    return {"semilla": semilla, "arranque": arranque, "liston_arranque": liston_arranque,
            "final": red.W[0][:, 0].copy(), "acierta": float((dice == (yte > UMBRAL)).mean()),
            "Xtr": Xtr}


def leer_tabla(ruta):
    """La fila «una capa de un solo comité» de la salida de que_mira_cada_una.py: media, peor y
    mejor. Desde el 9 de octubre de 2026, de su tabla editorial (la salida dejó de imprimir
    renglones sangrados y este lector se había quedado atrás)."""
    from formato import leer_tablas
    for rot, filas, _ in leer_tablas(Path(ruta).read_text(encoding="utf-8")).values():
        if rot and rot[1:] == ["media", "el peor", "el mejor"]:
            for f in filas:
                if f[0] == "una capa de un solo comité":
                    return tuple(float(x.replace(" %", "").replace(",", ".")) for x in f[1:])
    raise AssertionError(f"Se esperaba la fila «una capa de un solo comité» en {ruta}; no está")


def miles(n):
    return f"{n:,}".replace(",", ".")


def signo(x):
    return ("+" if x >= 0 else "-") + coma(abs(x), 2)


def tabla(tiradas, examenes, total):
    """Desde el 9 oct 2026, tablas de libro (formato.py; REGLAS 6 ter). Los tres pesos llevan un
    rótulo en dos pisos («peso al arrancar» y debajo «6 y 4»): con «fila 6, columna 4» entero en
    cada columna, la tabla no cabía en la página."""
    filas = []
    for k, t in enumerate(tiradas, 1):
        filas.append([str(k), miles(t["semilla"])]
                     + [signo(t["arranque"][indice(f, c)]) for f, c in PUNTOS]
                     + [pct(t["acierta"])])
    t1 = tabla_editorial(
        "Lo que pone al azar la semilla: los pesos con los que arranca",
        ["tirada", "semilla"]
        + [f"peso al arrancar: {f} y {c}" for f, c in PUNTOS]
        + ["acierta"], filas, "cd" + "d" * len(PUNTOS) + "d",
        [" ".join(SUBTITULO_1)[0].upper() + " ".join(SUBTITULO_1)[1:] + ".",
         "Peso al arrancar: «6 y 4» es el punto de la fila 6, columna 4, y así los otros."])
    a = [t["acierta"] for t in tiradas]
    t_resumen = tabla_editorial(
        "Lo que acierta, de las cinco tiradas", ["", "acierta"],
        [["el peor de los cinco", pct(min(a))], ["el mejor de los cinco", pct(max(a))],
         ["**la media de los cinco**", f"**{pct(float(np.mean(a)))}**"]], "id",
        ["De cada 100 dígitos del examen."])
    primero = examenes[0]
    t2 = tabla_editorial(
        "Y lo que también pone al azar: qué dígitos van al examen",
        ["tirada", f"de los {len(primero)} de su examen, cuántos son los de la tirada 1"],
        [[str(k), f"{len(ex & primero)} de {len(ex)}"] for k, ex in enumerate(examenes[1:], 2)],
        "cd", [SUBTITULO_2.format(total=miles(total), examen=len(primero)).capitalize() + "."])
    return [t1, t_resumen, t2]


def guardar(tiradas, ruta):
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tirada", "semilla", "cual", "punto", "valor"])
        for k, t in enumerate(tiradas, 1):
            w.writerow([k, t["semilla"], "acierta", "", f"{t['acierta']:.4f}"])
            for cual in ("arranque", "final"):
                for p, v in enumerate(t[cual], 1):
                    w.writerow([k, t["semilla"], cual, p, f"{v:.6f}"])


def medir():
    tiradas = [una_tirada(s) for s in semillas()]
    examenes, total = [], None
    for s in semillas():
        ex, total = examen(s)
        examenes.append(ex)
    return tiradas, examenes, total


# ============================ SELFTEST ============================

def selftest():
    fallos = []

    # [1] Test nulo: con las respuestas barajadas no hay nada que aprender, y el mismo comité,
    #     con el mismo sorteo, tiene que caerse al azar. Si no, el 91 % no mediría nada.
    nulo = una_tirada(SEMILLA, permutar=True)["acierta"]
    print(f"[1] test nulo         respuestas barajadas: acierta {pct(nulo)} (azar, 50 %)")
    if nulo > 0.65:
        fallos.append(f"test nulo: con respuestas barajadas acierta {pct(nulo)}")

    # [2] Señal: las cinco tiradas de aquí, con las cinco semillas de la tabla del capítulo,
    #     dan la media, el peor y el mejor que imprime que_mira_cada_una.py.
    tiradas, examenes, total = medir()
    a = [100 * t["acierta"] for t in tiradas]
    aqui = (float(np.mean(a)), min(a), max(a))
    alli = leer_tabla(AQUI / SALIDA_TABLA)
    print(f"[2] señal             media, peor y mejor: aquí "
          f"{', '.join(coma(x) for x in aqui)}; en la tabla {', '.join(coma(x) for x in alli)}")
    if any(abs(round(x, 1) - y) > 1e-9 for x, y in zip(aqui, alli)):
        fallos.append("señal: estas cinco tiradas no son las de la tabla del capítulo")

    # [3] Invariante: (a) la misma semilla da el mismo arranque, bit a bit, y otra semilla da
    #     otro; (b) un punto sin tinta en ningún dígito no recibe corrección ninguna: acaba con
    #     el peso exacto que le tocó en el sorteo; (c) todos los exámenes son del mismo tamaño
    #     y ninguno es igual a otro.
    otra = una_tirada(tiradas[0]["semilla"])
    p0 = indice(*PUNTO_SIN_TINTA)
    misma = np.array_equal(otra["arranque"], tiradas[0]["arranque"])
    distinta = not np.array_equal(tiradas[0]["arranque"], tiradas[1]["arranque"])
    sin_tinta = all(float(t["Xtr"][:, p0].max()) == 0.0 for t in tiradas)
    quieto = all(t["final"][p0] == t["arranque"][p0] for t in tiradas)
    tamanos = {len(e) for e in examenes}
    iguales = sum(e == examenes[0] for e in examenes[1:])
    liston = {t["liston_arranque"] for t in tiradas}
    print(f"[3] invariante        misma semilla, mismo arranque: {misma}; otra, otro: {distinta}; "
          f"fila 1, columna 1 sin tinta: {sin_tinta}, acaba con su peso del sorteo: {quieto}; "
          f"exámenes de {sorted(tamanos)} dígitos, {iguales} repetidos; listón al arrancar: "
          f"{sorted(liston)}")
    if not (misma and distinta):
        fallos.append("invariante: la semilla no fija el arranque")
    if not (sin_tinta and quieto):
        fallos.append("invariante: el punto sin tinta ha cambiado de peso al entrenar")
    if len(tamanos) != 1 or iguales or liston != {0.0}:
        fallos.append("invariante: los exámenes o el listón de arranque no son lo que se espera")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selftest", action="store_true")
    if p.parse_args().selftest:
        return selftest()
    tiradas, examenes, total = medir()
    for t in tabla(tiradas, examenes, total):
        print("\n".join(t))
        print()
    guardar(tiradas, SALIDA_CSV)
    return 0


if __name__ == "__main__":
    sys.exit(main())
