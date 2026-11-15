#!/usr/bin/env python3
"""
Capítulo 9 (Anexo A9, «Cómo se mide un tiempo») — cuánto se separan dos repeticiones enteras de la medición.

El anexo dice que la misma medición, repetida, casi siempre cambia muy poco, «en torno a un uno o dos por
ciento», pero que de vez en cuando alguna cifra se mueve «más de un tercio», y que la caída de la ventaja del
procesador, vista en cada repetición por separado, va «entre el 76 y el 85 %». Este programa saca esas tres
cosas de las mediciones ya guardadas (Orden 167).

Este programa NO mide: lee los csv de los barridos que ya hay en datos/salidas/, las dos repeticiones enteras
(A y B, Orden 104), cinco barridos cada una, del procesador (`en_serie_barridos_cpu-A.csv` y `-B.csv`) y de la
tarjeta (`en_serie_barridos-A.csv` y `-B.csv`). Para cada celda (tamaño de modelo, longitud del texto) y cada uno
de los dos tiempos (en serie y a la vez) toma la mediana de los barridos de A y la de B, y la diferencia entre las
dos es |A - B| / el menor de los dos. Da, por separado para el procesador y la tarjeta, la diferencia mediana y la
mayor de todas las celdas. Y, para el modelo de 256 números por posición, la parte de la ventaja del texto más
corto que queda en el más largo, en cada repetición por separado (la ventaja de cada celda es la mediana de las
ventajas de sus barridos).

POR QUÉ ESTA MEDICIÓN NO DEPENDE DE LA MÁQUINA. Se leen ficheros de texto ya guardados; no se cronometra nada.

Uso:
    python diferencia_entre_repeticiones.py --selftest
    python diferencia_entre_repeticiones.py > ../datos/salidas/diferencia_entre_repeticiones.txt
"""

# ======================= CONSTANTES =======================

CARPETA = "../datos/salidas"
# Las dos máquinas del portátil: nombre, y la base del csv de sus barridos («base-A.csv», «base-B.csv»).
BASES = (("el procesador", "en_serie_barridos_cpu"), ("la tarjeta gráfica", "en_serie_barridos"))
REPETICIONES = ("A", "B")      # las dos repeticiones enteras (Orden 104)
BARRIDOS_POR_REPETICION = 5    # los que están en el csv (los que se tiran antes no)
ANCHO_MODELO = 256             # el modelo con el que se cuenta «lo que queda»
LONGITUD_CORTA = 64            # el texto más corto de la medición
LONGITUD_LARGA = 1024          # el más largo

# ==========================================================

import argparse
import csv
import datetime
import statistics
import sys
import tempfile
from pathlib import Path

from formato import comprobar_ancho, miles, pct, tabla_editorial

AQUI = Path(__file__).resolve().parent


def leer(ruta):
    """Los barridos de un csv: {(ancho, longitud): [(segundos en serie, segundos a la vez), ...]}."""
    celdas = {}
    with open(ruta, encoding="utf-8") as fh:
        for f in csv.DictReader(fh):
            celdas.setdefault((int(f["ancho"]), int(f["longitud"])), []).append(
                (float(f["en_serie_s"]), float(f["a_la_vez_s"])))
    return celdas


def diferencias(a, b):
    """[(diferencia relativa, (ancho, longitud), qué tiempo)] de cada celda y cada tiempo. Revienta si A y B
    no tienen las mismas celdas o no tienen los mismos barridos."""
    assert a.keys() == b.keys(), "A y B no tienen las mismas celdas"
    out = []
    for celda in sorted(a):
        assert len(a[celda]) == len(b[celda]) == BARRIDOS_POR_REPETICION, (
            f"se esperaban {BARRIDOS_POR_REPETICION} barridos en cada repetición, celda {celda}: "
            f"{len(a[celda])} y {len(b[celda])}")
        for j, nombre in enumerate(("en serie", "a la vez")):
            x = statistics.median(t[j] for t in a[celda])
            y = statistics.median(t[j] for t in b[celda])
            out.append((abs(x - y) / min(x, y), celda, nombre))
    return out


def queda(celdas, ancho=ANCHO_MODELO, corto=LONGITUD_CORTA, largo=LONGITUD_LARGA):
    """La parte de la ventaja del texto más corto que sigue en el más largo, con las medianas de las ventajas de
    cada celda (sin redondear antes de dividir)."""
    def ventaja(longitud):
        return statistics.median(s / v for s, v in celdas[(ancho, longitud)])
    return ventaja(largo) / ventaja(corto)


def informe(carpeta=None):
    """Las líneas de la salida: una tabla de diferencias y las dos «queda» del procesador."""
    carpeta = Path(carpeta or AQUI / CARPETA)
    filas, detalle, quedas = [], [], {}
    for nombre, base in BASES:
        a = leer(carpeta / f"{base}-A.csv")
        b = leer(carpeta / f"{base}-B.csv")
        d = diferencias(a, b)
        mayor = max(d)
        filas.append([nombre, str(len(d)), pct(statistics.median(x[0] for x in d), 1), pct(mayor[0], 1)])
        detalle.append(f"La mayor, medida en {nombre}: {mayor[2]}, modelo de {mayor[1][0]} números, "
                       f"{miles(mayor[1][1])} posiciones.")
        if base == "en_serie_barridos_cpu":
            quedas = {"A": queda(a), "B": queda(b)}
    out = tabla_editorial(
        "Cuánto se separan las dos repeticiones enteras de la medición",
        ["medido en", "cifras comparadas", "diferencia mediana", "la mayor"], filas, "iddd",
        ["Cada cifra es la mediana de cinco barridos, una vez en la repetición A y otra en la B; la diferencia "
         "es la que hay entre las dos, dividida entre la menor. «Cifras comparadas» son las celdas (tamaño de "
         "modelo y longitud del texto) por los dos tiempos, en serie y a la vez."])
    out += [""] + detalle
    out += ["", "--- LO QUE QUEDA DE LA VENTAJA EN CADA REPETICIÓN ---", "",
            f"El procesador, el modelo de {ANCHO_MODELO} números, de {LONGITUD_CORTA} a {miles(LONGITUD_LARGA)} posiciones."]
    for r in REPETICIONES:
        out.append(f"repetición {r}: queda el {pct(quedas[r], 1)} de la ventaja del texto más corto.")
    return out


def selftest():
    fallos = []

    def escribir(carpeta, base, rep, factor, longitudes=(64, 1024)):
        with open(Path(carpeta) / f"{base}-{rep}.csv", "w", encoding="utf-8") as fh:
            fh.write("barrido,ancho,longitud,en_serie_s,a_la_vez_s,ventaja\n")
            for ancho in (256,):
                for longitud in longitudes:
                    for k in range(1, BARRIDOS_POR_REPETICION + 1):
                        s, v = 0.1 * longitud / 64 * factor, 0.05 * factor
                        fh.write(f"{k},{ancho},{longitud},{s},{v},{s / v}\n")

    # 1. TEST NULO — dos repeticiones idénticas: ninguna diferencia, y lo que queda es lo mismo en las dos.
    with tempfile.TemporaryDirectory() as t:
        for _, base in BASES:
            escribir(t, base, "A", 1.0)
            escribir(t, base, "B", 1.0)
        d = diferencias(leer(Path(t) / "en_serie_barridos_cpu-A.csv"), leer(Path(t) / "en_serie_barridos_cpu-B.csv"))
    ok = max(x[0] for x in d) == 0
    print(f"[1] test nulo         dos repeticiones idénticas: diferencia mayor {pct(max(x[0] for x in d), 1)}")
    if not ok:
        fallos.append("test nulo: dos repeticiones idénticas dan una diferencia")
    # 2. SEÑAL IMPLANTADA — B un 37 % más lenta en todo: la diferencia es 37 % en todas las celdas.
    with tempfile.TemporaryDirectory() as t:
        escribir(t, "en_serie_barridos_cpu", "A", 1.0)
        escribir(t, "en_serie_barridos_cpu", "B", 1.37)
        d = diferencias(leer(Path(t) / "en_serie_barridos_cpu-A.csv"), leer(Path(t) / "en_serie_barridos_cpu-B.csv"))
    ok = all(abs(x[0] - 0.37) < 1e-9 for x in d) and len(d) == 4
    print(f"[2] señal implantada  una repetición un 37 % más lenta: {pct(min(x[0] for x in d), 1)} en las "
          f"{len(d)} cifras")
    if not ok:
        fallos.append("señal implantada: no sale el 37 %")
    # 3. INVARIANTE — con los csv de verdad: las mismas celdas y los mismos barridos en A y en B, la diferencia
    #    mediana no pasa de la mayor y lo que queda es una proporción entre 0 y 2.
    informe_real = informe()
    carpeta = AQUI / CARPETA
    todo = []
    for _, base in BASES:
        todo += diferencias(leer(carpeta / f"{base}-A.csv"), leer(carpeta / f"{base}-B.csv"))
    mediana, mayor = statistics.median(x[0] for x in todo), max(x[0] for x in todo)
    ok = mediana <= mayor and all(0 < queda(leer(carpeta / f"en_serie_barridos_cpu-{r}.csv")) < 2 for r in REPETICIONES)
    print(f"[3] invariante        {len(todo)} cifras comparadas; mediana {pct(mediana, 1)} y mayor {pct(mayor, 1)}; "
          f"{len(informe_real)} líneas")
    if not ok:
        fallos.append("invariante: la diferencia mediana pasa de la mayor, o «queda» no es una proporción")
    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    print("--- selftest ---")
    codigo = selftest()
    if codigo or args.selftest:
        sys.exit(codigo)
    cabecera = ["", "##### cuánto se separan dos repeticiones de la medición #####",
                f"fecha: {datetime.date.today()}",
                "máquina: ninguna. Este programa NO cronometra: relee los csv",
                "de los barridos ya guardados (las dos repeticiones, A y B),",
                "así que da lo mismo en cualquier ordenador.", ""]
    for l in comprobar_ancho(cabecera):
        print(l)
    dentro = False
    for l in informe():
        if l in ("::: tabla", "::: muestra"):
            dentro = True
        elif l == ":::":
            dentro = False
        elif not dentro:
            comprobar_ancho([l], 90)
        print(l)


if __name__ == "__main__":
    main()
