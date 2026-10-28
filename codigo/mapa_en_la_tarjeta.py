#!/usr/bin/env python3
"""
Capítulo 9, C — el mapa de la tarjeta (Orden 25.3, 19 oct 2026).

La Orden 24 midió, en la tarjeta (mps), con 1.024 posiciones, que la ventaja de la que mira todo a la
vez baja con el lote (`lotes_en_la_tarjeta.txt`: de 1,44x con lote 16 a 0,32x con lote 128, con el
modelo de 256 números). Falta el mapa entero: ¿a qué longitud deja de ganar, según el lote y el
modelo? Este programa mide, en la tarjeta, longitudes de 64, 128, 256, 512 y 1.024 posiciones, lotes de
16, 32, 64 y 128 textos y modelos de 256 y de 1.024 números por posición. Usa, importadas de
`lotes_en_la_tarjeta.py` (y por ella de `en_serie_o_a_la_vez.py`), la misma función de cronometrar con
su sincronización de la tarjeta, el mismo calentamiento y las dos mismas máquinas (la que lee en orden
y la que mira a la vez, con la atención rápida de PyTorch, `need_weights=False`).

Tablas: la ventaja por longitud (columnas) y lote (filas), una por modelo; y, para cada modelo y lote,
la longitud más corta a la que la que mira a la vez deja de ganar (su ventaja baja de 1,0x) o «gana
en todas». Las casillas que no quepan en memoria dicen «no cabe», como en `lotes_en_la_tarjeta.py`.

Invariante: las casillas de 1.024 posiciones tienen que quedar a menos de un 30 % (arriba o abajo) de
las de `datos/salidas/lotes_en_la_tarjeta.txt`.

=====================================================================================
PREDICCIÓN DE CLAUDE, escrita antes de medir (Orden 25, 19 oct 2026, copiada tal cual):

  Con 64 posiciones y lote 128, la que mira a la vez sigue ganando (0,7); la longitud a la que deja
  de ganar baja cuando sube el lote (0,75); con el modelo de 1.024, a cada lote deja de ganar a una
  longitud igual o mayor que con el de 256 (0,7).
=====================================================================================

Uso:
    python mapa_en_la_tarjeta.py --selftest
    python mapa_en_la_tarjeta.py > ../datos/salidas/mapa_en_la_tarjeta.txt
    python mapa_en_la_tarjeta.py --desde-guardado > ../datos/salidas/mapa_en_la_tarjeta.txt

Con `--desde-guardado` (Orden 98) no se mide nada: se leen de `datos/salidas/mapa_en_la_tarjeta.csv`
las cifras que midió la Orden 25 (las de la salida guardada, con sus dos decimales y sus cabeceras:
fecha, máquina y versiones son las de aquella medición) y se escribe la salida entera. Sirve para
cambiar el texto de las notas sin que las cifras se muevan: una medición nueva en la tarjeta, aunque
sea con la misma máquina, da otras cifras. La medición de verdad es la de sin modo.
"""

# ======================= CONSTANTES =======================

LONGITUDES = [64, 128, 256, 512, 1024]
LOTES = [16, 32, 64, 128]
ANCHOS = [256, 1024]                # números por posición
MARGEN_INVARIANTE = 0.30            # las casillas de 1.024 posiciones, a menos de un 30 % de las guardadas
REFERENCIA = "../datos/salidas/lotes_en_la_tarjeta.txt"
GUARDADO = "../datos/salidas/mapa_en_la_tarjeta.csv"   # lo medido en la Orden 25, para --desde-guardado
GANA_EN_TODAS = "gana en todas"
ANCHO_SALIDA = 62                   # líneas de no más de 62 caracteres, fuera de las tablas

# ==========================================================

import argparse
import csv
import platform
import re
import sys
from datetime import date
from pathlib import Path

import torch

import en_serie_o_a_la_vez as E
import lotes_en_la_tarjeta as L
from formato import coma, comprobar_ancho, miles, tabla_editorial

AQUI = Path(__file__).resolve().parent


def comprobar_entrada():
    L.comprobar_entrada()
    assert 1024 in LONGITUDES and LONGITUDES == sorted(LONGITUDES)
    assert LOTES == L.LOTES and ANCHOS == L.ANCHOS, "Los lotes y los modelos son los de lotes_en_la_tarjeta.py"


def ventaja(t_s, t_v):
    return None if t_s is None or t_v is None else t_s / t_v


def deja_de_ganar(ventajas):
    """La longitud más corta a la que la ventaja baja de 1,0x, o None si no hay ninguna. `ventajas`:
    {longitud: ventaja o None}; las casillas que no caben (None) no cuentan."""
    perdiendo = [l for l in sorted(ventajas) if ventajas[l] is not None and ventajas[l] < 1.0]
    return perdiendo[0] if perdiendo else None


def texto_umbral(ventajas):
    u = deja_de_ganar(ventajas)
    if u is not None:
        return miles(u)
    return GANA_EN_TODAS + (" las que caben" if None in ventajas.values() else "")


def referencia(ruta=AQUI / REFERENCIA):
    """{(ancho, lote): ventaja con 1.024 posiciones} de la tabla guardada de lotes_en_la_tarjeta.txt."""
    out = {}
    for l in Path(ruta).read_text(encoding="utf-8").splitlines():
        m = re.match(r"\|\s*([\d.]+)\s*\|\s*(\d+)\s*\|[^|]*\|[^|]*\|\s*([\d,]+)x\s*\|", l)
        if m:
            out[(int(m.group(1).replace(".", "")), int(m.group(2)))] = float(m.group(3).replace(",", "."))
    assert len(out) == len(ANCHOS) * len(LOTES), f"Se esperaban {len(ANCHOS) * len(LOTES)} casillas; hay {len(out)}"
    return out


def cerca(valor, ref, margen=MARGEN_INVARIANTE):
    return ref * (1 - margen) <= valor <= ref * (1 + margen)


def leer_guardado(ruta=AQUI / GUARDADO):
    """(cabecera, {(ancho, lote, longitud): ventaja}) del csv de lo medido en la Orden 25. La cabecera
    son las cinco primeras líneas de la salida de entonces, que el csv guarda tras «# salida: »."""
    cabecera, filas = [], []
    for l in Path(ruta).read_text(encoding="utf-8").splitlines():
        if l.startswith("# salida: "):
            cabecera.append(l[len("# salida: "):])
        elif not l.startswith("#"):
            filas.append(l)
    v = {}
    for f in csv.DictReader(filas):
        v[(int(f["ancho"]), int(f["lote"]), int(f["longitud"]))] = float(f["ventaja"]) if f["ventaja"] else None
    esperadas = {(a, b, c) for a in ANCHOS for b in LOTES for c in LONGITUDES}
    assert set(v) == esperadas, f"Se esperaban las {len(esperadas)} casillas en {ruta}; hay {len(v)}"
    assert len(cabecera) == 5, f"Se esperaban cinco líneas de cabecera en {ruta}; hay {len(cabecera)}"
    return cabecera, v


def celda(v):
    return L.NO_CABE if v is None else coma(v, 2) + "x"


def informe(v, emitir):
    emitir(["--- LA VENTAJA, POR LONGITUD Y LOTE ---", ""])
    for ancho in ANCHOS:
        filas = [[str(lote)] + [celda(v[(ancho, lote, l)]) for l in LONGITUDES] for lote in LOTES]
        emitir(tabla_editorial(
            f"La ventaja, con el modelo de {miles(ancho)} números por posición",
            ["lote"] + [miles(l) for l in LONGITUDES], filas, "d" * (1 + len(LONGITUDES)),
            ["Columnas: posiciones del texto. Filas: textos a la vez.",
             "Ventaja: cuántas veces más rápida es la que mira todo a la vez; por debajo de 1,0x es "
             f"más lenta. «{L.NO_CABE}»: la tarjeta se quedó sin memoria."]))
        emitir([""])
    emitir(["--- DÓNDE DEJA DE GANAR ---", ""])
    filas = []
    for ancho in ANCHOS:
        for lote in LOTES:
            filas.append([miles(ancho), str(lote),
                          texto_umbral({l: v[(ancho, lote, l)] for l in LONGITUDES})])
    emitir(tabla_editorial(
        "La longitud más corta a la que deja de ganar",
        ["números por posición", "lote", "deja de ganar con"], filas, "dd" + "d",
        [# Orden 97: nota para el lector, en una sola frase larga.
         "En la tarjeta gráfica, para cada modelo y cada lote, es decir, cada número de textos a la "
         "vez: la longitud más corta, en posiciones, a la que la máquina que mira todo a la vez ya "
         f"no es más rápida. «{GANA_EN_TODAS.capitalize()}» quiere decir que es más rápida en "
         f"todas, de {miles(LONGITUDES[0])} a {miles(LONGITUDES[-1])} posiciones."]))
    emitir([""])


def selftest():
    fallos = []
    comprobar_entrada()

    # 1. TEST NULO — la misma máquina contra sí misma no tiene ventaja (razón 1,0 con un 30 % de margen),
    #    y una tabla donde la que mira a la vez gana siempre no tiene longitud a la que deje de ganar.
    L.calentar_la_tarjeta()
    E.ANCHO, E.LOTE = 256, 16
    torch.manual_seed(E.SEMILLA)
    serie = E.EnSerie(256).to(E.DISPOSITIVO)
    razon = E.cronometrar(serie, 64) / E.cronometrar(serie, 64)
    siempre = deja_de_ganar({64: 3.9, 128: 2.9, 256: 2.7, 512: 2.4, 1024: 1.2})
    print(f"[1] test nulo         la de en orden contra sí misma: {coma(razon, 2)}x (entre 0,70 y 1,30); "
          f"si gana siempre, deja de ganar con: {'ninguna' if siempre is None else siempre}")
    if not cerca(razon, 1.0) or siempre is not None:
        fallos.append("test nulo: la misma máquina contra sí misma, o una tabla que siempre gana, mal contadas")

    # 2. SEÑAL IMPLANTADA — con lote 16 y modelo de 256 la ventaja baja al crecer la longitud (3,9x con 64
    #    posiciones en la medición del portátil): con 64 posiciones tiene que ser mayor que con 1.024; y
    #    la función encuentra el umbral de una tabla fabricada.
    t64 = L.medir(256, 16, 64)
    t1024 = L.medir(256, 16, 1024)
    v64, v1024 = ventaja(*t64), ventaja(*t1024)
    fab = deja_de_ganar({64: 3.0, 128: 2.0, 256: 1.1, 512: 0.8, 1024: 0.5})
    print(f"[2] señal implantada  ventaja con lote 16 y modelo de 256: {coma(v64, 2)}x con 64 posiciones y "
          f"{coma(v1024, 2)}x con 1.024; en una tabla fabricada deja de ganar con {fab}")
    if not v64 > v1024 * 1.2 or fab != 512:
        fallos.append("señal implantada: la ventaja no baja con la longitud, o el umbral no se encuentra")

    # 3. INVARIANTE — la referencia (las ocho casillas de lotes_en_la_tarjeta.txt) se lee entera; las que no
    #    caben no cuentan al buscar el umbral, y se dice cuáles se saltan; y 1.024 está entre las longitudes.
    ref = referencia()
    con_hueco = deja_de_ganar({64: 2.0, 128: None, 256: 0.9})
    texto = texto_umbral({64: 2.0, 128: None, 256: 1.5})
    print(f"[3] invariante        casillas de referencia leídas: {len(ref)}; con una casilla que no cabe, "
          f"deja de ganar con {con_hueco}; y si gana en las demás: «{texto}»")
    if len(ref) != 8 or con_hueco != 256 or texto != GANA_EN_TODAS + " las que caben":
        fallos.append("invariante: la referencia, o el tratamiento de las casillas que no caben, no son los esperados")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--desde-guardado", action="store_true",
                    help="no mide: reimprime la salida con las cifras guardadas en el csv de la Orden 25")
    args = ap.parse_args()
    comprobar_entrada()
    if args.selftest:
        return selftest()

    def emitir(lineas):
        dentro = getattr(emitir, "dentro", False)
        for l in lineas:
            if l.startswith("::: "):
                dentro = True
            if not dentro:
                comprobar_ancho([l], ANCHO_SALIDA)
            if l == ":::":
                dentro = False
            print(l)
        emitir.dentro = dentro

    if args.desde_guardado:
        cabecera, v = leer_guardado()
        emitir(cabecera + ["Los tiempos solo valen en este portátil.", ""])
    else:
        emitir(["######## el mapa de la tarjeta (Orden 25) ########",
                f"fecha: {date.today().isoformat()}",
                f"máquina: {platform.system()} {platform.machine()}, Apple M4 Max, 36 GB",
                f"python {platform.python_version()}, torch {torch.__version__}",
                f"se usa la tarjeta ({E.DISPOSITIVO}, con sincronización).",
                "Los tiempos solo valen en este portátil.", ""])
        L.calentar_la_tarjeta()
        v, tiempos = {}, {}
        for ancho in ANCHOS:
            for lote in LOTES:
                for longitud in LONGITUDES:
                    t = L.medir(ancho, lote, longitud)
                    tiempos[(ancho, lote, longitud)] = t
                    v[(ancho, lote, longitud)] = ventaja(*t)
    informe(v, emitir)

    ref = referencia()
    filas, malas = [], []
    for (ancho, lote), r in sorted(ref.items()):
        mia = v[(ancho, lote, 1024)]
        ok = mia is not None and cerca(mia, r)
        filas.append([miles(ancho), str(lote), coma(r, 2) + "x", celda(mia), "sí" if ok else "NO"])
        if not ok:
            malas.append((ancho, lote, mia, r))
    emitir(tabla_editorial(
        "La comprobación contra lotes_en_la_tarjeta.txt",
        ["números por posición", "lote", "la guardada", "esta", "¿a menos de un 30 %?"], filas, "dddd" + "c",
        ["Las casillas de 1.024 posiciones. Si alguna no queda a menos de un 30 %, esta medición no vale."]))
    emitir([""])
    assert not malas, f"Casillas de 1.024 posiciones a más de un 30 % de las guardadas: {malas}"
    return 0


if __name__ == "__main__":
    sys.exit(main())
