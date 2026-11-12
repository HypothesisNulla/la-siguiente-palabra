#!/usr/bin/env python3
"""
Capítulo 9, B — ¿se sostiene la ventaja en la tarjeta con más trabajo? (Orden 24.3, 19 oct 2026).

El capítulo 9 dice que las tarjetas no explican quién gana. La medición de la tarjeta gráfica del
portátil (`en_serie_o_a_la_vez.txt`: 1,2x con 1.024 posiciones, lotes de 16 y 256 números por
posición) deja la ventaja casi en empate. ¿Se sostiene, o crece o encoge, si a la tarjeta se le da
más trabajo? Este programa mide, en la tarjeta (mps) y con 1.024 posiciones, lotes de 16, 32, 64 y
128 textos con modelos de 256 y de 1.024 números por posición: segundos en orden, segundos a la vez
y ventaja. Usa, importadas tal cual, las dos máquinas y la función de cronometrar de
`en_serie_o_a_la_vez.py`, con su sincronización de la tarjeta.

Si una combinación no cabe en la memoria, la tabla dice «no cabe» en esa casilla y se sigue con las
demás. Solo se da por «no cabe» la falta de memoria de la tarjeta; cualquier otro error revienta el
programa.

Invariante: con lote 16 y modelo de 256, la ventaja tiene que quedar a menos de un 30 % (arriba o
abajo) de la de `datos/salidas/en_serie_o_a_la_vez.txt` (1,2x con 1.024 posiciones; el
`en_serie_o_a_la_vez.csv` de `datos/salidas/` es otra medición).

=====================================================================================
PREDICCIÓN DE CLAUDE, escrita antes de medir (Orden 24, 19 oct 2026, copiada tal cual):

  La ventaja no crece con el lote; más bien baja (la que lee en orden hace 1.024 turnos de cuentas
  pequeñas y apenas nota el lote; la que mira a la vez crece con él). P(ventaja con lote 128 y
  modelo 256 por debajo de 1,2x) ≈ 0,6; P(por debajo de 1,0x) ≈ 0,35. Qué cambia el texto: si
  queda por encima de 1,0x, «en la tarjeta también gana» se sostiene, con el aviso de que por poco;
  si cae por debajo, la frase «las tarjetas no explican quién gana» no vale tal cual.
=====================================================================================

Uso:
    python lotes_en_la_tarjeta.py --selftest
    python lotes_en_la_tarjeta.py > ../datos/salidas/lotes_en_la_tarjeta.txt
"""

# ======================= CONSTANTES =======================

POSICIONES = 1024
LOTES = [16, 32, 64, 128]
ANCHOS = [256, 1024]                # números por posición
MARGEN_INVARIANTE = 0.30            # con lote 16 y modelo de 256, a menos de un 30 % de la guardada
REFERENCIA = "../datos/salidas/en_serie_o_a_la_vez.txt"   # el portátil, la tarjeta: 1,2x con 1.024
MARCA_SIN_MEMORIA = "out of memory"
NO_CABE = "no cabe"
UMBRAL_SELFTEST_LOTE = 1.2          # con 4 veces el lote, la que mira a la vez tiene que tardar más de esto
ANCHO_SALIDA = 62                   # líneas de no más de 62 caracteres, fuera de las tablas

# ==========================================================

import argparse
import gc
import platform
import re
import sys
from datetime import date
from pathlib import Path

import torch

import en_serie_o_a_la_vez as E
from formato import coma, comprobar_ancho, miles, tabla_editorial

AQUI = Path(__file__).resolve().parent


def comprobar_entrada():
    assert E.DISPOSITIVO.type == "mps", f"Se esperaba la tarjeta (mps); hay {E.DISPOSITIVO}"
    assert POSICIONES in E.LONGITUDES and LOTES == sorted(LOTES) and ANCHOS == sorted(ANCHOS)
    assert all(a % E.CABEZAS == 0 for a in ANCHOS), f"Los anchos tienen que ser múltiplos de {E.CABEZAS}"
    assert os_forzar_cpu_no(), "FORZAR_CPU=1 obliga al procesador: aquí se mide la tarjeta"


def os_forzar_cpu_no():
    import os
    return os.environ.get("FORZAR_CPU") != "1"


def calentar_la_tarjeta():
    """Una pasada de cada máquina con 64 posiciones, que se tira. La primera llamada de un proceso
    nuevo a la tarjeta tarda un 50 % más que las siguientes aunque `cronometrar` ya calienta una vez
    (medido el 19 oct 2026: 7,2 ms la primera vez con 64 posiciones y lote 16, y 4,7 ms las demás), y
    sin esto esa sobrecarga caería en la primera casilla de la tabla. No cambia nada de lo que se mide."""
    E.ANCHO, E.LOTE = ANCHOS[0], LOTES[0]
    torch.manual_seed(E.SEMILLA)
    for modelo in (E.EnSerie(E.ANCHO).to(E.DISPOSITIVO),
                   E.ALaVez(E.ANCHO, E.CABEZAS).to(E.DISPOSITIVO)):
        E.cronometrar(modelo, 64, 1)
        E.cronometrar(modelo, 64, 1)


def sin_memoria(error):
    """¿Es el error de falta de memoria de la tarjeta?"""
    return isinstance(error, RuntimeError) and MARCA_SIN_MEMORIA in str(error).lower()


def intentar(funcion):
    """Lo que devuelve `funcion()`, o None si la tarjeta se queda sin memoria. Cualquier otro error
    sube tal cual: no se tapa nada que no sea la falta de memoria."""
    try:
        return funcion()
    except RuntimeError as e:
        if not sin_memoria(e):
            raise
        return None
    finally:
        gc.collect()
        torch.mps.empty_cache()


def medir(ancho, lote, longitud=POSICIONES):
    """(segundos en orden, segundos a la vez) de una pasada con ese ancho y ese lote; None en la
    casilla que no cabe. Las dos máquinas, y el ancho y el lote que lee `cronometrar`, de
    en_serie_o_a_la_vez.py."""
    E.ANCHO, E.LOTE = ancho, lote
    torch.manual_seed(E.SEMILLA)
    serie = E.EnSerie(ancho).to(E.DISPOSITIVO)
    a_la_vez = E.ALaVez(ancho, E.CABEZAS).to(E.DISPOSITIVO)
    t_s = intentar(lambda: E.cronometrar(serie, longitud))
    t_v = intentar(lambda: E.cronometrar(a_la_vez, longitud))
    del serie, a_la_vez
    gc.collect()
    torch.mps.empty_cache()
    return t_s, t_v


def ventaja_de_referencia(ruta=AQUI / REFERENCIA, longitud=POSICIONES):
    for l in Path(ruta).read_text(encoding="utf-8").splitlines():
        m = re.match(rf"\s*{re.escape(miles(longitud))}\s+\S+\s+\S+\s+([\d,]+)x", l)
        if m:
            return float(m.group(1).replace(",", "."))
    raise AssertionError(f"No encuentro la ventaja con {longitud} posiciones en {ruta}")


def cerca(valor, referencia, margen=MARGEN_INVARIANTE):
    return referencia * (1 - margen) <= valor <= referencia * (1 + margen)


def celda(t):
    return NO_CABE if t is None else coma(t, 4)


def informe(r, emitir):
    emitir(["--- UNA PASADA EN LA TARJETA, SEGÚN EL LOTE ---", ""])
    filas = []
    for ancho in ANCHOS:
        for lote in LOTES:
            t_s, t_v = r[(ancho, lote)]
            filas.append([miles(ancho), str(lote), celda(t_s), celda(t_v),
                          NO_CABE if t_s is None or t_v is None else coma(t_s / t_v, 2) + "x"])
    emitir(tabla_editorial(
        f"Una pasada de entrenamiento con {miles(POSICIONES)} posiciones, por modelo y lote",
        ["números por posición", "lote", "en orden (s)", "a la vez (s)", "ventaja"], filas, "ddddd",
        ["Tarjeta gráfica del portátil. Lote: cuántos textos a la vez.",
         "Ventaja: cuántas veces más rápida es la que mira todo a la vez; por debajo de 1,0x es más "
         "lenta.",
         f"«{NO_CABE}»: la tarjeta se quedó sin memoria en esa casilla."]))
    emitir([""])
    filas = [[miles(a)] + [NO_CABE if None in r[(a, l)] else coma(r[(a, l)][0] / r[(a, l)][1], 2) + "x"
                           for l in LOTES] for a in ANCHOS]
    emitir(tabla_editorial(
        "La ventaja, por modelo y lote", ["números por posición"] + [f"lote {l}" for l in LOTES], filas,
        "d" * (1 + len(LOTES)), ["Es la última columna de la tabla de arriba."]))
    emitir([""])


def selftest():
    fallos = []
    comprobar_entrada()

    # 1. TEST NULO — la misma máquina contra sí misma no tiene ventaja: la que lee en orden, medida
    #    dos veces seguidas, da una razón de 1,0 con un margen del 30 %.
    calentar_la_tarjeta()
    E.ANCHO, E.LOTE = 256, 16
    torch.manual_seed(E.SEMILLA)
    serie = E.EnSerie(256).to(E.DISPOSITIVO)
    razon = E.cronometrar(serie, 64) / E.cronometrar(serie, 64)
    print(f"[1] test nulo         la de en orden contra sí misma: {coma(razon, 2)}x (tiene que estar entre "
          f"{coma(1 - MARGEN_INVARIANTE, 2)} y {coma(1 + MARGEN_INVARIANTE, 2)})")
    if not cerca(razon, 1.0):
        fallos.append(f"test nulo: la misma máquina contra sí misma da {razon}")

    # 2. SEÑAL IMPLANTADA — con cuatro veces el lote, la que mira a la vez tarda más (la tarjeta no se
    #    traga el lote gratis): la medición del lote responde a algo.
    t16 = medir(256, 16)[1]
    t64 = medir(256, 64)[1]
    print(f"[2] señal implantada  con 4 veces el lote, la que mira a la vez tarda {coma(t64 / t16, 2)} veces "
          f"más (se espera más de {coma(UMBRAL_SELFTEST_LOTE, 1)})")
    if t64 / t16 < UMBRAL_SELFTEST_LOTE:
        fallos.append("señal implantada: con cuatro veces el lote no tarda más; la medición no mide el lote")

    # 3. INVARIANTE — «no cabe» solo para la falta de memoria: un error de falta de memoria se traga y
    #    cualquier otro sube; y la ventaja de referencia sale del fichero.
    def sin_mem():
        raise RuntimeError("MPS backend out of memory (MPS allocated: 30 GB)")

    def otro_error():
        raise RuntimeError("shape mismatch")

    tragado = intentar(sin_mem) is None
    sube = False
    try:
        intentar(otro_error)
    except RuntimeError as e:
        sube = "shape mismatch" in str(e)
    ref = ventaja_de_referencia()
    print(f"[3] invariante        la falta de memoria da «{NO_CABE}»: {'sí' if tragado else 'NO'}; otro error "
          f"sube: {'sí' if sube else 'NO'}; la ventaja de referencia con {miles(POSICIONES)} posiciones es "
          f"{coma(ref, 1)}x")
    if not tragado or not sube or abs(ref - 1.2) > 1e-9:
        fallos.append("invariante: el tratamiento de la falta de memoria o la referencia no son los esperados")

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

    emitir(["######## más trabajo en la tarjeta (Orden 24) ########",
            f"fecha: {date.today().isoformat()}",
            f"máquina: {platform.system()} {platform.machine()}, Apple M4 Max, 36 GB",
            f"python {platform.python_version()}, torch {torch.__version__}",
            f"se usa la tarjeta ({E.DISPOSITIVO}, con sincronización).",
            "Los tiempos solo valen en este portátil.", ""])
    calentar_la_tarjeta()
    r = {}
    for ancho in ANCHOS:
        for lote in LOTES:
            r[(ancho, lote)] = medir(ancho, lote)
    informe(r, emitir)

    ref = ventaja_de_referencia()
    t_s, t_v = r[(256, 16)]
    assert t_s is not None and t_v is not None, "Con lote 16 y modelo de 256 tiene que caber"
    v = t_s / t_v
    ok = cerca(v, ref)
    emitir(tabla_editorial(
        "La comprobación contra la medición del portátil", ["", "ventaja"],
        [[f"la guardada, en la tarjeta, con {miles(POSICIONES)} posiciones", coma(ref, 1) + "x"],
         ["esta, con lote 16 y modelo de 256", coma(v, 2) + "x"],
         ["¿a menos de un 30 % de la guardada?", "sí" if ok else "NO"]], "id",
        ["Si no queda a menos de un 30 %, esta medición no vale."]))
    emitir([""])
    assert ok, f"con lote 16 y modelo de 256 la ventaja ({v}) no queda a menos de un 30 % de {ref}"
    return 0


if __name__ == "__main__":
    sys.exit(main())
