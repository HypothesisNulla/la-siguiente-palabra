#!/usr/bin/env python3
"""
Capítulo 9, A — ¿son los núcleos? (Orden 24.2, 19 oct 2026).

El texto del capítulo 9 dice que entre el ordenador de dos núcleos y el portátil «lo único que
cambia es cuántas cosas puede hacer cada uno a la vez: dos contra catorce núcleos». Este programa lo
pone a prueba en el procesador del portátil, con el mismo modelo (256 números por posición) y el
mismo lote (16) de `en_serie_o_a_la_vez.py`, que se importa tal cual, con su función de cronometrar
y su sincronización de la tarjeta (aquí no se usa: se fuerza el procesador). Las longitudes son las
suyas (64, 128, 256, 512 y 1.024) y lo único que cambia es cuántos hilos usa el procesador:
`torch.set_num_threads(n)` con n = 1, 2, 4, 8 y 14 (y `torch.set_num_interop_threads(1)` una vez, al
principio).

Tabla: segundos en orden, segundos a la vez y ventaja, por n y longitud.
Diagnóstico aparte: el tiempo de una multiplicación de matrices de 1.024 × 1.024 con n = 1 y con
n = 14. Si sale casi igual, en este Mac las multiplicaciones no las reparten los hilos (van a la
unidad de matrices) y hay que decirlo.
Invariante: con n = 14, la ventaja con 1.024 posiciones tiene que quedar a menos de un 30 % (arriba o
abajo) de la que dio el portátil en el procesador, 3,2x (`datos/salidas/en_serie_o_a_la_vez_cpu.txt`;
el `en_serie_o_a_la_vez.csv` de `codigo/` es otra medición, la de 4,83x).

=====================================================================================
PREDICCIÓN DE CLAUDE, escrita antes de medir (Orden 24, 19 oct 2026, copiada tal cual):

  Con 1 o 2 hilos la ventaja baja, pero puede no cruzar por debajo de 1,0x porque las
  multiplicaciones grandes van a la unidad de matrices. P(cruza por debajo de 1,0x con 1.024
  posiciones y 1 o 2 hilos) ≈ 0,45. Qué cambia el texto: si cruza, el texto se sostiene; si no
  cruza, la frase de los catorce núcleos es falsa.
=====================================================================================

Orden 101: una sola medición varía demasiado de una vez a otra, así que la medición son BARRIDOS barridos
completos (tras BARRIDOS_TIRADOS que se tiran, con las longitudes al revés si LONGITUDES_AL_REVES: el
protocolo de la Orden 100) y cada celda es la mediana de los BARRIDOS. Los barridos, con sus tiempos, se
guardan en `datos/salidas/cuantos_hilos.csv`.

Orden 105: la salida del libro sale de DOS repeticiones enteras, A y B (Orden 104), que `--desde-guardado`
lee juntas sin medir nada: `datos/salidas/cuantos_hilos-A.csv` y `-B.csv`. Cada celda es la mediana de las diez
mediciones (cinco barridos de cada una). La multiplicación de matrices no se guarda en el csv y ese modo no la
repite: su sección no sale.

Uso:
    python cuantos_hilos.py --selftest
    python cuantos_hilos.py > ../datos/salidas/cuantos_hilos.txt
    python cuantos_hilos.py --desde-guardado > ../datos/salidas/cuantos_hilos.txt
"""

# ======================= CONSTANTES =======================

BARRIDOS = 5                       # barridos completos; cada celda es la mediana de los BARRIDOS
BARRIDOS_TIRADOS = 1               # barridos enteros que se miden antes y se tiran
LONGITUDES_AL_REVES = True         # las longitudes, de la más larga a la más corta
CSV_BARRIDOS = "../datos/salidas/cuantos_hilos.csv"          # donde escribe una medición nueva (luego se guarda como A o B)
GUARDADOS = ["../datos/salidas/cuantos_hilos-A.csv",         # las dos repeticiones que lee --desde-guardado (Orden 105)
             "../datos/salidas/cuantos_hilos-B.csv"]
EN_LETRA = {5: "cinco", 10: "diez"}  # cifra que la nota escribe con letra; si no está, falla aquí y no en la página
HILOS = [1, 2, 4, 8, 14]           # el portátil tiene 14 núcleos: 10 de rendimiento y 4 de eficiencia
HILOS_ENTRE_PROCESOS = 1           # torch.set_num_interop_threads, una sola vez
LONGITUD_LARGA = 1024              # donde se mira la predicción
MATRIZ = 1024                      # la multiplicación del diagnóstico: MATRIZ × MATRIZ
REPETICIONES_MATRIZ = 40           # multiplicaciones por tanda
TANDAS_MATRIZ = 5                  # tandas; se da la mediana
CASI_IGUAL = 1.3                   # n = 1 contra n = 14: por debajo de esta razón, «casi igual»
MARGEN_INVARIANTE = 0.30           # la ventaja con 14 hilos, a menos de un 30 % de la de referencia
REFERENCIA = "../datos/salidas/en_serie_o_a_la_vez_cpu.txt"   # el portátil, procesador: 3,2x con 1.024
UMBRAL_SELFTEST_HILOS = 1.5        # el interruptor de hilos tiene que acelerar una operación repartible
ANCHO_SALIDA = 62                  # líneas de no más de 62 caracteres, fuera de las tablas

# ==========================================================

import os

# El cronómetro de en_serie_o_a_la_vez.py decide el dispositivo al importarlo: se le pide el
# procesador antes de importarlo (es lo que hace su variable FORZAR_CPU).
os.environ["FORZAR_CPU"] = "1"

import argparse
import csv
import platform
import re
import statistics
import sys
import time
from datetime import date
from pathlib import Path

import torch

import en_serie_o_a_la_vez as E
from formato import coma, comprobar_ancho, miles, tabla_editorial

AQUI = Path(__file__).resolve().parent


def comprobar_entrada():
    assert E.DISPOSITIVO.type == "cpu", f"Se esperaba el procesador; hay {E.DISPOSITIVO}"
    assert (E.ANCHO, E.LOTE) == (256, 16), f"Se esperaban 256 números y lotes de 16; hay {(E.ANCHO, E.LOTE)}"
    assert list(E.LONGITUDES) == [64, 128, 256, 512, 1024], f"Longitudes inesperadas: {E.LONGITUDES}"
    assert LONGITUD_LARGA in E.LONGITUDES and HILOS == sorted(HILOS) and HILOS[0] == 1
    assert HILOS[-1] <= (os.cpu_count() or 0), f"El portátil no tiene {HILOS[-1]} núcleos"


def poner_hilos(n):
    torch.set_num_threads(n)
    assert torch.get_num_threads() == n, f"Se pidieron {n} hilos y hay {torch.get_num_threads()}"


def modelos():
    torch.manual_seed(E.SEMILLA)
    return E.EnSerie(E.ANCHO).to(E.DISPOSITIVO), E.ALaVez(E.ANCHO, E.CABEZAS).to(E.DISPOSITIVO)


def un_barrido(serie, a_la_vez):
    """{(n, longitud): (segundos en orden, segundos a la vez)} con la función de cronometrar de
    en_serie_o_a_la_vez.py, para cada número de hilos."""
    r = {}
    longitudes = E.LONGITUDES[::-1] if LONGITUDES_AL_REVES else E.LONGITUDES
    for n in HILOS:
        poner_hilos(n)
        for longitud in longitudes:
            r[(n, longitud)] = (E.cronometrar(serie, longitud), E.cronometrar(a_la_vez, longitud))
    return r


def medir_pasadas(serie, a_la_vez):
    """Los BARRIDOS barridos (tras los tirados): {(barrido, n, longitud): (en orden, a la vez)}, con los
    segundos a seis decimales (los del csv; de ellos sale todo)."""
    for _ in range(BARRIDOS_TIRADOS):
        un_barrido(serie, a_la_vez)
    crudo = {}
    for b in range(1, BARRIDOS + 1):
        for (n, longitud), (t_s, t_v) in un_barrido(serie, a_la_vez).items():
            crudo[(b, n, longitud)] = (round(t_s, 6), round(t_v, 6))
    return crudo


def escribir_barridos(crudo, ruta=AQUI / CSV_BARRIDOS):
    with open(ruta, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["barrido", "hilos", "longitud", "t_en_orden", "t_a_la_vez", "ventaja"])
        for (b, n, longitud), (t_s, t_v) in sorted(crudo.items()):
            w.writerow([b, n, longitud, f"{t_s:.6f}", f"{t_v:.6f}", f"{t_s / t_v:.4f}"])


def leer_guardados(rutas=None):
    """{(barrido, n, longitud): (en orden, a la vez)} de los csv guardados (por omisión, GUARDADOS: las repeticiones
    A y B, Orden 105); los barridos del segundo siguen la numeración del primero (6 a 10 tras 1 a 5)."""
    rutas = rutas or [AQUI / g for g in GUARDADOS]
    crudo = {}
    for ruta in rutas:
        desplazamiento = max((k[0] for k in crudo), default=0)
        propios = {}
        for f in csv.DictReader(open(ruta, encoding="utf-8")):
            propios[(int(f["barrido"]) + desplazamiento, int(f["hilos"]), int(f["longitud"]))] = (
                float(f["t_en_orden"]), float(f["t_a_la_vez"]))
        barridos = sorted({k[0] for k in propios})
        esperadas = {(b, n, l) for b in barridos for n in HILOS for l in E.LONGITUDES}
        assert set(propios) == esperadas, f"Se esperaban todas las casillas en cada barrido de {ruta}; hay {len(propios)}"
        crudo.update(propios)
    return crudo


def medianas(crudo):
    """({(n, longitud): (mediana de los segundos en orden, mediana de los segundos a la vez)},
    {(n, longitud): mediana de las ventajas de los barridos}, sin redondear)."""
    seg, ventajas = {}, {}
    barridos = sorted({k[0] for k in crudo})
    for n in HILOS:
        for longitud in E.LONGITUDES:
            tiempos = [crudo[(b, n, longitud)] for b in barridos]
            seg[(n, longitud)] = (statistics.median(t[0] for t in tiempos), statistics.median(t[1] for t in tiempos))
            ventajas[(n, longitud)] = statistics.median(t[0] / t[1] for t in tiempos)
    return seg, ventajas


def tiempo_matmul(n, matriz=MATRIZ, repeticiones=REPETICIONES_MATRIZ, tandas=TANDAS_MATRIZ):
    """Segundos de una multiplicación de matrices matriz × matriz (float32) con n hilos: la mediana de
    las tandas, cada una la media de `repeticiones` multiplicaciones, tras calentar."""
    poner_hilos(n)
    torch.manual_seed(E.SEMILLA)
    a, b = torch.randn(matriz, matriz), torch.randn(matriz, matriz)
    for _ in range(5):
        torch.mm(a, b)
    medidas = []
    for _ in range(tandas):
        t = time.perf_counter()
        for _ in range(repeticiones):
            torch.mm(a, b)
        medidas.append((time.perf_counter() - t) / repeticiones)
    return statistics.median(medidas)


def ventaja_de_referencia(ruta=AQUI / REFERENCIA, longitud=LONGITUD_LARGA):
    """La ventaja con `longitud` posiciones de la salida guardada del portátil en el procesador."""
    for l in Path(ruta).read_text(encoding="utf-8").splitlines():
        m = re.match(rf"\s*{re.escape(miles(longitud))}\s+\S+\s+\S+\s+([\d,]+)x", l)
        if m:
            return float(m.group(1).replace(",", "."))
    raise AssertionError(f"No encuentro la ventaja con {longitud} posiciones en {ruta}")


def cerca(valor, referencia, margen=MARGEN_INVARIANTE):
    """¿A menos de `margen` (arriba o abajo) de la referencia?"""
    return referencia * (1 - margen) <= valor <= referencia * (1 + margen)


# ---------------------------------------------------------------- impresión

def informe(r, v, matmul, emitir, barridos=BARRIDOS):
    """`matmul` es None si no se midió (--desde-guardado): entonces no sale la sección de la multiplicación."""
    mediana = f" Cada cifra es la mediana de {EN_LETRA[barridos]} mediciones." if barridos > 1 else ""
    emitir(["--- TIEMPO DE UNA PASADA DE ENTRENAMIENTO, SEGÚN LOS HILOS ---", ""])
    filas = []
    for n in HILOS:
        for longitud in E.LONGITUDES:
            t_s, t_v = r[(n, longitud)]
            filas.append([str(n), miles(longitud), coma(t_s, 4), coma(t_v, 4), coma(v[(n, longitud)], 2) + "x"])
    emitir(tabla_editorial(
        "Una pasada de entrenamiento, por hilos y longitud",
        ["hilos", "posiciones", "en orden (s)", "a la vez (s)", "ventaja"], filas, "ddddd",
        ["Procesador del portátil, modelo de 256 números por posición, lotes de 16.",
         "Ventaja: cuántas veces más rápida es la que mira todo a la vez; por debajo de 1,0x es más "
         f"lenta.{mediana}"]))
    emitir([""])
    filas = [[str(n)] + [coma(v[(n, l)], 2) + "x" for l in E.LONGITUDES] for n in HILOS]
    emitir(tabla_editorial(
        "La ventaja, por hilos y longitud", ["hilos"] + [miles(l) for l in E.LONGITUDES], filas,
        "d" * (1 + len(E.LONGITUDES)),
        ["Columnas: posiciones del texto. Es la última columna de la tabla de arriba."]))
    emitir([""])
    if matmul is None:
        return
    emitir(["--- UNA MULTIPLICACIÓN DE MATRICES ---", ""])
    t1, tn = matmul[HILOS[0]], matmul[HILOS[-1]]
    flops = 2 * MATRIZ ** 3
    emitir(tabla_editorial(
        f"Una multiplicación de matrices de {miles(MATRIZ)} × {miles(MATRIZ)}",
        ["hilos", "milisegundos", "miles de millones de cuentas por segundo"],
        [[str(n), coma(matmul[n] * 1000, 2), miles(round(flops / matmul[n] / 1e9))] for n in HILOS[0:1] + HILOS[-1:]],
        "ddd",
        [f"Con {HILOS[0]} hilo tarda {coma(t1 / tn, 2)} veces lo que con {HILOS[-1]}.",
         ("Casi igual: en este Mac los hilos no reparten la multiplicación."
          if t1 / tn < CASI_IGUAL else
          "No es casi igual: los hilos sí reparten la multiplicación.")]))
    emitir([""])


# ---------------------------------------------------------------- selftest

def selftest():
    fallos = []
    comprobar_entrada()
    serie, a_la_vez = modelos()

    # 1. TEST NULO — la misma máquina contra sí misma no tiene ventaja: la que lee en orden, medida
    #    dos veces seguidas con los mismos hilos, da una razón de 1,0 con un margen del 30 %.
    poner_hilos(4)
    razon = E.cronometrar(serie, 64) / E.cronometrar(serie, 64)
    print(f"[1] test nulo         la de en orden contra sí misma: {coma(razon, 2)}x (tiene que estar entre "
          f"{coma(1 - MARGEN_INVARIANTE, 2)} y {coma(1 + MARGEN_INVARIANTE, 2)})")
    if not cerca(razon, 1.0):
        fallos.append(f"test nulo: la misma máquina contra sí misma da {razon}")

    # 2. SEÑAL IMPLANTADA — el interruptor de hilos hace algo: una operación que sí se reparte
    #    (una exponencial sobre 32 millones de números) tarda menos con 10 hilos que con 1.
    x = torch.randn(1 << 25)

    def exp_tarda():
        torch.exp(x)
        t = time.perf_counter()
        for _ in range(3):
            torch.exp(x)
        return (time.perf_counter() - t) / 3

    poner_hilos(1)
    t1 = exp_tarda()
    poner_hilos(10)
    t10 = exp_tarda()
    print(f"[2] señal implantada  una exponencial sobre 32 millones de números tarda "
          f"{coma(t1 / t10, 2)} veces más con 1 hilo que con 10 (se espera más de "
          f"{coma(UMBRAL_SELFTEST_HILOS, 1)})")
    if t1 / t10 < UMBRAL_SELFTEST_HILOS:
        fallos.append("señal implantada: con más hilos no se acelera ni una operación repartible")

    # 3. INVARIANTE — la ventaja de referencia sale del fichero (3,2x con 1.024 posiciones), el
    #    interruptor de hilos deja los hilos pedidos, y los pasos en serie no dependen de los hilos.
    ref = ventaja_de_referencia()
    for n in HILOS:
        poner_hilos(n)
    pasos = all(E.EnSerie.pasos_en_serie(l) == l and E.ALaVez.pasos_en_serie(l) == 1 for l in E.LONGITUDES)
    print(f"[3] invariante        la ventaja de referencia con {miles(LONGITUD_LARGA)} posiciones es "
          f"{coma(ref, 1)}x; los hilos pedidos son los que hay; pasos en serie como siempre: "
          f"{'sí' if pasos else 'NO'}")
    if abs(ref - 3.2) > 1e-9 or not pasos:
        fallos.append(f"invariante: la referencia es {ref} y se esperaba 3,2, o los pasos en serie cambian")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--desde-guardado", action="store_true",
                    help="no mide: lee los barridos guardados, A y B, y escribe la salida entera (sin la multiplicación)")
    args = ap.parse_args()
    torch.set_num_interop_threads(HILOS_ENTRE_PROCESOS)
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
        # Orden 105: no se mide nada; el csv no guarda ni la fecha ni las versiones, así que la máquina y las
        # versiones que salen son las del entorno donde se lee (el mismo portátil y el mismo entorno).
        emitir(["######## ¿son los núcleos? (Orden 24) ########",
                "se leen los barridos guardados, A y B (cuantos_hilos-A.csv",
                "y cuantos_hilos-B.csv); no se mide nada.",
                f"máquina: {platform.system()} {platform.machine()}, Apple M4 Max, "
                f"{os.cpu_count()} núcleos",
                f"python {platform.python_version()}, torch {torch.__version__}",
                "Los tiempos solo valen en este portátil.",
                "La multiplicación de matrices no se vuelve a medir: no",
                "sale en esta salida.", ""])
        crudo = leer_guardados()
        matmul = None
    else:
        emitir(["######## ¿son los núcleos? (Orden 24) ########",
                f"fecha: {date.today().isoformat()}",
                f"máquina: {platform.system()} {platform.machine()}, Apple M4 Max, "
                f"{os.cpu_count()} núcleos",
                f"python {platform.python_version()}, torch {torch.__version__}",
                f"se usa el procesador ({E.DISPOSITIVO}); la tarjeta no.",
                f"hilos entre procesos: {torch.get_num_interop_threads()}; hilos por omisión: 10",
                "Los tiempos solo valen en este portátil.", ""])
        serie, a_la_vez = modelos()
        crudo = medir_pasadas(serie, a_la_vez)
        escribir_barridos(crudo)
        matmul = {n: tiempo_matmul(n) for n in (HILOS[0], HILOS[-1])}
    r, v = medianas(crudo)
    informe(r, v, matmul, emitir, len({k[0] for k in crudo}))

    ref = ventaja_de_referencia()
    v14 = v[(HILOS[-1], LONGITUD_LARGA)]
    ok = cerca(v14, ref)
    emitir(tabla_editorial(
        "La comprobación contra la medición del portátil",
        ["", "ventaja"],
        [[f"la guardada, en el procesador, con {miles(LONGITUD_LARGA)} posiciones", coma(ref, 1) + "x"],
         [f"esta, con {HILOS[-1]} hilos", coma(v14, 2) + "x"],
         ["¿a menos de un 30 % de la guardada?", "sí" if ok else "NO"]], "id",
        ["Si no queda a menos de un 30 %, esta medición no vale."]))
    emitir([""])
    assert ok, f"con {HILOS[-1]} hilos la ventaja ({v14}) no queda a menos de un 30 % de {ref}"
    v_poco = {n: v[(n, LONGITUD_LARGA)] for n in HILOS}
    cruza = {n: v < 1.0 for n, v in v_poco.items()}
    emitir(tabla_editorial(
        f"¿Cae por debajo de 1,0x con {miles(LONGITUD_LARGA)} posiciones?", ["hilos", "ventaja", "¿por debajo de 1,0x?"],
        [[str(n), coma(v_poco[n], 2) + "x", "sí" if cruza[n] else "no"] for n in HILOS], "ddc",
        ["El texto del capítulo 9 dice que con dos núcleos la que mira todo a la vez pierde."]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
