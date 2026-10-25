#!/usr/bin/env python3
"""
Capítulo 9 — por qué lo decisivo fue quitar la lectura en orden.

Compara dos maneras de procesar una frase con el mismo tamaño de modelo:
  - EN SERIE: una red recurrente, que no puede empezar la palabra N sin haber terminado la
    N-1. Por muchos procesadores que haya, esperan.
  - A LA VEZ: un bloque de atención, donde todas las posiciones se calculan a la vez.

Mide el tiempo de una pasada de entrenamiento (adelante y atrás) para longitudes crecientes,
y cuenta los PASOS EN SERIE, que no dependen de la máquina.

Uso:
    python en_serie_o_a_la_vez.py
    python en_serie_o_a_la_vez.py --selftest
    python en_serie_o_a_la_vez.py --ficha > ../datos/salidas/en_serie_o_a_la_vez_ficha.txt
"""

# ======================= CONSTANTES =======================

SEMILLA = 20260914
ANCHO = 256                # números por posición, igual en las dos máquinas
CABEZAS = 8
LOTE = 16
LONGITUDES = [64, 128, 256, 512, 1024]
REPETICIONES = 3
CALENTAMIENTO = 1

# Cuánto tiene que separarse la ventaja medida con texto largo de la ventaja medida con UNA
# sola posición para que no se pueda explicar por la sobrecarga fija del montaje. 1,3 = un
# 30 % arriba o abajo. Ver fallo_del_test_nulo().
MARGEN_TEST_NULO = 1.3

SALIDA_CSV = "en_serie_o_a_la_vez.csv"

# ==========================================================

import argparse
import csv
import os
import sys
import time

from formato import comprobar_ancho, miles, coma, tabla_editorial
import numpy as np
import torch
import torch.nn as nn


def dispositivo():
    """El procesador que se usa. Con la variable de entorno FORZAR_CPU=1 se obliga a usar
    el procesador normal aunque haya tarjeta gráfica.

    Hace falta porque el capítulo compara TRES máquinas y dos de ellas son el mismo
    portátil: su procesador y su tarjeta. Sin este interruptor, esa comparación se haría en
    dos ordenadores distintos y no se sabría qué parte de la diferencia es el diseño y qué
    parte es la máquina. (Este comentario decía otra cosa —«primero en el procesador y luego
    en la tarjeta»— que describía el capítulo de antes de medir: fallo 4.20.)"""
    if os.environ.get("FORZAR_CPU") == "1":
        return torch.device("cpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DISPOSITIVO = dispositivo()


class EnSerie(nn.Module):
    """Recurrente: la posición N necesita el resultado de la N-1. PASOS EN SERIE = longitud."""

    def __init__(self, ancho=ANCHO):
        super().__init__()
        self.nucleo = nn.LSTM(ancho, ancho, batch_first=True)
        self.salida = nn.Linear(ancho, ancho)

    def forward(self, x):
        h, _ = self.nucleo(x)
        return self.salida(h)

    @staticmethod
    def pasos_en_serie(longitud):
        return longitud


class ALaVez(nn.Module):
    """Atención: todas las posiciones se comparan a la vez. PASOS EN SERIE = 1.

    Lleva la red densa que acompaña siempre a la atención en un bloque real. Sin ella, esta
    máquina tiene la MITAD de números ajustables que la recurrente y la comparación sale
    sesgada a su favor: el test nulo lo detectó (4,28x de ventaja con una sola posición,
    donde no hay nada que paralelizar). Con la red densa, los tamaños quedan parejos."""

    def __init__(self, ancho=ANCHO, cabezas=CABEZAS):
        super().__init__()
        self.atencion = nn.MultiheadAttention(ancho, cabezas, batch_first=True)
        self.densa = nn.Sequential(nn.Linear(ancho, 2 * ancho), nn.GELU(),
                                   nn.Linear(2 * ancho, ancho))
        self.salida = nn.Linear(ancho, ancho)

    def forward(self, x):
        h, _ = self.atencion(x, x, x, need_weights=False)
        return self.salida(self.densa(h))

    @staticmethod
    def pasos_en_serie(longitud):
        return 1


def cronometrar(modelo, longitud, repeticiones=REPETICIONES):
    torch.manual_seed(SEMILLA)
    x = torch.randn(LOTE, longitud, ANCHO, device=DISPOSITIVO)
    objetivo = torch.randn(LOTE, longitud, ANCHO, device=DISPOSITIVO)
    perdida = nn.MSELoss()

    def una_pasada():
        modelo.zero_grad()
        perdida(modelo(x), objetivo).backward()
        if DISPOSITIVO.type == "mps":
            torch.mps.synchronize()
        elif DISPOSITIVO.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(CALENTAMIENTO):
        una_pasada()
    t = time.perf_counter()
    for _ in range(repeticiones):
        una_pasada()
    return (time.perf_counter() - t) / repeticiones


def fallo_del_test_nulo(ventaja1, ventajaL, margen=MARGEN_TEST_NULO):
    """Decide si lo medido con texto largo se distingue de la sobrecarga fija del montaje.

    Con UNA SOLA posición no hay nada que repartir, así que la diferencia de tiempo que se
    mide ahí es sobrecarga del montaje y nada más. Lo medido con texto largo solo dice algo
    si se separa de esa cifra.

    Devuelve el motivo del fallo, o None si el resultado es informativo.

    OJO: esta comprobación estuvo mal desde el primer día y nadie lo notó (fallo 4.31).
    Solo miraba el caso en que la máquina «a la vez» GANA, así que en un ordenador de dos
    núcleos, donde pierde, el selftest imprimía «pasa» sin haber comprobado nada. Ahora hay
    dos motivos de fallo, y el segundo vale gane quien gane.
    """
    if ventajaL > 1.0 and ventajaL <= ventaja1:
        return (f"la ventaja a longitud larga ({coma(ventajaL, 2)}x) no supera la sobrecarga "
                f"fija medida con una sola posición ({coma(ventaja1, 2)}x); el resultado lo "
                "explica el montaje, no el paralelismo")
    if 1 / margen <= ventajaL / ventaja1 <= margen:
        return (f"la ventaja a longitud larga ({coma(ventajaL, 2)}x) no se distingue de la "
                f"sobrecarga fija medida con una sola posición ({coma(ventaja1, 2)}x): se "
                f"parecen en menos de un {round((margen - 1) * 100)} %, así que lo medido "
                "puede ser el montaje entero")
    return None


# Casos conocidos con los que se comprueba la comprobación, antes de usarla. Los tres
# primeros son mediciones reales de las tres máquinas del capítulo; los tres últimos son
# los casos que el test nulo roto dejaba pasar.
CASOS_TEST_NULO = [
    # (ventaja1, ventajaL, debe_fallar, de dónde sale)
    (1.69, 0.54, False, "contenedor de dos núcleos: pierde, y pierde mucho más que el montaje"),
    (0.81, 3.46, False, "portátil, procesador: gana muy por encima del montaje"),
    (0.82, 1.98, False, "portátil, tarjeta: gana muy por encima del montaje"),
    (3.00, 1.50, True, "gana, pero menos que la sobrecarga: es el montaje"),
    (1.00, 1.05, True, "empate largo indistinguible del empate corto"),
    (1.69, 1.60, True, "pierde ventaja, pero tan poco que no se separa del montaje"),
]


def comprobar_el_test_nulo():
    """El test nulo es código, y el código del capítulo 9 ya estuvo mal una vez."""
    for ventaja1, ventajaL, debe_fallar, motivo in CASOS_TEST_NULO:
        salio = fallo_del_test_nulo(ventaja1, ventajaL) is not None
        assert salio == debe_fallar, (
            f"el test nulo se equivoca con ventaja1={ventaja1} y ventajaL={ventajaL} "
            f"({motivo}): se esperaba {'fallo' if debe_fallar else 'que pasara'} y "
            f"{'falló' if salio else 'pasó'}")


def selftest():
    fallos = []
    comprobar_el_test_nulo()
    serie = EnSerie().to(DISPOSITIVO)
    a_la_vez = ALaVez().to(DISPOSITIVO)

    # 1. TEST NULO — ver fallo_del_test_nulo(), que es donde está la decisión.
    t_s1 = cronometrar(serie, 1, 5)
    t_v1 = cronometrar(a_la_vez, 1, 5)
    ventaja1 = t_s1 / t_v1
    t_sL = cronometrar(serie, 512, 2)
    t_vL = cronometrar(a_la_vez, 512, 2)
    ventajaL = t_sL / t_vL
    print(f"[1] test nulo         ventaja con 1 posición (pura sobrecarga) "
          f"{coma(ventaja1, 2)}x; con 512 posiciones {coma(ventajaL, 2)}x")
    motivo = fallo_del_test_nulo(ventaja1, ventajaL)
    if motivo:
        fallos.append("test nulo: " + motivo)

    # 2. SEÑAL IMPLANTADA — la máquina en serie tiene que tardar aproximadamente el doble al
    #    doblar la longitud: su trabajo crece con el número de pasos.
    t_a = cronometrar(serie, 128, 3)
    t_b = cronometrar(serie, 256, 3)
    razon = t_b / t_a
    print(f"[2] señal implantada  al doblar la longitud, la de en serie tarda "
          f"{coma(razon, 2)}x (se espera cerca de 2)")
    if not 1.3 <= razon <= 3.2:
        fallos.append(f"señal implantada: al doblar la longitud la razón es {coma(razon, 2)}, "
                      "fuera del intervalo esperado de 1,3 a 3,2")

    # 3. INVARIANTE DEL DOMINIO — el recuento de pasos en serie no depende de la máquina.
    ok = all(EnSerie.pasos_en_serie(n) == n and ALaVez.pasos_en_serie(n) == 1
             for n in LONGITUDES)
    print(f"[3] invariante        pasos en serie: en serie = longitud, a la vez = 1: "
          f"{'sí' if ok else 'NO'}")
    if not ok:
        fallos.append("invariante: el recuento de pasos en serie no es el esperado")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def ficha():
    """La ficha de las dos máquinas del capítulo (Orden 79, 25 oct 2026): cuántos números lleva
    cada posición, cuántas capas, cuántas miradas y cuántos pesos. Se cuenta de los propios
    modelos, recién construidos, sin medir ningún tiempo, con las constantes de arriba. No va al
    libro: es el respaldo de las cifras de la ficha del capítulo 9."""
    serie, a_la_vez = EnSerie(ancho=ANCHO), ALaVez(ancho=ANCHO, cabezas=CABEZAS)
    pesos = lambda m: sum(p.numel() for p in m.parameters())
    bloques = sum(isinstance(m, nn.MultiheadAttention) for m in a_la_vez.modules())
    filas = [
        ["qué lee", "una lista de números por posición", "una lista de números por posición"],
        ["números por posición", str(serie.nucleo.input_size), str(a_la_vez.atencion.embed_dim)],
        ["capas", str(serie.nucleo.num_layers), str(bloques)],
        ["miradas por capa", "—", str(a_la_vez.atencion.num_heads)],
        ["pesos", miles(pesos(serie)), miles(pesos(a_la_vez))],
    ]
    print("--- LA FICHA DE LAS DOS MÁQUINAS DEL CAPÍTULO 9 ---\n")
    print("\n".join(tabla_editorial(
        "La ficha de las dos máquinas del capítulo 9",
        ["", "la que lee en orden", "la que mira todo a la vez"], filas, "idd",
        ["Contado de los propios modelos, sin medir tiempos. «Capas» de la que mira todo a la "
         "vez: un bloque de atención con su red densa."])))


def main():
    global ANCHO, LOTE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    # El tamaño de la máquina de juguete se puede cambiar SIN tocar las constantes: las
    # de arriba siguen siendo las de la medición del libro, y una ejecución con otro
    # tamaño lo dice en su propia cabecera. Hacía falta porque no se sabía si la ventaja
    # depende del tamaño de lo que se manda hacer. Ya se sabe, y no en el mismo sentido en
    # las dos máquinas: con el modelo más grande, en la tarjeta la ventaja encoge y en el
    # procesador crece. Está medido en en_serie_512_* y en_serie_1024_*.
    ap.add_argument("--ancho", type=int, default=ANCHO,
                    help=f"números por posición (por omisión {ANCHO})")
    ap.add_argument("--lote", type=int, default=LOTE,
                    help=f"cuántos textos a la vez (por omisión {LOTE})")
    ap.add_argument("--ficha", action="store_true",
                    help="imprime la ficha de las dos máquinas (contada, sin medir) y sale")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    if args.ficha:
        ficha()
        return

    ANCHO, LOTE = args.ancho, args.lote
    assert ANCHO % CABEZAS == 0, \
        f"Se esperaba un ancho múltiplo de {CABEZAS} cabezas; se pidió {ANCHO}"
    assert LOTE >= 1, f"Se esperaba un lote de 1 o más; se pidió {LOTE}"

    print(f"Procesador usado: {DISPOSITIVO}")
    serie = EnSerie(ancho=ANCHO).to(DISPOSITIVO)
    a_la_vez = ALaVez(ancho=ANCHO, cabezas=CABEZAS).to(DISPOSITIVO)
    n_serie = sum(p.numel() for p in serie.parameters())
    n_vez = sum(p.numel() for p in a_la_vez.parameters())
    print(f"{ANCHO} números por posición, lotes de {LOTE}.")
    print(f"Números ajustables: en serie {miles(n_serie)} | a la vez {miles(n_vez)} "
          f"({coma(n_vez / n_serie, 2)} veces).\n")
    filas = []

    def imprimir(*lineas):
        """Imprime comprobando antes que cabe en la caja del libro (fallo 4.27)."""
        for l in comprobar_ancho(list(lineas)):
            print(l)

    print("--- PASOS QUE HAY QUE DAR UNO DETRÁS DE OTRO (no depende de la máquina) ---")
    # Los números van con el separador de miles del castellano, y TODAS las columnas
    # igual: antes la longitud salía «1024» y los pasos «1,024» en la misma fila.
    pasos = [f"{'longitud':>10}{'en serie':>12}{'a la vez':>12}"]
    for n in LONGITUDES:
        pasos.append(f"{miles(n):>10}{miles(EnSerie.pasos_en_serie(n)):>12}"
                     f"{miles(ALaVez.pasos_en_serie(n)):>12}")
    imprimir(*pasos)

    print("\n--- TIEMPO DE UNA PASADA DE ENTRENAMIENTO ---")
    imprimir(f"{'longitud':>10}{'en serie (s)':>15}{'a la vez (s)':>15}{'ventaja':>10}")
    for n in LONGITUDES:
        t_s = cronometrar(serie, n)
        t_v = cronometrar(a_la_vez, n)
        imprimir(f"{miles(n):>10}{coma(t_s, 4):>15}{coma(t_v, 4):>15}"
                 f"{coma(t_s / t_v, 1):>9}x")
        filas.append([str(DISPOSITIVO), n, f"{t_s:.6f}", f"{t_v:.6f}", f"{t_s/t_v:.2f}"])
    # La columna «ventaja» es una razón entre dos tiempos, y una razón sin decir de qué
    # es no es un dato (regla 9). La clave va pegada a la tabla, debajo de ella.
    imprimir("«ventaja»: cuántas veces más rápida es la de «a la vez»;",
             "por debajo de 1,0x es más lenta.")

    with open(SALIDA_CSV, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(
            [["procesador", "longitud", "en_serie_s", "a_la_vez_s", "ventaja"]] + filas)
    print(f"\nEscrito {SALIDA_CSV}")


if __name__ == "__main__":
    main()
