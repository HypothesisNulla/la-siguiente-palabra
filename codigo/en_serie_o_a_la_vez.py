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

# Orden 100: el modo --barridos repite la medición de siempre BARRIDOS veces, con los tres tamaños de
# modelo de la tarjeta (los de en_serie_o_a_la_vez.txt, en_serie_512_gpu.txt y en_serie_1024_gpu.txt), y
# guarda las BARRIDOS mediciones de cada celda: una sola varía demasiado de una vez a otra.
BARRIDOS = 5
BARRIDOS_TIRADOS = 1                # barridos enteros que se miden antes y se tiran: calientan la tarjeta
LONGITUDES_AL_REVES = True          # las longitudes, de la más larga a la más corta (la 64 deja de ir la primera)
ANCHOS_BARRIDOS = (256, 512, 1024)
CSV_BARRIDOS = "../datos/salidas/en_serie_barridos.csv"          # en la tarjeta
CSV_BARRIDOS_CPU = "../datos/salidas/en_serie_barridos_cpu.csv"  # con FORZAR_CPU=1, en el procesador

# ==========================================================

import argparse
import csv
import os
import subprocess
import sys
import time

from formato import comprobar_ancho, miles, coma, tabla_editorial


def lanzar_barridos(lote):
    """Orden 104: EL PROCESO QUE LANZA no importa `torch` (ni este mismo módulo más allá de aquí): solo
    lanza un proceso por ancho con `subprocess`, recoge sus líneas y escribe el csv. Con `torch` importado en
    el proceso que espera, el hijo salía un 19 % más lento en la tarjeta (256 números, lote 16, 1.024
    posiciones: de 0,044 s a 0,052 s en orden; Respuesta 103), sin que el padre hiciera nada más que esperar.

    BARRIDOS barridos (tras BARRIDOS_TIRADOS que se tiran) de cada ancho de ANCHOS_BARRIDOS, **cada ancho en
    su propio proceso** (Orden 103: después de las pasadas largas de un modelo grande, el pequeño se queda un
    35 % más lento en ese mismo proceso en el procesador; Respuesta 102). Las BARRIDOS mediciones de cada
    celda van a CSV_BARRIDOS (CSV_BARRIDOS_CPU, si el hijo dice que usó el procesador): segundos de las dos
    máquinas y ventaja, a seis decimales; la ventaja sale de los segundos ya redondeados."""
    assert "torch" not in sys.modules, "El proceso que lanza no puede haber importado torch (Orden 104)"
    filas, dispositivo = [], None
    for ancho in ANCHOS_BARRIDOS:
        hijo = subprocess.run([sys.executable, os.path.abspath(__file__), "--barridos-de-un-ancho", str(ancho),
                               "--lote", str(lote)], capture_output=True, text=True)
        assert hijo.returncode == 0, f"El proceso del ancho {ancho} falló:\n{hijo.stderr}"
        for l in hijo.stdout.splitlines():
            if l.startswith("# dispositivo "):
                assert dispositivo in (None, l.split()[2]), "Los procesos de los anchos usaron dispositivos distintos"
                dispositivo = l.split()[2]
        medidas = [l.split(",") for l in hijo.stdout.splitlines() if l.count(",") == 3]
        assert len(medidas) == BARRIDOS * len(LONGITUDES), (
            f"Se esperaban {BARRIDOS * len(LONGITUDES)} celdas del ancho {ancho}; hay {len(medidas)}")
        for b, n, t_s, t_v in medidas:
            t_s, t_v = float(t_s), float(t_v)
            filas.append([int(b), ancho, int(n), f"{t_s:.6f}", f"{t_v:.6f}", f"{t_s / t_v:.4f}"])
    print(f"Procesador usado: {dispositivo}", file=sys.stderr)
    destino = CSV_BARRIDOS_CPU if dispositivo == "cpu" else CSV_BARRIDOS
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), destino), "w",
              newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(
            [["barrido", "ancho", "longitud", "en_serie_s", "a_la_vez_s", "ventaja"]] + filas)
    print(f"Escrito {destino}: {BARRIDOS} barridos, anchos {ANCHOS_BARRIDOS} (cada uno en su proceso), lote {lote}.")


if __name__ == "__main__" and "--barridos" in sys.argv:
    _ap = argparse.ArgumentParser(add_help=False)
    _ap.add_argument("--barridos", action="store_true")
    _ap.add_argument("--lote", type=int, default=LOTE)
    sys.exit(lanzar_barridos(_ap.parse_known_args()[0].lote))

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


def un_barrido(lote, ancho):
    """La medición de siempre con un ancho: las dos máquinas, y las LONGITUDES de la más corta a la más
    larga, o al revés si LONGITUDES_AL_REVES. Devuelve {longitud: (en serie, a la vez)}."""
    global ANCHO, LOTE
    LOTE, ANCHO = lote, ancho
    serie = EnSerie(ancho=ancho).to(DISPOSITIVO)
    a_la_vez = ALaVez(ancho=ancho, cabezas=CABEZAS).to(DISPOSITIVO)
    out = {}
    for n in (LONGITUDES[::-1] if LONGITUDES_AL_REVES else LONGITUDES):
        out[n] = (cronometrar(serie, n), cronometrar(a_la_vez, n))
    return out


def barridos_de_un_ancho(lote, ancho):
    """Dentro de SU proceso: BARRIDOS_TIRADOS barridos que se tiran y BARRIDOS que cuentan, de un solo ancho.
    Escribe en la salida estándar una línea «barrido,longitud,segundos en serie,segundos a la vez» por celda."""
    print(f"# dispositivo {DISPOSITIVO.type}")
    for _ in range(BARRIDOS_TIRADOS):
        un_barrido(lote, ancho)
    for b in range(1, BARRIDOS + 1):
        for n, (t_s, t_v) in un_barrido(lote, ancho).items():
            print(f"{b},{n},{t_s:.6f},{t_v:.6f}")


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
    ap.add_argument("--barridos", action="store_true",
                    help=f"repite la medición {BARRIDOS} veces con los anchos {ANCHOS_BARRIDOS}, escribe "
                         f"{CSV_BARRIDOS} ({CSV_BARRIDOS_CPU} con FORZAR_CPU=1) y sale (no escribe {SALIDA_CSV})")
    ap.add_argument("--barridos-de-un-ancho", type=int, metavar="ANCHO",
                    help="(interno de --barridos) los barridos de un solo ancho, en este proceso")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    if args.ficha:
        ficha()
        return
    if args.barridos_de_un_ancho:
        barridos_de_un_ancho(args.lote, args.barridos_de_un_ancho)
        return
    if args.barridos:           # lo atiende el lanzador de arriba, antes de importar torch
        raise SystemExit("--barridos tiene que ejecutarse como programa: python en_serie_o_a_la_vez.py --barridos")

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
