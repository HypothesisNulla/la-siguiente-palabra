#!/usr/bin/env python3
"""
Capítulo 9 — ¿se estrecha la ventaja del «todo a la vez» cuando el texto crece?

El capítulo afirma dos cosas sobre la FORMA de sus tablas, y las dos se pueden comprobar
sin volver a medir nada, porque las mediciones ya están guardadas:

  1. «La ventaja se estrecha según crece el texto.»
  2. «Estoy usando modelos diminutos […] en el entrenamiento de verdad la diferencia es
     mucho mayor que la que yo he medido.»

Este programa NO mide: lee las tablas que ya hay en datos/salidas/ (siete ejecuciones, tres
máquinas y tres tamaños de modelo), recompone la ventaja longitud a longitud y dice, para
cada una, cuánto vale al principio, cuánto al final y en qué sentido va.

No lleva cronómetro a propósito: un tiempo medido aquí sería un tiempo de OTRA máquina
presentado como si fuera el del libro, que es el fallo que este mismo capítulo cuenta.

Uso:
    python como_cambia_la_ventaja.py
    python como_cambia_la_ventaja.py --selftest
"""

# ======================= CONSTANTES =======================

CARPETA = "../datos/salidas"

# Las siete ejecuciones guardadas, con lo que hay que saber de cada una para leer la tabla.
# La última columna dice CUÁL de las tablas de tiempos del fichero se lee. Casi todos los
# ficheros tienen una sola; el del contenedor tiene tres, porque esa máquina es compartida
# y la medición se repite tres veces seguidas, y el libro toma la segunda.
#
# Antes esto no estaba y el programa se tragaba TODAS las filas del fichero, fueran de la
# tabla que fueran. Con un fichero de una sola tabla no se nota; el día que uno tuvo tres,
# el programa juntó quince filas en una y sacó una tabla de 126 caracteres de ancho. Saltó
# por el ancho de la caja, no por leer mal: se salvó de casualidad.
EJECUCIONES = [
    ("en_serie_o_a_la_vez_2nucleos.txt", "contenedor, 2 núcleos", 256, 2),
    ("en_serie_o_a_la_vez_cpu.txt",      "portátil, procesador",  256, 1),
    ("en_serie_o_a_la_vez.txt",          "portátil, tarjeta",     256, 1),
    ("en_serie_512_cpu.txt",             "portátil, procesador",  512, 1),
    ("en_serie_512_gpu.txt",             "portátil, tarjeta",     512, 1),
    ("en_serie_1024_cpu.txt",            "portátil, procesador", 1024, 1),
    ("en_serie_1024_gpu.txt",            "portátil, tarjeta",    1024, 1),
]
CORTO, LARGO = 64, 1024      # las dos longitudes que el capítulo compara
# Cómo se llama cada ordenador en el título de su tabla.
NOMBRE_TABLA = {"contenedor, 2 núcleos": "el ordenador de 2 núcleos en la nube",
                "portátil, procesador": "el procesador del portátil",
                "portátil, tarjeta": "la tarjeta gráfica del portátil"}
# Cuánta de la ventaja del texto corto queda al llegar al texto largo. Los dos cortes
# están escritos aquí y en la leyenda impresa para que se vean: la columna «queda» es el
# dato, y la palabra de al lado solo lo resume.
QUEDA_CRECE = 1.10           # por encima de esto, la ventaja crece
QUEDA_CAE = 0.75             # por debajo de esto, la ventaja cae
# Orden 97: el lote con el que se midió todo lo de en_serie_*.txt (el LOTE de en_serie_o_a_la_vez.py),
# escrito con letra en la nota; si cambia, falla aquí y no en la página.
TEXTOS_POR_PASADA = 16
EN_LETRA = {16: "dieciséis"}
MEDIO_ULTIMO = 0.00005       # los segundos se imprimen con cuatro decimales: medio último

# ==========================================================

import argparse
import os
import re
import sys

from formato import comprobar_ancho, coma, miles, pct, tabla_editorial

FILA = re.compile(r"^\s*([\d.]+)\s+(\d+,\d+)\s+(\d+,\d+)\s+(\d+,\d+)x\s*$")


def numero(s):
    """Un número escrito en castellano, de vuelta a número: '1.024' -> 1024,0."""
    return float(s.replace(".", "").replace(",", "."))


def leer(ruta, cual=1):
    """La tabla número `cual` del fichero, como [(longitud, s en serie, s a la vez, ventaja)].

    Una «tabla» es una tanda de filas seguidas que casan con FILA. Entre dos tablas hay
    siempre alguna línea que no casa (la cabecera de columnas, una línea en blanco), así
    que la separación no necesita saber nada del formato del fichero.
    """
    tablas, actual = [], []
    for linea in open(ruta, encoding="utf-8"):
        m = FILA.match(linea.rstrip("\n"))
        if m:
            actual.append((int(numero(m.group(1))), numero(m.group(2)),
                           numero(m.group(3)), numero(m.group(4))))
        elif actual:
            tablas.append(actual)
            actual = []
    if actual:
        tablas.append(actual)

    assert tablas, f"Se esperaba al menos una tabla de tiempos en {ruta}; no hay ninguna"
    assert 1 <= cual <= len(tablas), (
        f"Se esperaba la tabla número {cual} de {ruta}; el fichero tiene {len(tablas)}")
    filas = tablas[cual - 1]
    longitudes = [n for n, _, _, _ in filas]
    assert len(set(longitudes)) == len(longitudes), (
        f"Se esperaban longitudes distintas en la tabla {cual} de {ruta}; se encontró "
        f"{longitudes}, que repite alguna: eso es señal de que se han pegado dos tablas")
    for n in (CORTO, LARGO):
        assert n in longitudes, (
            f"Se esperaba la longitud {miles(n)} en la tabla {cual} de {ruta}; solo hay "
            f"{longitudes}")
    return filas


def sentido(filas):
    """Ventaja al principio y al final, qué parte de ella queda, y cómo se llama eso."""
    ventajas = dict((n, v) for n, _, _, v in filas)
    corto, largo = ventajas[CORTO], ventajas[LARGO]
    queda = largo / corto
    if queda >= QUEDA_CRECE:
        palabra = "crece"
    elif queda < QUEDA_CAE:
        palabra = "cae"
    else:
        palabra = "se mantiene"
    return palabra, corto, largo, queda


def informe(ejecuciones=EJECUCIONES, carpeta=CARPETA):
    """La ventaja de cada ejecución, longitud a longitud, en tablas editoriales: una por
    ordenador, con una fila por tamaño de modelo (L24, 9 de octubre de 2026; antes, renglones
    sangrados que el capítulo copiaba)."""
    por_maquina = {}
    for fichero, maquina, ancho, cual in ejecuciones:
        filas = leer(os.path.join(carpeta, fichero), cual)
        palabra, _, _, queda = sentido(filas)
        por_maquina.setdefault(maquina, []).append((ancho, filas, palabra, queda))
    lineas = []
    for maquina, ejes in por_maquina.items():
        longitudes = [n for n, _, _, _ in ejes[0][1]]
        for _, filas, _, _ in ejes:
            assert [n for n, _, _, _ in filas] == longitudes, (
                f"Se esperaban las mismas longitudes en todas las tablas de {maquina}; "
                f"hay {longitudes} y {[n for n, _, _, _ in filas]}")
        lineas += tabla_editorial(
            f"La ventaja en {NOMBRE_TABLA[maquina]}, con tres tamaños de modelo"
            if len(ejes) == 3 else f"La ventaja en {NOMBRE_TABLA[maquina]}",
            ["modelo, números por posición"] + [f"longitud del texto: {miles(n)}" for n in longitudes]
            + ["queda"],
            [[miles(ancho)] + [coma(v, 1) + "x" for _, _, _, v in filas]
             + [f"{pct(queda, 0)}, {palabra}"] for ancho, filas, palabra, queda in ejes],
            "d" * (len(longitudes) + 1) + "i",
            ["Cuántas veces más rápida va la máquina que lo mira todo a la vez, con cada tamaño de "
             "modelo y cada longitud de texto. «Queda» es la parte de la ventaja del texto más corto "
             f"que sigue en el más largo: por debajo del {pct(QUEDA_CAE, 0)} se dice que cae, y entre "
             f"el {round(100 * QUEDA_CAE)} y el {pct(QUEDA_CRECE, 0)}, que se mantiene."]) + [""]
    return lineas


def selftest():
    fallos = []

    # 1. TEST NULO — una tabla donde la ventaja NO cambia tiene que salir «igual». Si el
    #    programa encuentra una tendencia aquí, la tendencia la pone él, no los datos.
    plana = [(64, 2.0, 1.0, 2.0), (1024, 4.0, 2.0, 2.0)]
    va_plana, _, _, queda_plana = sentido(plana)
    print(f"[1] test nulo         ventaja constante de 2,0x: el programa dice "
          f"«{va_plana}», queda {pct(queda_plana, 0)}")
    if va_plana != "se mantiene" or round(queda_plana, 6) != 1:
        fallos.append(f"test nulo: con una ventaja constante dice «{va_plana}», "
                      "y tiene que decir «se mantiene» con el 100 % en pie")

    # 2. SEÑAL IMPLANTADA — una caída puesta a mano tiene que salir, y con su tamaño.
    caida = [(64, 6.0, 1.0, 6.0), (1024, 1.5, 1.0, 1.5)]
    va_caida, c, l, queda_caida = sentido(caida)
    print(f"[2] señal implantada  caída de 6,0x a 1,5x: el programa dice "
          f"«{va_caida}», de {coma(c, 1)}x a {coma(l, 1)}x, queda {pct(queda_caida, 0)}")
    if va_caida != "cae" or (c, l) != (6.0, 1.5) or round(queda_caida, 4) != 0.25:
        fallos.append("señal implantada: no reconoce una caída puesta a mano")

    # 3. INVARIANTE DEL DOMINIO — la ventaja impresa en cada fichero tiene que ser el
    #    cociente de sus dos columnas de segundos. Si no lo es, se están leyendo columnas
    #    equivocadas y todo lo demás sobra.
    #
    #    El cociente no se compara contra un margen inventado: los segundos vienen ya
    #    redondeados a cuatro decimales, así que el cociente de verdad solo puede estar
    #    dentro del intervalo que permite ese redondeo, y lo que se exige es que la
    #    ventaja impresa —redondeada a un decimal— quepa en él. A 0,0020 segundos ese
    #    intervalo es ancho, y eso no es holgura: es la precisión que hay.
    fuera, revisadas = [], 0
    for fichero, _, _, cual in EJECUCIONES:
        for n, t_s, t_v, ventaja in leer(os.path.join(CARPETA, fichero), cual):
            revisadas += 1
            minimo = (t_s - MEDIO_ULTIMO) / (t_v + MEDIO_ULTIMO)
            maximo = (t_s + MEDIO_ULTIMO) / (t_v - MEDIO_ULTIMO)
            if not (minimo - 0.05 <= ventaja <= maximo + 0.05):
                fuera.append(f"{fichero}, longitud {miles(n)}: impresa {coma(ventaja, 1)}x, "
                             f"posible entre {coma(minimo, 2)}x y {coma(maximo, 2)}x")
    print(f"[3] invariante        la ventaja impresa cabe en el cociente de los dos "
          f"tiempos: {revisadas - len(fuera)} de {revisadas} filas")
    if fuera:
        fallos.append("invariante: la ventaja impresa no sale de los tiempos impresos en "
                      + "; ".join(fuera))

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


# ---- L24: los segundos, máquina a máquina (sección añadida al final; lo de arriba no cambia) ----
# El capítulo 9 enseña dos tablas de segundos y usa una columna «ventaja» que el lector no ha
# visto calcular nunca. Esto imprime, de los mismos ficheros, las dos tablas con un título que
# dice en qué máquina se midieron, dos filas con la cuenta hecha, los segundos de las tres
# máquinas con el texto más largo, y la tabla del procesador con el modelo grande.
TABLAS_DE_SEGUNDOS = [
    ("A", "en_serie_o_a_la_vez_2nucleos.txt", 2, "el ordenador de 2 núcleos en la nube", 256),
    ("B", "en_serie_o_a_la_vez_cpu.txt", 1, "el procesador del portátil", 256),
]
TRES_MAQUINAS = [
    ("ordenador de 2 núcleos", "en_serie_o_a_la_vez_2nucleos.txt", 2),
    ("portátil, procesador", "en_serie_o_a_la_vez_cpu.txt", 1),
    ("portátil, tarjeta gráfica", "en_serie_o_a_la_vez.txt", 1),
]
GRANDE = ("en_serie_1024_cpu.txt", 1, "el procesador del portátil", 1024)


NOTAS_SEGUNDOS = [
    # Orden 97: nota para el lector. Que cada número sea la media de tres pasadas tras una de
    # calentamiento que se tira está en en_serie_o_a_la_vez.py (REPETICIONES, CALENTAMIENTO).
    f"Lo que tarda, en segundos, una pasada de entrenamiento con {EN_LETRA[TEXTOS_POR_PASADA]} "
    "textos de cada longitud. La ventaja es el primer tiempo entre el segundo: por encima de 1,0x "
    "gana la que mira todo a la vez; por debajo, la que lee en orden."]


def tabla_de_segundos(titulo, filas, notas=NOTAS_SEGUNDOS):
    return tabla_editorial(
        titulo, ["longitud", "en orden (s)", "a la vez (s)", "ventaja"],
        [[miles(n), coma(t_s, 4), coma(t_v, 4), coma(v, 1) + "x"] for n, t_s, t_v, v in filas],
        "dddd", notas)


def segundos(carpeta=CARPETA):
    """Las tablas de segundos que enseña el capítulo 9, en tablas editoriales (L24, 9 de
    octubre de 2026; antes, renglones sangrados con una letra, A, B y C, delante)."""
    L = ["", "##### los segundos, ordenador a ordenador #####", ""]
    tablas = {}
    for letra, fichero, cual, maquina, ancho in TABLAS_DE_SEGUNDOS + [("C",) + GRANDE]:
        tablas[letra] = (leer(os.path.join(carpeta, fichero), cual), maquina, ancho)
    for letra in ("A", "B"):
        filas, maquina, ancho = tablas[letra]
        L += tabla_de_segundos(f"Los segundos en {maquina}, modelo de {miles(ancho)} números "
                               f"por posición", filas) + [""]
    # la cuenta de la ventaja, hecha, con la primera y la última fila del ordenador de 2 núcleos
    cuenta, quien = [], []
    for n, t_s, t_v, v in (tablas["A"][0][0], tablas["A"][0][-1]):
        cociente = t_s / t_v
        quien.append(f"con {miles(n)} posiciones, " +
                     (f"es {coma(cociente, 1)} veces más rápida" if cociente >= 1
                      else f"tarda {coma(1 / cociente, 1)} veces más"))
        cuenta.append([miles(n), coma(t_s, 4), coma(t_v, 4), coma(cociente, 2),
                       coma(v, 1) + "x"])
    L += tabla_editorial(
        "La cuenta de la ventaja, en el ordenador de 2 núcleos",
        ["posiciones", "en orden (s)", "a la vez (s)", "cociente", "redondeado"],
        cuenta, "ddddd",
        ["Cociente: los segundos de «en orden» entre los de «a la vez».",
         "La que mira todo a la vez: " + "; ".join(quien) + "."]) + [""]
    tres = []
    for nombre, fichero, cual in TRES_MAQUINAS:
        f = [r for r in leer(os.path.join(carpeta, fichero), cual) if r[0] == LARGO][0]
        tres.append([nombre, coma(f[1], 4), coma(f[2], 4)])
    L += tabla_editorial(
        f"Con {miles(LARGO)} posiciones y el modelo de 256 números, los tres ordenadores",
        ["ordenador", "en orden (s)", "a la vez (s)"], tres, "idd") + [""]
    filas, maquina, ancho = tablas["C"]
    L += tabla_de_segundos(f"Los segundos en {maquina}, modelo de {miles(ancho)} números "
                           f"por posición", filas) + [""]
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())

    cabecera = [
        "##### capítulo 9: cómo cambia la ventaja al alargar el texto #####",
        "máquina: ninguna. Este programa NO cronometra: relee las siete",
        "tablas ya guardadas en datos/salidas/, así que da lo mismo en",
        "cualquier ordenador y no hay máquina que declarar.",
        "",
        "«longitud del texto» son posiciones; «ventaja», cuántas veces",
        "más rápida es la máquina que lo mira todo a la vez, y por debajo",
        "de 1,0x es más lenta. De lo que queda de esa ventaja al pasar",
        "del texto más corto al más largo: por debajo del 75 % se llama",
        "«cae»; por encima del 110 %, «crece»; en medio, «se mantiene».",
        "",
    ]
    for l in comprobar_ancho(cabecera):
        print(l)
    # El ancho se comprueba fuera de las tablas editoriales: ésas las compone el libro.
    dentro = False
    for l in informe() + segundos():
        if l in ("::: tabla", "::: muestra"):
            dentro = True
        elif l == ":::":
            dentro = False
        elif not dentro:
            comprobar_ancho([l], 64)
        print(l)


if __name__ == "__main__":
    main()
