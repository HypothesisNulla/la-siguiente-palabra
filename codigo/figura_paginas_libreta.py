#!/usr/bin/env python3
"""
Capítulo 1 — dos hojas de la libreta de la máquina de letras, con el selector en 2.

La libreta tiene una hoja por cada contexto: arriba, de título, las dos letras que la máquina
tiene delante, y debajo un palito por cada vez que en el Quijote vino detrás cada letra. Esta
figura enseña dos de esas hojas una al lado de otra, para que el lector vea lo que hay dentro
de la libreta antes de que el capítulo hable de ella:

  - la de «xt», que solo tiene 24 palitos y caben todos dibujados, de cinco en cinco;
  - la de «qu», que tiene 32.709: no caben, y se ponen los números.

Los palitos no se escriben aquí: se cuentan en el Quijote con el mismo código del capítulo
(`casillas_vacias.tras`), y el programa revienta si lo contado no coincide con las dos tablas que
el capítulo imprime: la de «xt» (`datos/salidas/traza_escalera.txt`) y la de «qu»
(`datos/salidas/casillas_vacias.txt`).

Uso:
    python figura_paginas_libreta.py --selftest
    python figura_paginas_libreta.py
"""

# ======================= CONSTANTES =======================

SALIDA_XT = "../datos/salidas/traza_escalera.txt"        # la tabla de «xt»
SALIDA_QU = "../datos/salidas/casillas_vacias.txt"       # la tabla de «qu»
DESTINO = "../figuras/paginas_libreta.png"
CORPUS = "../datos/entrada/quijote.txt"

PAGINA_CON_PALITOS = "xt"       # la que se dibuja palito a palito
PAGINA_CONTADA = "qu"           # la que no cabe y va contada
MAXIMO_DIBUJABLE = 30           # más palitos que esto en una letra no se dibujan: van contados
RENGLONES = 7                   # seis letras como máximo, y el total
ALTO_RENGLON = 6.6              # unidades del lienzo
ANCHO_HOJA = 44.0

NOTA = "demasiados palitos para dibujarlos: van contados"   # debajo de la hoja de «qu», no dentro
TAM_NOTA = 6.4                  # cuerpo de la nota; en una línea si cabe al ancho de la hoja

# Selftest
MINI = "xtaxtaxte"              # contado a mano: detrás de «xt», dos «a» y una «e»

# ==========================================================

import argparse, sys
from collections import Counter
from pathlib import Path

from formato import leer_tablas, miles
import casillas_vacias as C
from hoja_libreta import hoja, palitos
from infografia import Lienzo, GRIS

AQUI = Path(__file__).resolve().parent


def titulo_tabla(par):
    return f"Qué letra viene detrás de «{par}»"


def pagina_de_la_salida(texto, par):
    """La hoja de `par` tal como la imprime el capítulo: [(letra, "25.972"), ...] en el orden
    de la tabla, y el total que dice la nota («Las 32.709 veces que aparece…»)."""
    tablas = leer_tablas(texto)
    assert titulo_tabla(par) in tablas, f"no encuentro la tabla de «{par}» en la salida"
    _, filas, notas = tablas[titulo_tabla(par)]
    total = next(n for n in notas if n.startswith("Las ") and f"aparece «{par}»" in n)
    return [(f[0], f[1]) for f in filas], total.split()[1]


def contar(secuencia, par):
    """Lo que viene detrás de `par` en el texto: el mismo recuento que usa el capítulo."""
    return C.tras(secuencia, par)


def ancho_texto(L, s, tam):
    """Lo que mide `s` de ancho, en unidades del lienzo, a ese cuerpo."""
    t = L.ax.text(0, 0, s, fontsize=tam)
    ancho_px = t.get_window_extent(L.fig.canvas.get_renderer()).width
    t.remove()
    return ancho_px / (L.fig.get_size_inches()[0] * L.fig.dpi) * 100


def poner_nota(L, x, y_base, ancho):
    """La nota de la hoja contada, debajo de ella y fuera de las filas: en una línea si cabe al
    ancho de la hoja; si no, en dos, partida en los dos puntos. Devuelve cuántas líneas."""
    lineas = [NOTA] if ancho_texto(L, NOTA, TAM_NOTA) <= ancho else NOTA.split(": ")
    for k, linea in enumerate(lineas):
        L.texto(x + 1.0, y_base - 3.0 - 3.0 * k, linea + (":" if k < len(lineas) - 1 else ""),
                tam=TAM_NOTA, color=L.p.suave)
    return len(lineas)


def dibujar_pagina(L, x, y_tope, par, cuentas, dibujar):
    """Una hoja con el título `par` y, en cada renglón, la letra, sus palitos (si `dibujar`) y su
    número. `cuentas` es [(letra, n)]. Devuelve los palitos dibujados."""
    ys, x_texto, x_fin, y_base = hoja(L, x, y_tope, ANCHO_HOJA, par, RENGLONES, ALTO_RENGLON)
    assert len(cuentas) < RENGLONES, f"se esperaban como mucho {RENGLONES - 1} letras; hay {len(cuentas)}"
    pintados = 0
    for (letra, n), y in zip(cuentas, ys):
        L.texto(x_texto, y, letra, tam=9.0, negrita=True)
        if dibujar:
            assert n <= MAXIMO_DIBUJABLE, f"«{letra}» tiene {n} palitos: no se dibujan"
            x_p = x_texto + 6.5
            palitos(L, x_p, y, n)
            pintados += n
        L.texto(x_fin, y, miles(n), tam=8.2, ha="right")
    if not dibujar:
        poner_nota(L, x, y_base, ANCHO_HOJA)
    L.texto(x_texto, ys[-1], "en total", tam=7.4, color=L.p.suave)
    L.texto(x_fin, ys[-1], miles(sum(n for _, n in cuentas)), tam=8.2, ha="right", negrita=True)
    return pintados, y_base


def dibujar(secuencia, paleta, ruta):
    L = Lienzo("Dos hojas de la libreta",
               "La libreta de la máquina de letras con el selector en 2: una hoja por cada\n"
               "pareja de letras, y un palito por cada vez que vino detrás cada letra.",
               paleta, alto=3.79)  # sin el hueco del pie (Orden 4; era 4,10)
    y_tope = L.y - 1.0
    xt = contar(secuencia, PAGINA_CON_PALITOS)
    qu = contar(secuencia, PAGINA_CONTADA)
    pintados, _ = dibujar_pagina(L, 4.0, y_tope, PAGINA_CON_PALITOS, xt.most_common(), True)
    assert pintados == sum(xt.values()), (pintados, sum(xt.values()))
    dibujar_pagina(L, 52.0, y_tope, PAGINA_CONTADA, qu.most_common(), False)
    L.guardar(ruta)
    return ruta, xt, qu


def cargar():
    from ngrama import ALFABETO, cargar_corpus, normalizar
    return normalizar(cargar_corpus(str(AQUI / CORPUS)), ALFABETO)


def selftest():
    fallos = []
    crudo_xt = (AQUI / SALIDA_XT).read_text(encoding="utf-8")
    crudo_qu = (AQUI / SALIDA_QU).read_text(encoding="utf-8")

    # 1. TEST NULO — una salida sin la tabla de la hoja tiene que reventar: si no, la figura
    #    dibujaría palitos de una hoja que la salida no respalda.
    try:
        pagina_de_la_salida(crudo_qu, PAGINA_CON_PALITOS); revienta = False
    except AssertionError:
        revienta = True
    print(f"[1] test nulo         la salida de «{PAGINA_CONTADA}» no tiene la tabla de "
          f"«{PAGINA_CON_PALITOS}»: {'revienta' if revienta else 'NO revienta'}")
    if not revienta:
        fallos.append("test nulo: se lee una tabla que no está en la salida")

    # 2. SEÑAL IMPLANTADA — en un texto de nueve letras contado a mano («xtaxtaxte»: detrás de
    #    «xt», dos «a» y una «e»), la hoja dibuja exactamente tres palitos.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mini = contar(MINI, PAGINA_CON_PALITOS)
    L = Lienzo("prueba", "", GRIS, alto=3.79)
    pintados, _ = dibujar_pagina(L, 4.0, 40.0, PAGINA_CON_PALITOS, mini.most_common(), True)
    plt.close(L.fig)
    print(f"[2] señal implantada  «{MINI}»: {dict(mini)}; palitos dibujados: {pintados}")
    if mini != Counter({"a": 2, "e": 1}) or pintados != 3:
        fallos.append(f"señal implantada: se esperaban 2 «a» y 1 «e» y 3 palitos; salió "
                      f"{dict(mini)} y {pintados}")

    # 3. INVARIANTE — lo contado en el Quijote es, letra por letra y en el mismo orden, lo que
    #    imprimen las dos tablas del capítulo; los palitos dibujados de «xt» son los 24 de su
    #    total; y la de «qu» no se dibuja porque tiene más de MAXIMO_DIBUJABLE palitos.
    secuencia = cargar()
    malas = []
    for par, crudo in ((PAGINA_CON_PALITOS, crudo_xt), (PAGINA_CONTADA, crudo_qu)):
        impresa, total = pagina_de_la_salida(crudo, par)
        contada = contar(secuencia, par)
        hecha = [(l, miles(n)) for l, n in sorted(contada.items(), key=lambda t: -t[1])]
        if sorted(impresa, key=lambda t: t[0]) != sorted(hecha, key=lambda t: t[0]) \
                or miles(sum(contada.values())) != total:
            malas.append(par)
        # el orden de la tabla es el de más a menos, con los empates en el orden en que aparecen
        if [l for l, _ in impresa] != [l for l, _ in contada.most_common()]:
            malas.append(par + " (orden)")
    xt, qu = contar(secuencia, PAGINA_CON_PALITOS), contar(secuencia, PAGINA_CONTADA)
    print(f"[3] invariante        «{PAGINA_CON_PALITOS}»: {sum(xt.values())} palitos, {len(xt)} "
          f"letras; «{PAGINA_CONTADA}»: {miles(sum(qu.values()))} palitos, {len(qu)} letras; "
          f"coinciden con las tablas impresas: {'sí' if not malas else 'NO'}")
    if malas:
        fallos.append(f"invariante: lo contado no coincide con la tabla impresa de {malas}")
    if max(xt.values()) > MAXIMO_DIBUJABLE or max(qu.values()) <= MAXIMO_DIBUJABLE:
        fallos.append("invariante: la hoja de «xt» debía caber dibujada y la de «qu» no")
    dibujar(secuencia, GRIS, "/dev/null")      # sus asserts cuentan los palitos
    L = Lienzo("prueba", "", GRIS, alto=3.79)
    ancho_nota = ancho_texto(L, NOTA, TAM_NOTA)
    lineas_nota = poner_nota(L, 52.0, 30.0, ANCHO_HOJA)
    plt.close(L.fig)
    ancho_txt = f"{ancho_nota:.1f}".replace(".", ",")
    print(f"[3] invariante        la nota mide {ancho_txt} de {ANCHO_HOJA:.0f} unidades de hoja: "
          f"va en {lineas_nota} línea{'s' if lineas_nota > 1 else ''}, debajo de la hoja")
    if lineas_nota > 2 or (ancho_nota <= ANCHO_HOJA) != (lineas_nota == 1):
        fallos.append(f"invariante: la nota mide {ancho_txt} y sale en {lineas_nota} líneas")

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
    print("Escrito", dibujar(cargar(), GRIS, DESTINO)[0])


if __name__ == "__main__":
    main()
