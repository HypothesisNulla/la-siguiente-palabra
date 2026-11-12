#!/usr/bin/env python3
"""
Capítulo 10 — la curva de lo mal que lo hace, con los cuatro retratos marcados (L24, D07).

La figura que dibujaba `en_que_orden_aprende.py` tenía la curva entera de 0 a 50.000 pasos y un
recuadro con el tramo final «sin el cero». Dos problemas: los retratos de 30 y 300 pasos caían
pegados al borde izquierdo, donde la caída que el texto pide mirar es una rayita, y el recuadro
hablaba de un cero que nadie había explicado. Esta la sustituye en el capítulo (la otra sigue
saliendo de su programa, sin tocar):

  - arriba, el entrenamiento entero; abajo, los primeros 3.000 pasos de cerca, con la MISMA
    altura, para que se vea la misma caída ancha;
  - los cuatro retratos marcados donde caen, con sus primeras palabras;
  - el suelo rotulado: cero es acertar todas las letras y con toda seguridad.

Los datos se leen de `datos/salidas/en_que_orden_aprende_curva.csv` y de la salida del capítulo; aquí no
hay ni un número escrito.

Uso:
    python figura_lo_mal_que_lo_hace.py --selftest
    python figura_lo_mal_que_lo_hace.py
"""

# ======================= CONSTANTES =======================

CURVA = "../datos/salidas/en_que_orden_aprende_curva.csv"
SALIDA_CAP10 = "../datos/salidas/en_que_orden_aprende.txt"
DESTINO = "../figuras/lo_mal_que_lo_hace.png"
ALTO = 4.85                 # pulgadas (Orden 13: era 4,5, y el rótulo del eje de abajo salía cortado)
ALTO_BASE = 4.5             # el alto con el que se midieron las posiciones de los ejes, en fracciones
CERCA = 3_000               # hasta dónde llega el dibujo de cerca
MEDIA_VENTANA = 20          # arriba: cada punto con los 20 de cada lado (el dibujo entero)
MEDIA_CERCA = 3             # abajo, de cerca: con los 3 de cada lado (la caída es rápida)
PALABRAS_ROTULO = 3         # primeras palabras de cada retrato en su rótulo
# Dónde va el rótulo de cada retrato: (panel, paso) -> (parte del ancho, parte del alto, alineación).
# Elegido para que quede por encima de la curva; la revisión de la figura comprueba que no la pisa.
SITIO_ROTULO = {(0, 3_000): (0.09, 0.70, "left"), (0, 50_000): (0.99, 0.60, "right"),
                (1, 30): (0.04, 0.93, "left"), (1, 300): (0.16, 0.82, "left"),
                (1, 3_000): (0.99, 0.70, "right")}

# ==========================================================

import argparse
import csv
import re
import sys

from infografia import COLOR, GRIS, Lienzo
from formato import miles


def leer_curva(ruta=CURVA):
    with open(ruta, newline="", encoding="utf-8") as fh:
        filas = list(csv.reader(fh))
    assert filas[0] == ["pasos", "lo_mal_que_lo_hace"], f"cabecera inesperada en {ruta}"
    return [(int(a), float(b)) for a, b in filas[1:]]


def leer_retratos(ruta=SALIDA_CAP10):
    t = open(ruta, encoding="utf-8").read()
    pares = re.findall(r"\[tras ([\d.]+) pasos[^\]]*\]\n(.*)\n", t)
    assert len(pares) == 4, "se esperaban cuatro retratos"
    return [(int(p.replace(".", "")), " ".join(m.split()[:PALABRAS_ROTULO]) + "…") for p, m in pares]


def suavizar(v, medio=MEDIA_VENTANA):
    """La media de cada punto con los `medio` que tiene a cada lado: centrada, para que la línea
    negra vaya por DENTRO de la gris y no detrás de ella. (La de `en_que_orden_aprende.py` es la
    media de los puntos anteriores, y en una curva que cae deprisa se queda por encima: la
    verificación del 28 vio la negra por encima de la gris en los primeros pasos.) En los bordes la
    ventana se encoge igual por los dos lados, para no arrastrar nada."""
    fuera = []
    n = len(v)
    for i in range(n):
        h = min(medio, i, n - 1 - i)
        tramo = v[i - h:i + h + 1]
        fuera.append(sum(tramo) / len(tramo))
    return fuera


def valor_en(curva, paso):
    """La medición cruda en ese paso, entre los dos puntos guardados más cercanos."""
    antes = [c for c in curva if c[0] <= paso]
    despues = [c for c in curva if c[0] >= paso]
    if not antes:
        return despues[0][1]
    if not despues:
        return antes[-1][1]
    (a, va), (b, vb) = antes[-1], despues[0]
    return va if a == b else va + (vb - va) * (paso - a) / (b - a)


def dibujar(curva, retratos, paleta, ruta):
    L = Lienzo("Lo mal que lo hace, paso a paso",
               "Gris: la medición de cada paso. Negro: la misma, suavizada. Arriba, el\n"
               "entrenamiento entero; abajo, sus primeros pasos de cerca, a la misma altura.",
               paleta, alto=ALTO)
    p = paleta
    W, H = L.fig.get_size_inches()
    alto_u = L.alto_u
    # Las posiciones de los ejes eran fracciones del alto de la figura, medidas con ALTO_BASE. Al
    # subir el alto para que quepa el rótulo del eje de abajo (Orden 13), se pasan a pulgadas desde
    # el borde de arriba y se vuelven a pasar a fracción: las dos gráficas miden y caen lo mismo,
    # y lo que se añade es margen por debajo. Ni un texto ni un número cambian.
    arriba_in = (alto_u - L.y) * (W / 100.0)             # de arriba a donde empiezan las gráficas
    fraccion = lambda pulgadas_desde_arriba: 1 - (arriba_in + pulgadas_desde_arriba) / H
    tope = max(v for _, v in curva) * 1.08
    ejes = []
    marcados = []
    for k, (x0, x1) in enumerate(((0, curva[-1][0]), (0, CERCA))):
        base = (0.02 + k * 0.42) * ALTO_BASE
        ax = L.fig.add_axes([0.14, fraccion(base + 0.31 * ALTO_BASE), 0.82, 0.30 * ALTO_BASE / H])
        tramo = [c for c in curva if c[0] <= x1]
        t = [c[0] for c in tramo]
        v = [c[1] for c in tramo]
        todo_v = [c[1] for c in curva]
        suave = suavizar(todo_v, MEDIA_VENTANA if k == 0 else MEDIA_CERCA)[:len(tramo)]
        ax.plot(t, v, color="0.72", linewidth=0.7)
        ax.plot(t, suave, color=p.tinta, linewidth=1.5)
        ax.set_xlim(x0, x1)
        ax.set_ylim(0, tope)
        ax.set_yticks([])
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=6.8, length=2.5)
        ax.xaxis.set_major_formatter(__import__("matplotlib.pyplot").pyplot.FuncFormatter(
            lambda x, _: miles(int(x))))
        ax.set_ylabel("lo mal que lo hace", fontsize=7.4, color=p.tinta)
        ax.text(x1, tope * 0.03, "el cero: acertaría todas las letras, con toda la probabilidad en la buena ",
                ha="right", va="bottom", fontsize=6.4, color=p.suave)
        for paso, rotulo in retratos:
            if not (x0 < paso <= x1):
                continue
            if k == 0 and paso < CERCA:
                continue
            yv = valor_en(curva, paso)
            ax.plot([paso], [yv], marker="o", markersize=3.6, color=p.tinta, zorder=5)
            # El rótulo va por encima de la curva, a una altura fija por retrato, y una raya lo une
            # a su punto: pegado al punto, la medición cruda (que tiembla) lo tachaba.
            fx, fy, ha = SITIO_ROTULO[(k, paso)]
            ax.annotate(f"tras {miles(paso)} pasos: «{rotulo}»", (paso, yv),
                        xytext=(x0 + (x1 - x0) * fx, tope * fy), fontsize=6.6, ha=ha, va="bottom",
                        color=p.tinta, arrowprops=dict(arrowstyle="-", color=p.suave, linewidth=0.6))
            marcados.append((k, paso))
        if k == 0:
            ax.axvspan(0, CERCA, color=p.neutro, alpha=0.6, linewidth=0)
            ax.text(CERCA * 1.3, tope * 0.93, "este tramo, abajo de cerca", fontsize=6.6,
                    ha="left", va="top", color=p.suave)
        ax.set_xlabel("pasos de aprendizaje", fontsize=7.4, labelpad=1.5)
        ejes.append(ax)
    # Nada de lo escrito sale de la figura: el rótulo del eje de abajo salía cortado (Orden 13).
    L.fig.canvas.draw()
    pintor = L.fig.canvas.get_renderer()
    fuera = L.fig.bbox
    for ax in ejes:
        for texto in [ax.xaxis.label, *ax.get_xticklabels()]:
            caja = texto.get_window_extent(pintor)
            assert caja.y0 >= fuera.y0 and caja.y1 <= fuera.y1, (
                f"«{texto.get_text()}» se sale de la figura por abajo o por arriba "
                f"({caja.y0:.1f} < {fuera.y0:.1f}): sube el alto")
    L.guardar(ruta)
    return marcados


def selftest():
    fallos = []
    curva, retratos = leer_curva(), leer_retratos()
    # 1. TEST NULO — una curva plana no tiene caída: lo suavizado es igual a lo medido.
    plana = [(p, 1.0) for p, _ in curva]
    s = suavizar([v for _, v in plana])
    print(f"[1] test nulo         curva plana, suavizada igual: {'sí' if max(abs(x - 1) for x in s) < 1e-12 else 'NO'}")
    if max(abs(x - 1) for x in s) > 1e-12:
        fallos.append("test nulo: suavizar cambia una curva plana")
    # 2. SEÑAL — los cuatro retratos se marcan: 3.000 y 50.000 arriba; 30, 300 y 3.000 abajo.
    m = dibujar(curva, retratos, GRIS, "/dev/null")
    esperado = [(0, 3000), (0, 50000), (1, 30), (1, 300), (1, 3000)]
    print(f"[2] señal             marcados {m}")
    if sorted(m) != sorted(esperado):
        fallos.append(f"señal: se esperaban {esperado}")
    # 3. INVARIANTE — la curva baja: el final suavizado está por debajo del principio, y el
    #    suelo del dibujo es el cero (lo que dice el rótulo).
    v = suavizar([x for _, x in curva])
    for pal in (COLOR, GRIS):
        dibujar(curva, retratos, pal, "/dev/null")
    print(f"[3] invariante        baja: {'sí' if v[-1] < v[MEDIA_VENTANA] else 'NO'}; mínimo medido "
          f"por encima del cero: {'sí' if min(x for _, x in curva) > 0 else 'NO'}")
    if not (v[-1] < v[MEDIA_VENTANA] and min(x for _, x in curva) > 0):
        fallos.append("invariante: la curva no baja o toca el cero")
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
    codigo = selftest()
    if codigo or args.selftest:
        sys.exit(codigo)
    dibujar(leer_curva(), leer_retratos(), GRIS, DESTINO)
    print(f"escrita {DESTINO}")


if __name__ == "__main__":
    main()
