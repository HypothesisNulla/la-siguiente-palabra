#!/usr/bin/env python3
"""
Capítulo 6 — la red que lee, desenrollada: un eslabón por símbolo (L24).

La red del capítulo 6 lee una prueba de memoria símbolo a símbolo, arrastrando un resumen. Si se
dibuja un eslabón por símbolo, con el resumen pasando de cada uno al siguiente, sale una pila de
capas tumbada: la forma del capítulo 3, y por eso le pasa lo mismo que en el capítulo 4 a la
culpa que tiene que volver hasta el primer eslabón.

La prueba no se inventa: se lee del bloque 1 de `datos/salidas/una_prueba_de_memoria.txt`, que
la sacó la misma función que genera las pruebas del entrenamiento.

Uso:
    python figura_red_desenrollada.py --selftest
    python figura_red_desenrollada.py
"""

# ======================= CONSTANTES =======================

SALIDA = "../datos/salidas/una_prueba_de_memoria.txt"
DESTINO = "../figuras/red_desenrollada.png"
ALTO = 2.9
PRUEBA = 1                       # la primera prueba del bloque (la de diez de paja)

# ==========================================================

import argparse
import re
import sys
from pathlib import Path

from matplotlib.patches import FancyArrow, FancyBboxPatch

from formato import leer_tablas
from infografia import COLOR, GRIS, Lienzo

AQUI = Path(__file__).resolve().parent


def leer(ruta):
    """La prueba, de sus tablas editoriales (L24, 9 de octubre)."""
    tablas = leer_tablas(Path(ruta).read_text(encoding="utf-8"))
    pruebas = [t for t in tablas if t.startswith("Una prueba, con ")]
    assert len(pruebas) >= PRUEBA, "no se encontró la prueba en el bloque 1"
    filas = dict((f[0], f[1]) for f in tablas[pruebas[PRUEBA - 1]][1])
    return [filas["se le enseña"]] + filas["luego, la paja"].split(), filas["respuesta buscada"]


def dibujar(simbolos, respuesta, paleta, ruta):
    L = Lienzo("La red leyendo una prueba, eslabón a eslabón",
               "Un eslabón por símbolo. Cada uno combina su símbolo con el resumen que le\n"
               "llega y pasa el resultado al siguiente.", paleta, alto=ALTO)
    p, ax = L.p, L.ax
    n = len(simbolos)
    paso = 58.0 / n
    y_sim, y_res = 12.0, 25.0
    x_ini = 21.0
    xs = [x_ini + paso * i + paso / 2 for i in range(n)]
    # a la izquierda, en pequeño, la pila de capas del capítulo 3: la misma cadena, de pie
    for k in range(4):
        yb = 9.0 + k * 6.0
        ax.add_patch(FancyBboxPatch((6.0, yb), 7.0, 3.6, boxstyle="round,pad=0,rounding_size=0.6",
                                    facecolor=p.fondo, edgecolor=p.tinta, linewidth=0.8))
        if k < 3:
            ax.add_patch(FancyArrow(9.5, yb + 3.7, 0, 2.2, width=0.2, head_width=1.0,
                                    head_length=0.8, length_includes_head=True,
                                    facecolor=p.tinta, edgecolor="none"))
    L.texto(9.5, 5.8, "capas del", tam=6.4, ha="center", color=p.suave)
    L.texto(9.5, 3.4, "capítulo 3", tam=6.4, ha="center", color=p.suave)
    ax.plot([16.5, 16.5], [4, 34], color=p.marco, linewidth=0.8)
    for i, (x, s) in enumerate(zip(xs, simbolos)):
        primero = i == 0
        L.ficha(x - 2.4, y_sim, s, 4.8, alto=4.6, relleno=p.acento if primero else "white",
                tinta="white" if primero else p.tinta, negrita=primero, tam=8.4)
        ax.add_patch(FancyArrow(x, y_sim + 2.5, 0, y_res - y_sim - 5.2, width=0.25, head_width=1.2,
                                head_length=1.0, length_includes_head=True, facecolor=p.suave,
                                edgecolor="none"))
        ax.add_patch(FancyBboxPatch((x - 2.2, y_res - 2.2), 4.4, 4.4,
                                    boxstyle="round,pad=0,rounding_size=0.8",
                                    facecolor=p.fondo, edgecolor=p.tinta, linewidth=0.9))
        if i < n - 1:
            ax.add_patch(FancyArrow(x + 2.3, y_res, paso - 4.6, 0, width=0.25, head_width=1.2,
                                    head_length=1.0, length_includes_head=True,
                                    facecolor=p.tinta, edgecolor="none"))
    xf = xs[-1] + 2.3
    ax.add_patch(FancyArrow(xf, y_res, 4.0, 0, width=0.25, head_width=1.2, head_length=1.0,
                            length_includes_head=True, facecolor=p.tinta, edgecolor="none"))
    L.texto(xf + 4.8, y_res + 1.6, "¿cuál era", tam=7.2)
    L.texto(xf + 4.8, y_res - 1.6, f"el primero? {respuesta}", tam=7.2, negrita=True)
    L.texto(x_ini, y_sim - 5.0, "símbolo que lee", tam=7.0, color=p.suave)
    L.texto(x_ini, y_res + 4.6, "resumen", tam=7.0, color=p.suave)
    # la culpa, de vuelta
    yc = y_res + 9.0
    ax.add_patch(FancyArrow(xs[-1], yc, xs[0] - xs[-1], 0, width=0.2, head_width=1.3,
                            head_length=1.2, length_includes_head=True, facecolor=p.suave,
                            edgecolor="none", linestyle="--"))
    L.texto((xs[0] + xs[-1]) / 2, yc + 2.2,
            f"la culpa de un fallo vuelve eslabón a eslabón: {n - 1} eslabones hasta la {simbolos[0]}",
            tam=7.0, ha="center", color=p.suave)
    L.guardar(ruta)
    return xs


def selftest():
    fallos = []
    sim, resp = leer(AQUI / SALIDA)
    # 1. TEST NULO — una prueba sin paja es un solo eslabón y ninguna flecha entre eslabones.
    xs = dibujar(sim[:1], resp, GRIS, "/dev/null")
    print(f"[1] test nulo         sin paja: {len(xs)} eslabón")
    if len(xs) != 1:
        fallos.append("test nulo: sin paja no sale un solo eslabón")
    # 2. SEÑAL — la prueba de la salida: un eslabón por símbolo, y la respuesta es el primero.
    xs = dibujar(sim, resp, GRIS, "/dev/null")
    print(f"[2] señal             {len(xs)} eslabones para {len(sim)} símbolos; respuesta {resp} = primero {sim[0]}")
    if len(xs) != len(sim) or resp != sim[0]:
        fallos.append("señal: eslabones o respuesta no casan con la prueba")
    # 3. INVARIANTE — los eslabones van de izquierda a derecha y caben en la página; color y gris.
    dibujar(sim, resp, COLOR, "/dev/null")
    ok = all(a < b for a, b in zip(xs, xs[1:])) and xs[0] > 16.5 and xs[-1] < 90
    print(f"[3] invariante        eslabones en orden y dentro de la página: {ok}")
    if not ok:
        fallos.append("invariante: eslabones fuera de orden o de la página")
    print()
    if fallos:
        for x in fallos:
            print("FALLA:", x)
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
    dibujar(*leer(AQUI / SALIDA), GRIS, str(AQUI / DESTINO))
    print(f"escrita {DESTINO}")


if __name__ == "__main__":
    main()
