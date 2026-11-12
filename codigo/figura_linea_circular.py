#!/usr/bin/env python3
"""
Capítulo 7 — el plano de metro de la máquina entera (F7-1, aprobada en CLARIDAD.md, D7-6).

Dos dibujos del mismo plano:
  - el grande: la línea circular con sus cinco estaciones y la vuelta dibujada como vuelta, para
    abrir el capítulo (`figuras/linea_circular.png`);
  - cinco tiras pequeñas, una por estación, con esa estación marcada «estás aquí», para abrir
    cada apartado (`figuras/linea_circular_1.png` a `_5.png`).

Cinco estaciones y no seis: es la propuesta del borrador del capítulo 7 (BORRADOR-CAP07.md,
apartado 1), pendiente de que Carlos la confirme. Si prefiere seis, se cambia ESTACIONES.

A lo sumo una cifra por estación, y se leen de `datos/salidas/maquina_entera.txt`: los números
por trozo, las capas y las entradas de la lista.

Uso:
    python figura_linea_circular.py --selftest
    python figura_linea_circular.py
"""

# ======================= CONSTANTES =======================

SALIDA = "../datos/salidas/maquina_entera.txt"
DESTINO = "../figuras/linea_circular.png"
DESTINO_TIRA = "../figuras/linea_circular_{}.png"
ALTO = 3.55
ALTO_TIRA = 1.25
# (nombre corto, qué pasa ahí, clave de la cifra en la salida o None)
# Orden 157 (12 nov 2026): los rótulos dicen lo mismo que el cap. 7 (trozos, «Encuesta», «Se escoge uno») y
# la estación 4 ya no lleva «151.936», que el cuerpo no explica (11.9). Las cinco tiras salen de esta misma
# lista, así que dicen lo mismo; hoy ningún capítulo las cita.
ESTACIONES = [("Se rompe", "el texto, en trozos", None),
              ("Números", "{} números en cada lista", "números por trozo"),
              ("Capas", "{} capas", "rondas, una detrás de otra"),
              ("Encuesta", "una probabilidad por trozo", None),
              ("Se escoge uno", "y se pega al texto", None)]

# ==========================================================

import argparse
import math
import re
import sys
from pathlib import Path

from matplotlib.patches import Arc, Circle, FancyArrowPatch

from formato import leer_tablas
from infografia import COLOR, GRIS, Lienzo

AQUI = Path(__file__).resolve().parent


def leer(ruta):
    """Las cifras de las estaciones, de la tabla «El tamaño de la máquina» (L24, 9 de octubre)."""
    tablas = leer_tablas(Path(ruta).read_text(encoding="utf-8"))
    assert "El tamaño de la máquina" in tablas, f"se esperaba la tabla del tamaño en {ruta}"
    filas = dict((f[0], f[1]) for f in tablas["El tamaño de la máquina"][1])
    cifras = {}
    for _, _, clave in ESTACIONES:
        if clave:
            assert clave in filas, f"no se encontró «{clave}» en {ruta}"
            cifras[clave] = filas[clave]
    return cifras


def rotulos(cifras):
    return [(n, q.format(cifras[c]) if c else q) for n, q, c in ESTACIONES]


def dibujar(cifras, paleta, ruta):
    L = Lienzo("La máquina entera, como una línea de metro",
               "El texto recorre las cinco estaciones y vuelve a la primera cada vez\n"
               "que la máquina escribe algo más.", paleta, alto=ALTO)
    p, ax = L.p, L.ax
    cx, cy, r = 50, 30, 17
    ax.add_patch(Circle((cx, cy), r, facecolor="none", edgecolor=p.tinta, linewidth=3.0))
    n = len(ESTACIONES)
    pos = []
    for k, (nombre, que) in enumerate(rotulos(cifras)):
        ang = math.pi / 2 - 2 * math.pi * k / n          # la 1 arriba; en el sentido del reloj
        x, y = cx + r * math.cos(ang), cy + r * math.sin(ang)
        pos.append((x, y))
        ax.add_patch(Circle((x, y), 3.1, facecolor="white", edgecolor=p.tinta, linewidth=1.6, zorder=3))
        L.texto(x, y, str(k + 1), tam=9, ha="center", negrita=True)
        dx = math.cos(ang)
        ha = "left" if dx > 0.3 else ("right" if dx < -0.3 else "center")
        tx = x + (5 if ha == "left" else -5 if ha == "right" else 0)
        ty = y + (6.8 if abs(dx) <= 0.3 else 1.8 * math.copysign(1, math.sin(ang)))
        L.texto(tx, ty + 1.5, nombre, tam=8.4, ha=ha, negrita=True)
        L.texto(tx, ty - 1.5, que, tam=7.2, ha=ha, color=p.suave)
    # flechas del sentido de la marcha, entre estación y estación
    for k in range(n):
        a1 = math.pi / 2 - 2 * math.pi * (k + 0.45) / n
        a2 = math.pi / 2 - 2 * math.pi * (k + 0.55) / n
        ax.add_patch(FancyArrowPatch((cx + r * math.cos(a1), cy + r * math.sin(a1)),
                                     (cx + r * math.cos(a2), cy + r * math.sin(a2)),
                                     arrowstyle="-|>", mutation_scale=11, color=p.tinta, zorder=4))
    L.texto(cx, cy + 2.0, "de la 5 a la 1:", tam=7.2, ha="center", color=p.suave)
    L.texto(cx, cy - 1.2, "vuelta a empezar, con el", tam=7.2, ha="center", color=p.suave)
    L.texto(cx, cy - 4.2, "texto un poco más largo", tam=7.2, ha="center", color=p.suave)
    L.guardar(ruta)
    return pos


def dibujar_tira(cifras, aqui, paleta, ruta):
    L = Lienzo("", "", paleta, alto=ALTO_TIRA)
    p, ax = L.p, L.ax
    n = len(ESTACIONES)
    y = 17.0
    xs = [12 + 19 * k for k in range(n)]
    ax.plot([xs[0], xs[-1]], [y, y], color=p.tinta, linewidth=2.4)
    ax.add_patch(FancyArrowPatch((xs[-1], y - 3.2), (xs[0], y - 3.2), connectionstyle="arc3,rad=0.12",
                                 arrowstyle="-|>", mutation_scale=9, color=p.suave, linewidth=0.9))
    for k, (nombre, _) in enumerate(rotulos(cifras)):
        mio = k + 1 == aqui
        ax.add_patch(Circle((xs[k], y), 2.6 if mio else 2.0, facecolor=p.tinta if mio else "white",
                            edgecolor=p.tinta, linewidth=1.4, zorder=3))
        L.texto(xs[k], y, str(k + 1), tam=7.6, ha="center", negrita=True,
                color="white" if mio else p.tinta)
        L.texto(xs[k], y + 5.0, nombre, tam=7.4, ha="center", negrita=mio,
                color=p.tinta if mio else p.suave)
    L.texto(xs[aqui - 1], y + 9.4, "estás aquí", tam=7.0, ha="center", negrita=True)
    L.guardar(ruta)
    return xs


def selftest():
    fallos = []
    cifras = leer(AQUI / SALIDA)
    # 1. TEST NULO — sin cifras que leer, el programa no se inventa ninguna: revienta.
    try:
        rotulos({})
        vacio = False
    except KeyError:
        vacio = True
    print(f"[1] test nulo         sin la salida, las estaciones con cifra no se dibujan: {vacio}")
    if not vacio:
        fallos.append("test nulo: dibuja cifras que no ha leído")
    # 2. SEÑAL — las cifras son las del capítulo: 896 y 24 (los 151.936 de la estación 4 se quitaron
    #    de la figura en la Orden 157: el cuerpo no los explica).
    ok = (cifras["números por trozo"], cifras["rondas, una detrás de otra"]) == ("896", "24")
    print(f"[2] señal             cifras leídas {cifras}: {ok}")
    if not ok:
        fallos.append("señal: las cifras no son las del capítulo")
    # 3. INVARIANTE — cinco estaciones a la misma distancia del centro, en el sentido del reloj; y
    #    en cada tira hay exactamente una marcada. En color y en gris.
    pos = dibujar(cifras, GRIS, "/dev/null")
    dibujar(cifras, COLOR, "/dev/null")
    dist = {round(math.hypot(x - 50, y - 30), 6) for x, y in pos}
    horario = all((pos[k][0] - 50) * (pos[k + 1][1] - 30) - (pos[k][1] - 30) * (pos[k + 1][0] - 50) < 0
                  for k in range(len(pos) - 1))
    for k in range(1, len(ESTACIONES) + 1):
        dibujar_tira(cifras, k, GRIS, "/dev/null")
    print(f"[3] invariante        {len(pos)} estaciones, misma distancia: {len(dist) == 1}; sentido del reloj: {horario}")
    if len(dist) != 1 or not horario:
        fallos.append("invariante: la línea no es un círculo recorrido en el sentido del reloj")
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
    cifras = leer(AQUI / SALIDA)
    dibujar(cifras, GRIS, str(AQUI / DESTINO))
    for k in range(1, len(ESTACIONES) + 1):
        dibujar_tira(cifras, k, GRIS, str(AQUI / DESTINO_TIRA.format(k)))
    print(f"escritas {DESTINO} y cinco tiras")


if __name__ == "__main__":
    main()
