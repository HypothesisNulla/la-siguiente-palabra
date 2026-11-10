#!/usr/bin/env python3
"""
Capítulo 4 — lo que deja pasar la rampa corta y lo que deja pasar el codo (L24, A20).

Dos dibujos lado a lado, **con la misma escala en los dos ejes y en los dos cuadros**: lo que le
llega a la neurona en horizontal (cuánto pasa del listón) y lo que dice en vertical. Así la
inclinación se compara a ojo: la rampa, en su centro, sube una cuarta parte de lo que avanza; el
codo, en su parte positiva, sube lo mismo que avanza. (En la primera versión cada cuadro tenía su
escala y la rampa parecía más empinada que el codo: la figura decía lo contrario que el texto.)
El codo se llamaba «la que corta» (segunda vuelta de L24: chocaba con «la rampa corta»).

Los números marcados se leen del bloque 6 de `datos/salidas/culpa_hacia_atras.txt`; las curvas se
dibujan con las mismas funciones que usa el programa, y el selftest comprueba que pasan por los
puntos de la salida.

Desde la Orden 148 (10 nov 2026, regla 11.9: una figura del cuerpo no lleva números que el cuerpo no explique) la
figura no lleva cifras: las marcas dicen «una cuarta parte», «casi nada» (la rampa) y «nada», «todo» (el codo); en el
eje horizontal, una sola marca, «el listón», con el rótulo «lo que le llega»; en el vertical, solo el 0 y el 1. Las
palabras no se escriben a ojo: `palabras_casan` comprueba que casan con los números de la salida (0,25, menos de
0,05, 0 y 1), y el selftest lo exige.

Uso:
    python figura_la_rampa_y_el_codo.py --selftest
    python figura_la_rampa_y_el_codo.py
"""

# ======================= CONSTANTES =======================

SALIDA = "../datos/salidas/culpa_hacia_atras.txt"
DESTINO = "../figuras/la_rampa_y_el_codo.png"
ALTO = 2.5                     # pulgadas
X_DESDE, X_HASTA = -4.5, 4.5   # la misma ventana en los dos cuadros
Y_DESDE, Y_HASTA = -0.6, 4.0
MARCA_RAMPA = [0, 4]           # dónde se marca lo que deja pasar la rampa
MARCA_CODO = [-2, 2]           # y el codo
LADO_TANGENTE = 1.0

# ==========================================================

import argparse
from formato import coma, leer_tablas
import re
import sys
from pathlib import Path

import numpy as np

from infografia import COLOR, GRIS, Lienzo
from retropropagacion import sigmoide

AQUI = Path(__file__).resolve().parent


def num(s):
    return float(s.replace("+", "").replace(",", "."))


def leer(ruta):
    """Del bloque 6: {x: (rampa dice, rampa deja pasar, codo dice, codo deja pasar o None)}.
    Desde el 9 de octubre de 2026, de la tabla editorial «Lo que deja pasar cada una»."""
    T = leer_tablas(Path(ruta).read_text(encoding="utf-8"))
    assert "Lo que deja pasar cada una" in T, f"se esperaba la tabla «Lo que deja pasar cada una» en {ruta}"
    rot, filas_t, _ = T["Lo que deja pasar cada una"]
    assert rot[0] == "pasa del listón por" and len(rot) == 5, rot
    filas = {}
    for x, rd, rp, cd, cp in filas_t:
        filas[num(x)] = (num(rd), num(rp), num(cd), None if cp.startswith("(") else num(cp))
    assert len(filas) >= 5, f"se esperaban al menos cinco filas; hay {len(filas)}"
    return filas


def c(x):
    return f"{x:.3f}".replace(".", ",")


def panel(L, x0, titulo, xs, ys):
    ancho = 0.40
    alto = ancho * 4.45 / ALTO * (Y_HASTA - Y_DESDE) / (X_HASTA - X_DESDE)
    ax = L.fig.add_axes([x0, 0.22, ancho, alto])
    ax.plot(xs, ys, color=L.p.tinta, linewidth=1.6)
    ax.set_xlim(X_DESDE, X_HASTA); ax.set_ylim(Y_DESDE, Y_HASTA)
    ax.set_aspect("equal")
    ax.set_title(titulo, fontsize=9.0, color=L.p.tinta, fontweight="bold", pad=4)
    ax.set_xlabel("lo que le llega", fontsize=7.2, color=L.p.tinta)
    ax.set_ylabel("lo que dice", fontsize=7.2, color=L.p.tinta)
    ax.set_xticks([0]); ax.set_xticklabels(["el listón"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["0", "1"])
    ax.tick_params(labelsize=7.0)
    ax.axvline(0, color=L.p.marco, linewidth=0.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax


def marca(ax, x, y, pendiente, texto, p, xt, yt):
    xs = np.array([x - LADO_TANGENTE, x + LADO_TANGENTE])
    ax.plot(xs, y + pendiente * (xs - x), color=p.acento, linewidth=3.0, alpha=0.55,
            solid_capstyle="round")
    ax.plot([x], [y], "o", color=p.tinta, markersize=3.5)
    ax.annotate(texto, xy=(x, y), xytext=(xt, yt), fontsize=6.9, ha="center", va="center",
                color=p.tinta, linespacing=1.15, arrowprops=dict(arrowstyle="-", color=p.suave, lw=0.6))


PALABRAS = {                   # lo que dice la figura, y la prueba de que casa con la salida
    "una cuarta parte": lambda d: abs(d[MARCA_RAMPA[0]][1] - 0.25) < 1e-3,
    "casi nada": lambda d: d[MARCA_RAMPA[1]][1] < 0.05,
    "nada": lambda d: d[MARCA_CODO[0]][3] == 0,
    "todo": lambda d: abs(d[MARCA_CODO[1]][3] - 1) < 1e-9,
}


def palabras_casan(filas):
    """Las palabras de la figura que NO casan con los números de la salida (lista vacía si casan todas)."""
    return [w for w, ok in PALABRAS.items() if not ok(filas)]


def dibujar(filas, paleta, ruta):
    L = Lienzo("La rampa y el codo",
               "Los dos con la misma escala. La inclinación es lo que deja pasar cada uno: por\n"
               "cada pelín que se mueve lo que le llega, cuánto se mueve lo que dice.",
               paleta, alto=ALTO)
    p = L.p
    xs = np.linspace(X_DESDE, X_HASTA, 400)
    a = panel(L, 0.08, "la rampa corta", xs, sigmoide(xs))
    d0, d1 = filas[MARCA_RAMPA[0]], filas[MARCA_RAMPA[1]]
    assert not palabras_casan(filas), f"las palabras de la figura no casan con la salida: {palabras_casan(filas)}"
    marca(a, MARCA_RAMPA[0], d0[0], d0[1], "en el centro: deja\npasar una cuarta parte", p, -1.8, 2.4)
    marca(a, MARCA_RAMPA[1], d1[0], d1[1], "en lo plano:\ncasi nada", p, 2.6, 2.6)
    b = panel(L, 0.56, "el codo", xs, np.maximum(0, xs))
    e0, e1 = filas[MARCA_CODO[0]], filas[MARCA_CODO[1]]
    marca(b, MARCA_CODO[0], e0[2], e0[3], "por debajo:\nnada", p, -2.6, 1.6)
    marca(b, MARCA_CODO[1], e1[2], e1[3], "por encima:\ntodo", p, 3.2, 0.7)
    L.guardar(ruta)
    return a, b


def selftest():
    fallos = []
    filas = leer(AQUI / SALIDA)

    # 1. TEST NULO — en lo plano de la rampa y por debajo del codo no pasa (casi) nada.
    nulos = [filas[4][1], filas[-4][1], filas[-2][3]]
    print(f"[1] test nulo         en lo plano deja pasar {', '.join(coma(x, 3) for x in nulos)}")
    if max(nulos) > 0.02:
        fallos.append("test nulo: en lo plano pasa algo")

    # 2. SEÑAL — la rampa deja pasar 0,25 en el centro; el codo, 1 por encima del listón.
    print(f"[2] señal             rampa en 0: {coma(filas[0][1], 2)}; codo en +2: {coma(filas[2][3], 1)}")
    if abs(filas[0][1] - 0.25) > 1e-3 or abs(filas[2][3] - 1) > 1e-3:
        fallos.append("señal: la rampa no deja pasar 0,25 en el centro, o el codo no deja pasar 1")

    # 3. INVARIANTE — las curvas pasan por los puntos de la salida, y los dos cuadros tienen la
    #    misma escala (se comprueba en el dibujo: la unidad mide lo mismo en x y en y, y en los
    #    dos cuadros).
    peor = max(max(abs(float(sigmoide(x)) - v[0]), abs(max(0, x) - v[2])) for x, v in filas.items())
    for pal in (COLOR, GRIS):
        a, b = dibujar(filas, pal, "/dev/null")
    # la 11.9 (Orden 148): las palabras casan con la salida, y los ejes no llevan más cifras que el 0 y el 1
    mal = palabras_casan(filas)
    marcas = [[t.get_text() for t in ax.get_xticklabels()] for ax in (a, b)]
    cifras_y = [[t.get_text() for t in ax.get_yticklabels()] for ax in (a, b)]
    print(f"[3] invariante        curvas contra salida, diferencia máxima {coma(peor, 4)}; misma escala en los dos cuadros; "
          f"palabras que no casan con la salida: {mal or 'ninguna'}; marcas del eje horizontal: {marcas[0]}; del vertical: {cifras_y[0]}")
    if peor > 0.001:
        fallos.append("invariante: las curvas no pasan por los puntos de la salida")
    if mal or marcas != [["el listón"]] * 2 or cifras_y != [["0", "1"]] * 2:
        fallos.append(f"invariante: las palabras no casan con la salida o los ejes llevan cifras: {mal} {marcas} {cifras_y}")
    # señal implantada: con una salida cambiada a propósito, las palabras tienen que dejar de casar
    cambiada = {k: tuple(0.9 if i == 1 else x for i, x in enumerate(v)) for k, v in filas.items()}
    detecta = palabras_casan(cambiada)
    print(f"      señal implantada  con lo que deja pasar la rampa cambiado a 0,9: casan mal {detecta}")
    if "una cuarta parte" not in detecta:
        fallos.append("señal implantada: «una cuarta parte» sigue casando con una salida de 0,9")
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
    dibujar(leer(AQUI / SALIDA), GRIS, str(AQUI / DESTINO))
    print(f"escrita {DESTINO}")


if __name__ == "__main__":
    main()
