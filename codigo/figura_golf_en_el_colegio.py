#!/usr/bin/env python3
"""
Orden 199 (22 nov 2026) y Orden 200 (22 nov 2026, segunda versión): la figura del golf, para el anexo A4.

El mismo peso de la red del colegio (el de la línea de la primera a la final), saliendo de 4, con cuatro
poquitos: 0,5, 100, 500 y 1.000. Dos paneles. A la izquierda (Orden 199), el tramo del terreno (de -5 a
30): la curva del error, el 0,5 con un punto cada 100 pasos y el 100 con sus diez primeros golpes. A la
derecha (Orden 200), dónde está el peso, paso a paso: en horizontal el paso (escala logarítmica, de 1 a
2.000), en vertical lo que vale el peso; el 500 y el 1.000 con todos sus pasos, sin flechas ni arcos, y
una línea fina en el fondo de la curva.

Todos los números se leen de `datos/salidas/golf_en_el_colegio_datos.csv` (los cuatro recorridos) y de
`datos/salidas/culpa_en_el_colegio_datos.csv` (la curva del terreno, la misma de
`figura_el_terreno_del_colegio.py`). Ningún número se escribe a mano; el fondo (14,121) es el peso de la fila
del CSV de golf con el error más bajo.

Uso:
    python figura_golf_en_el_colegio.py --selftest
    python figura_golf_en_el_colegio.py
"""

# ======================= CONSTANTES =======================

DATOS_GOLF = "../datos/salidas/golf_en_el_colegio_datos.csv"       # los recorridos, de golf_en_el_colegio.py
DATOS_TERRENO = "../datos/salidas/culpa_en_el_colegio_datos.csv"   # la curva del terreno
DESTINO = "../figuras/golf_en_el_colegio.png"
COPIA = "../../Claude outputs/reforma-nov/cap16/ver/golf_en_el_colegio.png"
ALTO = 3.5                      # pulgadas
IZQUIERDA = (0.5, 100.0)        # los poquitos del panel de la izquierda
DERECHA = (500.0, 1000.0)       # y los del de la derecha
TRAMO_IZQ = (-5.0, 30.0)        # el tramo de la figura del terreno
PASOS_DER = (1, 2000)           # la derecha: el paso, en escala logarítmica, de 1 a 2.000 (el paso 0 es la salida y no cabe en un logaritmo)
PESO_DER = (-10.0, 135.0)       # y lo que vale el peso, de unos -10 a unos 135
CADA_EN_0_5 = 100               # el 0,5: un punto cada tantos pasos (como en la figura del terreno)
GOLPES_EN_100 = 10              # el 100: los diez primeros golpes

# ==========================================================

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import FancyArrowPatch   # las flechas del 100, a la izquierda

from figura_el_terreno_del_colegio import curvas
from figura_la_red_del_colegio import leer_datos
from infografia import COLOR, GRIS, Lienzo

AQUI = Path(__file__).resolve().parent


def leer_golf(ruta):
    """{poquito: array de filas (paso, peso, error)} del CSV de golf_en_el_colegio.py."""
    d = {}
    with open(ruta, encoding="utf-8") as fh:
        for fila in csv.DictReader(fh):
            d.setdefault(float(fila["poquito"]), []).append((int(fila["paso"]), float(fila["peso"]), float(fila["error"])))
    assert d, "no hay datos de golf"
    return {p: np.array(f) for p, f in d.items()}


def que_se_dibuja(golf):
    """Lo que se dibuja: {('izquierda', poquito): [(peso, error), ...]} (solo los poquitos de IZQUIERDA) y
    {('derecha'|'final', poquito): [(paso, peso), ...]} (solo los de DERECHA: todos sus pasos y el último)."""
    sale = {}
    for p in IZQUIERDA:
        if p not in golf:
            continue
        f = golf[p]
        paso_final = int(f[-1, 0])
        if p == IZQUIERDA[0]:
            sel = f[(f[:, 0] % CADA_EN_0_5 == 0) & (f[:, 0] <= paso_final)]
        else:
            sel = f[f[:, 0] <= GOLPES_EN_100]
        sale[("izquierda", p)] = sel[:, 1:3]
    for p in DERECHA:
        if p not in golf:
            continue
        f = golf[p]
        sale[("derecha", p)] = f[f[:, 0] >= PASOS_DER[0]][:, 0:2]     # (paso, peso) de todos los pasos
        sale[("final", p)] = f[-1:, 0:2]                              # el último paso
    return sale


def fondo_del_csv(golf):
    """El fondo: el peso de la fila de todo el CSV con el error más bajo."""
    todo = np.vstack(list(golf.values()))
    return float(todo[int(np.argmin(todo[:, 2])), 1])


def coma(x, n=1):
    return f"{x:.{n}f}".replace(".", ",").replace("-", "−")


def poq(p):
    return f"{p:g}".replace(".", ",") if p < 1000 else "1.000"


def dibujar(golf, terreno, paleta, ruta):
    """Dibuja la figura. Devuelve (ax_izquierda, ax_derecha, lo_dibujado)."""
    sale = que_se_dibuja(golf)
    L = Lienzo("El golf, con un peso",
               "A la izquierda, la curva del error y dónde cae el peso. A la derecha, dónde está el peso\n"
               "en cada paso, con los dos poquitos que se salen del dibujo.",
               paleta, alto=ALTO)
    p = L.p
    yl = (-0.02, 0.64)
    ai = L.fig.add_axes([0.105, 0.14, 0.395, 0.52])
    ad = L.fig.add_axes([0.665, 0.14, 0.305, 0.52])
    ai.set_ylim(*yl)
    ai.plot(terreno[:, 0], terreno[:, 1], color=p.tinta, linewidth=1.3, zorder=2)
    ai.set_xlim(*TRAMO_IZQ)
    for a in (ai, ad):
        for s_ in ("top", "right"):
            a.spines[s_].set_visible(False)
        a.tick_params(labelsize=6.6)
    ai.set_xlabel("lo que vale el peso", fontsize=7.0, color=p.tinta, labelpad=2)
    ai.set_ylabel("error", fontsize=7.0, color=p.tinta)
    ai.set_yticks([0, 0.25, 0.5]); ai.set_yticklabels(["0", "0,25", "0,5"])
    ai.set_xticks([0, 10, 20, 30]); ai.set_xticklabels(["0", "10", "20", "30"])
    ai.set_title("poquito 0,5 y poquito 100", fontsize=7.2, color=p.tinta, loc="left", pad=3)

    # --- izquierda: 0,5 (círculos llenos) y 100 (cuadrados vacíos, con sus golpes unidos)
    pts = sale[("izquierda", 0.5)]
    ai.plot(pts[:, 0], pts[:, 1], "o", markersize=3.4, markerfacecolor=p.acento, markeredgecolor=p.tinta,
            markeredgewidth=0.5, zorder=4, label="0,5: un punto cada 100 pasos")
    pts = sale[("izquierda", 100.0)]
    for a_, b_ in zip(pts[:-1], pts[1:]):
        ai.add_patch(FancyArrowPatch(tuple(a_), tuple(b_), connectionstyle="arc3,rad=-0.2", arrowstyle="-|>",
                                     mutation_scale=4.5, lw=0.6, color=p.suave, zorder=3, shrinkA=1.5, shrinkB=1.5))
    ai.plot(pts[:, 0], pts[:, 1], "s", markersize=3.6, markerfacecolor="white", markeredgecolor=p.tinta,
            markeredgewidth=0.8, zorder=5, label="100: los 10 primeros golpes")
    k = int(np.argmin(terreno[:, 1]))
    ai.annotate("el fondo", xy=(terreno[k, 0], terreno[k, 1]), xytext=(terreno[k, 0] + 3.0, 0.12), fontsize=6.6,
                color=p.tinta, arrowprops=dict(arrowstyle="-", color=p.suave, lw=0.5))
    ai.legend(loc="upper right", fontsize=5.8, frameon=False, handletextpad=0.3, borderaxespad=0.1)

    # --- derecha: el peso en cada paso (escala logarítmica); sin flechas ni arcos
    fondo = fondo_del_csv(golf)
    ad.set_xscale("log")
    ad.set_xlim(*PASOS_DER)
    ad.set_ylim(*PESO_DER)
    ad.set_xticks([1, 10, 100, 1000]); ad.set_xticklabels(["1", "10", "100", "1.000"])
    ad.minorticks_off()
    ad.set_yticks([0, 50, 100])
    ad.set_xlabel("paso (escala logarítmica)", fontsize=7.0, color=p.tinta, labelpad=2)
    ad.set_ylabel("lo que vale el peso", fontsize=7.0, color=p.tinta)
    ad.set_title("poquito 500 y poquito 1.000:\ndónde está el peso, paso a paso", fontsize=7.2, color=p.tinta,
                 loc="left", pad=3, linespacing=1.15)
    ad.axhline(fondo, color=p.suave, linewidth=0.6, linestyle=(0, (4, 2)), zorder=1)
    ad.text(1450, fondo + 3, "el fondo", fontsize=6.6, color=p.tinta, ha="right", va="bottom")
    estilos = {500.0: (p.tinta, 1.0, "^", "white"), 1000.0: (p.contra, 1.7, "D", p.acento)}
    for q in DERECHA:
        color, grosor, marca, relleno = estilos[q]
        pts = sale[("derecha", q)]
        ad.plot(pts[:, 0], pts[:, 1], color=color, linewidth=grosor, drawstyle="steps-post", zorder=3 if q == 500.0 else 2)
        fin = sale[("final", q)]
        ad.plot(fin[:, 0], fin[:, 1], marca, markersize=5.0, markerfacecolor=relleno, markeredgecolor=p.tinta,
                markeredgewidth=0.9, zorder=6, linestyle="none", clip_on=False)
    p5 = sale[("derecha", 500.0)]
    g5 = p5[int(np.argmax(p5[:, 1]))]          # donde el 500 llega más lejos
    ad.text(g5[0] * 1.5, g5[1] + 7, "se pasa, vuelve y entra", fontsize=6.4, color=p.tinta, ha="left", va="bottom")
    f1 = sale[("final", 1000.0)][0]
    ad.text(f1[0], f1[1] - 9, "sale del campo\ny no ha vuelto", fontsize=6.4, color=p.tinta, ha="right", va="top",
            linespacing=1.1)
    L.guardar(ruta)
    return ai, ad, (sale, fondo)


def cargar():
    golf = leer_golf(AQUI / DATOS_GOLF)
    t, _, _ = curvas(leer_datos(AQUI / DATOS_TERRENO))
    return golf, t


def _datos_de_linea(ax, xy):
    """¿Alguna línea del eje tiene un punto en xy?"""
    for ln in ax.lines:
        x, y = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
        if np.any((np.abs(x - xy[0]) < 1e-9) & (np.abs(y - xy[1]) < 1e-9)):
            return True
    return False


def selftest():
    fallos = []
    golf, t = cargar()

    # 1. NULO — con el CSV de poquito 0,5 no se dibuja ningún punto fuera del tramo izquierdo, y a la derecha
    #    el 0,5 no aparece.
    solo = {0.5: golf[0.5]}
    sale = que_se_dibuja(solo)
    p05 = sale[("izquierda", 0.5)]
    dentro = bool(np.all((p05[:, 0] >= TRAMO_IZQ[0]) & (p05[:, 0] <= TRAMO_IZQ[1])))
    claves = sorted(sale)
    print(f"[1] test nulo         el CSV del 0,5 da {len(p05)} puntos, todos dentro del tramo de "
          f"{coma(TRAMO_IZQ[0], 0)} a {coma(TRAMO_IZQ[1], 0)}: {'sí' if dentro else 'NO'}; paneles con algo: {claves}")
    if not dentro or any(k[0] != "izquierda" for k in claves) or len(p05) != 21:
        fallos.append("nulo: el 0,5 dibuja algo fuera del tramo izquierdo, o fuera de su panel, o no son 21 puntos")

    # 2. SEÑAL — un CSV de prueba con un salto conocido aparece en su sitio (paso 1, peso 77,0) en la línea
    #    de la derecha, y se dibuja en la escala logarítmica.
    falso = {500.0: np.array([(0, 4.0, 0.49), (1, 77.0, 0.31), (2, 40.0, 0.2), (2000, 14.1, 0.0002)]),
             1000.0: np.array([(0, 4.0, 0.49), (1, 100.0, 0.31), (2000, 120.0, 0.3)]),
             0.5: golf[0.5], 100.0: golf[100.0]}
    ai, ad, _ = dibujar(falso, t, GRIS, "/dev/null")
    esta = _datos_de_linea(ad, (1, 77.0)) and _datos_de_linea(ad, (1, 100.0)) and _datos_de_linea(ad, (2, 40.0))
    print(f"[2] señal             el salto de prueba (paso 1, peso 77,0) está en la línea de la derecha: "
          f"{'sí' if esta else 'NO'}; el eje horizontal es {ad.get_xscale()}")
    if not esta or ad.get_xscale() != "log":
        fallos.append("señal: el salto de prueba no está en su sitio, o el eje no es logarítmico")

    # 3. INVARIANTE — el punto final de cada serie es el del CSV; la serie del 1.000 pasa por -5,063 en el
    #    paso 13 y por 130,866 en el paso 200 (leídos del CSV); el fondo es 14,121; en las dos paletas.
    dibujar(golf, t, COLOR, "/dev/null")
    ai, ad, (s3, fondo) = dibujar(golf, t, GRIS, "/dev/null")
    f5, f1 = s3[("final", 500.0)][0], s3[("final", 1000.0)][0]
    m = golf[1000.0]
    p13 = float(m[m[:, 0] == 13][0, 1])
    p200 = float(m[m[:, 0] == 200][0, 1])
    en_linea = _datos_de_linea(ad, (13, p13)) and _datos_de_linea(ad, (200, p200))
    ok = (abs(f5[1] - golf[500.0][-1, 1]) < 1e-12 and abs(f1[1] - golf[1000.0][-1, 1]) < 1e-12
          and coma(f5[1], 3) == "14,121" and coma(f1[1], 3) == "129,889" and int(f5[0]) == 2000 and int(f1[0]) == 2000
          and _datos_de_linea(ad, (f5[0], f5[1])) and _datos_de_linea(ad, (f1[0], f1[1]))
          and coma(p13, 3) == "−5,063" and coma(p200, 3) == "130,866" and en_linea and coma(fondo, 3) == "14,121")
    print(f"[3] invariante        final del 500: {coma(f5[1], 3)}; final del 1.000: {coma(f1[1], 3)} (los del CSV: "
          f"14,121 y 129,889); el 1.000 en el paso 13: {coma(p13, 3)} y en el 200: {coma(p200, 3)} "
          f"(esperados −5,063 y 130,866); el fondo: {coma(fondo, 3)}: {'sí' if ok else 'NO'}")
    if not ok:
        fallos.append("invariante: algún punto no es el del CSV")
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
    golf, t = cargar()
    _, _, (sale, fondo) = dibujar(golf, t, GRIS, str(AQUI / DESTINO))
    print(f"escrita {DESTINO}")
    copia = (AQUI / COPIA).resolve()
    copia.parent.mkdir(parents=True, exist_ok=True)
    dibujar(golf, t, GRIS, str(copia))
    print(f"escrita {copia}")
    for q in DERECHA:
        print(f"pasos dibujados a la derecha con el {q:g}: {len(sale[('derecha', q)])}")


if __name__ == "__main__":
    main()
