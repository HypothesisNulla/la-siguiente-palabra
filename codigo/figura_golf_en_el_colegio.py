#!/usr/bin/env python3
"""
Orden 199 (22 nov 2026): la figura del golf, para el anexo A4.

El mismo peso de la red del colegio (el de la línea de la primera a la final), saliendo de 4, con cuatro
poquitos: 0,5, 100, 500 y 1.000. Dos paneles. A la izquierda, el tramo del terreno (de -5 a 30): el 0,5
con un punto cada 100 pasos y el 100 con sus diez primeros golpes. A la derecha, un tramo ancho que
cabe todo: el 500 y el 1.000, con los golpes de los 20 primeros pasos unidos en orden y el punto de los
2.000 pasos.

Todos los números se leen de `datos/salidas/golf_en_el_colegio_datos.csv` (los cuatro recorridos) y de
`datos/salidas/culpa_en_el_colegio_datos.csv` (la curva del terreno, la misma de
`figura_el_terreno_del_colegio.py`). Ningún número se escribe a mano. La curva del terreno solo existe en
los datos de -5 a 30: a la derecha de 30 no se dibuja ninguna línea inventada, solo los puntos.

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
TRAMO_DER = (-10.0, 135.0)      # el tramo ancho de la derecha
CADA_EN_0_5 = 100               # el 0,5: un punto cada tantos pasos (como en la figura del terreno)
GOLPES_EN_100 = 10              # el 100: los diez primeros golpes
GOLPES_DERECHA = 20             # el 500 y el 1.000: los veinte primeros golpes
SALTO_CON_ARCO = 1.5            # a la derecha, un golpe se une con arco si el peso se mueve más que esto

# ==========================================================

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import FancyArrowPatch

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
    """Los puntos de cada panel: {('izquierda'|'derecha', poquito): array [(peso, error), ...]}.
    A la izquierda solo los poquitos de IZQUIERDA y a la derecha solo los de DERECHA."""
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
        sale[("derecha", p)] = f[f[:, 0] <= GOLPES_DERECHA][:, 1:3]
        sale[("final", p)] = f[-1:, 1:3]
    return sale


def coma(x, n=1):
    return f"{x:.{n}f}".replace(".", ",").replace("-", "−")


def poq(p):
    return f"{p:g}".replace(".", ",") if p < 1000 else "1.000"


def dibujar(golf, terreno, paleta, ruta):
    """Dibuja la figura. Devuelve (ax_izquierda, ax_derecha, lo_dibujado)."""
    sale = que_se_dibuja(golf)
    L = Lienzo("El golf, con un peso",
               "El mismo peso, saliendo de 4, con cuatro poquitos distintos.\n"
               "La línea es el error; los puntos, dónde cae el peso en cada golpe.",
               paleta, alto=ALTO)
    p = L.p
    yl = (-0.02, 0.64)
    ai = L.fig.add_axes([0.105, 0.14, 0.395, 0.52])
    ad = L.fig.add_axes([0.600, 0.14, 0.370, 0.52], sharey=ai)
    ai.set_ylim(*yl)
    for a, tramo in ((ai, TRAMO_IZQ), (ad, TRAMO_DER)):
        # la curva del terreno, solo donde hay datos
        a.plot(terreno[:, 0], terreno[:, 1], color=p.tinta, linewidth=1.3, zorder=2)
        a.set_xlim(*tramo)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.tick_params(labelsize=6.6)
        a.set_xlabel("lo que vale el peso", fontsize=7.0, color=p.tinta, labelpad=2)
    ai.set_ylabel("error", fontsize=7.0, color=p.tinta)
    ai.set_yticks([0, 0.25, 0.5]); ai.set_yticklabels(["0", "0,25", "0,5"])
    plt_ocultar = ad.get_yticklabels()
    for t in plt_ocultar:
        t.set_visible(False)
    ai.set_xticks([0, 10, 20, 30]); ai.set_xticklabels(["0", "10", "20", "30"])
    ad.set_xticks([0, 50, 100]); ad.set_xticklabels(["0", "50", "100"])
    ai.set_title("poquito 0,5 y poquito 100", fontsize=7.2, color=p.tinta, loc="left", pad=11)
    ad.set_title("poquito 500 y poquito 1.000", fontsize=7.2, color=p.tinta, loc="left", pad=11)

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

    # --- derecha: 500 (triángulos vacíos) y 1.000 (rombos llenos); golpes unidos con arco si el salto se ve
    estilos = {500.0: ("^", "white", "500"), 1000.0: ("D", p.acento, "1.000")}
    cuantos = {}
    for q in DERECHA:
        marca, relleno, rotulo = estilos[q]
        pts = sale[("derecha", q)]
        arcos = 0
        for a_, b_ in zip(pts[:-1], pts[1:]):
            if abs(b_[0] - a_[0]) >= SALTO_CON_ARCO:
                ad.add_patch(FancyArrowPatch(tuple(a_), tuple(b_), connectionstyle="arc3,rad=-0.12",
                                             arrowstyle="-|>", mutation_scale=4.5, lw=0.6, color=p.suave,
                                             zorder=3, shrinkA=1.5, shrinkB=1.5))
                arcos += 1
        cuantos[q] = (len(pts) - 1, arcos)
        ad.plot(pts[:, 0], pts[:, 1], marca, markersize=3.4, markerfacecolor=relleno, markeredgecolor=p.tinta,
                markeredgewidth=0.6, zorder=5, linestyle="none", label=f"{rotulo}: los {GOLPES_DERECHA} primeros golpes")
        fin = sale[("final", q)]
        ad.plot(fin[:, 0], fin[:, 1], marca, markersize=6.0, markerfacecolor=relleno, markeredgecolor=p.tinta,
                markeredgewidth=1.1, zorder=6, linestyle="none")
        ad.annotate("", xy=(fin[0, 0], fin[0, 1]), xytext=(pts[-1, 0], pts[-1, 1]),
                    arrowprops=dict(arrowstyle="-|>", color=p.tinta, lw=0.8, linestyle=(0, (3, 2)),
                                    connectionstyle="arc3,rad=" + ("0.25" if q == DERECHA[0] else "-0.22"),
                                    shrinkA=3, shrinkB=4, mutation_scale=5))
    f5, f1 = sale[("final", 500.0)][0], sale[("final", 1000.0)][0]
    ad.annotate("el fondo", xy=(f5[0], f5[1]), xytext=(f5[0] + 34, 0.015), fontsize=6.6, color=p.tinta,
                arrowprops=dict(arrowstyle="-", color=p.suave, lw=0.5))
    p5 = sale[("derecha", 500.0)]
    g5 = p5[int(np.argmax(p5[:, 0]))]          # donde el 500 llega más lejos
    ad.text(g5[0] + 2, g5[1] + 0.045, "se pasa y vuelve", fontsize=6.6, color=p.tinta, ha="center", va="bottom")
    p1 = sale[("derecha", 1000.0)]
    g1 = p1[int(np.argmin(p1[:, 0]))]          # donde el 1.000 llega más a la izquierda
    ad.annotate("sale del campo", xy=(g1[0], g1[1]), xytext=(g1[0] + 14.0, g1[1] + 0.06), fontsize=6.6,
                color=p.tinta, ha="left", arrowprops=dict(arrowstyle="-", color=p.suave, lw=0.5))
    ad.text(f1[0] + 4, f1[1] - 0.04, "a los 2.000 pasos,\nsin haber vuelto", fontsize=6.2, color=p.tinta,
            ha="right", va="top", linespacing=1.1)
    h, e = ad.get_legend_handles_labels()
    ad.legend(h, e, loc="lower right", fontsize=5.8, frameon=False, handletextpad=0.3, borderaxespad=0.1,
              bbox_to_anchor=(1.0, 0.105))
    L.guardar(ruta)
    return ai, ad, (sale, cuantos)


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

    # 2. SEÑAL — un CSV de prueba con un salto conocido aparece en su sitio (se dibuja y está en la línea)
    falso = {500.0: np.array([(0, 4.0, 0.49), (1, 77.0, 0.31), (2, 40.0, 0.2), (2000, 14.1, 0.0002)]),
             1000.0: np.array([(0, 4.0, 0.49), (1, 100.0, 0.31), (2000, 120.0, 0.3)]),
             0.5: golf[0.5], 100.0: golf[100.0]}
    ai, ad, (s2, cu) = dibujar(falso, t, GRIS, "/dev/null")
    esta = _datos_de_linea(ad, (77.0, 0.31)) and _datos_de_linea(ad, (100.0, 0.31))
    print(f"[2] señal             el salto de prueba a 77,0 (error 0,31) está en la línea de la derecha: "
          f"{'sí' if esta else 'NO'}; arcos del 500: {cu[500.0][1]} (esperados 2: de 4 a 77 y de 77 a 40)")
    if not esta or cu[500.0][1] != 2:
        fallos.append("señal: el salto de prueba no está en su sitio o no se une con arco")

    # 3. INVARIANTE — el punto final de cada poquito es el del CSV; en las dos paletas
    dibujar(golf, t, COLOR, "/dev/null")
    ai, ad, (s3, cu) = dibujar(golf, t, GRIS, "/dev/null")
    f5, f1 = s3[("final", 500.0)][0], s3[("final", 1000.0)][0]
    ok = (abs(f5[0] - golf[500.0][-1, 1]) < 1e-12 and abs(f1[0] - golf[1000.0][-1, 1]) < 1e-12
          and coma(f5[0], 3) == "14,121" and coma(f1[0], 3) == "129,889"
          and _datos_de_linea(ad, (f5[0], f5[1])) and _datos_de_linea(ad, (f1[0], f1[1])))
    print(f"[3] invariante        el punto final del 500 es {coma(f5[0], 3)} y el del 1.000 es {coma(f1[0], 3)} "
          f"(los del CSV: 14,121 y 129,889): {'sí' if ok else 'NO'}; golpes dibujados (golpes, con arco): "
          f"500 {cu[500.0]}, 1.000 {cu[1000.0]}")
    if not ok:
        fallos.append("invariante: el punto final no es el del CSV")
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
    _, _, (_, cuantos) = dibujar(golf, t, GRIS, str(AQUI / DESTINO))
    print(f"escrita {DESTINO}")
    copia = (AQUI / COPIA).resolve()
    copia.parent.mkdir(parents=True, exist_ok=True)
    dibujar(golf, t, GRIS, str(copia))
    print(f"escrita {copia}")
    for q in DERECHA:
        print(f"golpes dibujados a la derecha con el {q:g}: {cuantos[q][0]} (con arco: {cuantos[q][1]})")


if __name__ == "__main__":
    main()
