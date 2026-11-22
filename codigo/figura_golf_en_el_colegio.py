#!/usr/bin/env python3
"""
Orden 199 (22 nov 2026), Orden 200 y Orden 201 (22 nov 2026, la versión de un solo panel): la figura del golf
con un peso, para el anexo A4.

El mismo peso de la red del colegio (el de la línea de la primera a la final), saliendo de 4, con dos
poquitos: 0,5 y 100. Un solo panel: el tramo del terreno (de -5 a 30), con la curva del error, el 0,5 con un
punto cada 100 pasos y el 100 con sus diez primeros golpes unidos con flechas. El 500 y el 1.000 siguen en
la salida de `golf_en_el_colegio.py` (Orden 198) y en su CSV; solo salen del dibujo (Orden 201).

Todos los números se leen de `datos/salidas/golf_en_el_colegio_datos.csv` (los recorridos) y de
`datos/salidas/culpa_en_el_colegio_datos.csv` (la curva del terreno, la misma de
`figura_el_terreno_del_colegio.py`). Ningún número se escribe a mano.

Uso:
    python figura_golf_en_el_colegio.py --selftest
    python figura_golf_en_el_colegio.py
"""

# ======================= CONSTANTES =======================

DATOS_GOLF = "../datos/salidas/golf_en_el_colegio_datos.csv"       # los recorridos, de golf_en_el_colegio.py
DATOS_TERRENO = "../datos/salidas/culpa_en_el_colegio_datos.csv"   # la curva del terreno
DESTINO = "../figuras/golf_en_el_colegio.png"
COPIA = "../../Claude outputs/reforma-nov/cap16/ver/golf_en_el_colegio.png"
ALTO = 3.2                      # pulgadas
POQUITOS = (0.5, 100.0)         # los que se dibujan
TRAMO = (-5.0, 30.0)            # el tramo de la figura del terreno
CADA_EN_0_5 = 100               # el 0,5: un punto cada tantos pasos (como en la figura del terreno)
GOLPES_EN_100 = 10              # el 100: los diez primeros golpes

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
    """Los puntos que se dibujan: {poquito: array [(peso, error), ...]}. Solo los poquitos de POQUITOS."""
    sale = {}
    for p in POQUITOS:
        if p not in golf:
            continue
        f = golf[p]
        if p == POQUITOS[0]:
            sel = f[f[:, 0] % CADA_EN_0_5 == 0]
        else:
            sel = f[f[:, 0] <= GOLPES_EN_100]
        sale[p] = sel[:, 1:3]
    return sale


def coma(x, n=1):
    return f"{x:.{n}f}".replace(".", ",").replace("-", "−")


def dibujar(golf, terreno, paleta, ruta):
    """Dibuja la figura. Devuelve (eje, lo_dibujado)."""
    sale = que_se_dibuja(golf)
    L = Lienzo("El golf, con un peso",
               "El mismo peso, saliendo de 4, con dos poquitos.\n"
               "La línea es el error; los puntos, dónde cae el peso.",
               paleta, alto=ALTO)
    p = L.p
    a = L.fig.add_axes([0.105, 0.17, 0.865, 0.50])
    a.set_ylim(-0.02, 0.64)
    a.plot(terreno[:, 0], terreno[:, 1], color=p.tinta, linewidth=1.3, zorder=2)
    a.set_xlim(*TRAMO)
    for s_ in ("top", "right"):
        a.spines[s_].set_visible(False)
    a.tick_params(labelsize=6.6)
    a.set_xlabel("lo que vale el peso", fontsize=7.0, color=p.tinta, labelpad=2)
    a.set_ylabel("error", fontsize=7.0, color=p.tinta)
    a.set_yticks([0, 0.25, 0.5]); a.set_yticklabels(["0", "0,25", "0,5"])
    a.set_xticks([0, 10, 20, 30]); a.set_xticklabels(["0", "10", "20", "30"])

    # 0,5 (círculos llenos) y 100 (cuadrados vacíos, con sus golpes unidos)
    pts = sale[0.5]
    a.plot(pts[:, 0], pts[:, 1], "o", markersize=3.6, markerfacecolor=p.acento, markeredgecolor=p.tinta,
           markeredgewidth=0.5, zorder=4, label="poquito 0,5: un punto cada 100 pasos")
    pts = sale[100.0]
    for a_, b_ in zip(pts[:-1], pts[1:]):
        a.add_patch(FancyArrowPatch(tuple(a_), tuple(b_), connectionstyle="arc3,rad=-0.2", arrowstyle="-|>",
                                    mutation_scale=4.5, lw=0.6, color=p.suave, zorder=3, shrinkA=1.5, shrinkB=1.5))
    a.plot(pts[:, 0], pts[:, 1], "s", markersize=3.8, markerfacecolor="white", markeredgecolor=p.tinta,
           markeredgewidth=0.8, zorder=5, label="poquito 100: los 10 primeros golpes")
    k = int(np.argmin(terreno[:, 1]))
    a.annotate("el fondo", xy=(terreno[k, 0], terreno[k, 1]), xytext=(terreno[k, 0] + 3.0, 0.12), fontsize=6.6,
               color=p.tinta, arrowprops=dict(arrowstyle="-", color=p.suave, lw=0.5))
    a.legend(loc="upper right", fontsize=6.2, frameon=False, handletextpad=0.3, borderaxespad=0.1)
    L.guardar(ruta)
    return a, sale


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

    # 1. NULO — con el CSV de poquito 0,5 no se dibuja ningún punto fuera del tramo, y el 100 no aparece.
    sale = que_se_dibuja({0.5: golf[0.5]})
    p05 = sale[0.5]
    dentro = bool(np.all((p05[:, 0] >= TRAMO[0]) & (p05[:, 0] <= TRAMO[1])))
    print(f"[1] test nulo         el CSV del 0,5 da {len(p05)} puntos, todos dentro del tramo de "
          f"{coma(TRAMO[0], 0)} a {coma(TRAMO[1], 0)}: {'sí' if dentro else 'NO'}; poquitos dibujados: {sorted(sale)}")
    if not dentro or sorted(sale) != [0.5] or len(p05) != 21:
        fallos.append("nulo: el 0,5 dibuja algo fuera del tramo, o aparece otro poquito, o no son 21 puntos")

    # 2. SEÑAL — un CSV de prueba con un golpe conocido aparece en su sitio (paso 1, peso 20,0, error 0,1)
    falso = {0.5: golf[0.5], 100.0: np.array([(0, 4.0, 0.49), (1, 20.0, 0.1), (2, 14.0, 0.0002)])}
    a, _ = dibujar(falso, t, GRIS, "/dev/null")
    esta = _datos_de_linea(a, (20.0, 0.1))
    print(f"[2] señal             el golpe de prueba (peso 20,0, error 0,1) está en la línea: {'sí' if esta else 'NO'}")
    if not esta:
        fallos.append("señal: el golpe de prueba no está en su sitio")

    # 3. INVARIANTE — el último punto del 0,5 (paso 2.000) y el décimo golpe del 100 son los del CSV; el 500 y el
    #    1.000, que están en el CSV, no se dibujan; en las dos paletas.
    dibujar(golf, t, COLOR, "/dev/null")
    a, s3 = dibujar(golf, t, GRIS, "/dev/null")
    f05, g10 = s3[0.5][-1], s3[100.0][-1]
    csv05 = golf[0.5][-1]
    csv100 = golf[100.0][golf[100.0][:, 0] == GOLPES_EN_100][0]
    ok = (abs(f05[0] - csv05[1]) < 1e-12 and abs(f05[1] - csv05[2]) < 1e-12 and coma(f05[0], 2) == "13,12"
          and abs(g10[0] - csv100[1]) < 1e-12 and sorted(s3) == [0.5, 100.0] and 500.0 in golf and 1000.0 in golf
          and _datos_de_linea(a, (f05[0], f05[1])) and _datos_de_linea(a, (g10[0], g10[1])))
    print(f"[3] invariante        el último punto del 0,5 es {coma(f05[0], 2)} y el décimo golpe del 100, {coma(g10[0], 2)} "
          f"(los del CSV); poquitos dibujados: {sorted(s3)}; el 500 y el 1.000 siguen en el CSV: "
          f"{'sí' if 500.0 in golf and 1000.0 in golf else 'NO'}: {'sí' if ok else 'NO'}")
    if not ok:
        fallos.append("invariante: algún punto no es el del CSV, o se dibuja otro poquito")
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
    dibujar(golf, t, GRIS, str(AQUI / DESTINO))
    print(f"escrita {DESTINO}")
    copia = (AQUI / COPIA).resolve()
    copia.parent.mkdir(parents=True, exist_ok=True)
    dibujar(golf, t, GRIS, str(copia))
    print(f"escrita {copia}")


if __name__ == "__main__":
    main()
