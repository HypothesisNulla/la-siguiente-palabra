#!/usr/bin/env python3
"""
Orden 201 (22 nov 2026): el terreno de los dos valles (el de `dos_valles.py`) en relieve, para el anexo A4 y
el capítulo 16.

Una superficie en tres dimensiones, vista en perspectiva, con malla, en grises como el resto del libro, y debajo,
en el suelo, el mapa de curvas de nivel del mismo terreno visto desde arriba. Los caminos se dibujan sobre la
superficie y su sombra, sobre el mapa. Si un camino sale del cuadrado que se dibuja (de -LIMITE a LIMITE en
los dos números), se dibuja hasta el borde y se marca por dónde sale con una flecha corta y la etiqueta
«sale del campo».

Esta figura dibuja las dos bolas de `dos_valles.py`, leídas de `datos/salidas/dos_valles.csv`. El terreno es
inventado (dos números no hacen una máquina de verdad); la altura es la función de `dos_valles.py`. El módulo
también lo usa `figura_golf_en_dos_valles.py` (`dibujar_relieve`).

Uso:
    python figura_dos_valles_relieve.py --selftest
    python figura_dos_valles_relieve.py
"""

# ======================= CONSTANTES =======================

CAMINOS = "../datos/salidas/dos_valles.csv"
DESTINO = "../figuras/dos_valles_relieve.png"
COPIA = "../../Claude outputs/reforma-nov/cap16/ver/dos_valles_relieve.png"
ALTO = 4.4                     # pulgadas
REJILLA = 70                   # puntos por lado de la superficie
REJILLA_SUELO = 140            # y del mapa del suelo
NIVELES = 12                   # curvas de nivel
ELEVACION, ACIMUT = 36, 10     # cómo se mira la superficie. Orden 202: con el valle hondo del lado del lector (probados 4 puntos de vista a 10, 35 y 60 grados de acimut); se ven las dos salidas, los dos fondos y el mapa, y el final de la bola 2 queda dentro del valle
CAJA = [-0.06, -0.10, 1.12, 0.92]   # dónde va el relieve en la figura (fracciones)
SUBIDA = 0.10                  # lo que se levanta el camino sobre la superficie, para que no se entierre
BAJADA_SUELO = 1.6             # a qué distancia por debajo del punto más bajo del terreno está el mapa del suelo

# ==========================================================

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from dos_valles import LIMITE, VALLES, altura, en_que_valle
from infografia import COLOR, GRIS, Lienzo

AQUI = Path(__file__).resolve().parent


def leer(ruta=AQUI / CAMINOS):
    caminos = {}
    with open(ruta, encoding="utf-8") as fh:
        for f in csv.DictReader(fh):
            caminos.setdefault(int(f["bola"]), []).append((float(f["x"]), float(f["y"]), float(f["altura"])))
    assert sorted(caminos) == [1, 2], f"se esperaban dos bolas; hay {sorted(caminos)}"
    return {k: np.array(v) for k, v in caminos.items()}


def recortar(camino):
    """El camino hasta el borde del cuadrado: (puntos, salida) donde salida es None o (x, y, dx, dy): por dónde
    sale (el punto del borde) y hacia dónde va. Si no sale, el camino entero."""
    c = np.asarray(camino, float)
    dentro = (np.abs(c[:, 0]) <= LIMITE) & (np.abs(c[:, 1]) <= LIMITE)
    if np.all(dentro):
        return c, None
    k = int(np.argmin(dentro))                    # el primer punto fuera
    a, b = c[k - 1], c[k]
    t = 1.0
    for eje in (0, 1):
        for borde in (-LIMITE, LIMITE):
            if (a[eje] - borde) * (b[eje] - borde) < 0:      # cruza este borde
                t = min(t, (borde - a[eje]) / (b[eje] - a[eje]))
    x, y = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
    n = np.hypot(b[0] - a[0], b[1] - a[1])
    dx, dy = (b[0] - a[0]) / n, (b[1] - a[1]) / n
    return np.vstack([c[:k], [[x, y, altura(x, y)]]]), (x, y, dx, dy)


def dibujar_relieve(fig, caja, paleta, caminos, etiquetas, valles=True, estilos=None,
                    corrimiento=((1.3, -1.6, 0.9), (-1.3, 1.8, 0.9))):
    """Dibuja el relieve en `fig`, en el rectángulo `caja` = [izq, abajo, ancho, alto] (fracciones de la figura).
    `caminos`: {nombre: array [(x, y, altura), ...]}. `etiquetas`: {nombre: texto} (la etiqueta va en la salida) o
    {nombre: {"texto": t, "frac": f, "desp": (dx, dy, dz)}} (en el punto del camino que lleva la fracción f de su
    longitud, corrida en el terreno (dx, dy, dz)). `estilos`: {nombre: {"color": c, "grosor": g, "marcas": [índices]}}
    (opcional: el color y el grosor del camino, los puntos que se marcan y cómo: "marcador" y "relleno"; "sin_final": True no pone el punto negro del final). Devuelve (eje, dibujado) donde
    dibujado = {nombre: (puntos dibujados sobre la superficie, salida o None)}."""
    estilos = estilos or {}
    p = paleta
    ax = fig.add_axes(caja, projection="3d", computed_zorder=False)
    g = np.linspace(-LIMITE, LIMITE, REJILLA)
    X, Y = np.meshgrid(g, g)
    Z = np.vectorize(altura)(X, Y)
    zmin, zmax = float(Z.min()), float(Z.max())
    suelo = zmin - BAJADA_SUELO
    gs = np.linspace(-LIMITE, LIMITE, REJILLA_SUELO)
    XS, YS = np.meshgrid(gs, gs)
    ZS = np.vectorize(altura)(XS, YS)
    # el mapa del suelo
    ax.contourf(XS, YS, ZS, levels=NIVELES, zdir="z", offset=suelo, cmap="Greys_r", vmin=zmin - 1.5, vmax=zmax + 0.8,
                zorder=0)
    ax.contour(XS, YS, ZS, levels=NIVELES, zdir="z", offset=suelo, colors=["white"], linewidths=0.35, zorder=1)
    # la superficie, con malla
    ax.plot_surface(X, Y, Z, rcount=40, ccount=40, cmap="Greys_r", vmin=zmin - 1.8, vmax=zmax + 0.6,
                    edgecolor=(0.25, 0.25, 0.25, 0.55), linewidth=0.28, antialiased=True, shade=False, alpha=1.0,
                    zorder=2)
    dibujado = {}
    for nombre, c in caminos.items():
        pts, salida = recortar(c)
        dibujado[nombre] = (pts, salida)
        # su sombra en el mapa y el camino sobre la superficie (con un borde blanco para que se lea)
        ax.plot(pts[:, 0], pts[:, 1], np.full(len(pts), suelo), color="white", linewidth=2.4, zorder=3)
        ax.plot(pts[:, 0], pts[:, 1], np.full(len(pts), suelo), color=p.tinta, linewidth=1.1,
                linestyle=(0, (2, 1.5)), zorder=4)
        est = estilos.get(nombre, {})
        color, grosor = est.get("color", p.tinta), est.get("grosor", 1.5)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2] + SUBIDA, color="white", linewidth=grosor + 1.5, zorder=5)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2] + SUBIDA, color=color, linewidth=grosor, zorder=6)
        marcas = [i for i in est.get("marcas", []) if 0 < i < len(pts) - 1]
        if marcas:
            ax.plot(pts[marcas, 0], pts[marcas, 1], pts[marcas, 2] + SUBIDA, est.get("marcador", "o"), markersize=3.4,
                    markerfacecolor=est.get("relleno", "white"), markeredgecolor=p.tinta, markeredgewidth=0.8,
                    zorder=7, linestyle="none")
        ax.plot([pts[0, 0]], [pts[0, 1]], [pts[0, 2] + SUBIDA], "o", markersize=5.5, markerfacecolor="white",
                markeredgecolor=p.tinta, zorder=7)
        if not est.get("sin_final"):
            ax.plot([pts[-1, 0]], [pts[-1, 1]], [pts[-1, 2] + SUBIDA], "o", markersize=5.5, markerfacecolor=p.tinta,
                    markeredgecolor="white", zorder=7)
        et = etiquetas.get(nombre, "")
        if isinstance(et, dict):
            largo = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])))])
            k = int(np.argmin(np.abs(largo - et["frac"] * largo[-1])))
            dx, dy, dz = et["desp"]
            ax.plot([pts[k, 0], pts[k, 0] + dx], [pts[k, 1], pts[k, 1] + dy],
                    [pts[k, 2] + SUBIDA, pts[k, 2] + SUBIDA + dz], color=p.suave, linewidth=0.6, zorder=8)
            ax.text(pts[k, 0] + dx, pts[k, 1] + dy, pts[k, 2] + SUBIDA + dz, et["texto"], fontsize=7.2,
                    color=p.tinta, fontweight="bold", ha="center", zorder=9)
        elif et:
            ax.text(pts[0, 0], pts[0, 1], pts[0, 2] + SUBIDA + 0.45, et, fontsize=7.4,
                    color=p.tinta, fontweight="bold", ha="center", zorder=9)
        if salida is not None:
            x, y, dx, dy = salida
            z = altura(x, y) + SUBIDA
            ax.quiver(x - 0.5 * dx, y - 0.5 * dy, z, 1.0 * dx, 1.0 * dy, 0.0, color=p.tinta, linewidth=1.3,
                      arrow_length_ratio=0.5, zorder=8)
            ax.text(x + 0.6 * dx, y + 0.6 * dy, z + 0.3, "sale del campo", fontsize=7.0, color=p.tinta, ha="center",
                    zorder=9)
    if valles:
        nombres = ("valle hondo", "valle poco hondo")
        # `corrimiento`: (x, y, z) a donde se corre la etiqueta de cada valle, para no tapar el camino
        for i, (cx, cy, _, _) in enumerate(VALLES):
            dx, dy, dz = corrimiento[i]
            ax.plot([cx, cx + dx], [cy, cy + dy], [altura(cx, cy), altura(cx, cy) + dz], color=p.suave, linewidth=0.7, zorder=8)
            ax.text(cx + dx, cy + dy, altura(cx, cy) + dz, nombres[i], fontsize=7.4, color=p.tinta, fontweight="bold",
                    ha="center", zorder=9)
    ax.set_xlim(-LIMITE, LIMITE); ax.set_ylim(-LIMITE, LIMITE); ax.set_zlim(suelo, zmax + 0.3)
    ax.view_init(elev=ELEVACION, azim=ACIMUT)
    ax.set_box_aspect((1, 1, 0.8))
    ax.set_axis_off()
    return ax, dibujado


def dibujar(caminos, paleta, ruta):
    L = Lienzo("Dos bolas, dos valles",
               "Un terreno inventado con dos pesos. La altura es lo mal que lo hace la red; las curvas\n"
               "de abajo son el mismo terreno visto desde arriba, como en un mapa.",
               paleta, alto=ALTO)
    ax, dib = dibujar_relieve(L.fig, CAJA, paleta, {k: v for k, v in caminos.items()},
                              {1: "bola 1", 2: "bola 2"})
    L.guardar(ruta)
    return ax, dib


def selftest():
    fallos = []
    caminos = leer()

    # 1. NULO — un camino que no se mueve dentro del cuadrado no tiene salida ni se recorta; y un camino
    #    que se queda dentro no pone ninguna etiqueta «sale del campo».
    quieto = np.array([[1.0, 1.0, altura(1.0, 1.0)]] * 5)
    pts, sal = recortar(quieto)
    ax, dib = dibujar_relieve(Lienzo("t", "", GRIS, alto=ALTO).fig, CAJA, GRIS, {1: quieto}, {1: "x"})
    textos = [t.get_text() for t in ax.texts]
    ok1 = sal is None and len(pts) == 5 and "sale del campo" not in textos
    print(f"[1] test nulo         un camino quieto dentro del cuadrado: {len(pts)} puntos, sin salida: {'sí' if ok1 else 'NO'}")
    if not ok1:
        fallos.append("nulo: un camino quieto se recorta o marca salida")

    # 2. SEÑAL — un camino de prueba que sale del cuadrado por x = 4 se dibuja hasta el borde (x = 4), y se marca
    #    la salida con su etiqueta.
    sale = np.array([[2.0, 0.0, altura(2.0, 0.0)], [3.0, 0.5, altura(3.0, 0.5)], [5.0, 1.5, altura(5.0, 1.5)],
                     [8.0, 3.0, altura(8.0, 3.0)]])
    pts, sal = recortar(sale)
    ax, dib = dibujar_relieve(Lienzo("t", "", GRIS, alto=ALTO).fig, CAJA, GRIS, {1: sale}, {1: "x"})
    textos = [t.get_text() for t in ax.texts]
    ok2 = (sal is not None and abs(pts[-1, 0] - LIMITE) < 1e-9 and len(pts) == 3 and abs(sal[0] - LIMITE) < 1e-9
           and "sale del campo" in textos)
    print(f"[2] señal             un camino que sale por x = 4: se dibuja hasta ({pts[-1, 0]:.2f}; {pts[-1, 1]:.2f}) y "
          f"lleva «sale del campo»: {'sí' if ok2 else 'NO'}")
    if not ok2:
        fallos.append("señal: el camino que sale no se recorta en el borde o no se marca")

    # 3. INVARIANTE — la altura de la salida es la del terreno en cada punto, las dos bolas acaban en valles
    #    distintos, y la figura sale en las dos paletas.
    casa = all(abs(h - altura(x, y)) < 1e-4 for c in caminos.values() for x, y, h in c)
    valles = sorted(en_que_valle(c[-1, 0], c[-1, 1]) for c in caminos.values())
    dibujar(caminos, COLOR, "/dev/null")
    dibujar(caminos, GRIS, "/dev/null")
    ok3 = casa and valles == [0, 1]
    print(f"[3] invariante        las alturas de la salida son las del terreno: {'sí' if casa else 'NO'}; valles donde "
          f"acaban las bolas: {valles}")
    if not ok3:
        fallos.append("invariante: la salida y el terreno no casan, o las bolas acaban en el mismo valle")
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
    caminos = leer()
    dibujar(caminos, GRIS, str(AQUI / DESTINO))
    print(f"escrita {DESTINO}")
    copia = (AQUI / COPIA).resolve()
    copia.parent.mkdir(parents=True, exist_ok=True)
    dibujar(caminos, GRIS, str(copia))
    print(f"escrita {copia}")


if __name__ == "__main__":
    main()
