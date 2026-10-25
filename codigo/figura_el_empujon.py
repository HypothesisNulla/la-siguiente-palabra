#!/usr/bin/env python3
"""
Capítulo 14 — el empujón, dibujado (L24, E16).

Una red de tres neuronas en fila (entrada, en medio, salida) con conexiones de ida y de vuelta,
en tres momentos: antes del empujón, ya asentada; después de empujar la salida hacia arriba (la
respuesta correcta estaba más arriba); y después de empujarla hacia abajo. Dentro de cada neurona,
su actividad; debajo, el producto de las dos que une la unión que se mira, y qué le pasa.

Los números NO se escriben aquí: se leen de `datos/salidas/cuentas_a_mano.csv`, que escribe
`cuentas_a_mano.py`. En gris: el libro se imprime en negro.

Uso:
    python figura_el_empujon.py --selftest
    python figura_el_empujon.py
"""

# ======================= CONSTANTES =======================

DATOS = "../datos/salidas/cuentas_a_mano.csv"
DESTINO = "../figuras/el_empujon.png"
ALTO = 5.25                 # pulgadas (sin el hueco del pie, Orden 4; era 5,5)
ALTO_PANEL = 27             # unidades del lienzo
X = (20, 50, 80)            # dónde van las tres neuronas
R = 5.0                     # radio de una neurona

# ==========================================================

import argparse
import csv
import sys
from pathlib import Path

from matplotlib.patches import Circle, FancyArrowPatch

from infografia import COLOR, GRIS, Lienzo

AQUI = Path(__file__).resolve().parent


def coma(x, d):
    return f"{x:.{d}f}".replace(".", ",")


def leer(ruta=AQUI / DATOS):
    casos = []
    with open(ruta, encoding="utf-8") as fh:
        for f in csv.DictReader(fh):
            if f["caso"].startswith("la respuesta correcta"):
                casos.append((f["caso"], (float(f["a"]), float(f["b"])),
                              (float(f["c"]), float(f["d"])), float(f["e"])))
    assert len(casos) == 2, f"se esperaban dos casos de empujón; hay {len(casos)}"
    assert casos[0][1] == casos[1][1], "los dos casos tienen que partir del mismo antes"
    return casos


def red(L, y, valores, empujon=None):
    """Tres neuronas en fila con sus flechas de ida (arriba) y de vuelta (abajo)."""
    p, ax = L.p, L.ax
    nombres = ("entrada", "centro", "salida")
    for i, (x, n) in enumerate(zip(X, nombres)):
        ax.add_patch(Circle((x, y), R, facecolor="white", edgecolor=p.tinta, linewidth=1.1, zorder=3))
        v = valores[i]
        L.texto(x, y, "" if v is None else coma(v, 1), tam=10, ha="center", negrita=True)
        L.texto(x, y - R - 2.6, n, tam=7.4, ha="center", color=p.suave)
    for a, b in zip(X, X[1:]):
        grueso = 1.8 if a == X[1] else 0.9
        ax.add_patch(FancyArrowPatch((a + R + 0.8, y + 1.6), (b - R - 0.8, y + 1.6),
                                     arrowstyle="-|>", mutation_scale=8, color=p.tinta,
                                     linewidth=grueso))
        ax.add_patch(FancyArrowPatch((b - R - 0.8, y - 1.6), (a + R + 0.8, y - 1.6),
                                     arrowstyle="-|>", mutation_scale=8, color=p.contra,
                                     linewidth=grueso))
    if empujon:
        sube = empujon == "sube"
        ax.add_patch(FancyArrowPatch((X[2] + R + 5, y + (-5 if sube else 5)),
                                     (X[2] + R + 5, y + (5 if sube else -5)),
                                     arrowstyle="-|>", mutation_scale=12, color=p.acento, linewidth=2.2))
        L.texto(X[2] + R + 5, y + (7.5 if sube else -7.5), "empujón", tam=7.4, ha="center",
                negrita=True)


def dibujar(casos, paleta, ruta):
    L = Lienzo("El empujón, en los dos sentidos",
               "Tres neuronas en fila, con conexiones de ida y de vuelta. Se mira la unión\n"
               "entre la del centro y la de salida (la flecha gruesa).",
               paleta, alto=ALTO)
    p = L.p
    # clave
    yc = L.y - 0.5
    L.ax.add_patch(FancyArrowPatch((4, yc), (12, yc), arrowstyle="-|>", mutation_scale=8, color=p.tinta))
    L.texto(13.5, yc, "ida", tam=7.2, color=p.suave)
    L.ax.add_patch(FancyArrowPatch((30, yc), (22, yc), arrowstyle="-|>", mutation_scale=8, color=p.contra))
    L.texto(31.5, yc, "vuelta", tam=7.2, color=p.suave)
    L.texto(45, yc, "número dentro: la actividad de la neurona", tam=7.2, color=p.suave)
    L.y -= 4.5
    (_, antes, _, _) = casos[0]
    filas = [("Antes del empujón: la red se asienta", (None, antes[0], antes[1]), None,
              f"producto de la unión: {coma(antes[0], 1)} por {coma(antes[1], 1)}, {coma(antes[0] * antes[1], 2)}")]
    for caso, a, d, cambio in casos:
        sube = cambio > 0
        titulo = "Si la respuesta correcta estaba más " + ("arriba" if "arriba" in caso else "abajo")
        filas.append((titulo, (None, d[0], d[1]), "sube" if d[1] > a[1] else "baja",
                      f"producto: {coma(d[0] * d[1], 2)}; {'sube' if sube else 'baja'} "
                      f"{coma(abs(cambio), 2)}: la unión se {'refuerza' if sube else 'debilita'}"))
    for n, (titulo, valores, emp, pie) in enumerate(filas, 1):
        x0, ytop, _ = L.panel(n, titulo, ALTO_PANEL)
        yr = ytop - 6.0
        red(L, yr, valores, emp)
        L.texto(50, ytop - 16.8, pie, tam=7.8, ha="center")
    L.guardar(ruta)
    return [f[1] for f in filas]


def selftest():
    fallos = []
    casos = leer()
    # 1. TEST NULO — sin empujón (después igual que antes), la figura no dice que la unión cambie.
    (c, a, _, _) = casos[0]
    v = dibujar([(c, a, a, 0.0), (c.replace("arriba", "abajo"), a, a, 0.0)], GRIS, "/dev/null")
    print(f"[1] test nulo         sin empujón, las tres filas dibujan la misma red: "
          f"{'sí' if v[0] == v[1] == v[2] else 'NO'}")
    if not v[0] == v[1] == v[2]:
        fallos.append("test nulo: la figura cambia la red sin empujón")
    # 2. SEÑAL IMPLANTADA — lo del capítulo: 0,5 y 0,4 antes; 0,6 y 0,6 si sube; el producto
    #    sube si la correcta estaba arriba y baja si estaba abajo.
    ok = (casos[0][1] == (0.5, 0.4) and casos[0][2] == (0.6, 0.6)
          and casos[0][3] > 0 and casos[1][3] < 0)
    print(f"[2] señal implantada  antes {casos[0][1]}, después {casos[0][2]} y {casos[1][2]}; "
          f"refuerza y debilita: {'sí' if ok else 'NO'}")
    if not ok:
        fallos.append("señal: los casos no son los del capítulo")
    # 3. INVARIANTE DEL DOMINIO — el cambio que trae la salida es el producto de después menos el
    #    de antes, y la figura se dibuja en las dos paletas.
    casa = all(abs((d[0] * d[1] - a[0] * a[1]) - cm) < 1e-4 for _, a, d, cm in casos)
    for pal in (COLOR, GRIS):
        dibujar(casos, pal, "/dev/null")
    print(f"[3] invariante        el cambio es producto después menos antes: {'sí' if casa else 'NO'}")
    if not casa:
        fallos.append("invariante: el cambio de la salida no es la diferencia de productos")
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
    dibujar(leer(), GRIS, str(AQUI / DESTINO))
    print(f"escrita {DESTINO}")


if __name__ == "__main__":
    main()
