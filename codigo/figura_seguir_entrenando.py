#!/usr/bin/env python3
"""
Capítulo 10 — la máquina del capítulo, dejada el triple de pasos (L24, D08).

La pregunta final del capítulo —¿qué pasaría si la dejáramos entrenando mucho más?— se contestaba
con palabras. `seguir_entrenando.py` lo mide: el mismo modelo, la misma semilla, el triple de
pasos, y cada mil pasos lo mal que lo hace en dos textos, el suyo y uno que no lee nunca. Esta
figura dibuja las dos líneas. En su propio texto sigue bajando; en el nuevo, se aplana.

Los datos se leen de `datos/salidas/seguir_entrenando.csv`; aquí no hay ningún número.

Uso:
    python figura_seguir_entrenando.py --selftest
    python figura_seguir_entrenando.py
"""

# ======================= CONSTANTES =======================

CSV = "../datos/salidas/seguir_entrenando.csv"
DESTINO = "../figuras/seguir_entrenando.png"
ALTO = 3.2
DESDE = 1_000               # la medida del paso 0 (el azar) aplastaría el dibujo; se empieza aquí
FIN_CAPITULO = 50_000       # donde acababa el entrenamiento del capítulo

# ==========================================================

import argparse
import csv
import sys

import matplotlib.pyplot as plt

from infografia import COLOR, GRIS, Lienzo
from formato import miles


def leer(ruta=CSV):
    with open(ruta, newline="", encoding="utf-8") as fh:
        f = list(csv.reader(fh))
    assert f[0] == ["pasos", "visto", "nuevo", "acierto_visto", "acierto_nuevo"], "cabecera inesperada"
    d = [(int(a), float(b), float(c)) for a, b, c, _, _ in f[1:]]
    assert len(d) > 20 and d[-1][0] > FIN_CAPITULO, "la medición no llega más allá del capítulo"
    return d


def mejor_nuevo(d):
    return min(d, key=lambda x: x[2])


def dibujar(d, paleta, ruta):
    L = Lienzo("La misma red, el triple de pasos",   # Orden 167: «red», como en el texto
               "Lo mal que lo hace en dos textos: los libros con los que aprende, y otros libros\n"
               "que no lee nunca. Cuanto más abajo, mejor; el cero es el suelo del dibujo.",
               paleta, alto=ALTO)
    p = paleta
    arriba = L.y / L.alto_u
    ax = L.fig.add_axes([0.1, 0.16, 0.86, arriba - 0.21])
    dd = [x for x in d if x[0] >= DESDE]
    t = [x[0] for x in dd]
    ax.plot(t, [x[1] for x in dd], color="0.6", linewidth=1.6, label="en los libros con los que aprende")
    ax.plot(t, [x[2] for x in dd], color=p.tinta, linewidth=1.8, label="en libros que no lee nunca")
    tope = max(max(x[1], x[2]) for x in dd) * 1.15
    ax.set_ylim(0, tope)
    ax.set_xlim(0, d[-1][0])
    # Regla 4 del capítulo 10 (dibujos sin cifras nuevas, 28 sep): en el eje, solo el 50.000 que el
    # capítulo ya da (su último retrato) y «el triple», que es lo que dice el texto.
    ax.set_xticks([FIN_CAPITULO, d[-1][0]])
    ax.set_xticklabels([miles(FIN_CAPITULO), "el triple"])
    ax.axvline(FIN_CAPITULO, color=p.suave, linewidth=0.7, linestyle=(0, (3, 2)))
    ax.text(FIN_CAPITULO, tope * 0.97, " aquí acababa el capítulo", fontsize=6.8, va="top",
            color=p.suave)
    m = mejor_nuevo(d)
    ax.plot([m[0]], [m[2]], marker="o", markersize=3.6, color=p.tinta)
    ax.annotate("lo mejor que llega a hacerlo en texto nuevo", (m[0], m[2]),
                xytext=(m[0] + 6000, m[2] * 0.55), ha="center", fontsize=6.8, color=p.tinta,
                arrowprops=dict(arrowstyle="-", color=p.suave, linewidth=0.6))
    ax.set_yticks([])
    ax.set_ylabel("lo mal que lo hace", fontsize=7.4)
    ax.set_xlabel("pasos de aprendizaje", fontsize=7.4)
    ax.tick_params(labelsize=6.8, length=2.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=6.9, bbox_to_anchor=(1.0, 0.9))
    L.guardar(ruta)
    return m[0]


def selftest():
    fallos = []
    d = leer()
    # 1. TEST NULO — si los dos textos dieran lo mismo, no habría distancia entre las líneas.
    igual = [(a, b, b) for a, b, _ in d]
    ok = all(x[1] == x[2] for x in igual)
    dibujar(igual, GRIS, "/dev/null")
    print(f"[1] test nulo         con los dos textos iguales, las líneas coinciden: {'sí' if ok else 'NO'}")
    if not ok:
        fallos.append("test nulo")
    # 2. SEÑAL — el punto marcado es el mínimo del texto nuevo en la salida.
    paso = dibujar(d, GRIS, "/dev/null")
    print(f"[2] señal             lo mejor en texto nuevo, en el paso {miles(paso)}")
    if paso != mejor_nuevo(d)[0]:
        fallos.append("señal: el punto marcado no es el mínimo")
    # 3. INVARIANTE — al final, en su propio texto lo hace mejor que en el nuevo.
    dibujar(d, COLOR, "/dev/null")
    ok = d[-1][1] < d[-1][2]
    print(f"[3] invariante        al final, mejor en lo visto que en lo nuevo: {'sí' if ok else 'NO'}")
    if not ok:
        fallos.append("invariante: al final no lo hace mejor en su propio texto")
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
    dibujar(leer(), GRIS, DESTINO)
    print(f"escrita {DESTINO}")


if __name__ == "__main__":
    main()
