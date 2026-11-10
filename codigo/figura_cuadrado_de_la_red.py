#!/usr/bin/env python3
"""
Capítulo 4, anexo A4 — el cuadrado de la red del colegio (Orden 148, 10 nov 2026).

Las trece edades colocadas en las esquinas de un cuadrado según lo que dicen las dos neuronas de la primera
capa de la red entrenada: a lo ancho, la primera («¿tiene 6 o más?»); a lo alto, la segunda («¿tiene 13 o más?»).
Una neurona dice «sí» si dice más de 0,5. Es el cuadrado de `figura_cuadrado_del_colegio.py` (capítulo 3), con las
preguntas que ha aprendido la red en lugar de las que se le pusieron.

Las esquinas salen de lo que dice cada neurona entrenada (`final_dice_medio` de
`datos/salidas/culpa_en_el_colegio_datos.csv`), no están escritas a mano; los rótulos de los ejes salen de los
números de la red, como en `figura_la_red_del_colegio.py`. De 3 a 5 años caen en «no, no»; de 6 a 12, en «sí, no»;
de 13 a 15, en «sí, sí»; y la cuarta esquina («no, sí») queda vacía, con el rótulo «nadie». Una raya recta deja sola
la esquina de «sí, no».

Uso:
    python figura_cuadrado_de_la_red.py --selftest
    python figura_cuadrado_de_la_red.py
"""

# ======================= CONSTANTES =======================

SALIDA = "../figuras/cuadrado_de_la_red.png"
ANCHO_ALTO = (6.2, 4.0)          # pulgadas, como el cuadrado del capítulo 3
PUNTOS = 200                     # puntos por pulgada
EDADES = range(3, 16)
UMBRAL = 0.5                     # una neurona dice «sí» si dice más de esto
LIMITE_X = (-0.55, 1.55)         # lo que se ve del cuadrado, a lo ancho
LIMITE = (-0.45, 1.45)           # y a lo alto
RAYA = (1.0, -0.5)               # (pendiente, altura): y = x - 0,5 deja sola la esquina de «sí, no»
SEPARACION = 0.063               # entre marcas de una misma esquina, en unidades del cuadrado
TAMANO_MARCA = 9
ROTULO_VACIA = "nadie"
ESPERADO = {(0, 0): [3, 4, 5], (1, 0): list(range(6, 13)), (1, 1): [13, 14, 15], (0, 1): []}

# ==========================================================

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figura_la_red_del_colegio as R

AQUI = Path(__file__).resolve().parent


def esquinas(d, umbral=UMBRAL):
    """{(primera, segunda): [(edad, entra)]}: 0 o 1 según lo que dice cada neurona de en medio, a cada edad."""
    e = {(a, b): [] for a in (0, 1) for b in (0, 1)}
    for edad in EDADES:
        a = int(R.v(d, "final_dice_medio", edad, 0) > umbral)
        b = int(R.v(d, "final_dice_medio", edad, 1) > umbral)
        e[(a, b)].append((edad, 6 <= edad <= 12))
    return e


def lado_de_la_raya(x, y):
    m, h = RAYA
    return y - (m * x + h) < 0          # por debajo de la raya: el lado donde entran


def dibujar(d, ruta):
    pesos, listones = R.red_de(d)
    q = R.preguntas(pesos, listones)
    plt.rcParams["font.size"] = 9
    fig, ax = plt.subplots(figsize=ANCHO_ALTO)
    m, h = RAYA
    xs = [LIMITE_X[0], LIMITE_X[1]]
    ax.fill_between(xs, [m * x + h for x in xs], LIMITE[0], color="0.90", zorder=0)
    ax.plot(xs, [m * x + h for x in xs], "--", color="0.35", linewidth=1.3, zorder=2)
    ax.text(1.52, -0.40, "de este lado,\nlos que entran", fontsize=8, color="0.25", ha="right", va="bottom", zorder=4)
    for (a, b), edades in esquinas(d).items():
        if not edades:
            ax.text(a, b, ROTULO_VACIA, fontsize=9, color="0.4", style="italic", ha="center", va="center", zorder=4)
            continue
        n = len(edades)
        for k, (e, entra) in enumerate(edades):
            x = a + (k - (n - 1) / 2) * SEPARACION * 1.9
            ax.plot(x, b, "o", markersize=TAMANO_MARCA, markerfacecolor="black" if entra else "white",
                    markeredgecolor="black", markeredgewidth=1.4, zorder=3)
            ax.text(x, b - 0.075, str(e), fontsize=8, ha="center", va="top", zorder=4)
    ax.set_xlim(*LIMITE_X)
    ax.set_ylim(*LIMITE)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["no", "sí"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["no", "sí"])
    ax.set_xlabel(f"lo que dice la primera: {q[0]}")
    ax.set_ylabel(f"lo que dice la segunda: {q[1]}")
    ax.set_aspect("equal")
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.tick_params(length=0)
    fig.text(0.5, 0.012, "relleno: entra en el colegio    ·    vacío: no entra", ha="center", fontsize=8.5, color="0.3")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(ruta, dpi=PUNTOS, format="png")
    plt.close(fig)
    return q


def selftest():
    fallos = []
    d = R.leer_datos()
    es = esquinas(d)

    # 1. TEST NULO — con un umbral por las nubes ninguna neurona dice «sí»: las trece caen en «no, no», y eso no
    #    es lo esperado (la comprobación de las esquinas se queja).
    nulo = esquinas(d, umbral=1.1)
    print(f"[1] test nulo         con el umbral en 1,1 caen en «no, no» {len(nulo[(0, 0)])} edades, y las "
          f"esquinas esperadas se cumplen: {'sí' if {k: [e for e, _ in v] for k, v in nulo.items()} == ESPERADO else 'no'}")
    if {k: [e for e, _ in v] for k, v in nulo.items()} == ESPERADO or len(nulo[(0, 0)]) != 13:
        fallos.append("test nulo: con un umbral imposible las esquinas siguen siendo las esperadas")

    # 2. SEÑAL — las esquinas de la red entrenada: de 3 a 5 en «no, no»; de 6 a 12 en «sí, no»; de 13 a 15 en
    #    «sí, sí»; y la cuarta, vacía.
    real = {k: [e for e, _ in v] for k, v in es.items()}
    print(f"[2] señal             esquinas: «no, no» {real[(0, 0)]}; «sí, no» {real[(1, 0)]}; «sí, sí» {real[(1, 1)]}; "
          f"«no, sí» {real[(0, 1)] or 'vacía'}")
    if real != ESPERADO:
        fallos.append(f"señal: las esquinas son {real}, y se esperaba {ESPERADO}")

    # 3. INVARIANTE — la raya deja a un lado exactamente las siete que entran, la marca rellena es la de las que
    #    entran, las preguntas de los ejes salen de los números, y la figura se dibuja.
    puestos = [(a, b, e) for (a, b), v in es.items() for e, _ in v]
    al_lado = sorted(e for a, b, e in puestos if lado_de_la_raya(a, b))
    rellenas = sorted(e for v in es.values() for e, s in v if s)
    q = dibujar(d, "/dev/null")
    print(f"[3] invariante        la raya deja a un lado {al_lado}; rellenas {rellenas}; ejes: {q}")
    if al_lado != list(range(6, 13)) or rellenas != list(range(6, 13)):
        fallos.append("invariante: la raya no deja solas a las siete que entran")
    if q != ["«¿tiene 6 o más?»", "«¿tiene 13 o más?»"]:
        fallos.append(f"invariante: las preguntas de los ejes no son las de los números: {q}")
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
    dibujar(R.leer_datos(), str(AQUI / SALIDA))
    print(f"escrita {SALIDA}")


if __name__ == "__main__":
    main()
