#!/usr/bin/env python3
"""
Capítulo 14 — un ejercicio de la prueba de Chollet, dibujado (L24, E22).

Chollet (2019, «On the Measure of Intelligence», https://arxiv.org/abs/1911.01547) propuso medir
la eficiencia con que se aprende una habilidad nueva y construyó una prueba para eso: cada
ejercicio trae unos pocos pares de cuadrículas «antes → después» (3,3 de media, dice el artículo) y
una cuadrícula nueva, y hay que adivinar la regla y aplicarla. Esta figura enseña uno de verdad,
el 25ff71a9 del conjunto público de entrenamiento, tal como está en
https://github.com/fchollet/ARC-AGI (data/training/25ff71a9.json), copiado en
`datos/entrada/arc_25ff71a9.json`: tres ejemplos y la cuadrícula nueva, con su respuesta tapada.

En gris: el libro se imprime en negro. En el original las casillas llevan colores; aquí una casilla
con color es oscura y una vacía es blanca, porque en este ejercicio el color no cuenta.

Uso:
    python figura_prueba_de_chollet.py --selftest
    python figura_prueba_de_chollet.py
"""

# ======================= CONSTANTES =======================

DATOS = "../datos/entrada/arc_25ff71a9.json"
DESTINO = "../figuras/prueba_de_chollet.png"
ALTO = 3.75                # pulgadas (sin el hueco del pie, Orden 4; era 3,9)
EJEMPLOS = 3               # cuántos pares de ejemplo se dibujan
LADO = 3.6                 # lado de una casilla, en unidades del lienzo

# ==========================================================

import argparse
import json
import sys
from pathlib import Path

from matplotlib.patches import FancyArrow, Rectangle

from infografia import COLOR, GRIS, Lienzo

AQUI = Path(__file__).resolve().parent
EN_LETRA = {1: "uno", 2: "dos", 3: "tres", 4: "cuatro", 5: "cinco", 6: "seis"}


def leer(ruta=AQUI / DATOS):
    d = json.loads(Path(ruta).read_text(encoding="utf-8"))
    return [(p["input"], p["output"]) for p in d["train"]], d["test"][0]["input"], d["test"][0]["output"]


def baja_una_fila(antes, despues):
    """La regla del ejercicio: todo baja una fila."""
    n = len(antes)
    return all(despues[f] == (antes[f - 1] if f > 0 else [0] * len(antes[0])) for f in range(n)) \
        and not any(antes[-1])


def cuadricula(L, x, y, g, tapada=False):
    p = L.p
    for f, fila in enumerate(g):
        for c, v in enumerate(fila):
            L.ax.add_patch(Rectangle((x + c * LADO, y - (f + 1) * LADO), LADO, LADO,
                                     facecolor=(p.fondo if tapada else (p.tinta if v else "white")),
                                     edgecolor=p.marco, linewidth=0.8))
    if tapada:
        L.texto(x + 1.5 * LADO, y - 1.5 * LADO, "?", tam=16, ha="center", negrita=True)
    return len(g[0]) * LADO, len(g) * LADO


def dibujar(ejemplos, nueva, paleta, ruta):
    L = Lienzo("Un ejercicio de la prueba de Chollet",
               f"{EN_LETRA[EJEMPLOS].capitalize()} de sus {EN_LETRA[len(ejemplos)]} ejemplos, antes y después. "
               "¿Qué va en el hueco de abajo?",
               paleta, alto=ALTO)
    p = L.p
    # la clave
    y = L.y - 1.0
    L.ax.add_patch(Rectangle((4, y - 1.6), 3.2, 3.2, facecolor=p.tinta, edgecolor=p.marco))
    L.texto(8.4, y, "casilla con color", tam=7.4, color=p.suave)
    L.ax.add_patch(Rectangle((36, y - 1.6), 3.2, 3.2, facecolor="white", edgecolor=p.marco))
    L.texto(40.4, y, "casilla vacía", tam=7.4, color=p.suave)
    y -= 6.5
    filas = [(f"ejemplo {i}", a, b, False) for i, (a, b) in enumerate(ejemplos[:EJEMPLOS], 1)]
    filas.append(("la nueva", nueva, nueva, True))
    dibujadas = 0
    for rotulo, a, b, tapada in filas:
        L.texto(4, y - 1.5 * LADO, rotulo, tam=8.2, negrita=True)
        w, h = cuadricula(L, 30, y, a)
        L.ax.add_patch(FancyArrow(30 + w + 3, y - h / 2, 10, 0, width=0.5, head_width=2.2,
                                  head_length=1.8, length_includes_head=True,
                                  facecolor=p.acento, edgecolor="none"))
        cuadricula(L, 30 + w + 16, y, b, tapada=tapada)
        L.texto(30 + 2 * w + 20, y - 1.5 * LADO, "antes → después" if not tapada else
                "¿después?", tam=7.4, color=p.suave)
        dibujadas += 1
        y -= h + 4.5
    L.guardar(ruta)
    return dibujadas


def selftest():
    fallos = []
    ejemplos, nueva, respuesta = leer()
    # 1. TEST NULO — un par en que nada se mueve no cumple la regla: la comprobación no la regala.
    nulo = baja_una_fila(ejemplos[0][0], ejemplos[0][0])
    print(f"[1] test nulo         un par sin cambio cumple la regla: {'SÍ' if nulo else 'no'}")
    if nulo:
        fallos.append("test nulo: la comprobación da por buena la regla sin que nada baje")
    # 2. SEÑAL IMPLANTADA — todos los ejemplos del ejercicio, y la respuesta tapada, cumplen la
    #    regla que el libro dice: todo baja una fila.
    todos = all(baja_una_fila(a, b) for a, b in ejemplos) and baja_una_fila(nueva, respuesta)
    print(f"[2] señal implantada  los {len(ejemplos)} ejemplos y la respuesta bajan una fila: "
          f"{'sí' if todos else 'NO'}")
    if not todos:
        fallos.append("señal: el ejercicio no sigue la regla que cuenta el libro")
    # 3. INVARIANTE DEL DOMINIO — en cada par, las casillas con color son las mismas en número
    #    antes y después, y la figura se dibuja en color y en gris con las cuatro filas.
    conserva = all(sum(map(bool, sum(a, []))) == sum(map(bool, sum(b, []))) for a, b in ejemplos)
    n = [dibujar(ejemplos, nueva, pal, "/dev/null") for pal in (COLOR, GRIS)]
    print(f"[3] invariante        las casillas con color se conservan: {'sí' if conserva else 'NO'}; "
          f"filas dibujadas: {n}")
    if not conserva or n != [EJEMPLOS + 1] * 2:
        fallos.append("invariante: el ejercicio o la figura no son lo que se espera")
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
    ejemplos, nueva, _ = leer()
    dibujar(ejemplos, nueva, GRIS, str(AQUI / DESTINO))
    print(f"escrita {DESTINO}")


if __name__ == "__main__":
    main()
