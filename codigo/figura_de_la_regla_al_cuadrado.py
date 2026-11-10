#!/usr/bin/env python3
"""
Capítulo 4 — de la regla al cuadrado (Orden 20, 18 oct 2026).

A la izquierda, la regla de edades de 3 a 15, con las que entran en el colegio rellenas: sobre ella
no hay corte que las deje solas (hacen falta dos). A la derecha, un cuadrado cuyos ejes son lo que
dice cada una de las dos neuronas de en medio de la red entrenada (de 0 a 1), con cada edad en su
punto y la raya de la neurona final. Las siete que entran quedan juntas a un lado de la raya.

Los números se leen de `datos/salidas/culpa_en_el_colegio_datos.csv` (lo que dice cada neurona de en
medio a cada edad, y los pesos y el listón de la final); la raya es la de la final: los puntos en que
lo que le llega (peso por lo que dice cada una, sumado) es justo su listón.

Uso:
    python figura_de_la_regla_al_cuadrado.py --selftest
    python figura_de_la_regla_al_cuadrado.py
"""

# ======================= CONSTANTES =======================

DATOS = "../datos/salidas/culpa_en_el_colegio_datos.csv"   # los números, que escribe culpa_en_el_colegio.py
DESTINO = "../figuras/de_la_regla_al_cuadrado.png"
ALTO = 4.2                     # pulgadas
ALTO_EJES = 0.61               # alto de la regla y del cuadrado, en fracción de la figura
LIM_X = (-0.10, 1.50)          # el cuadrado: lo que se ve a lo ancho (a la derecha caben los rótulos)
LIM_Y = (-0.10, 1.10)          # y a lo alto
CERCA = 0.05                   # dos marcas a menos de esto, en las unidades del cuadrado, se pisan
EDADES = list(range(3, 16))    # las de los datos
RAYA_TRAMOS = 2                # cuántos cortes hacen falta en la regla para dejar solas a las que entran

# ==========================================================

import argparse
import sys
from pathlib import Path

import numpy as np

from figura_la_red_del_colegio import f, leer_datos, red_de, v
from infografia import COLOR, GRIS, Lienzo

AQUI = Path(__file__).resolve().parent


def puntos(d):
    """{edad: (lo que dice la primera, lo que dice la segunda, entra)} de los datos."""
    return {e: (v(d, "final_dice_medio", e, 0), v(d, "final_dice_medio", e, 1),
                v(d, "deberia", e) == 1.0) for e in EDADES}


def cortes_en_la_regla(entran):
    """Cuántas veces cambia la respuesta de una edad a la siguiente, en la regla de edades."""
    return sum(1 for a, b in zip(entran, entran[1:]) if a != b)


def lado_de_la_raya(pesos, listones, x, y):
    """Lo que le llega a la final menos su listón, con lo que dicen las dos de en medio: positivo,
    a un lado de la raya; negativo, al otro."""
    p = pesos["la final"]
    return p["la primera"] * x + p["la segunda"] * y - listones["la final"]


def grupos_cercanos(pts, cerca=CERCA):
    """Las edades cuyos puntos del cuadrado quedan a menos de `cerca` unos de otros (una marca pisa a la
    otra) van juntas: [[edades]], en orden de edad, y cada grupo lleva el punto de su primera edad."""
    grupos = []
    for e in EDADES:
        for g in grupos:
            x0, y0, _ = pts[g[0]]
            if ((pts[e][0] - x0) ** 2 + (pts[e][1] - y0) ** 2) ** 0.5 < cerca:
                g.append(e)
                break
        else:
            grupos.append([e])
    return grupos


def rotulo_de_edades(g):
    """«6», «3 y 4», «7 a 11»: las edades de un grupo, dichas como se dirían."""
    if len(g) == 1:
        return str(g[0])
    seguidas = all(b == a + 1 for a, b in zip(g, g[1:]))
    if len(g) == 2:
        return f"{g[0]} y {g[1]}"
    return f"{g[0]} a {g[-1]}" if seguidas else ", ".join(map(str, g[:-1])) + f" y {g[-1]}"


def dibujar(d, paleta, ruta):
    pesos, listones = red_de(d)
    pts = puntos(d)
    L = Lienzo("De la regla al cuadrado",
               "A la izquierda, las edades en una regla: las edades de 6 a 12 quedan en medio y ningún\n"
               "corte los deja solos. A la derecha, las mismas edades colocadas por lo que dicen las\n"
               "dos neuronas de la primera capa: ahora una raya los deja solos.", paleta, alto=ALTO)
    p, fig = L.p, L.fig
    # la regla, en vertical
    a1 = fig.add_axes([0.04, 0.15, 0.22, ALTO_EJES])
    for e in EDADES:
        a1.plot(0, e, "o", markersize=7, markerfacecolor=p.tinta if pts[e][2] else "white",
                markeredgecolor=p.tinta, markeredgewidth=1.2, zorder=3)
        a1.text(0.5, e, str(e), fontsize=7.0, va="center", color=p.tinta)
    a1.set_xlim(-0.5, 1.2)
    a1.set_ylim(EDADES[0] - 0.7, EDADES[-1] + 2.0)
    a1.axis("off")
    a1.text(-0.45, EDADES[-1] + 1.0, "la regla", fontsize=7.6, fontweight="bold", color=p.tinta, va="bottom")
    # el cuadrado
    a2 = fig.add_axes([0.32, 0.15, 0.64, ALTO_EJES])
    xs = np.linspace(LIM_X[0], LIM_X[1], 60)
    pa, pl = pesos["la final"], listones["la final"]
    ys = (pl - pa["la primera"] * xs) / pa["la segunda"]       # la raya: pa1*x + pa2*y = listón
    a2.fill_between(xs, ys, LIM_Y[0], where=ys > LIM_Y[0], color="0.90", zorder=0)
    a2.plot(xs, ys, "--", color=p.suave, linewidth=1.3, zorder=2)
    for e in EDADES:
        x, y, entra = pts[e]
        a2.plot(x, y, "o", markersize=5.5, markerfacecolor=p.tinta if entra else "white",
                markeredgecolor=p.tinta, markeredgewidth=1.1, zorder=3)
    # las edades, escritas junto a cada marca o cada montón de marcas: a la derecha las de la derecha,
    # encima las de la izquierda y las de en medio
    for g in grupos_cercanos(pts):
        x, y, _ = pts[g[0]]
        if x > 0.9:
            x_derecha = max(pts[e][0] for e in g)
            a2.text(x_derecha + 0.07, y, rotulo_de_edades(g), fontsize=7.0, color=p.tinta, ha="left",
                    va="center")
        else:
            a2.text(x, y + 0.075, rotulo_de_edades(g), fontsize=7.0, color=p.tinta, ha="center", va="bottom")
    a2.annotate("la raya de la final", xy=(0.62, (pl - pa["la primera"] * 0.62) / pa["la segunda"]),
                xytext=(0.18, 0.40), fontsize=7.0, color=p.suave, ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", color=p.suave, lw=0.8))
    a2.set_xlim(*LIM_X)
    a2.set_ylim(*LIM_Y)
    a2.set_xticks([0, 1]); a2.set_xticklabels(["no", "sí"], fontsize=7)
    a2.set_yticks([0, 1]); a2.set_yticklabels(["no", "sí"], fontsize=7)
    a2.set_xlabel("lo que dice la primera (de 0 a 1)", fontsize=7.4, color=p.tinta)
    a2.set_ylabel("lo que dice la segunda (de 0 a 1)", fontsize=7.4, color=p.tinta)
    a2.set_aspect("equal", adjustable="box")
    for s in ("top", "right"):
        a2.spines[s].set_visible(False)
    a2.tick_params(length=2)
    L.pie("Relleno: entra en el colegio; vacío: no entra. Sombreado: el lado de la raya donde la final dice sí.\n"
          "Donde varias edades caen casi en el mismo punto, sus marcas se montan y van con todas sus edades.")
    L.guardar(ruta)


def selftest():
    fallos = []
    d = leer_datos(AQUI / DATOS)
    pesos, listones = red_de(d)
    pts = puntos(d)
    entran = [pts[e][2] for e in EDADES]

    # 1. TEST NULO — con una sola neurona (la primera) no se separan: mirando solo lo que dice la
    #    primera, las que entran y las que no se mezclan (las de 13 a 15 dicen lo mismo que las de
    #    6 a 12): ningún corte en ese eje deja a un lado exactamente las siete.
    primera = np.array([pts[e][0] for e in EDADES])
    separa_sola = any(set(np.flatnonzero(primera > t)) == {i for i, s in enumerate(entran) if s}
                      for t in np.linspace(0, 1, 1001))
    print(f"[1] test nulo         con solo lo que dice la primera, hay un corte que deja solas a las siete: "
          f"{'sí' if separa_sola else 'no'}")
    if separa_sola:
        fallos.append("test nulo: una sola neurona ya separaba a las siete")

    # 2. SEÑAL IMPLANTADA — en la regla hacen falta dos cortes, y en el cuadrado la raya de la final
    #    deja a un lado exactamente las siete que entran.
    cortes = cortes_en_la_regla(entran)
    lado = {e: lado_de_la_raya(pesos, listones, pts[e][0], pts[e][1]) > 0 for e in EDADES}
    al_lado = [e for e in EDADES if lado[e]]
    solas = [e for e in EDADES if pts[e][2]]
    print(f"[2] señal implantada  cortes que hacen falta en la regla: {cortes}; la raya deja a un lado "
          f"{al_lado[0]} a {al_lado[-1]} ({len(al_lado)} edades), las que entran")
    if cortes != RAYA_TRAMOS or al_lado != solas or len(solas) != 7:
        fallos.append("señal implantada: la regla no pide dos cortes o la raya no deja solas a las siete")

    # 3. INVARIANTE — lo que dice cada neurona está entre 0 y 1, las trece edades están, y la figura
    #    se dibuja en las dos paletas.
    fuera = sum(1 for e in EDADES for k in (0, 1) if not 0 <= pts[e][k] <= 1)
    for pal in (COLOR, GRIS):
        dibujar(d, pal, "/dev/null")
    grupos = grupos_cercanos(pts)
    rotulos = [rotulo_de_edades(g) for g in grupos]
    todas = sorted(e for g in grupos for e in g)
    print(f"[3] invariante        edades: {len(pts)}; respuestas fuera de 0 a 1: {fuera}; marcas que se "
          f"pisan, con sus edades: {', '.join(rotulos)}")
    if todas != EDADES:
        fallos.append("invariante: alguna edad se queda sin rótulo, o con dos")
    if fuera or len(pts) != 13:
        fallos.append("invariante: faltan edades o hay respuestas fuera de 0 a 1")
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
    dibujar(leer_datos(AQUI / DATOS), GRIS, str(AQUI / DESTINO))
    print(f"escrita {DESTINO}")


if __name__ == "__main__":
    main()
