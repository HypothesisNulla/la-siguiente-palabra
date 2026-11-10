#!/usr/bin/env python3
"""
Capítulo 4 — el terreno del error, con un peso, en la red del colegio (Orden 20, 18 oct 2026).

La red entrenada del colegio, moviendo solo la línea de la primera a la final. En horizontal, lo que
vale ese peso; en vertical, el error (la media de las trece edades). Encima, los pasos cuesta abajo
desde 4, moviendo solo ese peso, marcados cada 100.

Todos los números se leen de `datos/salidas/culpa_en_el_colegio_datos.csv` (apartado d de
`culpa_en_el_colegio.txt`).

Uso:
    python figura_el_terreno_del_colegio.py --selftest
    python figura_el_terreno_del_colegio.py
"""

# ======================= CONSTANTES =======================

DATOS = "../datos/salidas/culpa_en_el_colegio_datos.csv"   # los números, que escribe culpa_en_el_colegio.py
SALIDA = "../datos/salidas/culpa_en_el_colegio.txt"
DESTINO = "../figuras/el_terreno_del_colegio.png"
ALTO = 3.3                     # pulgadas
ROTULO_X = "el peso de la línea de la primera a la final:  menos  ←  →  más"

# ==========================================================

import argparse
import sys
from pathlib import Path

import numpy as np

from figura_la_red_del_colegio import leer_datos, v
from formato import coma as coma_fija, leer_tablas
from infografia import COLOR, GRIS, Lienzo

AQUI = Path(__file__).resolve().parent


def curvas(d):
    """(terreno, camino, cada): el terreno [(peso, error)], el camino [(paso, peso, error)] y cada
    cuántos pasos se marca, de los datos."""
    terreno = sorted((float(a), val) for (que, a, b), val in d.items() if que == "terreno")
    camino = sorted((int(a), float(b), val) for (que, a, b), val in d.items() if que == "camino")
    return np.array(terreno), np.array(camino), int(v(d, "cada_cuantos_camino"))


def coma(x, n=1):
    return f"{x:.{n}f}".replace(".", ",").replace("-", "−")


def dibujar(d, paleta, ruta):
    t, c, cada = curvas(d)
    # Orden 148 (regla 11.9): sin cifras que el cuerpo no explique. El eje horizontal, sin cifras y con un rótulo
    # que dice hacia dónde crece el peso; el vertical, solo con el 0; y fuera «uno cada 100».
    L = Lienzo("El terreno del error, con un peso",
               "La red del colegio, ya entrenada. Se mueve solo la línea de la primera a la final;\n"
               "los demás pesos se quedan quietos. La altura es el error, con las trece edades.\n"
               "Los puntos, los pasos cuesta abajo moviendo solo ese peso.",
               paleta, alto=ALTO)
    p = L.p
    a = L.fig.add_axes([0.12, 0.17, 0.84, 0.50])
    a.plot(t[:, 0], t[:, 1], color=p.tinta, linewidth=1.5)
    marcas = c[::cada]
    a.plot(marcas[:, 1], marcas[:, 2], "o", markersize=4.2, markerfacecolor=p.acento,
           markeredgecolor=p.tinta, markeredgewidth=0.6, zorder=4)
    a.annotate("empieza aquí", xy=(marcas[0, 1], marcas[0, 2]), xytext=(marcas[0, 1] - 4.5, 0.40),
               fontsize=7.0, color=p.tinta, arrowprops=dict(arrowstyle="-", color=p.suave, lw=0.6))
    k = int(np.argmin(t[:, 1]))
    a.annotate("el fondo", xy=(t[k, 0], t[k, 1]), xytext=(t[k, 0] + 1.0, 0.16), fontsize=7.0,
               color=p.tinta, arrowprops=dict(arrowstyle="-", color=p.suave, lw=0.6))
    a.text(-4.7, 0.70, "llano: la culpa\nes casi cero", fontsize=7.0, color=p.suave, va="top",
           linespacing=1.1)
    a.set_xlabel(ROTULO_X, fontsize=7.4, color=p.tinta)
    a.set_ylabel("error", fontsize=7.4, color=p.tinta)
    a.set_ylim(-0.02, 0.71)
    a.set_yticks([0]); a.set_yticklabels(["0"])
    a.set_xticks([])
    a.tick_params(labelsize=7.0)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    L.guardar(ruta)
    return a


def selftest():
    fallos = []
    d = leer_datos(AQUI / DATOS)
    t, c, cada = curvas(d)
    T = leer_tablas((AQUI / SALIDA).read_text(encoding="utf-8"))

    # 1. TEST NULO — a la izquierda hay un llano: el error casi no cambia entre dos pesos lejanos
    #    de ese lado (la culpa es casi cero), y el camino no se mueve de ahí sin pendiente.
    izq = t[t[:, 0] <= 0]
    llano = float(izq[:, 1].max() - izq[:, 1].min())
    print(f"[1] test nulo         en el llano de la izquierda el error varía {coma_fija(llano, 4)} en "
          f"{coma_fija(izq[-1, 0] - izq[0, 0], 0)} unidades de peso")
    if llano > 0.005:
        fallos.append("test nulo: la izquierda no es un llano")

    # 2. SEÑAL — el camino baja siempre y acaba cerca del fondo de la curva; el fondo de la curva está
    #    cerca de donde dejó el peso el entrenamiento; y los puntos del camino son los de la tabla
    #    «Cuesta abajo» de la salida.
    k = int(np.argmin(t[:, 1]))
    baja = bool(np.all(np.diff(c[:, 2]) <= 1e-12))
    peso_red = v(d, "final_peso_final", 0)
    tabla = T[[k_ for k_ in T if k_.startswith("Cuesta abajo")][0]][1]
    en_tabla = all(f[1] == coma_fija(c[int(f[0].replace(".", "")), 1], 2) and
                   f[2] == coma_fija(c[int(f[0].replace(".", "")), 2], 4) for f in tabla)
    print(f"[2] señal             el camino baja siempre: {'sí' if baja else 'NO'}; acaba con error "
          f"{coma_fija(c[-1, 2], 4)} (el fondo: {coma_fija(t[k, 1], 5)}); el fondo está a "
          f"{coma_fija(abs(t[k, 0] - peso_red), 2)} del peso de la red; la tabla de la salida es el camino: "
          f"{'sí' if en_tabla else 'NO'}")
    if not baja or c[-1, 2] > 0.001 or abs(t[k, 0] - peso_red) > 0.5 or not en_tabla:
        fallos.append("señal: el camino no baja hasta cerca del fondo, o la figura no es la tabla")

    # 3. INVARIANTE — lo que mide cada punto de la curva es un error (entre 0 y 0,25 ·4 = 1 como mucho
    #    para respuestas entre 0 y 1) y se dibuja en las dos paletas.
    ok = bool(np.all((t[:, 1] >= 0) & (t[:, 1] <= 1)) and np.all((c[:, 2] >= 0) & (c[:, 2] <= 1)))
    for pal in (COLOR, GRIS):
        ejes = dibujar(d, pal, "/dev/null")
    # la 11.9 (Orden 148): el eje horizontal no lleva cifras, y el vertical, solo el 0
    cifras_x = [x.get_text() for x in ejes.get_xticklabels()]
    cifras_y = [x.get_text() for x in ejes.get_yticklabels()]
    print(f"[3] invariante        {len(t)} puntos de curva y {len(c)} del camino, todos con error entre 0 y 1: "
          f"{'sí' if ok else 'NO'}; cifras del eje horizontal: {cifras_x or 'ninguna'}; del vertical: {cifras_y}")
    if not ok:
        fallos.append("invariante: algún error está fuera de 0 a 1")
    if cifras_x or cifras_y != ["0"]:
        fallos.append(f"invariante: la figura lleva cifras en los ejes ({cifras_x}, {cifras_y})")
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
