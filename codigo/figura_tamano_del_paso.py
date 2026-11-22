#!/usr/bin/env python3
"""
Orden 206 (22 nov 2026): «El tamaño del paso». Sustituye a las figuras del golf, que no irán al libro.

Una curva inventada con forma de cuenco (error = peso al cuadrado: curvatura c = 2), vista de lado, y la misma salida
(-2,6) con tres tamaños de paso. En cada paso, el punto sobre la curva y una flecha curva, por encima de la curva, del
punto al siguiente; las flechas, de más oscuras a más claras según el orden. Tres paneles iguales, sin ejes ni cifras.

Las cuentas son exactas y se hacen aquí mismo: la regla de paso es la del libro, nuevo = viejo - poquito × pendiente, con
pendiente = c × peso. Los tres tamaños se fijan por el producto del poquito por la curvatura, que es lo que decide
(Respuesta 198, punto 1):
  0,3  paso pequeño: cada paso multiplica la distancia al fondo por 0,7; llega sin cruzarlo, cada vez más despacio;
  1,6  paso grande: la multiplica por -0,6; cruza el fondo en cada paso y la distancia baja; llega;
  2,2  paso demasiado grande: la multiplica por -1,2; cruza el fondo en cada paso y la distancia sube.
Con c = 2 los poquitos son 0,15, 0,8 y 1,1.

Se dibujan los pasos hasta que la distancia al fondo baja del FRACCION_FIN de la de salida (el 2 %), o hasta que el punto se
sale del dibujo (de -XLIM a XLIM): entonces la última flecha se dibuja entera y el borde del panel la corta.

Uso:
    python figura_tamano_del_paso.py --selftest
    python figura_tamano_del_paso.py
"""

# ======================= CONSTANTES =======================

DESTINO = "../figuras/tamano_del_paso.png"
COPIA = "../../Claude outputs/reforma-nov/cap16/ver/tamano_del_paso.png"
ALTO = 4.1                      # pulgadas
CURVATURA = 2.0                 # error = (CURVATURA / 2) × peso², así que la pendiente es CURVATURA × peso
SALIDA = -2.6                   # el peso del que salen los tres
PRODUCTOS = (0.3, 1.6, 2.2)     # poquito × curvatura de los tres paneles
FRACCION_FIN = 0.02             # se para cuando la distancia al fondo baja de esta fracción de la de salida
XLIM = 4.9                      # el dibujo va de -XLIM a XLIM en horizontal (Orden 208: era 4,5 y el último punto del paso demasiado grande, 4,49, quedaba cortado por el borde)
MAX_PASOS = 60                  # tope de seguridad
ALTURA_FLECHA = 0.5             # lo que se levantan los extremos de cada flecha sobre la curva
RADIO_FLECHA = 0.45             # cuánto se curva cada flecha
GRIS_FLECHAS = (0.15, 0.70)     # de la primera (más oscura) a la última (más clara)
TITULOS = ("paso pequeño", "paso grande", "paso demasiado grande")
PIES = ("llega, cada vez\\nmás despacio", "se pasa y vuelve,\\ny acaba llegando", "se pasa cada vez\\nmás lejos")

# ==========================================================

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import FancyArrowPatch

from infografia import COLOR, GRIS, Lienzo

AQUI = Path(__file__).resolve().parent


def error(x, c=CURVATURA):
    """La curva: el error de un peso."""
    return 0.5 * c * np.asarray(x, float) ** 2


def pasos(salida, poquito, c=CURVATURA):
    """Las posiciones x0, x1, …: el peso de salida y tras cada paso, con nuevo = viejo - poquito × c × viejo, hasta que la
    distancia al fondo baja de FRACCION_FIN de la de salida o el punto se sale del dibujo. Con poquito 0 no hay pasos."""
    xs = [float(salida)]
    x = float(salida)
    d0 = abs(x)
    for _ in range(MAX_PASOS):
        nuevo = x - poquito * (c * x)
        if nuevo == x:
            break                                   # no se mueve: no hay paso
        xs.append(nuevo)
        x = nuevo
        if abs(x) > XLIM or abs(x) < FRACCION_FIN * d0:
            break
    return xs


def poquitos():
    return tuple(p / CURVATURA for p in PRODUCTOS)


def panel(fig, caja, paleta, xs, titulo, pie):
    """Un panel: la curva de lado, los puntos y las flechas. Devuelve (eje, lista_de_flechas, lista_de_puntos)."""
    p = paleta
    ax = fig.add_axes(caja)
    g = np.linspace(-XLIM, XLIM, 400)
    ax.plot(g, error(g), color=p.tinta, linewidth=2.0, zorder=2)
    n = len(xs) - 1
    flechas = []
    for i, (a, b) in enumerate(zip(xs[:-1], xs[1:])):
        tono = GRIS_FLECHAS[0] + (GRIS_FLECHAS[1] - GRIS_FLECHAS[0]) * i / max(n - 1, 1)
        arr = FancyArrowPatch((a, float(error(a)) + ALTURA_FLECHA), (b, float(error(b)) + ALTURA_FLECHA),
                              connectionstyle="arc3,rad=-%g" % RADIO_FLECHA if b > a else "arc3,rad=%g" % RADIO_FLECHA,
                              arrowstyle="-|>", mutation_scale=7, color=(tono, tono, tono), linewidth=1.1,
                              shrinkA=4, shrinkB=5, zorder=4)
        ax.add_patch(arr)
        flechas.append(arr)
    puntos = []
    for i, x in enumerate(xs):
        if abs(x) > XLIM:
            continue                                  # un punto fuera del dibujo no se dibuja
        (pt,) = ax.plot([x], [float(error(x))], "o", markersize=4.8, markerfacecolor="white" if i == 0 else "0.35",
                        markeredgecolor="black", markeredgewidth=0.8, zorder=5)
        puntos.append(pt)
    # el fondo: una rayita y su etiqueta
    ax.plot([0, 0], [-0.9, 0.0], color=p.tinta, linewidth=1.1, zorder=3, solid_capstyle="butt")
    ax.text(0.0, -1.6, "el fondo", ha="center", va="top", fontsize=6.8, color=p.suave)
    ax.set_xlim(-XLIM, XLIM)
    ax.set_ylim(-3.6, float(error(XLIM)) + 1.8)
    ax.axis("off")
    ax.set_title(titulo, fontsize=7.6, fontweight="bold", color=p.tinta, pad=2)
    ax.text(0.5, -0.02, pie.replace("\\n", "\n"), transform=ax.transAxes, ha="center", va="top", fontsize=7.4,
            color=p.tinta, linespacing=1.2)
    return ax, flechas, puntos


def dibujar(paleta, ruta, salida=SALIDA, productos=PRODUCTOS):
    """Dibuja la figura. Devuelve (ejes, posiciones, figura): las posiciones de cada panel (x0, x1, …)."""
    L = Lienzo("El tamaño del paso",
               "Una curva inventada, con forma de cuenco, y la misma salida.\n"
               "Cada flecha es un paso; lo único que cambia es su tamaño.",
               paleta, alto=ALTO)
    ancho, hueco, margen = 0.29, 0.03, 0.05
    ejes, posiciones = [], []
    for i, prod in enumerate(productos):
        xs = pasos(salida, prod / CURVATURA)
        caja = [margen + i * (ancho + hueco), 0.17, ancho, 0.52]
        ax, _, _ = panel(L.fig, caja, paleta, xs, TITULOS[i], PIES[i])
        ejes.append(ax)
        posiciones.append(xs)
    L.guardar(ruta)
    return ejes, posiciones, L.fig


def textos_de(fig):
    """Todos los textos de la figura: título, subtítulo, rótulos y marcas de los ejes visibles."""
    sal = []
    for ax in fig.axes:
        sal += [t.get_text() for t in ax.texts]
        if ax.axison:
            sal += [t.get_text() for t in ax.get_xticklabels() + ax.get_yticklabels()]
        sal += [ax.get_title()]
    sal += [t.get_text() for t in fig.texts]
    return [t for t in sal if t]


def selftest():
    fallos = []

    # 1. NULO — con poquito 0 no hay pasos (la posición no cambia: solo la salida).
    sin = pasos(SALIDA, 0.0)
    ok1 = sin == [SALIDA]
    print(f"[1] test nulo         con poquito 0 hay {len(sin) - 1} pasos: {'sí' if ok1 else 'NO'}")
    if not ok1:
        fallos.append("nulo: con poquito 0 hay pasos")

    # 2. SEÑAL — la posición tras n pasos es exactamente salida × (1 − poquito × c)^n, a 1e-12, en los tres tamaños.
    peor = 0.0
    for prod in PRODUCTOS:
        p = prod / CURVATURA
        xs = pasos(SALIDA, p)
        for n, x in enumerate(xs):
            peor = max(peor, abs(x - SALIDA * (1 - p * CURVATURA) ** n))
    ok2 = peor < 1e-12
    print(f"[2] señal             la posición tras n pasos es salida × (1 − poquito × c)^n: diferencia máxima {peor:.1e} "
          f"(tope 1e-12): {'sí' if ok2 else 'NO'}")
    if not ok2:
        fallos.append("señal: la posición no es salida × (1 − poquito × c)^n")

    # 3. INVARIANTE — con 0,3 no cruza nunca el fondo (el peso no cambia de signo) y la distancia baja; con 1,6 lo cruza en
    #    cada paso y la distancia baja; con 2,2 lo cruza en cada paso y la distancia sube. Y la figura no lleva textos con cifras
    #    (ni ejes con marcas), en las dos paletas.
    res = []
    for prod in PRODUCTOS:
        xs = np.array(pasos(SALIDA, prod / CURVATURA))
        cruza = int(np.sum(np.sign(xs[:-1]) != np.sign(xs[1:])))
        d = np.abs(xs)
        res.append((cruza, bool(np.all(np.diff(d) < 0)), bool(np.all(np.diff(d) > 0)), len(xs) - 1))
    ok_pequeno = res[0][0] == 0 and res[0][1]
    ok_grande = res[1][0] == res[1][3] and res[1][1]
    ok_demasiado = res[2][0] == res[2][3] and res[2][2]
    dibujar(COLOR, "/dev/null")
    ejes, posiciones, fig = dibujar(GRIS, "/dev/null")
    con_cifras = [t for t in textos_de(fig) if re.search(r"\d", t)]
    marcas = [t for a in fig.axes if a.axison for t in a.get_xticks().tolist() + a.get_yticks().tolist()]
    ok3 = ok_pequeno and ok_grande and ok_demasiado and not con_cifras and not marcas
    print(f"[3] invariante        con 0,3: cruces {res[0][0]}, la distancia baja: {'sí' if res[0][1] else 'NO'}; con 1,6: cruza en "
          f"{res[1][0]} de {res[1][3]} pasos y la distancia baja: {'sí' if res[1][1] else 'NO'}; con 2,2: cruza en "
          f"{res[2][0]} de {res[2][3]} pasos y la distancia sube: {'sí' if res[2][2] else 'NO'}; textos con cifras: "
          f"{len(con_cifras)}; ejes con marcas: {len(marcas)}: {'sí' if ok3 else 'NO'}")
    if not ok3:
        fallos.append("invariante: el paso pequeño cruza, el grande no cruza o no baja, el demasiado grande no sube, o hay cifras")
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
    _, posiciones, _ = dibujar(GRIS, str(AQUI / DESTINO))
    print(f"escrita {DESTINO}")
    copia = (AQUI / COPIA).resolve()
    copia.parent.mkdir(parents=True, exist_ok=True)
    dibujar(GRIS, str(copia))
    print(f"escrita {copia}")
    for tit, prod, xs in zip(TITULOS, PRODUCTOS, posiciones):
        print(f"{tit}: producto {prod:g}, poquito {prod / CURVATURA:g}; {len(xs) - 1} pasos; última posición {xs[-1]:.3f}")


if __name__ == "__main__":
    main()
