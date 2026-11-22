#!/usr/bin/env python3
"""
Orden 201 (22 nov 2026) y Orden 202 (otra regla para elegir las bolas, y un recuadro): el golf, con dos pesos, en
relieve (el terreno de los dos valles de `dos_valles.py`).

El mismo terreno inventado y la misma salida (la de la bola 2), con tres bolas. La regla de la Orden 201 (que (c)
saliera del cuadrado) no servía: (c) no existía y (a) y (b) salían como una sola línea. Regla vigente (Orden 202),
sobre la tabla de `golf_en_dos_valles.txt`, en el orden de la rejilla:
  (a) «entra»: el primer poquito que acaba en el valle hondo sin cruzar el fondo;
  (b) «se pasa y vuelve»: el último poquito que todavía acaba asentado en el valle hondo;
  (c) «no se asienta nunca»: el primero que acaba «yendo y viniendo sin asentarse».
La elección la hace `golf_en_dos_valles.elegir` sobre los caminos leídos de `datos/salidas/golf_en_dos_valles_datos.csv`
(la tabla y los caminos son los de la Orden 201: la medición no se repitió). Salen 0,1, 1,9 y 2,1 veces 1/c.

De (c) se dibujan solo los primeros PASOS_C pasos (si no, el ir y venir llena el fondo del valle). Un recuadro, en una
esquina, muestra el fondo del valle hondo visto desde arriba, en curvas de nivel, ampliado lo justo para que se vean
el zigzag de (b) y el ir y venir de (c): la ventana es MARGEN veces la mayor distancia al fondo que alcanzan (b) y (c)
desde que entran a menos de ENTRADA del fondo. Cada bola lleva en el recuadro la misma marca que en el relieve.

El relieve es el de `figura_dos_valles_relieve.py`, con el mismo punto de vista que «Dos bolas, dos valles».

Uso:
    python figura_golf_en_dos_valles.py --selftest
    python figura_golf_en_dos_valles.py
"""

# ======================= CONSTANTES =======================

DATOS = "../datos/salidas/golf_en_dos_valles_datos.csv"
DESTINO = "../figuras/golf_en_dos_valles.png"
COPIA = "../../Claude outputs/reforma-nov/cap16/ver/golf_en_dos_valles.png"
ALTO = 5.1                      # pulgadas
CAJA_RELIEVE = [-0.06, -0.06, 1.12, 0.80]      # dónde va el relieve (fracciones de la figura)
CAJA_RECUADRO = [0.715, 0.545, 0.265, 0.235]   # y el recuadro, en la esquina de arriba a la derecha, bajo el subtítulo
MARCA_CADA = 10                 # en la bola (a), un punto cada tantos pasos, mientras no ha llegado al fondo (a más de EPS_MARCA de él)
EPS_MARCA = 1e-2
MARCAS_B = 25                   # en la bola (b), un punto en cada uno de los primeros golpes
PASOS_C = 40                    # de la bola (c) se dibujan los primeros pasos
MARCAS_C = 14                   # y se marcan con un rombo los primeros golpes (si se marcaran los 40 se vuelve una mancha)
ENTRADA = 0.5                   # «entra»: a menos de esto del fondo (para la ventana del recuadro)
MARGEN = 1.4                    # la ventana del recuadro: media anchura = MARGEN × la mayor distancia de (b) y (c) tras entrar
NIVELES = 9                     # curvas de nivel del recuadro
PUNTOS_RECUADRO = 160           # puntos por lado de la rejilla del recuadro
TEXTOS = {"a": "entra", "b": "se pasa y vuelve", "c": "no se asienta nunca"}
ESPERADAS = (0.1, 1.9, 2.1)     # las tres bolas que dice la Orden 202 (veces 1/c); el invariante comprueba que la regla las da

# ==========================================================

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from dos_valles import altura
import golf_en_dos_valles as G
from figura_dos_valles_relieve import dibujar_relieve
from infografia import COLOR, GRIS, Lienzo

AQUI = Path(__file__).resolve().parent


def leer(ruta=AQUI / DATOS):
    """{multiplo: (poquito, array [(x, y, altura), ...])} del CSV de golf_en_dos_valles.py."""
    d = {}
    with open(ruta, encoding="utf-8") as fh:
        for f in csv.DictReader(fh):
            m = float(f["multiplo"])
            d.setdefault(m, (float(f["poquito"]), []))[1].append((float(f["x"]), float(f["y"]), float(f["altura"])))
    assert d, "no hay datos"
    return {m: (p, np.array(c)) for m, (p, c) in d.items()}


def filas_de(datos):
    """Las filas de golf_en_dos_valles.py (con su resumen), recalculadas sobre los caminos leídos."""
    _, _, fondo = G.curvatura()
    filas = []
    for m in sorted(datos):
        p, c = datos[m]
        camino = [tuple(r) for r in c]
        paro = (len(camino) - 1) if len(camino) - 1 < G.PASOS_FIJOS else None
        filas.append({"mult": m, "poquito": p, "camino": camino, **G.resumen(camino, paro, fondo)})
    return filas, fondo


def pq(x):
    return f"{x:.2g}".replace(".", ",")


def etiqueta(k, r):
    return f"{TEXTOS[k]}\npoquito {pq(r['poquito'])}\n({pq(r['mult'])} veces)"


def ventana(el, fondo):
    """Media anchura de la ventana del recuadro: MARGEN × la mayor distancia al fondo de (b) y (c) desde que entran
    a menos de ENTRADA del fondo (en los pasos que se dibujan)."""
    m = 0.0
    for k in "bc":
        if el[k] is None:
            continue
        c = np.array(el[k]["camino"])
        if k == "c":
            c = c[:PASOS_C + 1]
        d = np.hypot(c[:, 0] - fondo[0], c[:, 1] - fondo[1])
        if (d < ENTRADA).any():
            m = max(m, float(d[int(np.argmax(d < ENTRADA)):].max()))
    return MARGEN * m


def estilos_de(p, el, fondo):
    marcas_a = []
    if el["a"] is not None:
        ca = np.array(el["a"]["camino"])
        da = np.hypot(ca[:, 0] - fondo[0], ca[:, 1] - fondo[1])
        marcas_a = [i for i in range(MARCA_CADA, len(ca), MARCA_CADA) if da[i] > EPS_MARCA]
    return {"a": {"color": p.suave, "grosor": 1.8, "marcas": marcas_a, "marcador": "s", "relleno": p.suave},
            "b": {"color": p.tinta, "grosor": 1.2, "marcas": list(range(1, MARCAS_B)), "marcador": "o", "relleno": "white"},
            "c": {"color": p.tinta, "grosor": 0.9, "marcas": list(range(1, MARCAS_C)), "marcador": "D", "relleno": p.tinta,
                  "sin_final": True}}


def recuadro(fig, caja, paleta, el, fondo, estilos):
    """El fondo del valle hondo, desde arriba, en curvas de nivel. Devuelve (eje, media anchura)."""
    p = paleta
    W = ventana(el, fondo)
    ax = fig.add_axes(caja)
    g = np.linspace(-W, W, PUNTOS_RECUADRO)
    X, Y = np.meshgrid(fondo[0] + g, fondo[1] + g)
    Z = np.vectorize(altura)(X, Y)
    ax.contourf(X, Y, Z, levels=NIVELES, cmap="Greys_r", vmin=Z.min() - 0.5 * (Z.max() - Z.min()),
                vmax=Z.max() + 0.2 * (Z.max() - Z.min()))
    ax.contour(X, Y, Z, levels=NIVELES, colors=["white"], linewidths=0.4)
    for k in "abc":
        if el[k] is None:
            continue
        c = np.array(el[k]["camino"])
        if k == "c":
            c = c[:PASOS_C + 1]
        est = estilos[k]
        if k == "c":     # el ir y venir son 40 pasos entre dos sitios: línea fina y sin borde blanco, o se vuelve una mancha
            ax.plot(c[:, 0], c[:, 1], color="white", linewidth=0.3, zorder=4)
        else:
            ax.plot(c[:, 0], c[:, 1], color="white", linewidth=est["grosor"] * 0.55 + 1.0, zorder=3, solid_capstyle="round")
            ax.plot(c[:, 0], c[:, 1], color=est["color"], linewidth=est["grosor"] * 0.55, zorder=4)
        mk = [i for i in est["marcas"] if 0 < i < len(c)]
        if k == "a":
            mk = [i for i in mk if abs(c[i, 0] - fondo[0]) < W and abs(c[i, 1] - fondo[1]) < W]
        ax.plot(c[mk, 0], c[mk, 1], est["marcador"], markersize=2.4 if k != "c" else 2.0, markerfacecolor=est["relleno"],
                markeredgecolor=p.tinta, markeredgewidth=0.5, zorder=5, linestyle="none")
    ax.plot([fondo[0]], [fondo[1]], "+", markersize=5, color=p.tinta, markeredgewidth=0.8, zorder=6)
    ax.set_xlim(fondo[0] - W, fondo[0] + W)
    ax.set_ylim(fondo[1] - W, fondo[1] + W)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(p.tinta); s.set_linewidth(0.8)
    ax.set_title("el fondo del valle hondo,\nvisto desde arriba", fontsize=6.4, color=p.tinta, pad=2, linespacing=1.1)
    return ax, W


def dibujar(datos, paleta, ruta):
    filas, fondo = filas_de(datos)
    el = G.elegir(filas)
    L = Lienzo("El golf, con dos pesos",
               "El mismo terreno inventado y la misma salida, con tres poquitos.\n"
               "El recuadro es el fondo del valle hondo, visto desde arriba.",
               paleta, alto=ALTO)
    p = L.p
    estilos = estilos_de(p, el, fondo)
    caminos, etiquetas = {}, {}
    desp = {"a": (-0.2, 1.7, 1.2), "b": (-0.5, -2.8, 1.9), "c": (0.0, -3.4, 1.5)}
    frac = {"a": 0.25, "b": 0.35, "c": 0.0}
    for k in "abc":
        r = el[k]
        if r is None:
            continue
        c = np.array(r["camino"])
        caminos[k] = c[:PASOS_C + 1] if k == "c" else c
        etiquetas[k] = {"texto": etiqueta(k, r), "frac": frac[k], "desp": desp[k]}
    ax, dib = dibujar_relieve(L.fig, CAJA_RELIEVE, paleta, caminos, etiquetas, valles=True, estilos=estilos,
                            corrimiento=((1.3, -1.6, 0.9), (0.6, -2.4, 1.0)))
    axr, W = recuadro(L.fig, CAJA_RECUADRO, paleta, el, fondo, estilos)
    L.guardar(ruta)
    return ax, dib, el, W


def _sintetico(datos, fondo):
    """Una copia de los datos con (b) y (c) sustituidos por caminos de amplitud conocida alrededor del fondo:
    (b) se encoge ±0,1·(−0,5)^k y se queda en el fondo; (c) salta ±0,2 sin parar."""
    falso = dict(datos)
    ini = np.array(datos[1.9][1][0])
    fx, fy = fondo
    b = [tuple(ini)] + [(fx + 0.1 * (-0.5) ** k, fy, altura(fx + 0.1 * (-0.5) ** k, fy)) for k in range(1, 40)]
    b += [(fx, fy, altura(fx, fy))] * (G.PASOS_FIJOS + 1 - len(b))
    c = [tuple(ini)] + [(fx + 0.2 * (-1) ** k, fy, altura(fx + 0.2 * (-1) ** k, fy)) for k in range(1, G.PASOS_FIJOS + 1)]
    falso[1.9] = (datos[1.9][0], np.array(b))
    falso[2.1] = (datos[2.1][0], np.array(c))
    return falso


def selftest():
    fallos = []
    datos = leer()
    filas, fondo = filas_de(datos)

    # 1. NULO — con poquito 0 el peso no se mueve: el recuadro no tiene nada que mostrar de él, y una bola que
    #    se queda en el fondo no ensancha la ventana.
    quieto, _ = G.recorrer(G.SALIDA, 0.0, pasos=30)
    cq = np.array(quieto)
    mov = float(np.max(np.hypot(cq[:, 0] - cq[0, 0], cq[:, 1] - cq[0, 1])))
    en_fondo = {"b": {"camino": [(fondo[0], fondo[1], altura(*fondo))] * 50}, "c": None}
    w0 = ventana(en_fondo, fondo)
    print(f"[1] test nulo         con poquito 0 el camino se mueve {mov:.1f}; una bola que ya está en el fondo da una "
          f"ventana de {w0:.1f}: {'sí' if mov == 0.0 and w0 == 0.0 else 'NO'}")
    if mov != 0.0 or w0 != 0.0:
        fallos.append("nulo: un camino quieto se mueve, o una bola en el fondo ensancha la ventana")

    # 2. SEÑAL — (b) y (c) de amplitud conocida (0,1 que se encoge y ±0,2 sin parar): la ventana del recuadro es
    #    MARGEN × 0,2 y los puntos del ir y venir de (c) están dentro.
    falso = _sintetico(datos, fondo)
    ax, dib, el, W = dibujar(falso, GRIS, "/dev/null")
    c = np.array(el["c"]["camino"])[:PASOS_C + 1]
    dentro = bool(np.all(np.abs(c[6:, 0] - fondo[0]) <= W + 1e-12))
    ok2 = abs(W - MARGEN * 0.2) < 1e-9 and dentro
    print(f"[2] señal             (c) de amplitud 0,2: la ventana del recuadro mide {W:.3f} de media anchura "
          f"(esperada {MARGEN * 0.2:.3f}) y el ir y venir cabe: {'sí' if ok2 else 'NO'}")
    if not ok2:
        fallos.append("señal: la ventana del recuadro no es la esperada o el ir y venir no cabe")

    # 3. INVARIANTE — la regla da las tres bolas de la Orden 202 (0,1; 1,9 y 2,1 veces 1/c), leídas de la tabla y no
    #    escritas a mano; las tres acaban donde dice la regla; de (c) se dibujan PASOS_C pasos; en las dos paletas.
    dibujar(datos, COLOR, "/dev/null")
    ax, dib, el, W = dibujar(datos, GRIS, "/dev/null")
    mults = tuple(el[k]["mult"] if el[k] else None for k in "abc")
    por_tabla = (all(abs(a - b) < 1e-12 for a, b in zip(mults, ESPERADAS)) and el["a"]["cruces"] == 0
                 and el["b"]["acaba"] == G.NOMBRES[0] and el["c"]["acaba"] == G.ACABA_SIN_ASENTARSE
                 and all(r["acaba"] != G.NOMBRES[0] for r in filas if r["mult"] > mults[1]))
    pasos_c = len(dib["c"][0]) - 1
    ok3 = por_tabla and pasos_c == PASOS_C
    print(f"[3] invariante        las tres bolas son {', '.join(G.coma_fija(m, 1) for m in mults)} veces 1/c (esperadas "
          f"{', '.join(G.coma_fija(m, 1) for m in ESPERADAS)}) y cumplen la regla sobre la tabla; pasos de (c) dibujados: "
          f"{pasos_c}; media anchura del recuadro: {G.coma_fija(W, 3)}: {'sí' if ok3 else 'NO'}")
    if not ok3:
        fallos.append("invariante: la regla no da las tres bolas esperadas, o no se dibujan los pasos de (c)")
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
    datos = leer()
    _, _, el, W = dibujar(datos, GRIS, str(AQUI / DESTINO))
    print(f"escrita {DESTINO}")
    copia = (AQUI / COPIA).resolve()
    copia.parent.mkdir(parents=True, exist_ok=True)
    dibujar(datos, GRIS, str(copia))
    print(f"escrita {copia}")
    for k in "abc":
        r = el[k]
        print(f"({k}) " + (f"{r['mult']:g}/c, poquito {r['poquito']:.4f}, {r['acaba']}" if r else "no existe en la rejilla"))
    print(f"pasos dibujados de (c): {PASOS_C}; media anchura del recuadro: {W:.3f}")


if __name__ == "__main__":
    main()
