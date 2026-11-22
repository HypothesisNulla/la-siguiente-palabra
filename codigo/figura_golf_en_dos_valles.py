#!/usr/bin/env python3
"""
Orden 201 (22 nov 2026): el golf, con dos pesos, en relieve (el terreno de los dos valles de `dos_valles.py`).

El mismo terreno inventado y la misma salida (la de la bola 2), con las tres bolas que elige la regla de
`golf_en_dos_valles.py` (fijada antes de medir, para que nadie escoja a ojo):
  (a) el primer poquito que acaba en el valle hondo sin cruzar el fondo («entra»);
  (b) el primer poquito que cruza el fondo al menos dos veces y acaba asentado en el valle hondo
      («se pasa y vuelve»);
  (c) el primer poquito que sale del cuadrado («sale del campo»).
Si alguna no existe en la rejilla, no se amplía: se dibujan las que haya.

Los caminos se leen de `datos/salidas/golf_en_dos_valles_datos.csv` (los escribe `golf_en_dos_valles.py`); la
elección se hace con las mismas funciones de ese programa sobre esos caminos. El relieve es el de
`figura_dos_valles_relieve.py`.

Uso:
    python figura_golf_en_dos_valles.py --selftest
    python figura_golf_en_dos_valles.py
"""

# ======================= CONSTANTES =======================

DATOS = "../datos/salidas/golf_en_dos_valles_datos.csv"
DESTINO = "../figuras/golf_en_dos_valles.png"
COPIA = "../../Claude outputs/reforma-nov/cap16/ver/golf_en_dos_valles.png"
ALTO = 4.4                      # pulgadas
MARCA_CADA = 10                 # en la bola (a), un punto cada tantos pasos, mientras no ha llegado al fondo (a más de EPS_MARCA de él)
EPS_MARCA = 1e-2
MARCAS_B = 12                   # en la bola (b), un punto en cada uno de los primeros golpes
TEXTOS = {"a": "entra", "b": "se pasa y vuelve", "c": "sale del campo"}

# ==========================================================

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from dos_valles import LIMITE, SALIDAS, altura
import golf_en_dos_valles as G
from figura_dos_valles_relieve import CAJA, dibujar_relieve, recortar
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


def poq(x):
    return f"{x:.3g}".replace(".", ",")


def dibujar(datos, paleta, ruta):
    filas, fondo = filas_de(datos)
    el = G.elegir(filas)
    cuantos = {1: "un poquito", 2: "dos poquitos", 3: "tres poquitos"}[sum(1 for k in "abc" if el[k] is not None)]
    L = Lienzo("El golf, con dos pesos",
               f"El mismo terreno inventado y la misma salida, con {cuantos}.",
               paleta, alto=ALTO)
    p = L.p
    caminos, etiquetas, estilos = {}, {}, {}
    desp = {"a": (-1.2, 1.4, 1.2), "b": (1.3, 1.1, 1.3), "c": (0.0, 0.0, 0.0)}
    frac = {"a": 0.45, "b": 0.40, "c": 0.0}
    for k in "abc":
        r = el[k]
        if r is None:
            continue
        c = np.array(r["camino"])
        caminos[k] = c
        etiquetas[k] = {"texto": f"{TEXTOS[k]}\n(poquito {poq(r['poquito'])})", "frac": frac[k], "desp": desp[k]}
    marcas_a = []
    if el["a"] is not None:
        ca = np.array(el["a"]["camino"])
        da = np.hypot(ca[:, 0] - fondo[0], ca[:, 1] - fondo[1])
        marcas_a = [i for i in range(MARCA_CADA, len(ca), MARCA_CADA) if da[i] > EPS_MARCA]
    estilos["a"] = {"color": p.suave, "grosor": 1.8, "marcas": marcas_a,
                    "marcador": "s", "relleno": p.suave}
    estilos["b"] = {"color": p.tinta, "grosor": 1.2, "marcas": list(range(1, MARCAS_B)), "marcador": "o", "relleno": "white"}
    estilos["c"] = {"color": p.tinta, "grosor": 1.2, "marcas": []}
    ax, dib = dibujar_relieve(L.fig, CAJA, paleta, caminos, etiquetas, valles=True, estilos=estilos)
    L.guardar(ruta)
    return ax, dib, el


def selftest():
    fallos = []
    datos = leer()

    # 1. NULO — con poquito 0 el camino no se mueve: se dibuja un solo punto, sin salida.
    quieto, _ = G.recorrer(SALIDAS[1], 0.0, pasos=30)
    pts, sal = recortar(np.array(quieto))
    mov = float(np.max(np.hypot(pts[:, 0] - pts[0, 0], pts[:, 1] - pts[0, 1])))
    print(f"[1] test nulo         con poquito 0 el camino dibujado se mueve {mov:.1f} y no tiene salida: "
          f"{'sí' if mov == 0.0 and sal is None else 'NO'}")
    if mov != 0.0 or sal is not None:
        fallos.append("nulo: un camino quieto se mueve o sale")

    # 2. SEÑAL — con un poquito enorme el peso sale del cuadrado: sale la bola (c), el camino se recorta en el
    #    borde y la figura pone «sale del campo».
    c, _, fondo = G.curvatura()
    cam, paro = G.recorrer(SALIDAS[1], 80.0 / c, pasos=300)
    falso = dict(datos)
    falso[80.0] = (80.0 / c, np.array(cam))
    ax, dib, el = dibujar(falso, GRIS, "/dev/null")
    textos = [t.get_text() for t in ax.texts]
    sale = dib.get("c", (None, None))[1]
    fin = dib["c"][0][-1]
    en_borde = abs(abs(fin[0]) - LIMITE) < 1e-9 or abs(abs(fin[1]) - LIMITE) < 1e-9
    ok2 = (el["c"] is not None and abs(el["c"]["mult"] - 80.0) < 1e-12 and sale is not None and en_borde
           and any("sale del campo" in t for t in textos))
    print(f"[2] señal             un poquito de 80/c sale del cuadrado: la bola (c) existe: "
          f"{'sí' if el['c'] is not None else 'NO'}; se recorta en el borde ({dib['c'][0][-1, 0]:.2f}; {dib['c'][0][-1, 1]:.2f}) "
          f"y lleva «sale del campo»: {'sí' if ok2 else 'NO'}")
    if not ok2:
        fallos.append("señal: el camino que sale no se recorta en el borde o no se marca")

    # 3. INVARIANTE — la elección es la de la regla sobre los caminos del CSV; cada bola termina en el fondo del
    #    valle hondo (la que termina ahí); sale en las dos paletas.
    dibujar(datos, COLOR, "/dev/null")
    ax, dib, el = dibujar(datos, GRIS, "/dev/null")
    filas, fondo = filas_de(datos)
    a_ok = el["a"] is not None and abs(el["a"]["mult"] - 0.1) < 1e-12
    b_ok = el["b"] is not None and abs(el["b"]["mult"] - 1.2) < 1e-12
    c_no = el["c"] is None
    fin_a = np.array(el["a"]["camino"][-1][:2]); fin_b = np.array(el["b"]["camino"][-1][:2])
    en_fondo = float(np.hypot(*(fin_a - np.array(fondo)))) < 1e-3 and float(np.hypot(*(fin_b - np.array(fondo)))) < 1e-3
    ok3 = a_ok and b_ok and c_no and en_fondo
    print(f"[3] invariante        (a) = {G.coma_fija(el['a']['mult'], 1)}/c, (b) = {G.coma_fija(el['b']['mult'], 1)}/c, (c) "
          f"{'no existe' if c_no else 'existe'}; (a) y (b) acaban a menos de 0,001 del fondo ({G.coma_fija(fondo[0], 4)}; "
          f"{G.coma_fija(fondo[1], 4)}): {'sí' if en_fondo else 'NO'}: {'sí' if ok3 else 'NO'}")
    if not ok3:
        fallos.append("invariante: la elección o el final de las bolas no es la esperada")
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
    _, _, el = dibujar(datos, GRIS, str(AQUI / DESTINO))
    print(f"escrita {DESTINO}")
    copia = (AQUI / COPIA).resolve()
    copia.parent.mkdir(parents=True, exist_ok=True)
    dibujar(datos, GRIS, str(copia))
    print(f"escrita {copia}")
    for k in "abc":
        r = el[k]
        print(f"({k}) " + (f"{r['mult']:g}/c, poquito {r['poquito']:.4f}, {r['acaba']}" if r else "no existe en la rejilla"))


if __name__ == "__main__":
    main()
