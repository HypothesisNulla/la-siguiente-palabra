#!/usr/bin/env python3
"""
Orden 204 (22 nov 2026, sustituye al dibujo de las Órdenes 201 a 203): el golf, con dos pesos, dibujado de forma
conceptual, como tres greens vistos desde arriba.

Tres paneles iguales, uno al lado del otro. Cada uno es el mismo trozo del terreno inventado de `dos_valles.py` visto
desde arriba, en curvas de nivel grises: la zona del valle hondo, con la salida dentro. En el fondo, un hoyo con su
banderita. En cada panel, un solo camino, el de una de las tres bolas, con los mismos datos del CSV: una línea negra
continua con una flechita pequeña al principio para indicar el sentido, un punto blanco en la salida y ninguna marca en
cada paso. Ningún número en la figura (ni poquitos, ni «veces», ni ejes con cifras).

Las tres bolas son las de la Orden 202 (la regla sigue siendo la misma, sobre la tabla de `golf_en_dos_valles.txt`, que no
cambia):
  (a) «entra»: el primer poquito que acaba en el valle hondo sin cruzar el fondo;
  (b) «se pasa y vuelve»: el último poquito que todavía acaba asentado en el valle hondo;
  (c) «no para nunca» («no se asienta nunca» en la regla): el primero que acaba yendo y viniendo sin asentarse.
Salen 0,1, 1,9 y 2,1 veces 1/c. La elección la hace `golf_en_dos_valles.elegir` sobre los caminos leídos de
`datos/salidas/golf_en_dos_valles_datos.csv`.

Orden 205 (segunda fila): debajo, tres paneles más, «el hoyo, de cerca»: el fondo del valle hondo ampliado, con el mismo zoom en
los tres, el hoyo y su banderita. En cada uno, la bola de su columna desde que entra en esa ventana, con cada golpe dibujado como un
arco (el vuelo de la bola), del sitio donde cae al siguiente. Los arcos de un mismo panel van de más oscuros a más claros, para
que se vea el orden sin numerarlos. Para que la ida y la vuelta de (b) y (c) no se pisen se dibuja un esquema: cada posición, a partir
de la primera vez que la bola pasa al otro lado del hoyo, se corre SEPARACION en perpendicular a la línea por la que va y viene,
siempre hacia el mismo lado y acumulando uno por golpe; lo que cae dentro del hoyo no se corre. El desplazamiento se cuenta hacia atrás desde el último golpe (HACIA_EL_HOYO), para que la cadena termine en el hoyo. El desplazamiento es solo de
dibujo (con SEPARACION = 0 los arcos se pisan; las posiciones medidas no se tocan). El zoom sale de una regla: MARGEN_ZOOM veces
la mayor distancia al fondo de los arcos de (b) y (c) desde que entran a menos de ENTRADA_CERCA del fondo. Se dibujan:
  (a) los golpes desde que entra hasta que cae en el hoyo;
  (b) todos hasta que un golpe mide menos que el hoyo dibujado;
  (c) GOLPES_C_CERCA golpes desde que entra.

Los tres paneles enseñan lo mismo, y solo cambia la fuerza del golpe. De (c) se dibujan los PASOS_C primeros pasos; (a) y (b),
enteros. La ventana de cada panel es un cuadrado que contiene la salida y el fondo con un margen (MARGEN_VENTANA); no se
amplía ni se exagera ninguna parte del dibujo (el rebasamiento de (b) es lo que es a esa escala).

Uso:
    python figura_golf_en_dos_valles.py --selftest
    python figura_golf_en_dos_valles.py
"""

# ======================= CONSTANTES =======================

DATOS = "../datos/salidas/golf_en_dos_valles_datos.csv"
DESTINO = "../figuras/golf_en_dos_valles.png"
COPIA = "../../Claude outputs/reforma-nov/cap16/ver/golf_en_dos_valles.png"
ALTO = 4.6                      # pulgadas
PASOS_C = 12                    # de la bola (c) se dibujan los primeros pasos (probados 8, 14, 24 y 40: con 24 o más el ir y venir se vuelve una mancha negra)
MARGEN_VENTANA = 0.55           # lo que se añade a cada lado de la caja que contiene la salida y el fondo (unidades del terreno)
NIVELES = 10                    # curvas de nivel
PUNTOS = 220                    # puntos por lado de la rejilla de cada panel
FLECHA_DESDE, FLECHA_HASTA = 0.18, 0.52   # la flechita del principio: de aquí a aquí, medido a lo largo del camino
RADIO_HOYO = 0.085              # el hoyo
ALTO_PALO = 0.62                # el palo de la banderita (unidades del terreno)
TEXTOS = {"a": ("golpe suave", "entra"),
          "b": ("golpe fuerte", "se pasa y vuelve"),
          "c": ("golpe demasiado fuerte", "no para nunca")}
ESPERADAS = (0.1, 1.9, 2.1)     # las tres bolas que dice la Orden 202 (veces 1/c); el invariante comprueba que la regla las da
# --- Orden 205: la segunda fila, «el hoyo, de cerca»
TITULO_CERCA = "el hoyo, de cerca"
HACIA_EL_HOYO = True            # el desplazamiento se cuenta desde el último golpe (el que cae en el hoyo, sin desplazar) hacia atrás: la cadena
                                # de golpes termina en el hoyo y no se va al lado (contado desde el primer cruce, el final de (b) se alejaba
                                # 0,34 del hoyo; probado y descartado)
SEPARACION = 0.02               # cuánto se corre, en perpendicular y por golpe, cada posición (unidades del terreno); el mismo en los tres paneles
ALTURA_ARCO = 0.30              # la altura de cada arco, como fracción de su longitud
ENTRADA_CERCA = 0.5             # «entra en la ventana»: a menos de esto del fondo
MARGEN_ZOOM = 1.25              # el zoom: media anchura = esto × la mayor distancia al fondo de los arcos de (b) y (c) tras entrar
RADIO_HOYO_CERCA = 0.04         # el hoyo de cerca (unidades del terreno)
GOLPES_C_CERCA = 12             # de (c), los golpes que se dibujan desde que entra
GRIS_ARCOS = (0.0, 0.62)        # el gris de los arcos: del primero (más oscuro) al último (más claro)
EPS_CRUCE = 1e-6                # como en golf_en_dos_valles.py: a menos de esto del fondo no se cuenta un cruce
PUNTOS_ARCO = 60                # puntos con que se dibuja cada arco
TOLERANCIA_PISAR = 0.01         # dos arcos «se pisan» si un punto de uno está a menos de esto del otro

# ==========================================================

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import Circle, Polygon

from dos_valles import altura
import golf_en_dos_valles as G
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


def ventana(salida, fondo):
    """El cuadrado de cada panel: (x0, x1, y0, y1), centrado entre la salida y el fondo, que los contiene con MARGEN_VENTANA."""
    cx, cy = (salida[0] + fondo[0]) / 2, (salida[1] + fondo[1]) / 2
    media = max(abs(salida[0] - fondo[0]), abs(salida[1] - fondo[1])) / 2 + MARGEN_VENTANA
    return cx - media, cx + media, cy - media, cy + media


def punto_a_lo_largo(c, largo):
    """El punto del camino c (array [(x, y, ...)]) que está a `largo` de la salida, medido a lo largo del camino."""
    d = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(c[:, 0]), np.diff(c[:, 1])))])
    k = int(np.argmin(np.abs(d - largo)))
    return c[k, 0], c[k, 1]


def panel(fig, caja, paleta, camino, fondo, ventana_, textos):
    """Un panel: el terreno desde arriba en curvas de nivel grises, el hoyo con su banderita, el camino y la salida.
    Devuelve (eje, linea_del_camino)."""
    p = paleta
    ax = fig.add_axes(caja)
    x0, x1, y0, y1 = ventana_
    gx, gy = np.linspace(x0, x1, PUNTOS), np.linspace(y0, y1, PUNTOS)
    X, Y = np.meshgrid(gx, gy)
    Z = np.vectorize(altura)(X, Y)
    ax.contourf(X, Y, Z, levels=NIVELES, cmap="Greys_r", vmin=Z.min() - 0.6 * (Z.max() - Z.min()),
                vmax=Z.max() + 0.15 * (Z.max() - Z.min()))
    ax.contour(X, Y, Z, levels=NIVELES, colors=["white"], linewidths=0.35)
    # el hoyo, con su banderita (un palo y un banderín)
    fx, fy = fondo
    ax.add_patch(Circle((fx, fy), RADIO_HOYO, facecolor="black", edgecolor="white", linewidth=0.8, zorder=6))
    ax.plot([fx, fx], [fy, fy + ALTO_PALO], color="black", linewidth=0.9, zorder=6)
    ax.add_patch(Polygon([[fx, fy + ALTO_PALO], [fx + 0.30, fy + ALTO_PALO - 0.10], [fx, fy + ALTO_PALO - 0.20]],
                         closed=True, facecolor="black", edgecolor="black", linewidth=0.4, zorder=6))
    # el camino: una línea negra continua, sin marcas en cada paso, con una flechita al principio y un punto blanco en la salida
    c = np.asarray(camino, float)
    (linea,) = ax.plot(c[:, 0], c[:, 1], color="black", linewidth=1.1, zorder=5, solid_joinstyle="round")
    if len(c) > 1 and np.hypot(np.diff(c[:, 0]), np.diff(c[:, 1])).sum() > FLECHA_HASTA:
        a, b = punto_a_lo_largo(c, FLECHA_DESDE), punto_a_lo_largo(c, FLECHA_HASTA)
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color="black", lw=1.1, mutation_scale=7,
                                                        shrinkA=0, shrinkB=0), zorder=7)
    ax.plot([c[0, 0]], [c[0, 1]], "o", markersize=5.2, markerfacecolor="white", markeredgecolor="black",
            markeredgewidth=0.9, zorder=8)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_xticks([]); ax.set_yticks([])                 # sin ejes
    for s in ax.spines.values():
        s.set_color(p.tinta); s.set_linewidth(0.8)
    ax.text(0.5, -0.04, textos[0], transform=ax.transAxes, ha="center", va="top", fontsize=7.6, fontweight="bold",
            color=p.tinta)
    ax.text(0.5, -0.155, textos[1], transform=ax.transAxes, ha="center", va="top", fontsize=7.6, color=p.tinta)
    return ax, linea


def golpes_de_cerca(camino, fondo, tipo, separacion, hacia_el_hoyo=HACIA_EL_HOYO):
    """Los golpes que se dibujan de cerca para una bola: un dict con
      'idx'      los índices de las posiciones, desde la última antes de entrar hasta donde acaba el dibujo;
      'real'     esas posiciones, tal como las midió el programa (n×2);
      'desp'     el desplazamiento de dibujo que se les aplica (n×2): cero antes del primer cruce y dentro del hoyo;
      'dibujada' real + desp;
      'normal'   el vector perpendicular, siempre del mismo lado, en el que se corren;
      'entra'    el índice de la primera posición a menos de ENTRADA_CERCA del fondo."""
    c = np.asarray(camino, float)
    d = np.hypot(c[:, 0] - fondo[0], c[:, 1] - fondo[1])
    dentro = np.nonzero(d < ENTRADA_CERCA)[0]
    if len(dentro) == 0:
        return None
    k0 = int(dentro[0])
    paso = np.hypot(np.diff(c[:, 0]), np.diff(c[:, 1]))
    if tipo == "a":          # hasta que cae en el hoyo
        en = np.nonzero(d[k0:] < RADIO_HOYO_CERCA)[0]
        e = k0 + int(en[0]) if len(en) else len(c) - 1
    elif tipo == "b":        # hasta que un golpe mide menos que el hoyo dibujado
        corto = np.nonzero(paso[k0:] < 2 * RADIO_HOYO_CERCA)[0]
        e = k0 + int(corto[0]) if len(corto) else len(c) - 1
    else:                    # (c): unos cuantos
        e = min(k0 + GOLPES_C_CERCA, len(c) - 1)
    idx = list(range(max(k0 - 1, 0), e + 1))
    real = c[idx, :2]
    # la línea por la que va y viene (la de los primeros golpes desde que entra) y su perpendicular, del lado de abajo
    u = c[min(k0 + 1, len(c) - 1), :2] - c[k0, :2]
    if np.hypot(*u) < 1e-12:
        u = c[k0, :2] - c[max(k0 - 1, 0), :2]
    u = u / max(np.hypot(*u), 1e-12)
    n = np.array([u[1], -u[0]])
    if n[1] > 1e-9 or (abs(n[1]) <= 1e-9 and n[0] > 0):
        n = -n                                  # siempre hacia abajo (la banderita va hacia arriba)
    # cuántas veces ha cruzado el fondo hasta cada posición
    cruces = np.zeros(len(idx), int)
    cuenta = 0
    for m in range(1, len(idx)):
        a, b = real[m - 1] - fondo, real[m] - fondo
        if np.hypot(*a) > EPS_CRUCE and np.hypot(*b) > EPS_CRUCE and a @ b < 0:
            cuenta += 1
        cruces[m] = cuenta
    dist = np.hypot(real[:, 0] - fondo[0], real[:, 1] - fondo[1])
    if hacia_el_hoyo:
        cruces = cruces[-1] - cruces
    desp = np.outer(cruces * separacion, n)
    desp[dist <= RADIO_HOYO_CERCA] = 0.0          # lo que cae dentro del hoyo, sin desplazar
    return {"idx": idx, "real": real, "desp": desp, "dibujada": real + desp, "normal": n, "entra": k0}


def arco(a, b, normal, altura_arco=ALTURA_ARCO, n=PUNTOS_ARCO):
    """El arco de a a b: una curva de Bézier con el punto de control a media distancia, subido altura_arco × la longitud,
    siempre del lado de `normal` (el vuelo de una bola de golf visto desde arriba, hecho esquema)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ctrl = (a + b) / 2 + altura_arco * np.hypot(*(b - a)) * np.asarray(normal)
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) ** 2 * a + 2 * (1 - t) * t * ctrl + t ** 2 * b


def arcos_de(g):
    """Los arcos de un golpe a otro: lista de arrays (PUNTOS_ARCO × 2)."""
    return [arco(a, b, g["normal"]) for a, b in zip(g["dibujada"][:-1], g["dibujada"][1:])]


def media_anchura_zoom(golpes, fondo):
    """La media anchura del zoom: MARGEN_ZOOM × la mayor distancia al fondo de los arcos de (b) y (c) desde que entran (sin
    contar el golpe que entra, que viene de fuera)."""
    m = 0.0
    for k in "bc":
        g = golpes[k]
        arcos = arcos_de(g)[1:] if g["idx"][0] < g["entra"] else arcos_de(g)
        for a in arcos:
            m = max(m, float(np.hypot(a[:, 0] - fondo[0], a[:, 1] - fondo[1]).max()))
    return MARGEN_ZOOM * m


def fraccion_pisada(arcos, tolerancia=TOLERANCIA_PISAR):
    """Qué fracción de los puntos de cada arco (salvo el primero, y sin contar los que están pegados a su propio punto de
    salida) cae a menos de `tolerancia` de algún arco anterior. Con arcos que se pisan es alta; con arcos separados, baja."""
    if len(arcos) < 2:
        return 0.0
    total = pisados = 0
    for m in range(1, len(arcos)):
        previos = np.vstack(arcos[:m])
        for pt in arcos[m]:
            if np.hypot(*(pt - arcos[m][0])) < 3 * tolerancia:
                continue
            total += 1
            if np.min(np.hypot(previos[:, 0] - pt[0], previos[:, 1] - pt[1])) < tolerancia:
                pisados += 1
    return pisados / max(total, 1)


def panel_cerca(fig, caja, paleta, golpes, fondo, media):
    """El hoyo de cerca: el fondo del valle hondo ampliado, en curvas de nivel grises (más claras que arriba, para que se
    lean los arcos), el hoyo con su banderita y los golpes de la bola como arcos de más oscuros a más claros.
    Devuelve (eje, lista_de_lineas_de_los_arcos)."""
    p = paleta
    ax = fig.add_axes(caja)
    gx = np.linspace(fondo[0] - media, fondo[0] + media, PUNTOS)
    gy = np.linspace(fondo[1] - media, fondo[1] + media, PUNTOS)
    X, Y = np.meshgrid(gx, gy)
    Z = np.vectorize(altura)(X, Y)
    ax.contourf(X, Y, Z, levels=NIVELES, cmap="Greys_r", vmin=Z.min() - 1.4 * (Z.max() - Z.min()),
                vmax=Z.max() + 0.25 * (Z.max() - Z.min()))
    ax.contour(X, Y, Z, levels=NIVELES, colors=["white"], linewidths=0.35)
    fx, fy = fondo
    ax.add_patch(Circle((fx, fy), RADIO_HOYO_CERCA, facecolor="black", edgecolor="white", linewidth=0.8, zorder=6))
    palo = 0.62 * media
    ax.plot([fx, fx], [fy, fy + palo], color="black", linewidth=0.9, zorder=6)
    ax.add_patch(Polygon([[fx, fy + palo], [fx + 0.30 * media, fy + palo - 0.10 * media], [fx, fy + palo - 0.20 * media]],
                         closed=True, facecolor="black", edgecolor="black", linewidth=0.4, zorder=6))
    lineas = []
    arcos = arcos_de(golpes)
    for m, a in enumerate(arcos):
        g = GRIS_ARCOS[0] + (GRIS_ARCOS[1] - GRIS_ARCOS[0]) * m / max(len(arcos) - 1, 1)
        (ln,) = ax.plot(a[:, 0], a[:, 1], color=(g, g, g), linewidth=1.0, zorder=5, solid_capstyle="round")
        lineas.append(ln)
    ax.set_xlim(fx - media, fx + media); ax.set_ylim(fy - media, fy + media)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(p.tinta); s.set_linewidth(0.8)
    return ax, lineas


def dibujar(datos, paleta, ruta, separacion=SEPARACION):
    """Dibuja la figura. Devuelve (ejes, lineas, bolas_elegidas, ventana, figura, cerca) donde cerca es un dict con lo de
    la segunda fila: {'golpes', 'media', 'ejes', 'lineas'}."""
    filas, fondo = filas_de(datos)
    el = G.elegir(filas)
    L = Lienzo("El golf, con dos pesos",
               "El mismo terreno inventado, visto desde arriba, y la misma salida; lo único\n"
               "que cambia es la fuerza del golpe. Abajo, el hoyo de cerca: cada arco es un golpe,\n"
               "dibujados separados para que se vean, aunque van y vuelven por la misma línea.",
               paleta, alto=ALTO)
    salida = tuple(np.array(el["a"]["camino"][0][:2]))
    v = ventana(salida, fondo)
    lado, hueco, margen = 0.285, 0.03, 0.05
    alto_frac = lado * 4.45 / ALTO
    y_fila2 = 0.12 / ALTO                          # en pulgadas: margen de abajo 0,12; luego la fila 2, su título y los rótulos de la fila 1
    y_fila1 = y_fila2 + alto_frac + (0.22 + 0.50) / ALTO
    ejes, lineas = {}, {}
    for i, k in enumerate("abc"):
        c = np.array(el[k]["camino"])
        if k == "c":
            c = c[:PASOS_C + 1]
        caja = [margen + i * (lado + hueco), y_fila1, lado, alto_frac]
        ejes[k], lineas[k] = panel(L.fig, caja, paleta, c, fondo, v, TEXTOS[k])
    # la segunda fila
    golpes = {k: golpes_de_cerca(el[k]["camino"], fondo, k, separacion) for k in "abc"}
    media = media_anchura_zoom(golpes, fondo)
    L.fig.text(margen, y_fila2 + alto_frac + 0.04 / ALTO, TITULO_CERCA, fontsize=7.8, fontweight="bold", color=paleta.tinta,
               ha="left", va="bottom")
    ej2, ln2 = {}, {}
    for i, k in enumerate("abc"):
        caja = [margen + i * (lado + hueco), y_fila2, lado, alto_frac]
        ej2[k], ln2[k] = panel_cerca(L.fig, caja, paleta, golpes[k], fondo, media)
    L.guardar(ruta)
    return ejes, lineas, el, v, L.fig, {"golpes": golpes, "media": media, "ejes": ej2, "lineas": ln2}


def textos_de(fig):
    """Todos los textos de la figura: título, subtítulo, rótulos de los paneles y marcas de los ejes."""
    sal = []
    for ax in fig.axes:
        sal += [t.get_text() for t in ax.texts]
        if ax.axison:      # el eje del Lienzo (título y subtítulo) está apagado: sus marcas no se dibujan
            sal += [t.get_text() for t in ax.get_xticklabels() + ax.get_yticklabels()]
        sal += [ax.get_title()]
    sal += [t.get_text() for t in fig.texts]
    return [t for t in sal if t]


def _sintetico(datos):
    """Una copia de los datos con (c) sustituida por un camino que salta ±0,2 sin parar a un lado y otro del fondo."""
    _, _, fondo = G.curvatura()
    falso = dict(datos)
    ini = np.array(datos[2.1][1][0])
    fx, fy = fondo
    c = [tuple(ini)] + [(fx + 0.2 * (-1) ** k, fy, altura(fx + 0.2 * (-1) ** k, fy)) for k in range(1, G.PASOS_FIJOS + 1)]
    falso[2.1] = (datos[2.1][0], np.array(c))
    return falso


def selftest():
    fallos = []
    datos = leer()
    filas, fondo = filas_de(datos)

    # 1. NULO — con poquito 0 el peso no se mueve: la línea de arriba no tiene longitud, y de cerca no hay ningún golpe
    #    (la bola no entra en ninguna ventana: está en la salida).
    quieto, _ = G.recorrer(G.SALIDA, 0.0, pasos=30)
    cq = np.array(quieto)
    L = Lienzo("t", "", GRIS, alto=ALTO)
    ax, linea = panel(L.fig, [0.1, 0.2, 0.3, 0.4], GRIS, cq, fondo, ventana(tuple(cq[0, :2]), fondo), ("a", "b"))
    xs, ys = np.asarray(linea.get_xdata(), float), np.asarray(linea.get_ydata(), float)
    largo = float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))
    sin_golpes = golpes_de_cerca(cq, fondo, "b", SEPARACION) is None
    print(f"[1] test nulo         con poquito 0 el camino dibujado mide {largo:.1f} de largo y de cerca no hay golpes: "
          f"{'sí' if largo == 0.0 and sin_golpes else 'NO'}")
    if largo != 0.0 or not sin_golpes:
        fallos.append("nulo: un camino quieto tiene longitud o golpes de cerca")

    # 2. SEÑAL — con (c) sustituido por un camino que salta ±0,2 a cada lado del fondo, la línea del tercer panel llega
    #    a 0,2 del fondo por los dos lados, y no más; y de cerca los golpes de (c) son GOLPES_C_CERCA, los arcos unen
    #    esas posiciones, y se desplazan SEPARACION por golpe (en la dirección perpendicular).
    ejes, lineas, el, v, fig, cerca = dibujar(_sintetico(datos), GRIS, "/dev/null")
    xs = np.asarray(lineas["c"].get_xdata(), float)
    dx = xs[6:] - fondo[0]
    gc = cerca["golpes"]["c"]
    desp_c = np.hypot(gc["desp"][:, 0], gc["desp"][:, 1])
    # cada golpe cruza el fondo: el desplazamiento cambia SEPARACION por golpe (cuando la posición no cae en el hoyo)
    pasos_desp = np.diff(desp_c[1:])
    ok2 = (abs(dx.max() - 0.2) < 1e-9 and abs(dx.min() + 0.2) < 1e-9 and len(xs) == PASOS_C + 1
           and len(gc["idx"]) - 1 >= GOLPES_C_CERCA - 1 and np.allclose(np.abs(pasos_desp), SEPARACION, atol=1e-12))
    print(f"[2] señal             (c) que salta ±0,2: la línea llega a {dx.min():+.2f} y {dx.max():+.2f} del fondo "
          f"(esperado ∓0,20); de cerca, {len(gc['idx']) - 1} golpes, cada uno desplazado {SEPARACION:g} menos que el anterior (cuenta atrás hasta el hoyo): "
          f"{'sí' if ok2 else 'NO'}")
    if not ok2:
        fallos.append("señal: el ir y venir de prueba no sale a ±0,2 del fondo, o el desplazamiento no es SEPARACION por golpe")

    # 3. INVARIANTE — la regla da las tres bolas de la Orden 202 (0,1; 1,9 y 2,1 veces 1/c), leídas de la tabla; cada
    #    línea de arriba es el camino del CSV; la figura no contiene ningún texto con cifras ni ejes con marcas; los
    #    extremos de cada arco coinciden con las posiciones del CSV una vez quitado el desplazamiento; el zoom es el mismo
    #    en los tres paneles; con SEPARACION = 0 los arcos de (b) se pisan; y sale en las dos paletas.
    dibujar(datos, COLOR, "/dev/null")
    ejes, lineas, el, v, fig, cerca = dibujar(datos, GRIS, "/dev/null")
    mults = tuple(el[k]["mult"] if el[k] else None for k in "abc")
    por_tabla = (all(abs(a - b) < 1e-12 for a, b in zip(mults, ESPERADAS)) and el["a"]["cruces"] == 0
                 and el["b"]["acaba"] == G.NOMBRES[0] and el["c"]["acaba"] == G.ACABA_SIN_ASENTARSE
                 and all(r["acaba"] != G.NOMBRES[0] for r in filas if r["mult"] > mults[1]))
    csv_ok = True
    extremos_ok = True
    for k in "abc":
        ref = np.array(datos[el[k]["mult"]][1])
        r = ref[:PASOS_C + 1] if k == "c" else ref
        xs, ys = np.asarray(lineas[k].get_xdata(), float), np.asarray(lineas[k].get_ydata(), float)
        csv_ok &= len(xs) == len(r) and np.allclose(xs, r[:, 0], atol=1e-12) and np.allclose(ys, r[:, 1], atol=1e-12)
        g = cerca["golpes"][k]
        # los extremos dibujados de cada arco, menos el desplazamiento, son las posiciones del CSV
        arcos = arcos_de(g)
        for m, a in enumerate(arcos):
            ini = a[0] - g["desp"][m]
            fin = a[-1] - g["desp"][m + 1]
            extremos_ok &= bool(np.allclose(ini, ref[g["idx"][m], :2], atol=1e-12) and np.allclose(fin, ref[g["idx"][m + 1], :2], atol=1e-12))
    con_cifras = [t for t in textos_de(fig) if re.search(r"\d", t)]
    marcas = [t for a in fig.axes if a.axison for t in a.get_xticks().tolist() + a.get_yticks().tolist()]
    zoom = {k: (cerca["ejes"][k].get_xlim(), cerca["ejes"][k].get_ylim()) for k in "abc"}
    mismo_zoom = all(zoom[k] == zoom["a"] for k in "bc")
    # con SEPARACION = 0 los arcos de (b) se pisan (y con SEPARACION, mucho menos)
    _, _, _, _, _, cero = dibujar(datos, GRIS, "/dev/null", separacion=0.0)
    p_con = fraccion_pisada(arcos_de(cerca["golpes"]["b"]))
    p_cero = fraccion_pisada(arcos_de(cero["golpes"]["b"]))
    pisan = p_cero > 0.5 and p_con < 0.5 * p_cero
    ok3 = por_tabla and csv_ok and extremos_ok and not con_cifras and not marcas and mismo_zoom and pisan
    print(f"[3] invariante        las tres bolas son {', '.join(G.coma_fija(m, 1) for m in mults)} veces 1/c (esperadas "
          f"{', '.join(G.coma_fija(m, 1) for m in ESPERADAS)}) y cumplen la regla sobre la tabla; las líneas de arriba son "
          f"los caminos del CSV: {'sí' if csv_ok else 'NO'}; los extremos de los arcos, sin el desplazamiento, son las "
          f"posiciones del CSV: {'sí' if extremos_ok else 'NO'}; textos con cifras: {len(con_cifras)}; ejes con marcas: "
          f"{len(marcas)}; el mismo zoom en los tres: {'sí' if mismo_zoom else 'NO'}; fracción de puntos de los arcos de (b) "
          f"que pisan a los anteriores: {G.coma_fija(p_cero, 2)} con SEPARACION = 0 y {G.coma_fija(p_con, 2)} con "
          f"{SEPARACION:g}: {'sí' if ok3 else 'NO'}")
    if not ok3:
        fallos.append("invariante: la regla no da las bolas esperadas, o las líneas, los extremos, las cifras, el zoom o el pisarse fallan")
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
    _, _, el, v, _, cerca = dibujar(datos, GRIS, str(AQUI / DESTINO))
    print(f"escrita {DESTINO}")
    copia = (AQUI / COPIA).resolve()
    copia.parent.mkdir(parents=True, exist_ok=True)
    dibujar(datos, GRIS, str(copia))
    print(f"escrita {copia}")
    for k in "abc":
        r = el[k]
        g = cerca["golpes"][k]
        print(f"({k}) " + (f"{r['mult']:g}/c, poquito {r['poquito']:.4f}, {r['acaba']}" if r else "no existe en la rejilla")
              + f"; golpes de cerca dibujados: {len(g['idx']) - 1} (entra en el paso {g['entra']})")
    print(f"pasos dibujados de (c) arriba: {PASOS_C}; media anchura del zoom: {cerca['media']:.3f}")


if __name__ == "__main__":
    main()
