#!/usr/bin/env python3
"""
Orden 210 (23 nov 2026) y Orden 212 (23 nov 2026): «Tareas cada vez más largas». El horizonte temporal de METR, un punto por máquina.

Los datos son los de `horizonte_metr.py` (que lee el fichero de resultados de METR, TH1.1, guardado en `datos/entrada/`):
nada se escribe a mano. En horizontal, la fecha de salida de cada máquina; en vertical, la tarea más larga que termina bien la
mitad de las veces, medida en lo que le costaría a una persona experta, en escala logarítmica. Orden 212: el eje no lleva
«minutos» (quien mira deprisa lee que la máquina tarda más): lleva un rótulo propio, «la tarea más larga que termina», con una
flecha hacia arriba y «más capaz», y marcas en tiempo de una persona en su sitio exacto de la escala: una tarea de 10 segundos,
de 1 minuto, de 10 minutos, de 1 hora y de una jornada (8 horas).

Solo se dibujan las máquinas de la frontera (las que marcan el récord en su fecha) y, con el mismo trazo, la de más de 16 horas
(Claude Mythos Preview), que lleva un borde. Los modelos que no batieron el récord no se dibujan. La recta fina es el ajuste de
`horizonte_metr.py`: modelos de la frontera, sin los de más de 16 horas, todo el periodo (como hace METR). Con nombre, solo
GPT-2 y GPT-4, y el último. Grises y sin leyenda.

Uso:
    python figura_horizonte_metr.py --selftest
    python figura_horizonte_metr.py
"""

# ======================= CONSTANTES =======================

DESTINO = "../figuras/horizonte_metr.png"
COPIA = "../../Claude outputs/reforma-nov/epilogo/ver/horizonte_metr.png"
ALTO = 4.6                      # pulgadas
TITULO = "Tareas cada vez más largas"
SUBTITULO = ("Cada punto es una máquina, por la fecha en que salió. Más arriba, la tarea\n"
             "más larga que termina bien la mitad de las veces, medida en lo que le\n"
             "costaría a una persona experta. Más arriba es más capaz. Datos de METR.")
ETIQUETA_Y = "la tarea más larga que termina"
FLECHA_Y = "más capaz"
CON_NOMBRE = ("gpt2", "gpt_4", "claude_mythos_preview_early_inspect")     # el primero, uno de en medio y el último
ROTULOS = {"gpt2": "GPT-2", "gpt_4": "GPT-4", "claude_mythos_preview_early_inspect": "Claude Mythos Preview,\nabril de 2026"}
# dónde va el rótulo de cada modelo con nombre, respecto a su punto: (desplazamiento x, y en puntos, alineación horizontal, vertical).
# Se colocó a la vista (Orden 210): cada rótulo queda pegado a su punto y lejos de los vecinos.
POSICION_ROTULO = {"gpt2": (-3, 8, "left", "bottom"), "gpt_4": (-7, 7, "right", "bottom"),
                   "claude_mythos_preview_early_inspect": (-2, 6, "right", "bottom")}
# Las marcas del eje vertical (Orden 212): lo que dura la tarea para una persona, en minutos, en su sitio exacto de la escala.
MARCAS_Y = ((10 / 60, "una tarea de\n10 segundos"), (1.0, "de 1 minuto"), (10.0, "de 10 minutos"), (60.0, "de 1 hora"),
            (480.0, "de una jornada\n(8 horas)"))
LIMITE_Y = (0.03, 3000.0)       # minutos
ANIOS = tuple(range(2019, 2027))        # las marcas de la escala horizontal
FINAL_RECTA = "2026-06-30"      # la recta se dibuja hasta aquí (un poco más allá del último modelo, para verla llegar)

# ==========================================================

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np

import horizonte_metr as H
from infografia import COLOR, GRIS, Lienzo

AQUI = Path(__file__).resolve().parent


def cargar():
    _, modelos = H.leer(AQUI / H.ENTRADA)
    return modelos


def recta_de(modelos):
    """El ajuste de la figura: (duplicación en días, pendiente, ordenada) con los modelos de la frontera sin los de más de 16 h."""
    sel = H.seleccion(modelos)
    return H.ajustar([H.dias(m["fecha"]) for m in sel], [m["p50"] for m in sel])


def dibujar(modelos, paleta, ruta):
    """Dibuja la figura. Devuelve (eje, info): info = {'puntos': [(clave, días, minutos)], 'recta': (días0, días1, duplicación, pendiente, ordenada),
    'rotulos': [textos]}."""
    L = Lienzo(TITULO, SUBTITULO, paleta, alto=ALTO)
    p = L.p
    ax = L.fig.add_axes([0.31, 0.09, 0.62, 0.64])
    dup, b, a = recta_de(modelos)
    t0, t1 = H.dias(modelos[0]["fecha"]), H.dias(dt.date.fromisoformat(FINAL_RECTA))
    tt = np.array([t0, t1], float)
    ax.plot(tt, 2.0 ** (a + b * tt), color=p.suave, linewidth=0.8, zorder=2, solid_capstyle="butt")
    puntos, rotulos = [], []
    for m in modelos:
        if not m["sota"]:
            continue                      # Orden 212: los que no batieron el récord de su momento no se dibujan
        t = H.dias(m["fecha"])
        fuera = m["p50"] > H.LIMITE_MINUTOS
        ax.plot([t], [m["p50"]], "o", color=p.tinta, markersize=4.4, markerfacecolor=p.tinta, markeredgecolor=p.tinta,
                markeredgewidth=0.6, zorder=4)
        if fuera:                         # el de más de 16 horas: el mismo punto, con un borde (un aro alrededor)
            ax.plot([t], [m["p50"]], "o", color=p.tinta, markersize=8.2, markerfacecolor="none", markeredgecolor=p.tinta,
                    markeredgewidth=0.8, zorder=4)
        puntos.append((m["clave"], t, m["p50"]))
        if m["clave"] in CON_NOMBRE:
            texto = ROTULOS[m["clave"]]
            dx, dy, ha, va = POSICION_ROTULO[m["clave"]]
            ax.annotate(texto, xy=(t, m["p50"]), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=7.2,
                        color=p.tinta, linespacing=1.15, zorder=5)
            rotulos.append(texto)
    ax.set_yscale("log")
    ax.set_ylim(*LIMITE_Y)
    ax.set_xlim(t0 - 120, H.dias(dt.date(2026, 12, 31)))
    ax.set_xticks([H.dias(dt.date(a_, 1, 1)) for a_ in ANIOS])
    ax.set_xticklabels([str(a_) for a_ in ANIOS], fontsize=6.8, color=p.tinta)
    ax.set_yticks([v for v, _ in MARCAS_Y])
    ax.set_yticklabels([s for _, s in MARCAS_Y], fontsize=6.8, color=p.tinta, linespacing=1.1)
    ax.minorticks_off()
    ax.tick_params(axis="both", length=2.5, width=0.5, color=p.suave)
    ax.grid(axis="y", color=p.neutro, linewidth=0.4, zorder=0)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_linewidth(0.6)
        ax.spines[lado].set_color(p.suave)
    # el rótulo del eje vertical y la flecha hacia arriba (en coordenadas de la figura)
    y_medio = 0.09 + 0.64 / 2
    L.fig.text(0.040, y_medio, ETIQUETA_Y, rotation=90, ha="center", va="center", fontsize=7.4, color=p.tinta)
    ax.annotate("", xy=(0.085, y_medio + 0.22), xytext=(0.085, y_medio - 0.22), xycoords="figure fraction",
                arrowprops=dict(arrowstyle="-|>", color=p.suave, linewidth=0.9, mutation_scale=8), annotation_clip=False)
    L.fig.text(0.112, y_medio, FLECHA_Y, rotation=90, ha="center", va="center", fontsize=7.0, color=p.suave)
    L.guardar(ruta)
    return ax, {"puntos": puntos, "recta": (t0, t1, dup, b, a), "rotulos": rotulos}


# ---------------------------------------------------------------- selftest

def _es_gris(c):
    r, g, b = c[:3]
    return abs(r - g) < 1e-6 and abs(g - b) < 1e-6


def selftest():
    print("=== SELFTEST de figura_horizonte_metr.py ===")
    ok = True
    modelos = cargar()

    # [1] NULO — con unos datos sin tendencia (todos los horizontes iguales) la recta dibujada es horizontal.
    planos = [dict(m, p50=30.0, p50_bajo=20.0, p50_alto=45.0, sota=True) for m in modelos]
    ax, info = dibujar(planos, GRIS, "/dev/null")
    recta = [ln for ln in ax.lines if len(ln.get_xdata()) == 2 and ln.get_linestyle() == "-" and ln.get_marker() == "None"][0]
    y = np.asarray(recta.get_ydata(), float)
    n1 = bool(abs(y[1] / y[0] - 1.0) < 1e-9)
    print(f"[1] test nulo         con datos sin tendencia, la recta sube {y[1] / y[0] - 1:.1e} entre sus dos extremos: {'OK' if n1 else 'MAL'}")
    ok &= n1

    # [2] SEÑAL IMPLANTADA — unos datos inventados que se duplican cada 100 días: la recta dibujada se duplica cada 100 días.
    t0 = dt.date(2020, 1, 1)
    inventados = [dict(modelos[i], fecha=t0 + dt.timedelta(days=37 * i), p50=3.0 * 2.0 ** (37 * i / 100.0), p50_bajo=1.0, p50_alto=9.0,
                       sota=True, clave=modelos[i]["clave"]) for i in range(len(modelos))]
    inventados = [m for m in inventados if m["p50"] <= H.LIMITE_MINUTOS]
    ax, info = dibujar(inventados, GRIS, "/dev/null")
    recta = [ln for ln in ax.lines if len(ln.get_xdata()) == 2 and ln.get_linestyle() == "-" and ln.get_marker() == "None"][0]
    x, y = np.asarray(recta.get_xdata(), float), np.asarray(recta.get_ydata(), float)
    dup_dibujada = (x[1] - x[0]) / np.log2(y[1] / y[0])
    n2 = bool(abs(dup_dibujada - 100.0) < 1e-6)
    print(f"[2] señal implantada  datos que se duplican cada 100 días: la recta dibujada se duplica cada {dup_dibujada:.6f} días: {'OK' if n2 else 'MAL'}")
    ok &= n2

    # [3] INVARIANTE (Orden 212) — los puntos dibujados son los modelos de la frontera del fichero de METR (los marcados `is_sota`,
    #     Mythos incluido) y ninguno más, con sus valores sin retocar; la recta es la de horizonte_metr.py (187,8 días); las marcas del
    #     eje vertical están donde les toca (10 segundos, 1, 10, 60 y 480 minutos) y dicen lo que dicen; solo hay tres modelos con
    #     nombre; no hay leyenda; el de más de 16 horas lleva borde (un aro); todo es gris; y las dos paletas se dibujan.
    from matplotlib.colors import to_rgba
    dibujar(modelos, COLOR, "/dev/null")
    ax, info = dibujar(modelos, GRIS, "/dev/null")
    frontera = [m for m in modelos if m["sota"]]
    puntos = {c: (t, v) for c, t, v in info["puntos"]}
    iguales = (set(puntos) == {m["clave"] for m in frontera} and len(puntos) == len(frontera)
               and all(abs(puntos[m["clave"]][1] - m["p50"]) < 1e-12 and puntos[m["clave"]][0] == H.dias(m["fecha"]) for m in frontera))
    sin_grises = not any(m["clave"] in puntos for m in modelos if not m["sota"])
    publicados, _ = H.leer(AQUI / H.ENTRADA)
    dup = info["recta"][2]
    misma_recta = abs(dup - publicados["todo"]) < H.TOPE_DIFERENCIA
    marcas_ok = (np.allclose(ax.get_yticks(), [10 / 60, 1.0, 10.0, 60.0, 480.0], rtol=1e-9)
                 and [t.get_text() for t in ax.get_yticklabels()] == [s_ for _, s_ in MARCAS_Y] and ax.get_yscale() == "log")
    sin_minutos_sueltos = not any(("minutos" in t.get_text() and not t.get_text().startswith(("de ", "una "))) for t in ax.get_yticklabels())
    nombres = sum(1 for t in ax.texts if t.get_text() in {ROTULOS[c] for c in CON_NOMBRE})
    sin_leyenda = ax.get_legend() is None
    aros = [ln for ln in ax.lines if ln.get_marker() == "o" and ln.get_markerfacecolor() == "none"]
    borde_ok = len(aros) == sum(1 for m in frontera if m["p50"] > H.LIMITE_MINUTOS)
    textos_fig = {t.get_text() for t in ax.figure.texts}
    colores = []
    for ln in ax.lines:
        for c in (ln.get_color(), ln.get_markerfacecolor(), ln.get_markeredgecolor()):
            if isinstance(c, str) and c in ("none", "None"):
                continue
            colores.append(to_rgba(c))
    grises = all(_es_gris(c) for c in colores)
    rotulo_y = ETIQUETA_Y in textos_fig and FLECHA_Y in textos_fig
    n3 = bool(iguales and sin_grises and misma_recta and marcas_ok and sin_minutos_sueltos and nombres == len(CON_NOMBRE)
              and sin_leyenda and borde_ok and grises and rotulo_y)
    print(f"[3] invariante        {len(puntos)} puntos = los {len(frontera)} modelos de la frontera del fichero, con sus valores: {'sí' if iguales else 'NO'}; "
          f"ningún modelo gris dibujado: {'sí' if sin_grises else 'NO'}; recta de {dup:.1f} días (METR {publicados['todo']:.1f}): {'sí' if misma_recta else 'NO'}; "
          f"marcas del eje en su sitio y con su texto: {'sí' if marcas_ok else 'NO'}; sin «minutos» a secas: {'sí' if sin_minutos_sueltos else 'NO'}; "
          f"modelos con nombre: {nombres}; sin leyenda: {'sí' if sin_leyenda else 'NO'}; el de más de 16 horas, con aro: {'sí' if borde_ok else 'NO'}; "
          f"rótulo del eje y flecha: {'sí' if rotulo_y else 'NO'}; todo en gris: {'sí' if grises else 'NO'}: {'OK' if n3 else 'MAL'}")
    ok &= n3
    print("SELFTEST OK (3 pruebas)" if ok else "SELFTEST FALLA")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    codigo = selftest()
    if codigo or args.selftest:
        sys.exit(codigo)
    modelos = cargar()
    dibujar(modelos, GRIS, str(AQUI / DESTINO))
    print(f"escrita {DESTINO}")
    copia = (AQUI / COPIA).resolve()
    copia.parent.mkdir(parents=True, exist_ok=True)
    dibujar(modelos, GRIS, str(copia))
    print(f"escrita {copia}")


if __name__ == "__main__":
    main()
