#!/usr/bin/env python3
"""
Capítulo 1 — la hoja de «el caballero» en la libreta de la máquina de palabras.

La máquina de palabras con el selector en 2 tiene una hoja por cada pareja de palabras seguidas.
Esta figura enseña una entera, la de «el caballero»: de título, esas dos palabras, y debajo una
línea por cada palabra o signo que vino detrás, con sus palitos de cinco en cinco y el número al
final. Son 24 líneas y 97 palitos; no se quita ninguna.

Las cuentas no se escriben aquí: salen de `pagina_el_caballero.pagina` (el mismo recuento que usa
el capítulo), y el programa revienta si no coinciden, línea por línea, con la tabla que imprime
`datos/salidas/pagina_el_caballero.txt`.

Las 24 líneas caben en una sola columna dentro del alto máximo de una figura. Si algún día no
cupieran (otra pareja, otro corpus), el dibujo revienta en vez de quitar líneas.

Uso:
    python figura_pagina_el_caballero.py --selftest
    python figura_pagina_el_caballero.py
"""

# ======================= CONSTANTES =======================

SALIDA = "../datos/salidas/pagina_el_caballero.txt"
DESTINO = "../figuras/pagina_el_caballero.png"
CORPUS = "../datos/entrada/quijote.txt"

ALTO_RENGLON = 4.2          # unidades del lienzo; con menos, los palitos se pegan a la raya
PASO_PALITO = 1.15
HUECO_GRUPO = 1.9
ALTO_PALITO = 2.9
ANCHO_HOJA = 92.0
ANCHO_PALABRA = 17.0        # lo que se deja a la palabra antes de que empiecen los palitos

# ==========================================================

import argparse, math, sys
from pathlib import Path

import ngrama as N
from formato import leer_tablas, miles
from hoja_libreta import ancho_palitos, hoja, palitos
from infografia import ALTO_MAXIMO, ANCHO_PAGINA, Lienzo, GRIS
from pagina_el_caballero import PAGINA, pagina

AQUI = Path(__file__).resolve().parent
SUBTITULO = "Un palito por cada vez que vino algo detrás de esas dos palabras, en todo el Quijote."


def pagina_de_la_salida(texto):
    """La tabla que imprime `pagina_el_caballero.py`: [(palabra, n)] en su orden."""
    titulo = f"Qué viene detrás de «{' '.join(PAGINA)}», de más a menos"
    tablas = leer_tablas(texto)
    assert titulo in tablas, f"no encuentro la tabla «{titulo}» en la salida"
    _, filas, _ = tablas[titulo]
    return [(f[0].strip("«»"), int(f[1].replace(".", ""))) for f in filas]


def alto_necesario(renglones):
    """Pulgadas de alto que pide la figura con `renglones` líneas más el total, en una columna."""
    unidades = (4 + 6.2 + 4.0 + 1.5) + 1.0 + (9.5 + (renglones + 1) * ALTO_RENGLON + 1.5) + 4.0   # 4,0 de margen abajo; con el pie eran 8,0 (Orden 4)
    return math.ceil(unidades / 100 * ANCHO_PAGINA * 100) / 100


def dibujar(pag, paleta, ruta):
    """Dibuja la hoja `pag` = [(palabra, n)] y devuelve (ruta, palitos dibujados)."""
    alto = alto_necesario(len(pag))
    assert alto <= ALTO_MAXIMO, (
        f"las {len(pag)} líneas piden {alto} pulgadas y caben {ALTO_MAXIMO}: pasa la hoja a dos "
        f"columnas; no se quita ninguna línea")
    titulo = " ".join(PAGINA)
    L = Lienzo(f"La hoja de «{titulo}»", SUBTITULO, paleta, alto=alto)
    ys, x_texto, x_fin, _ = hoja(L, 4.0, L.y - 1.0, ANCHO_HOJA, titulo, len(pag) + 1,
                                 ALTO_RENGLON)
    pintados = 0
    for (palabra, n), y in zip(pag, ys):
        L.texto(x_texto, y, f"«{palabra}»", tam=7.6, negrita=(n == pag[0][1]))
        palitos(L, x_texto + ANCHO_PALABRA, y, n, paso=PASO_PALITO, hueco=HUECO_GRUPO,
                alto=ALTO_PALITO)
        pintados += n
        L.texto(x_fin, y, miles(n), tam=7.6, ha="right")
    L.texto(x_texto, ys[-1], "en total", tam=7.2, color=paleta.suave)
    L.texto(x_fin, ys[-1], miles(sum(n for _, n in pag)), tam=7.6, ha="right", negrita=True)
    L.guardar(ruta)
    return ruta, pintados


def selftest():
    fallos = []
    texto = (AQUI / SALIDA).read_text(encoding="utf-8")

    # 1. TEST NULO — una salida sin la tabla de la hoja tiene que reventar: si no, la figura
    #    dibujaría una hoja que la salida no respalda.
    try:
        pagina_de_la_salida(texto.replace("de más a menos", "de menos a más")); revienta = False
    except AssertionError:
        revienta = True
    print(f"[1] test nulo         sin la tabla de la hoja, {'revienta' if revienta else 'NO revienta'}")
    if not revienta:
        fallos.append("test nulo: se lee una tabla que no está en la salida")

    # 2. SEÑAL IMPLANTADA — una hoja inventada de tres palabras (4 + 1 + 2 palitos) se dibuja con
    #    exactamente 7 palitos; y una de más líneas de las que caben revienta en vez de recortarse.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _, pintados = dibujar([("de", 4), ("y", 1), ("la", 2)], GRIS, "/dev/null")
    plt.close("all")
    try:
        dibujar([(f"p{i}", 1) for i in range(60)], GRIS, "/dev/null"); revienta2 = False
    except AssertionError:
        revienta2 = True
    plt.close("all")
    print(f"[2] señal implantada  hoja de 3 líneas: {pintados} palitos dibujados; con 60 "
          f"líneas, {'revienta' if revienta2 else 'NO revienta'}")
    if pintados != 7 or not revienta2:
        fallos.append(f"señal implantada: se esperaban 7 palitos y que 60 líneas reventaran; "
                      f"salieron {pintados} y {'revienta' if revienta2 else 'no revienta'}")

    # 3. INVARIANTE — lo contado en el Quijote es, línea por línea, lo que imprime la tabla; son 24
    #    líneas y 97 palitos; todos se dibujan; y la palabra más larga cabe antes de los palitos
    #    y el grupo más largo de palitos antes del número.
    secuencia = N.normalizar(N.cargar_corpus(str(AQUI / CORPUS)), N.ALFABETO)
    contada = pagina(N.trocear(secuencia))
    impresa = pagina_de_la_salida(texto)
    igual = contada == impresa
    _, pintados = dibujar(contada, GRIS, "/dev/null")
    plt.close("all")
    mayor = max(n for _, n in contada)
    cabe = ANCHO_PALABRA + ancho_palitos(mayor, PASO_PALITO, HUECO_GRUPO) < ANCHO_HOJA - 20
    print(f"[3] invariante        {len(contada)} líneas y {sum(n for _, n in contada)} palitos "
          f"({pintados} dibujados); igual que la tabla impresa: {'sí' if igual else 'NO'}; "
          f"el grupo más largo cabe: {'sí' if cabe else 'NO'}")
    if not igual:
        fallos.append("invariante: lo contado no coincide con la tabla impresa")
    if pintados != sum(n for _, n in contada) or len(contada) != 24 or pintados != 97:
        fallos.append(f"invariante: se esperaban 24 líneas y 97 palitos dibujados; salen "
                      f"{len(contada)} y {pintados}")
    if not cabe:
        fallos.append("invariante: los palitos de la línea más larga no caben en la hoja")

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
    if args.selftest:
        sys.exit(selftest())
    secuencia = N.normalizar(N.cargar_corpus(str(AQUI / CORPUS)), N.ALFABETO)
    contada = pagina(N.trocear(secuencia))
    assert contada == pagina_de_la_salida((AQUI / SALIDA).read_text(encoding="utf-8")), (
        "lo contado no coincide con la tabla impresa de pagina_el_caballero.txt")
    print("Escrito", dibujar(contada, GRIS, DESTINO)[0])


if __name__ == "__main__":
    main()
