#!/usr/bin/env python3
"""
Capítulo 5 — de cuándo son los trescientos libros.

El capítulo dice, al explicar por qué «mujer» sale rodeada de «criatura», «casada» y
«virtuosa», que los libros bajados «son de dominio público, lo que en la práctica significa
que casi todos son anteriores a 1930». Es la única cifra del capítulo que no salía de
ningún sitio: `palabras_numeros.py` mide lo que hacen los libros, no de cuándo son.

Esto lo mide, con lo que los propios ficheros traen escrito. Proyecto Gutenberg pone en la
cabecera de parte de sus textos una línea «Original publication» con el año de la edición
de la que se transcribió. No la traen todos, así que lo que sale es una MUESTRA, y el
programa dice de qué tamaño: un porcentaje sobre 46 libros no es lo mismo que sobre 300 y
el libro no puede escribirlo como si lo fuera.

De propina, el año de nacimiento y de muerte del autor, que `datos/entrada/candidatos_es.txt`
lleva para casi todos. Es un indicio más flojo —un autor puede publicar a los veinte años
y morir a los noventa— y por eso va aparte y rotulado como lo que es.

Uso:
    python de_cuando_es_la_biblioteca.py
    python de_cuando_es_la_biblioteca.py --selftest
"""

# ======================= CONSTANTES =======================

LISTA = "../datos/entrada/candidatos_es.txt"
CABECERA = 4000              # bytes del principio del fichero donde vive la cabecera
CORTES = (1900, 1930, 1950)  # los años contra los que se cuenta; 1930 es el del capítulo

AÑO_MIN, AÑO_MAX = 1400, 2100   # fuera de aquí, lo leído no es un año de publicación

# ==========================================================

import argparse
import glob
import os
import platform
import re
import sys
from datetime import date

from formato import comprobar_ancho, pct

from palabras_numeros import CORPUS_BIBLIOTECA

RE_PUBLICACION = re.compile(r"Original publication:([^\n]*)")
RE_AÑO = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")


def año_de_publicacion(cabecera):
    """El año de la edición original, si el fichero lo declara. Se coge el ÚLTIMO año de
    la línea porque ahí van primero la ciudad y la editorial, que a veces llevan cifras."""
    m = RE_PUBLICACION.search(cabecera)
    if not m:
        return None
    años = [int(a) for a in RE_AÑO.findall(m.group(1))]
    años = [a for a in años if AÑO_MIN <= a <= AÑO_MAX]
    return años[-1] if años else None


def años_de_la_biblioteca(directorio=CORPUS_BIBLIOTECA):
    ficheros = sorted(glob.glob(os.path.join(directorio, "*.txt")))
    assert ficheros, f"Se esperaban ficheros .txt en «{directorio}»; no se encontró ninguno"
    años = []
    for ruta in ficheros:
        with open(ruta, encoding="utf-8", errors="replace") as fh:
            a = año_de_publicacion(fh.read(CABECERA))
        if a is not None:
            años.append(a)
    return len(ficheros), años


def años_de_los_autores(directorio=CORPUS_BIBLIOTECA, lista=LISTA):
    """Nacimiento y muerte del autor, de la lista de candidatos. Solo de los libros que
    de verdad están bajados: la lista trae más de los que se consiguieron."""
    ficha = {}
    with open(lista, encoding="utf-8") as fh:
        for linea in fh:
            partes = [p.strip() for p in linea.rstrip("\n").split("\t")]
            if len(partes) >= 3:
                ficha[partes[0]] = partes[2]
    muertes = []
    for ruta in sorted(glob.glob(os.path.join(directorio, "*.txt"))):
        autor = ficha.get(os.path.splitext(os.path.basename(ruta))[0], "")
        m = re.search(r"(\d{4})\s*-\s*(\d{4})", autor)
        if m:
            muertes.append(int(m.group(2)))
    return muertes


def reparto(años, cortes=CORTES):
    return [(c, sum(1 for a in años if a < c)) for c in cortes]


def selftest():
    fallos = []

    # 1. TEST NULO — una cabecera sin año de publicación no puede devolver ninguno. Es el
    #    fallo que importa aquí: si el buscador se inventara un año donde no lo hay, el
    #    porcentaje saldría sobre 300 libros en vez de sobre los que lo declaran.
    sin = ["Title: Nada\nAuthor: Nadie\nRelease date: January 1, 2004\nLanguage: Spanish\n",
           "Original publication: Spain: Imprenta de la calle\n",
           ""]
    encontrados = [año_de_publicacion(t) for t in sin]
    print(f"[1] test nulo         cabeceras sin año de publicación: {encontrados}")
    if any(a is not None for a in encontrados):
        fallos.append(f"test nulo: se ha leído un año donde no lo hay: {encontrados}")

    # 2. SEÑAL IMPLANTADA — una cabecera con un año conocido, y con basura numérica
    #    delante, tiene que dar ese año y no la basura.
    casos = [("Original publication: Madrid: Imprenta Real, 1887\n", 1887),
             ("Original publication: Spain: Casa 2 de Velasco, 1902\n", 1902),
             ("Original publication: Mexico: Lecturas 1946\nRelease date: 2004\n", 1946)]
    leidos = [(año_de_publicacion(t), esperado) for t, esperado in casos]
    print(f"[2] señal implantada  años implantados: "
          f"{', '.join(f'{l} (esperado {e})' for l, e in leidos)}")
    if any(l != e for l, e in leidos):
        fallos.append(f"señal implantada: {leidos}")

    # 3. INVARIANTE DEL DOMINIO — sobre la biblioteca de verdad: ningún año fuera de un
    #    rango con sentido, nunca más años que libros, y el reparto tiene que ser
    #    creciente (los anteriores a 1900 son un subconjunto de los anteriores a 1930).
    libros, años = años_de_la_biblioteca()
    creciente = all(b <= s for (_, b), (_, s) in zip(reparto(años), reparto(años)[1:]))
    fuera = [a for a in años if not (AÑO_MIN <= a <= AÑO_MAX)]
    print(f"[3] invariante        {len(años)} años leídos de {libros} libros; fuera de "
          f"{AÑO_MIN}-{AÑO_MAX}: {len(fuera)}; reparto creciente: "
          f"{'sí' if creciente else 'NO'}")
    if fuera or len(años) > libros or not creciente:
        fallos.append(f"invariante: {len(fuera)} años imposibles, {len(años)} años para "
                      f"{libros} libros, reparto creciente: {creciente}")

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

    libros, años = años_de_la_biblioteca()
    muertes = años_de_los_autores()

    def fila(etiqueta, valor):
        return f"{etiqueta:<56}{valor:>6}"

    lineas = [f"Máquina: {platform.machine()}, {platform.system()} {platform.release()}.",
              f"Medido el {date.today().isoformat()}.",
              "",
              "¿DE CUÁNDO SON LOS LIBROS DE LA BIBLIOTECA?",
              "",
              fila("libros bajados", libros),
              fila("de ésos, los que declaran el año de su edición original", len(años)),
              ""]
    if años:
        lineas += [fila("el más antiguo de los que lo declaran", min(años)),
                   fila("el más reciente de los que lo declaran", max(años)),
                   "",
                   "año de corte   libros   de cada 100 de los que lo declaran",
                   "------------   ------   ----------------------------------"]
        for corte, cuantos in reparto(años):
            lineas.append(f"{corte:<12}   {cuantos:>6}   "
                          f"{pct(cuantos / len(años), 1):>34}")
    lineas += ["",
               "indicio aparte y más flojo: el año en que murió el autor",
               "(de la lista de candidatos; un autor publica mucho antes",
               "de morirse, así que esto es una cota, no una fecha)",
               "",
               fila("libros con año de muerte del autor", len(muertes))]
    for corte, cuantos in reparto(muertes):
        lineas.append(fila(f"de ésos, autores muertos antes de {corte}", cuantos))
    for l in comprobar_ancho(lineas):
        print(l)


if __name__ == "__main__":
    main()
