#!/usr/bin/env python3
"""Comprueba que ningún bloque monoespaciado del libro se sale de la caja de texto.

    verificar_anchos.py raiz [--selftest]

Por qué existe este verificador
-------------------------------
Las tablas y las salidas de programa van en monoespaciado, y el monoespaciado no se
parte solo: si una línea es más ancha que la caja, LaTeX la imprime pisando el margen
o se la lleva fuera de la página. En pantalla, escribiendo, no se ve. En el libro
impreso se ve siempre, y lo que se ve es una tabla rota. Una tabla rota no es «poco
clara»: es una tabla que el lector no puede leer.

De dónde salen los números
--------------------------
No son una estimación. Se compuso una página de prueba con el preámbulo real del libro
(papel de 6 x 9 pulgadas, caja de 4,45 pulgadas, los bloques a \\footnotesize) y se fue
subiendo el ancho de una línea de caracteres iguales hasta que xelatex avisó:

  - bloque normal:            con 68 no avisa; con 69 avisa (se sale 4,42 pt).
  - bloque dentro de una cita: con 62 no avisa; con 63 avisa (se sale 4,07 pt).
    (Medido el 20 sep 2026; antes de que la cita llevara 1,4 em de margen por lado
    eran 64/65. Cada vez que cambie libro.tex hay que volver a medir: fallo 4.44.)

La cita es más estrecha porque el entorno de cita mete margen por los dos lados.
Si algún día cambia el tamaño del papel, la caja o el cuerpo de letra, estos dos
números hay que volver a MEDIRLOS, no ajustarlos a ojo.
"""

# ======================= CONSTANTES =======================

# Los dos anchos viven en formato.py, con los programas que imprimen las tablas: el que
# escribe la tabla y el que la revisa tienen que estar midiendo con la misma vara.
from formato import ANCHO_CAJA, ANCHO_CAJA_CITA

LIBRO = "manuscript/Book.txt"

# Cuántas líneas monoespaciadas caben en una página. El libro mete cada bloque de
# programa en una minipágina para que no se parta entre dos páginas (`construir/libro.tex`);
# eso solo funciona mientras el bloque quepa entero. Medido igual que el ancho: a
# \footnotesize, en la caja de 6 x 9, entran unas 50 líneas. Se deja margen y se avisa
# antes: un bloque de más de 40 líneas hay que partirlo a mano.
ALTO_CAJA = 40

# ==========================================================

import argparse
import io
import os
import re
import sys


def capitulos(raiz):
    ruta = os.path.join(raiz, LIBRO)
    assert os.path.exists(ruta), f"Se esperaba encontrar {ruta}; no existe"
    base = os.path.dirname(ruta)
    nombres = [l.strip() for l in io.open(ruta, encoding="utf-8") if l.strip()]
    return [os.path.join(base, n) for n in nombres]


def lineas_de_bloque(texto):
    """Devuelve (numero_de_linea, contenido, limite) de cada línea monoespaciada.

    Monoespaciado en este libro es una de tres cosas: una línea sangrada con cuatro
    espacios, una línea dentro de una valla de tres comillas inversas, o cualquiera de
    las dos anteriores dentro de una cita. La prosa de una cita normal no cuenta: esa
    se parte sola y nunca se sale."""
    fuera = []
    en_valla = False
    for n, linea in enumerate(texto.splitlines(), 1):
        cita = linea.startswith(">")
        cuerpo = re.sub(r"^>\s?", "", linea) if cita else linea
        limite = ANCHO_CAJA_CITA if cita else ANCHO_CAJA

        if cuerpo.strip().startswith("```"):
            en_valla = not en_valla
            continue
        if en_valla:
            fuera.append((n, cuerpo.rstrip(), limite))
        elif cuerpo.startswith("    ") and cuerpo.strip():
            fuera.append((n, cuerpo[4:].rstrip(), limite))
    return fuera


def bloques_seguidos(lineas):
    """Agrupa las líneas monoespaciadas en bloques: líneas de número consecutivo, o
    separadas solo por una línea en blanco, son el mismo bloque para pandoc."""
    grupos, actual = [], []
    for item in lineas:
        if actual and item[0] - actual[-1][0] > 2:
            grupos.append(actual); actual = []
        actual.append(item)
    if actual:
        grupos.append(actual)
    return grupos


def revisar(raiz):
    avisos = []
    contados = 0
    for ruta in capitulos(raiz):
        texto = io.open(ruta, encoding="utf-8").read()
        lineas = lineas_de_bloque(texto)
        for n, contenido, limite in lineas:
            contados += 1
            if len(contenido) > limite:
                avisos.append((os.path.basename(ruta), n, len(contenido), limite, contenido))
        for grupo in bloques_seguidos(lineas):
            alto = grupo[-1][0] - grupo[0][0] + 1
            if alto > ALTO_CAJA:
                avisos.append((os.path.basename(ruta), grupo[0][0], alto, ALTO_CAJA,
                               f"bloque de {alto} líneas: no cabe entero en una página"))
    return avisos, contados


def imprimir(avisos, contados):
    if not avisos:
        print(f"PASA: {contados} líneas monoespaciadas; ninguna se sale de la caja "
              f"({ANCHO_CAJA} normales, {ANCHO_CAJA_CITA} en cita) y ningún bloque pasa "
              f"de {ALTO_CAJA} líneas.")
        return 0
    print(f"FALLA: {len(avisos)} problemas de caja sobre {contados} líneas monoespaciadas.\n")
    for f, n, ancho, limite, contenido in avisos:
        print(f"  {f}:{n}  {ancho} caracteres, caben {limite}")
        print(f"      {contenido}")
    return 1


def selftest():
    fallos = []

    # 1. TEST NULO — un capítulo sin un solo bloque: no puede salir ningún aviso.
    prosa = "\n".join(["Una línea de prosa muy larga que se parte sola porque no va "
                       "en monoespaciado y por eso nunca se sale de la caja."] * 20)
    n_nulo = len(lineas_de_bloque(prosa))
    print(f"[1] test nulo         prosa sin bloques: {n_nulo} líneas monoespaciadas (esperado 0)")
    if n_nulo != 0:
        fallos.append(f"test nulo: se esperaban 0 líneas de bloque; se encontraron {n_nulo}")

    # 2. SEÑAL IMPLANTADA — se mete una línea de un carácter de más, y hay que
    #    encontrarla en el sitio y con el ancho exactos.
    largo = "M" * (ANCHO_CAJA + 1)
    texto = f"Prosa.\n\n    corta\n    {largo}\n\nMás prosa.\n"
    halladas = [(n, len(c)) for n, c, lim in lineas_de_bloque(texto) if len(c) > lim]
    # Y un bloque de una línea más de las que caben en una página: el otro defecto que
    # este verificador busca. Si no se implanta aquí, ese camino no lo prueba nadie.
    alto_texto = "Prosa.\n\n" + "".join(f"    linea {i}\n" for i in range(ALTO_CAJA + 1))
    grupos = bloques_seguidos(lineas_de_bloque(alto_texto))
    alto = grupos[0][-1][0] - grupos[0][0][0] + 1 if grupos else 0
    print(f"[2] señal implantada  línea de {ANCHO_CAJA + 1} en la línea 4: {halladas}; "
          f"bloque implantado de {alto} líneas (caben {ALTO_CAJA})")
    if halladas != [(4, ANCHO_CAJA + 1)]:
        fallos.append(f"señal implantada: se esperaba [(4, {ANCHO_CAJA + 1})]; se obtuvo {halladas}")
    if len(grupos) != 1 or alto != ALTO_CAJA + 1:
        fallos.append(f"señal implantada: se esperaba un solo bloque de {ALTO_CAJA + 1} líneas; "
                      f"se obtuvieron {len(grupos)} y el primero de {alto}")

    # 3. INVARIANTE DEL DOMINIO — el límite de la cita es MÁS ESTRECHO que el normal:
    #    la misma línea que cabe suelta no cabe dentro de una cita. Si un día estos dos
    #    números se cruzan o se igualan, el verificador está midiendo otra cosa.
    justa = "M" * ANCHO_CAJA
    suelta = [c for n, c, lim in lineas_de_bloque(f"    {justa}\n") if len(c) > lim]
    citada = [c for n, c, lim in lineas_de_bloque(f">     {justa}\n") if len(c) > lim]
    print(f"[3] invariante        {ANCHO_CAJA} caracteres: suelta pasa ({not suelta}), "
          f"en cita falla ({bool(citada)})")
    if suelta or not citada or ANCHO_CAJA_CITA >= ANCHO_CAJA:
        fallos.append("invariante: la caja de la cita tiene que ser más estrecha que la normal, "
                      f"y una línea de {ANCHO_CAJA} tiene que pasar suelta y fallar en cita")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("raiz", nargs="?", default=".")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    return imprimir(*revisar(a.raiz))


if __name__ == "__main__":
    sys.exit(main())
