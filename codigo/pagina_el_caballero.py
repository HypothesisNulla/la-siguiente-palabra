#!/usr/bin/env python3
"""
Capítulo 1 — la hoja de «el caballero», entera.

La máquina de palabras con el selector en 2 (la de la Tabla 7 del capítulo: cada signo de
puntuación cuenta como una palabra más) tiene en su libreta una hoja por cada pareja de
palabras seguidas. Esta es la de «el caballero»: cada palabra o signo que vino detrás en el
Quijote, y cuántas veces, de más a menos. Es la tercera fila de esa tabla (97 palitos repartidos
entre 24 palabras, la primera «de» con el 42 %), pero con todas las líneas.

El recuento es el de `ngrama.construir` sobre `ngrama.trocear`, el mismo que usa
`traza_escalera.py` para la Tabla 7; el selftest comprueba que sale lo mismo que imprime esa
tabla (`datos/salidas/traza_escalera.txt`).

POR QUÉ ESTA MEDICIÓN NO DEPENDE DE LA MÁQUINA. Se cuentan palabras de un fichero de texto. El
resultado es el mismo en cualquier ordenador y en cualquier día.

Uso:
    python pagina_el_caballero.py --selftest
    python pagina_el_caballero.py > ../datos/salidas/pagina_el_caballero.txt
"""

# ======================= CONSTANTES =======================

CORPUS = "../datos/entrada/quijote.txt"
TABLA_7 = "../datos/salidas/traza_escalera.txt"
TITULO_TABLA_7 = "Los 4 primeros sorteos de la máquina de palabras, con el selector en 2"

PAGINA = ("el", "caballero")     # el título de la hoja
SELECTOR = 2

# Lo que el capítulo afirma de esta hoja, y el programa comprueba al medir
TOTAL_ESPERADO = 97              # palitos en la hoja
DISTINTAS_ESPERADAS = 24         # palabras o signos distintos
PRIMERA_ESPERADA = "de"          # la que más vino
PORCENTAJE_PRIMERA = 42          # de cada cien, redondeado

# Selftest
COCIENTE_NULO_MAX = 0.20         # barajado: la hoja sale de pocas apariciones y la más repetida es
                                 # una al azar, con la coma (9 % de los trozos) a la cabeza; con esta
                                 # semilla salió 14 %. El umbral es 20 %: el real es 42 %
COCIENTE_REAL_MIN = 0.30         # real: la más repetida pasa del 30 %
MINI = "el caballero de la mancha el caballero de la triste el caballero andante".split()

# ==========================================================

import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import ngrama as N
from formato import ANCHO_CAJA, coma, comprobar_ancho, leer_tablas, miles, pct, tabla_editorial

AQUI = Path(__file__).resolve().parent


def pagina(trozos, titulo=PAGINA):
    """La hoja de `titulo` en la libreta de la máquina de palabras con el selector en 2: qué
    vino detrás de esas dos palabras y cuántas veces, de más a menos (los empates, por orden
    alfabético, para que la lista salga siempre igual)."""
    cuenta = N.construir(trozos, len(titulo))[tuple(titulo)]
    return sorted(cuenta.items(), key=lambda t: (-t[1], t[0]))


def la_que_mas_sale(pag):
    """La primera de la hoja y su parte de cada cien, tal como la imprime la Tabla 7."""
    total = sum(n for _, n in pag)
    assert total, "Se esperaba una hoja con algo apuntado; está en blanco"
    return pag[0][0], pag[0][1] / total


# ---- selftest ------------------------------------------------------------------------

def selftest(trozos):
    fallos = []

    # 1. TEST NULO — con las palabras barajadas no queda castellano: detrás de «el caballero» ya
    #    no manda ninguna palabra («de» no llega al 42 %), mientras que en el Quijote real sí.
    barajado = list(trozos)
    random.Random(N.SEMILLA).shuffle(barajado)
    _, parte_nula = la_que_mas_sale(pagina(barajado))
    _, parte_real = la_que_mas_sale(pagina(trozos))
    print(f"[1] test nulo         la que más sale en la hoja de «{' '.join(PAGINA)}»: "
          f"barajado={pct(parte_nula, 0)}  real={pct(parte_real, 0)}")
    if not (parte_nula < COCIENTE_NULO_MAX and parte_real > COCIENTE_REAL_MIN):
        fallos.append(f"test nulo: se esperaba barajado < {COCIENTE_NULO_MAX} y real > "
                      f"{COCIENTE_REAL_MIN}; salió {parte_nula:.3f} y {parte_real:.3f}")

    # 2. SEÑAL IMPLANTADA — un texto de quince palabras contado a mano: «el caballero» aparece
    #    tres veces, y detrás va «de» dos veces y «andante» una.
    p = pagina(MINI)
    print(f"[2] señal implantada  texto de {len(MINI)} palabras: {p}")
    if p != [("de", 2), ("andante", 1)]:
        fallos.append(f"señal implantada: se esperaba [('de', 2), ('andante', 1)]; salió {p}")

    # 3. INVARIANTES — la hoja tiene los 97 palitos y las 24 distintas que dice el capítulo, la
    #    primera es «de» con el 42 %, la lista va de más a menos y suma el total, y todo coincide
    #    con la fila de «el caballero» de la Tabla 7 que imprime traza_escalera.py.
    pag = pagina(trozos)
    total = sum(n for _, n in pag)
    primera, parte = la_que_mas_sale(pag)
    orden = all(pag[i][1] >= pag[i + 1][1] for i in range(len(pag) - 1))
    tabla = leer_tablas((AQUI / TABLA_7).read_text(encoding="utf-8"))[TITULO_TABLA_7]
    fila = next(f for f in tabla[1] if f[0] == f"«{' '.join(PAGINA)}»")
    coincide = fila[1:4] == [miles(total), str(len(pag)), f"«{primera}», {pct(parte, 0)}"]
    print(f"[3] invariante        {total} palitos en {len(pag)} palabras o signos; la primera, "
          f"«{primera}», {pct(parte, 0)}; de más a menos: {'sí' if orden else 'NO'}; "
          f"igual que la Tabla 7: {'sí' if coincide else 'NO'}")
    if (total, len(pag), primera, round(100 * parte)) != (
            TOTAL_ESPERADO, DISTINTAS_ESPERADAS, PRIMERA_ESPERADA, PORCENTAJE_PRIMERA):
        fallos.append(f"invariante: se esperaban {TOTAL_ESPERADO} palitos, {DISTINTAS_ESPERADAS} "
                      f"distintas y «{PRIMERA_ESPERADA}» con el {PORCENTAJE_PRIMERA} %; salieron "
                      f"{total}, {len(pag)} y «{primera}» con el {100 * parte:.1f} %")
    if not orden:
        fallos.append("invariante: la hoja no va de más a menos")
    if not coincide:
        fallos.append(f"invariante: la fila de la Tabla 7 dice {fila}, y aquí salen "
                      f"{miles(total)}, {len(pag)} y «{primera}»")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


# ---- la salida -------------------------------------------------------------------------

def main():
    secuencia = N.normalizar(N.cargar_corpus(str(AQUI / CORPUS)), N.ALFABETO)
    trozos = N.trocear(secuencia)

    print("--- selftest ---")
    codigo = selftest(trozos)
    if codigo or "--selftest" in sys.argv:
        sys.exit(codigo)

    pag = pagina(trozos)
    total = sum(n for _, n in pag)
    titulo = " ".join(PAGINA)
    L = [
        "",
        "########## capítulo 1: la hoja de «el caballero» ##########",
        f"fecha: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC",
        f"máquina: {platform.system()} {platform.machine()}. La medición no",
        "depende de la máquina: cuenta palabras de un fichero de texto",
        "corpus: datos/entrada/quijote.txt,",
        "limpiado y troceado como en ngrama.py:",
        "cada palabra es un trozo, y cada signo de puntuación, otro",
        "",
    ]
    E = tabla_editorial(
        f"Qué viene detrás de «{titulo}», de más a menos",
        ["palabra o signo", "veces", "de cada cien veces"],
        [[f"«{t}»", miles(n), pct(n / total, 1)] for t, n in pag], "idd",
        [f"Las {miles(total)} veces que aparece «{titulo}» en el Quijote, con el selector en "
         f"{SELECTOR}: un palito en la hoja de «{titulo}» por cada una.",
         f"Palabras o signos distintos detrás de «{titulo}»: {len(pag)}."])
    for l in comprobar_ancho([l.rstrip() for l in L], ANCHO_CAJA):
        print(l)
    for l in E:
        print(l)


if __name__ == "__main__":
    main()
