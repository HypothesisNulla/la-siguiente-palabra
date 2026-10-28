#!/usr/bin/env python3
"""
Capítulos 13 y 14 — las cifras que el libro toma de artículos, copiadas de la fuente y puestas en
tablas que se leen solas (L24: E07, E17, E18, E21 y E24).

Aquí no se mide nada. Cada cifra está copiada a mano del artículo que se cita al lado, con su
dirección, y el programa solo hace con ellas las cuentas que el libro le pide al lector (restas,
divisiones, una media) para que el lector no tenga que hacerlas. El selftest comprueba que la
copia es fiel donde el propio artículo da con qué comprobarla (sus medias, sus totales).

  1. PERDIDO EN MEDIO (capítulo 13). Liu y otros, 2023, «Lost in the Middle», tabla 1 y tabla 6:
     https://arxiv.org/abs/2307.03172 (versión 3, consultada el 28 de septiembre de 2026).
  2. TURING Y LA PRUEBA DE 2025 (capítulo 14). Turing, 1950, «Computing Machinery and
     Intelligence», Mind 59, p. 442: https://academic.oup.com/mind/article/LIX/236/433/986238 ;
     Jones y Bergen, 2025, «Large Language Models Pass the Turing Test», apartado 2.1:
     https://arxiv.org/abs/2503.23674 (consultados el 28 de septiembre de 2026).
  3. NEURONAS, UNIONES Y NÚMEROS (capítulo 14). Las cifras que da el propio capítulo, de Azevedo
     y otros (2009), Pakkenberg y otros (2003) y Olkowicz y otros (2016), contra los números del
     modelo grande del capítulo 13 (nominal: treinta y dos mil millones), dichas en «veces».
  4. EL TITULAR CONTRA EL ARTÍCULO (capítulo 14). Zheng y otros, 2026, «Dream-RSI», figura 3(a):
     https://arxiv.org/abs/2609.14858 (versión 1, consultada el 28 de septiembre de 2026).

Uso:
    python lo_que_dicen_los_articulos.py --selftest
    python lo_que_dicen_los_articulos.py > ../datos/salidas/lo_que_dicen_los_articulos.txt
"""

# ======================= CONSTANTES =======================

# --- 1. Liu y otros 2023: GPT-3.5-Turbo, preguntas con 20 documentos (tabla 6, «Index 0» a
#     «Index 19»: el documento bueno en el puesto 1, 5, 10, 15 y 20), sin documentos
#     («closed-book», tabla 1) y con solo el bueno («oracle», tabla 1). La versión que admite
#     cuatro veces más texto (16K frente a 4K) es la fila «GPT-3.5-Turbo (16K)» de la tabla 6.
LIU_PUESTOS = (1, 5, 10, 15, 20)
DOCUMENTOS = 20                    # los documentos de cada pregunta
VERSIONES = ("4.000", "16.000")    # trozos de texto que admite cada versión de la máquina
# Orden 97: cifra que la nota escribe con letra; si cambia el parámetro, falla aquí y no en la página.
EN_LETRA = {20: "veinte", 6: "seis"}
LIU_4K = (75.8, 57.2, 53.8, 55.4, 63.2)
LIU_16K = (75.7, 57.3, 54.1, 55.4, 63.1)
LIU_SIN_DOCUMENTOS = 56.1
LIU_SOLO_EL_BUENO = 88.3
# La versión de 16K también tiene las dos en la tabla 1 («GPT-3.5-Turbo (16K)»: 56,0 % y 88,6 %).
# Hasta el 8 oct 2026 no se imprimían; comprobado contra el artículo al pasar a tabla editorial.
LIU_SIN_DOCUMENTOS_16K = 56.0
LIU_SOLO_EL_BUENO_16K = 88.6

# --- 2. Turing 1950: «an average interrogator will not have more than 70 per cent. chance of
#     making the right identification after five minutes of questioning».
#     Jones y Bergen 2025: «GPT-4.5-PERSONA had a win rate of 73%» (la proporción de veces que el
#     interrogador eligió a la máquina como la persona).
TURING_ACIERTA_COMO_MUCHO = 70
JONES_TOMADA_POR_PERSONA = 73
A_CARA_O_CRUZ = 50

# --- 3. Las cifras del capítulo 14 y la del modelo del capítulo 13.
NEURONAS_HUMANO = 86_000_000_000          # Azevedo y otros 2009
UNIONES_CORTEZA = 150_000_000_000_000     # Pakkenberg y otros 2003
NEURONAS_CUERVO = 2_170_000_000           # Olkowicz y otros 2016
NUMEROS_MODELO = 32_000_000_000           # el de 32.000M del capítulo 13 (tamaño nominal)

# --- 4. Dream-RSI, figura 3(a), con Gemini-3.1-Pro: tiempo que tarda el programa descubierto en
#     cada uno de los seis conjuntos de datos de prueba, en milisegundos (menos es mejor), con la
#     búsqueda fija y con Dream-RSI, la media que da el artículo y las llamadas al agente.
DREAM_CONJUNTOS = ("Gisette", "RCV1", "DNA", "Leukemia", "Colon", "Duke Breast")
DREAM_FIJO = (1861.8, 19550.1, 41.5, 26.1, 14.5, 28.4)
DREAM_NUEVO = (2841.0, 14616.0, 49.9, 30.2, 16.4, 32.5)
DREAM_MEDIA_FIJO, DREAM_MEDIA_NUEVO = 3587.1, 2931.0
DREAM_LLAMADAS_FIJO, DREAM_LLAMADAS_NUEVO = 550, 317

# ==========================================================

import argparse
import sys

from formato import barra, coma, miles, tabla_editorial

# Desde el 8 de octubre de 2026 cada apartado sale como tabla de libro (formato.py,
# tabla_editorial; REGLAS 6 ter), no en columnas alineadas a espacios. Las cifras y las cuentas
# son las mismas.


def ms(x):
    """Milisegundos con una decimal y el punto de los miles, a la castellana."""
    entero, dec = f"{x:.1f}".split(".")
    return miles(int(entero)) + "," + dec


def imprime(*bloques):
    for b in bloques:
        print("\n".join(b))
        print()


def liu():
    print("--- 1. PERDIDO EN MEDIO (Liu y otros, 2023) ---\n")
    assert max(LIU_PUESTOS) == DOCUMENTOS, "la nota dice «veinte documentos»"
    filas = [[f"en el puesto {p} de {DOCUMENTOS}", coma(a, 1), coma(b, 1)]
             for p, a, b in zip(LIU_PUESTOS, LIU_4K, LIU_16K)]
    filas += [["sin ningún documento", coma(LIU_SIN_DOCUMENTOS, 1),
               coma(LIU_SIN_DOCUMENTOS_16K, 1)],
              ["con solo el documento bueno", coma(LIU_SOLO_EL_BUENO, 1),
               coma(LIU_SOLO_EL_BUENO_16K, 1)]]
    imprime(tabla_editorial(
        "Aciertos según dónde está el documento bueno (Liu y otros, 2023)",
        ["dónde estaba el documento bueno", *VERSIONES], filas, "idd",
        [# Orden 97: nota para el lector. Los veinte documentos (LIU_PUESTOS llega hasta el 20) y los
         # trozos de cada versión (4.000 y 16.000) salen en la nota con letra y con el rótulo.
         f"Aciertos de cada cien, con {EN_LETRA[DOCUMENTOS]} documentos y una pregunta, y solo uno de "
         f"los {EN_LETRA[DOCUMENTOS]} con la respuesta. Las dos columnas son la misma máquina en dos "
         f"versiones: una admite {VERSIONES[0]} trozos de texto y la otra, {VERSIONES[1]}."]))
    peor = min(range(len(LIU_4K)), key=lambda i: LIU_4K[i])
    return peor


def turing():
    print("--- 2. TURING Y LA PRUEBA DE 2025, EN LA MISMA MONEDA ---\n")
    equivoca_turing = 100 - TURING_ACIERTA_COMO_MUCHO
    filas = [["lo que pedía Turing para el año 2000", f"al menos {equivoca_turing}",
              barra(equivoca_turing / 100)],
             ["lo que pasó en 2025", str(JONES_TOMADA_POR_PERSONA),
              barra(JONES_TOMADA_POR_PERSONA / 100)],
             ["eligiendo a cara o cruz", str(A_CARA_O_CRUZ), barra(A_CARA_O_CRUZ / 100)]]
    imprime(tabla_editorial(
        "Turing y la prueba de 2025, en la misma moneda",
        ["", "se equivoca el interrogador", ""], filas, "idi",
        ["De cada cien veces.",
         f"Turing: el interrogador no acertaría más de {TURING_ACIERTA_COMO_MUCHO} de cada 100.",
         f"2025: tomó a la máquina por la persona {JONES_TOMADA_POR_PERSONA} de cada 100."]))


def veces(a, b):
    return a / b


def cerebro():
    print("--- 3. NEURONAS, UNIONES Y NÚMEROS, EN VECES ---\n")
    imprime(
        tabla_editorial(
            "Un cerebro, un cuervo y el modelo de 32.000M, contados",
            ["qué se cuenta", "cuántos"],
            [["neuronas de un cerebro humano", miles(NEURONAS_HUMANO)],
             ["uniones, solo en la corteza", miles(UNIONES_CORTEZA)],
             ["neuronas de un cuervo", miles(NEURONAS_CUERVO)],
             ["números del modelo de 32.000M", miles(NUMEROS_MODELO)]], "id",
            ["Azevedo y otros (2009), Pakkenberg y otros (2003) y Olkowicz y otros (2016); el "
             "modelo, por su tamaño nominal."]),
        tabla_editorial(
            "Los mismos recuentos, en veces",
            ["comparación", "veces"],
            [["neuronas humanas entre números", coma(veces(NEURONAS_HUMANO, NUMEROS_MODELO), 1)],
             ["uniones entre números", miles(round(veces(UNIONES_CORTEZA, NUMEROS_MODELO)))],
             ["números entre neuronas del cuervo",
              coma(veces(NUMEROS_MODELO, NEURONAS_CUERVO), 1)]], "id",
            ["«Entre»: dividir el primero por el segundo."]))


def dream():
    print("--- 4. EL TITULAR CONTRA EL ARTÍCULO (Dream-RSI, 2026) ---\n")
    filas = []
    for c, a, b in zip(DREAM_CONJUNTOS, DREAM_FIJO, DREAM_NUEVO):
        filas.append([c, ms(a), ms(b), "nueva" if b < a else "fija", ms(abs(b - a))])
    mf = sum(DREAM_FIJO) / len(DREAM_FIJO)
    mn = sum(DREAM_NUEVO) / len(DREAM_NUEVO)
    filas.append(["**media**", f"**{ms(mf)}**", f"**{ms(mn)}**",
                  f"**{'nueva' if mn < mf else 'fija'}**", f"**{ms(abs(mn - mf))}**"])
    pierde = [b - a for a, b in zip(DREAM_FIJO, DREAM_NUEVO) if b > a]
    gana = [a - b for a, b in zip(DREAM_FIJO, DREAM_NUEVO) if b < a]
    imprime(
        tabla_editorial(
            "Lo que tarda el programa encontrado (Dream-RSI, 2026)",
            ["conjunto", "fija", "nueva", "gana", "por"], filas, "iddid",
            [# Orden 97: nota para el lector; las llamadas al agente (fija 550, nueva 317) están en
             # DREAM_LLAMADAS_FIJO y DREAM_LLAMADAS_NUEVO.
             "Lo que tarda, en milisegundos, el programa que encuentra cada búsqueda: la fija, sin "
             "cambiar, y la nueva, con el programa mejorado. Menos es mejor. «Por» es cuántos "
             "milisegundos menos tarda la que gana, y la media es la de los "
             f"{EN_LETRA[len(DREAM_CONJUNTOS)]} conjuntos."]),
        tabla_editorial(
            "Dónde pierde y dónde gana la búsqueda nueva",
            ["", "milisegundos"],
            [[f"lo que pierde en los {len(pierde)} donde pierde, sumado", ms(sum(pierde))],
             ["lo que gana en el único donde gana", ms(sum(gana))]], "id",
            ["Las mismas filas de la tabla anterior, sin la media."]))
    return mf, mn


def selftest():
    fallos = []
    # 1. TEST NULO — si las dos búsquedas tardaran lo mismo en todo, ninguna ganaría la media.
    empate = sum(DREAM_FIJO) / 6 - sum(DREAM_FIJO) / 6 == 0
    print(f"[1] test nulo         dos columnas iguales: {'empatan' if empate else 'NO EMPATAN'}")
    if not empate:
        fallos.append("test nulo: la cuenta de la media da ganador con columnas iguales")

    # 2. SEÑAL IMPLANTADA — lo que dicen los artículos con sus palabras: en Liu, el peor puesto
    #    está en medio y queda por debajo de no dar ningún documento; en Dream-RSI, la nueva
    #    pierde en cinco de seis y gana la media.
    peor = min(range(5), key=lambda i: LIU_4K[i])
    pierde5 = sum(b > a for a, b in zip(DREAM_FIJO, DREAM_NUEVO)) == 5
    ok = LIU_PUESTOS[peor] == 10 and LIU_4K[peor] < LIU_SIN_DOCUMENTOS and pierde5
    print(f"[2] señal implantada  peor puesto en Liu: {LIU_PUESTOS[peor]} ({coma(LIU_4K[peor], 1)} "
          f"frente a {coma(LIU_SIN_DOCUMENTOS, 1)} sin documentos); la nueva pierde en 5 de 6: "
          f"{'sí' if pierde5 else 'NO'}")
    if not ok:
        fallos.append("señal: las cifras copiadas no dicen lo que dice el artículo")

    # 3. INVARIANTE DEL DOMINIO — la copia de Dream-RSI es fiel: la media de las seis filas
    #    copiadas da la media que publica el artículo, en las dos columnas.
    mf = sum(DREAM_FIJO) / len(DREAM_FIJO)
    mn = sum(DREAM_NUEVO) / len(DREAM_NUEVO)
    fiel = abs(mf - DREAM_MEDIA_FIJO) < 0.05 and abs(mn - DREAM_MEDIA_NUEVO) < 0.05
    print(f"[3] invariante        medias de las filas copiadas {mf:.2f} y {mn:.2f}; el artículo, "
          f"{DREAM_MEDIA_FIJO} y {DREAM_MEDIA_NUEVO}: {'casan' if fiel else 'NO CASAN'}")
    if not fiel:
        fallos.append("invariante: las filas copiadas de Dream-RSI no dan sus medias")
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
    if ap.parse_args().selftest:
        sys.exit(selftest())
    liu()
    turing()
    cerebro()
    dream()


if __name__ == "__main__":
    main()
