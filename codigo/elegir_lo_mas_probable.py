#!/usr/bin/env python3
"""
Capítulo 7 — qué pasa de verdad cuando se selecciona siempre lo más probable.
(L25, 3 de octubre: los rótulos dicen «seleccionar», no «elegir»; REGLAS 5 quater. El nombre del
fichero se queda, para no romper las referencias.)

El capítulo 7 hace dos afirmaciones sobre el bucle de la máquina, y las dos se pueden
medir con el mismo programa y con la misma máquina que el capítulo usa:

  1. «La mayor parte de lo que hace esta máquina no son decisiones: es terminar palabras
     que ya había empezado. De las ocho, quizá tres son elecciones de verdad.» La tabla
     del capítulo tiene ocho pasos y uno solo —«ia» detrás de «Franc»— es terminar una
     palabra ya empezada. Ocho pasos son muy pocos para sostener un «la mayor parte»,
     así que aquí se miden ciento cincuenta por frase, en seis frases.

  2. «Seleccionar siempre lo más probable produce textos correctos y aburridos, que además
     tienden a atascarse repitiendo.» Esa es la respuesta a la pregunta del final del
     capítulo, o sea el criterio de aceptación de la regla 3, y en el repositorio no hay
     ninguna medición que la sostenga: la única que mira los bucles es la del capítulo 13
     (`romper_la_maquina.txt`), que no encontró ni uno.

Se mide sobre la MISMA máquina del capítulo (Qwen2.5-0.5B en float32, procesador), con la
misma manera de elegir: siempre el trozo más probable. Se importan de `maquina_entera.py`
el modelo y la función que pide la lista de probabilidades, para que no haya dos maneras
de hacer lo mismo en el mismo capítulo.

Uso:
    python elegir_lo_mas_probable.py
    python elegir_lo_mas_probable.py --selftest
"""

# ======================= CONSTANTES =======================

# Seis frases de arranque, todas en castellano y de clases distintas: un enunciado de
# hecho (el del capítulo), una narración, una enumeración, una definición, una receta y
# una noticia. Si el bucle dependiera del tipo de texto, con una sola no se vería.
FRASES = [
    "La capital de Francia es",
    "Era una mañana de invierno y el tren",
    "Los tres ingredientes principales son",
    "Un termómetro sirve para",
    "Para hacer una tortilla de patatas hay que",
    "El ayuntamiento anunció ayer que",
]
PASOS = 150            # trozos generados por frase

# Un bucle es un tramo final que se repite. PERIODO_MAXIMO es lo más largo que puede ser
# el trozo que se repite, y REPETICIONES cuántas veces seguidas tiene que repetirse para
# llamarlo bucle. Tres repeticiones seguidas de hasta veinte trozos es lo que cualquiera
# llamaría «se ha atascado»; con dos, una anáfora legítima («y más, y más») ya contaría.
PERIODO_MAXIMO = 20
REPETICIONES = 3

CASI_SEGURO = 0.90     # a partir de aquí el paso no es una elección, es un trámite

SALIDA_CSV = "../datos/salidas/elegir_lo_mas_probable.csv"

# ==========================================================

from formato import coma, comprobar_ancho, miles, pct
from maquina_entera import MODELO, DTYPE, cargar, siguientes

import argparse
import csv
import platform
import sys
from datetime import date


def continua_palabra(anterior, trozo):
    """¿Este trozo termina una palabra que el anterior ya había empezado?

    El troceador de esta máquina marca con un espacio delante los trozos que empiezan
    palabra. Así que un trozo que NO lleva espacio delante, que empieza por letra o
    cifra, y que cae detrás de otro que tampoco terminaba en espacio ni en signo, se
    pega al de antes y alarga la misma palabra: «Franc» + «ia».
    """
    if not trozo or trozo.startswith((" ", "\n", "\t")):
        return False
    if not trozo[0].isalnum():
        return False
    return bool(anterior) and anterior[-1].isalnum()


def bucle(trozos, periodo_maximo=PERIODO_MAXIMO, repeticiones=REPETICIONES):
    """Devuelve el periodo del bucle en que termina la lista, o 0 si no termina en bucle.

    Se mira el final, no el medio: lo que dice el capítulo es que el texto «se atasca»,
    y atascarse es no salir. Se prueba cada periodo de uno en adelante y se devuelve el
    más corto que explique el final, que es el que se ve al leer.
    """
    for p in range(1, periodo_maximo + 1):
        if len(trozos) < p * repeticiones:
            break
        ultimo = trozos[-p:]
        if all(trozos[-p * (k + 1):len(trozos) - p * k] == ultimo
               for k in range(repeticiones)):
            return p
    return 0


def recorrer(tok, modelo, frase, pasos=PASOS):
    """Genera `pasos` trozos eligiendo siempre el más probable y anota cada paso."""
    texto, elegidos, probabilidades, continuaciones = frase, [], [], []
    for _ in range(pasos):
        top, _ = siguientes(tok, modelo, texto, 1)
        trozo, prob = top[0]
        continuaciones.append(continua_palabra(texto, trozo))
        elegidos.append(trozo)
        probabilidades.append(prob)
        texto += trozo
    return elegidos, probabilidades, continuaciones, texto


def selftest(tok, modelo):
    fallos = []

    # 1. TEST NULO — sobre listas hechas a mano, sin modelo: un texto que no se repite no
    #    tiene bucle, y unos trozos que todos empiezan palabra no continúan ninguna.
    sin_bucle = bucle(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"])
    nuevas = sum(continua_palabra("es", t) for t in [" la", " ciudad", " más", " de"])
    print(f"[1] test nulo         texto que no se repite: periodo {sin_bucle} (tiene que "
          f"ser 0); trozos que empiezan palabra y continúan otra: {nuevas} (tiene que ser 0)")
    if sin_bucle or nuevas:
        fallos.append(f"test nulo: periodo {sin_bucle} y {nuevas} continuaciones donde no "
                      f"tenía que haber ninguna")

    # 2. SEÑAL IMPLANTADA — un bucle de tres trozos repetido cuatro veces al final, y la
    #    continuación de palabra del propio capítulo: «Franc» + «ia».
    con_bucle = bucle(["x", "y"] + ["uno", "dos", "tres"] * 4)
    pegado = continua_palabra("La capital de Francia es la ciudad más grande de Franc", "ia")
    print(f"[2] señal implantada  bucle de tres repetido cuatro veces: periodo "
          f"{con_bucle} (tiene que ser 3); «Franc»+«ia»: {'se pega' if pegado else 'NO SE PEGA'}")
    if con_bucle != 3 or not pegado:
        fallos.append(f"señal implantada: periodo {con_bucle} (esperado 3); «ia» detrás de "
                      f"«Franc» {'no ' if not pegado else ''}cuenta como continuación")

    # 3. INVARIANTE DEL DOMINIO — elegir siempre lo más probable es determinista, así que
    #    dos recorridos de la misma frase tienen que dar los mismos trozos; y cada paso es
    #    o continuación de palabra o principio de otra, sin terceros: las dos cuentas
    #    tienen que sumar los pasos dados.
    a, pa, ca, _ = recorrer(tok, modelo, FRASES[0], 8)
    b, _, _, _ = recorrer(tok, modelo, FRASES[0], 8)
    suman = sum(ca) + sum(not c for c in ca)
    print(f"[3] invariante        dos recorridos iguales: {'sí' if a == b else 'NO'}; "
          f"{sum(ca)} continuaciones + {sum(not c for c in ca)} principios = {suman} de 8 pasos")
    if a != b:
        fallos.append("invariante: dos recorridos de la misma frase dan trozos distintos")
    if suman != 8:
        fallos.append(f"invariante: las dos cuentas suman {suman} y los pasos son 8")

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

    tok, modelo = cargar()
    if args.selftest:
        sys.exit(selftest(tok, modelo))

    print(f"máquina: {platform.machine()}, {platform.system()} {platform.release()}, "
          f"procesador")
    print(f"modelo: {MODELO} en {DTYPE}   fecha: {date.today().isoformat()}")
    print(f"se seleccionan siempre los trozos más probables, {PASOS} por frase")
    print()

    filas, total_pasos, total_cont, total_seguro, con_bucle = [], 0, 0, 0, 0

    print("--- 1. DE CADA CIEN PASOS, ¿CUÁNTOS ESTÁN ABIERTOS? ---")
    lineas = [
        f"{'frase de arranque':<30} {'termina':>9} {'casi':>7} {'acaba en':>9}",
        f"{'':<30} {'palabra':>9} {'seguro':>7} {'bucle':>9}",
        f"{'-' * 30} {'-' * 9} {'-' * 7} {'-' * 9}",
    ]
    for frase in FRASES:
        trozos, probs, conts, _ = recorrer(tok, modelo, frase)
        p = bucle(trozos)
        total_pasos += len(trozos)
        total_cont += sum(conts)
        seguros = sum(1 for v in probs if v >= CASI_SEGURO)
        total_seguro += seguros
        con_bucle += 1 if p else 0
        corta = (frase[:27] + "…") if len(frase) > 28 else frase
        lineas.append(f"{corta:<30} {pct(sum(conts) / len(trozos), 0):>9} "
                      f"{pct(seguros / len(probs), 0):>7} "
                      f"{('sí, de ' + str(p)) if p else 'no':>9}")
        filas += [[frase, i + 1, t, f"{v:.6f}", int(c)]
                  for i, (t, v, c) in enumerate(zip(trozos, probs, conts))]
    lineas.append(f"{'-' * 30} {'-' * 9} {'-' * 7} {'-' * 9}")
    lineas.append(f"{'las seis juntas':<30} {pct(total_cont / total_pasos, 0):>9} "
                  f"{pct(total_seguro / total_pasos, 0):>7} "
                  f"{str(con_bucle) + ' de 6':>9}")
    lineas.append("")
    lineas.append("«termina palabra»: el trozo se pega al anterior y alarga la")
    lineas.append("misma palabra, como «ia» detrás de «Franc».")
    lineas.append(f"«casi seguro»: el trozo seleccionado tenía {pct(CASI_SEGURO, 0)} o más.")
    lineas.append(f"«acaba en bucle»: los últimos trozos son un tramo repetido")
    lineas.append(f"{REPETICIONES} veces seguidas; el número es el largo de ese tramo.")
    comprobar_ancho(lineas)
    for l in lineas:
        print(l)
    print()

    print("--- 2. LOS OCHO PASOS QUE ENSEÑA EL CAPÍTULO ---")
    trozos, probs, conts, texto = recorrer(tok, modelo, FRASES[0], 8)
    l2 = [f"{'paso':>4}  {'trozo':>10}  {'probabilidad':>12}  ¿qué es?",
          f"{'-' * 4}  {'-' * 10}  {'-' * 12}  {'-' * 22}"]
    for i, (t, v, c) in enumerate(zip(trozos, probs, conts), 1):
        que = "termina una palabra" if c else "hay varias posibles"
        l2.append(f"{i:>4}  {t.replace(' ', '_'):>10}  {pct(v, 2):>12}  {que}")
    l2.append("")
    l2.append(f"de los ocho pasos, {sum(conts)} termina una palabra y en "
              f"{sum(not c for c in conts)} hay varias posibles.")
    comprobar_ancho(l2)
    for l in l2:
        print(l)

    with open(SALIDA_CSV, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(
            [["frase", "paso", "trozo", "probabilidad", "termina_palabra"]] + filas)
    print(f"\nEscrito {SALIDA_CSV}")


if __name__ == "__main__":
    main()
