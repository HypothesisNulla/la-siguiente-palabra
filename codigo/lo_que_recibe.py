#!/usr/bin/env python3
"""
Capítulo 14 — lo que recibe la máquina, letra por letra, cuando la calculadora le cuela la frase.

No carga ningún modelo: solo su troceador, que trae la plantilla del formato de conversación.
Monta el texto de la segunda vuelta de la pregunta «¿Cuánto es 6396 por 7576? Contesta solo con
el número.» con calculadora y frase colada, tal como lo montó `ordenes_escritas.py` (la misma
función, `enunciado`): la descripción de la calculadora que añade la plantilla, la pregunta, la
orden que escribió la máquina en la primera vuelta y lo que devolvió la calculadora, con la frase
colada detrás. Es lo que la máquina tenía delante antes de escribir «48456096».

La orden no se reescribe a mano: se lee, literal, de la muestra «Una vuelta entera» de
`ordenes_escritas.txt`, que es la de esa misma pregunta (el CSV dice que, con la frase colada, la
primera vuelta fue la misma orden: misma pregunta, mismo texto delante, siempre el trozo más
probable).

Auditoría del capítulo 14, 10 de octubre de 2026 (F1, P11): enseñar lo que recibe la máquina en
vez de contarlo.

Uso (desde codigo/, con HF_HUB_OFFLINE=1 para no descargar nada):
    python lo_que_recibe.py --selftest
    python lo_que_recibe.py > ../datos/salidas/lo_que_recibe.txt
"""

# ======================= CONSTANTES =======================

MUESTRA = "../datos/salidas/ordenes_escritas.txt"
CSV = "../datos/salidas/ordenes_escritas.csv"
TITULO_VUELTA = "Una vuelta entera: la orden, lo que hace el programa y la respuesta"
A, B, PRODUCTO = 6396, 7576, 48456096
MAX_RENGLONES = 40          # lo que cabe de una muestra en una página (formato.py)

# ==========================================================

import argparse
import csv
import importlib.metadata
import sys

from formato import muestra_editorial
from ordenes_escritas import (FRASE_COLADA, MODELO_GRANDE, PREGUNTA, calcular, enunciado,
                              renglones)


def orden_literal(ruta=MUESTRA):
    """La orden que escribió la máquina, tal cual está en la muestra «Una vuelta entera»."""
    t = open(ruta, encoding="utf-8").read()
    i = t.find(TITULO_VUELTA)
    assert i >= 0, f"Se esperaba la muestra «{TITULO_VUELTA}» en {ruta}; no está"
    bloque = t[i:t.index(":::", i)]
    assert f"¿Cuánto es {A} por {B}?" in bloque, (
        f"Se esperaba la pregunta {A} por {B} en la muestra; dice:\n{bloque}")
    ini = bloque.index("Escribe la máquina:\n") + len("Escribe la máquina:\n")
    fin = bloque.index("\n    \n    Devuelve el programa:", ini)
    lineas = bloque[ini:fin].split("\n")
    assert all(l.startswith("    ") for l in lineas), "Se esperaban renglones sangrados"
    orden = "\n".join(l[4:] for l in lineas)
    assert orden.startswith("<tool_call>") and orden.endswith("</tool_call>"), (
        f"Se esperaba una orden entre «<tool_call>» y «</tool_call>»; se encontró:\n{orden}")
    return orden


def mensajes(orden, colar=True):
    devuelto = calcular(f"{A}*{B}") + ("\n" + FRASE_COLADA if colar else "")
    return [{"role": "user", "content": PREGUNTA.format(a=A, b=B)},
            {"role": "assistant", "content": orden},
            {"role": "tool", "content": devuelto}]


def troceador():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODELO_GRANDE[1])


def selftest():
    ok = True
    tok = troceador()
    orden = orden_literal()
    texto = enunciado(tok, mensajes(orden), True)
    # [1] test nulo: sin frase colada, la frase no aparece; sin herramienta, ni la calculadora
    #     ni la orden (enunciado() revienta si la descripción sale donde no debe)
    limpio = enunciado(tok, mensajes(orden, colar=False), True)
    solo = enunciado(tok, mensajes(orden)[:1], False)
    p1 = FRASE_COLADA not in limpio and "<tools>" not in solo
    ok &= p1
    print(f"[1] test nulo         sin colar, la frase está: {FRASE_COLADA in limpio}; sin "
          f"herramienta, la descripción está: {'<tools>' in solo}: {'bien' if p1 else 'MAL'}")
    # [2] señal implantada: la orden, el producto y la frase colada están, una vez cada una, y
    #     la frase va dentro de lo que devuelve la calculadora
    r_ini, r_fin = texto.find("<tool_response>"), texto.find("</tool_response>")
    p2 = (texto.count(orden) == 1 and texto.count(FRASE_COLADA) == 1
          and r_ini < texto.find(str(PRODUCTO)) < texto.find(FRASE_COLADA) < r_fin)
    ok &= p2
    print(f"[2] señal implantada  orden, producto y frase, en su sitio: {'bien' if p2 else 'MAL'}")
    # [3] invariante: el texto de la primera vuelta es el principio del de la segunda (a la
    #     máquina no se le cambia nada de lo que ya tenía delante; solo se le añade), y lo que
    #     devuelve la calculadora entra con la misma marca que la pregunta, «user»
    primera = enunciado(tok, mensajes(orden)[:1], True)
    previo = primera[:primera.rindex("<|im_start|>assistant")]
    marca = texto.count("<|im_start|>user")
    p3 = texto.startswith(previo) and marca == 2
    ok &= p3
    print(f"[3] invariante        la primera vuelta es el principio de la segunda: "
          f"{texto.startswith(previo)}; turnos marcados «user»: {marca} (se esperaban 2): "
          f"{'bien' if p3 else 'MAL'}")
    # y el CSV confirma que, con la frase colada, esa pregunta escribió una orden
    fila = [f for f in csv.DictReader(open(CSV, encoding="utf-8"))
            if f["condicion"] == "con restricción, colada" and f["a"] == str(A)
            and f["b"] == str(B)]
    p4 = len(fila) == 1 and fila[0]["ordenes"] == "1"
    ok &= p4
    print(f"    (en el CSV, con la frase colada, {A} por {B} escribió una orden: "
          f"{'sí' if p4 else 'NO'})")
    print()
    print("SELFTEST: las tres pruebas pasan." if ok else "SELFTEST: FALLA.")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        sys.exit(0 if selftest() else 1)
    tok = troceador()
    texto = enunciado(tok, mensajes(orden_literal()), True)
    lineas = renglones(texto.rstrip("\n"))
    print(f"troceador: {MODELO_GRANDE[1]}; transformers "
          f"{importlib.metadata.version('transformers')}")
    print()
    # En dos muestras si no cabe en una: la primera, hasta la pregunta; la segunda, el resto.
    corte = next(i for i, l in enumerate(lineas) if l.startswith("<|im_start|>assistant"))
    partes = [lineas] if len(lineas) <= MAX_RENGLONES else [lineas[:corte], lineas[corte:]]
    assert all(len(p) <= MAX_RENGLONES for p in partes), (
        f"Se esperaban como mucho {MAX_RENGLONES} renglones por muestra; hay "
        f"{[len(p) for p in partes]}")
    titulos = (["Lo que recibe la máquina, con la calculadora y la frase colada"]
               if len(partes) == 1 else
               ["Lo que recibe la máquina, con la calculadora y la frase colada (1 de 2)",
                "Lo que recibe la máquina, con la calculadora y la frase colada (2 de 2)"])
    # Orden 97: nota para el lector. La pregunta (6396 por 7576, con restricción) y que el texto es
    # literal, con los renglones largos partidos, están en las constantes y en `renglones`.
    nota = ("Todo lo que tiene delante la máquina antes de su segunda respuesta, con las marcas del "
            "formato de conversación. Arriba, en inglés, la descripción de la calculadora que pone el "
            "programa; abajo, el resultado de la calculadora con la frase colada detrás.")
    for titulo, parte in zip(titulos, partes):
        print("\n".join(muestra_editorial(titulo, parte, [nota])))
        print()


if __name__ == "__main__":
    main()
