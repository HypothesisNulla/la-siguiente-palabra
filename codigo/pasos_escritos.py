#!/usr/bin/env python3
"""
Capítulo 14 — escribir los pasos también es escribir.

Sin calculadora, el modelo de 32.000 millones falla las 40 multiplicaciones de `ordenes_escritas`
(20 con «Contesta solo con el número», 20 sin restricción): contesta de una vez, con el tamaño
bueno y las cifras del medio mal. Aquí se le pide lo contrario: que escriba la cuenta paso a paso
antes del resultado. Lo que escribe en cada paso queda delante para el paso siguiente; no hay nada
más. Si así acierta más, la diferencia la ha hecho escribir más texto, no otra máquina.

Las mismas 20 preguntas (la misma semilla que `ordenes_escritas`), sin herramienta, y siempre el
trozo más probable. No se mide ningún tiempo.

Predicciones, escritas ANTES de ejecutar (10 de octubre de 2026):
  - acierta entre 8 y 18 de 20 (sin pasos fueron 0 de 20 las dos veces);
  - en 18 o más termina con «Resultado:» como se le pide;
  - ninguna respuesta llega al límite de trozos sin terminar.

El resultado se lee del número que sigue al último «Resultado:»; si no lo hay, del último número
de la respuesta, y se cuenta aparte.

Rótulo (auditoría del 10 de octubre de 2026): la primera columna de la tabla, que no tenía
cabecera, pasa a llamarse «lo que se cuenta». No se volvió a ejecutar: se cambió aquí y, con la
misma sustitución de texto, en la salida guardada. Es el procedimiento de L25.

Uso (desde codigo/, con HF_HUB_OFFLINE=1 para no descargar nada):
    python pasos_escritos.py --selftest
    python pasos_escritos.py > ../datos/salidas/pasos_escritos.txt
"""

# ======================= CONSTANTES =======================

PREGUNTA_PASOS = ("¿Cuánto es {a} por {b}? Escribe la cuenta paso a paso y termina con una "
                  "línea «Resultado: » y el número.")
MAX_NUEVOS_PASOS = 900      # una multiplicación de cuatro por cuatro cifras, escrita entera
SALIDA_CSV = "../datos/salidas/pasos_escritos.csv"

# ==========================================================

import argparse
import csv
import importlib.metadata
import platform
import random
import re
import sys

from formato import miles, muestra_editorial, tabla_editorial
from pasos_a_la_vista import faltan
from ordenes_escritas import (MODELO_GRANDE, N_PREGUNTAS, SEMILLA, cargar, enunciado, escribir,
                              liberar, preguntas, renglones, ultimo_numero)

# Orden 97: cifra que la nota escribe con letra; si cambia el parámetro, falla aquí y no en la página.
EN_LETRA = {20: "veinte"}

RESULTADO = re.compile(r"Resultado:\s*\**\s*([\d.,   ]*\d)")


def nota_cuenta(f, titulo):
    """La nota de la muestra de una cuenta escrita paso a paso (Orden 97: para el lector)."""
    if "falla" in titulo:
        if faltan(f["texto"], f["a"], f["b"]):
            return (f"Falta algún producto parcial: {miles(f['numero'])} en lugar de "
                    f"{miles(f['prod'])}.")
        return (f"Los cuatro productos parciales están bien; el error está en la suma final: "
                f"{miles(f['numero'])} en lugar de {miles(f['prod'])}.")
    # (la muestra que acierta no está en el borrador del capítulo: se queda con su nota de siempre)
    return (f"Modelo de 32.000M, adiestrado, sin herramienta. Producto exacto: "
            f"{f['prod']}; leído: {f['numero']}. Texto literal; los renglones largos se parten "
            "donde caben.")


def leer_resultado(texto):
    """(número, de dónde): del último «Resultado:», o del último número si no lo hay."""
    ms = RESULTADO.findall(texto)
    if ms:
        return ultimo_numero(ms[-1]), "Resultado"
    return ultimo_numero(texto), "último número"


def selftest():
    ok = True
    ps = preguntas()

    # [1] test nulo: los resultados de otra pregunta no aciertan ninguna
    rng = random.Random(SEMILLA + 1)
    orden = list(range(N_PREGUNTAS))
    while any(i == j for i, j in enumerate(orden)):
        rng.shuffle(orden)
    falsos = sum(ps[i][2] == ps[j][2] for i, j in enumerate(orden))
    p1 = falsos == 0
    ok &= p1
    print(f"[1] test nulo         resultados de otra pregunta: {falsos} aciertos de "
          f"{N_PREGUNTAS}: {'bien' if p1 else 'MAL'}")

    # [2] señal implantada: una respuesta con pasos y el producto exacto al final se lee como
    #     acierto, aunque los pasos lleven otros números detrás
    a, b, prod = ps[0]
    texto = (f"{a} × {b}:\n{a} × 4000 = {a * 4000}\n{a} × 800 = {a * 800}\n"
             f"{a} × 20 = {a * 20}\nSuma: {a * 4820}\nResultado: {prod:,}".replace(",", ".")
             + "\n(comprobado en 2 pasos)")
    n, de = leer_resultado(texto)
    p2 = n == prod and de == "Resultado"
    ok &= p2
    print(f"[2] señal implantada  respuesta con pasos y «Resultado: {prod}» y un número detrás: "
          f"leído {n} (de {de}): {'bien' if p2 else 'MAL'}")

    # [3] invariante: las preguntas son las de ordenes_escritas, el enunciado no lleva
    #     herramienta, y sin «Resultado:» se lee el último número y se dice
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(MODELO_GRANDE[1], local_files_only=True)
    t = enunciado(tk, [{"role": "user", "content": PREGUNTA_PASOS.format(a=a, b=b)}], False)
    sin = leer_resultado("La cuenta da 123 y luego 456")
    p3 = (ps == preguntas() and "paso a paso" in t and "calculadora" not in t
          and sin == (456, "último número"))
    ok &= p3
    print(f"[3] invariante        mismas preguntas, enunciado sin herramienta, sin «Resultado:» "
          f"lee {sin}: {'bien' if p3 else 'MAL'}")

    print()
    print("SELFTEST: las tres pruebas pasan." if ok else "SELFTEST: FALLA.")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)

    print(f"máquina: {platform.platform()}; Python {platform.python_version()}")
    print(f"mlx {importlib.metadata.version('mlx')}; "
          f"mlx-lm {importlib.metadata.version('mlx-lm')}")
    print()
    nombre, repo = MODELO_GRANDE
    modelo, tok, _ = cargar(repo)
    filas = []
    for a, b, prod in preguntas():
        texto = enunciado(tok, [{"role": "user", "content": PREGUNTA_PASOS.format(a=a, b=b)}],
                          False)
        escrito, _ = escribir(modelo, tok, texto, MAX_NUEVOS_PASOS)
        n, de = leer_resultado(escrito)
        largo = len(tok.encode(escrito, add_special_tokens=False))
        filas.append({"a": a, "b": b, "prod": prod, "numero": n, "de": de, "trozos": largo,
                      "cortada": largo >= MAX_NUEVOS_PASOS, "texto": escrito})
    del modelo
    liberar()

    aciertos = sum(f["numero"] == f["prod"] for f in filas)
    con_resultado = sum(f["de"] == "Resultado" for f in filas)
    cortadas = sum(f["cortada"] for f in filas)
    largos = sorted(f["trozos"] for f in filas)
    mediana = (largos[9] + largos[10]) // 2
    print("\n".join(tabla_editorial(
        f"Multiplicar escribiendo la cuenta paso a paso ({nombre.split()[0]})",
        ["lo que se cuenta", "de 20"],
        [["aciertos", f"{aciertos}"],
         ["terminan con «Resultado:»", f"{con_resultado}"],
         ["se cortan sin terminar", f"{cortadas}"]], "id",
        [# Orden 97: notas para el lector. El modelo (adiestrado y comprimido), la pregunta exacta
         # (PREGUNTA_PASOS), el largo de las respuestas (mediana 277, de 161 a 771, límite
         # MAX_NUEVOS_PASOS) y que se lea el último número si falta «Resultado:» (`leer_resultado`) ya
         # no van en la nota.
         f"El modelo grande, sin calculadora, con las mismas {EN_LETRA[N_PREGUNTAS]} multiplicaciones, "
         "pidiéndole que escriba la cuenta paso a paso y acabe con «Resultado:» y el número.",
         "Acierta: el número de detrás de «Resultado:» es el producto exacto."])))
    print()
    ej = next((f for f in filas if f["numero"] == f["prod"]), None)
    mal = next((f for f in filas if f["numero"] != f["prod"]), None)
    for f, titulo in [(ej, "Una cuenta escrita paso a paso que acierta"),
                      (mal, "Una cuenta escrita paso a paso que falla")]:
        if f is None:
            print(f"({titulo}: no hay ninguna.)")
            print()
            continue
        print("\n".join(muestra_editorial(
            titulo,
            [f"Pregunta: ¿Cuánto es {f['a']} por {f['b']}?", ""] + renglones(f["texto"].strip()),
            [nota_cuenta(f, titulo)])))
        print()

    with open(SALIDA_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["a", "b", "producto", "leido", "de_donde", "trozos", "cortada", "respuesta"])
        for f in filas:
            w.writerow([f["a"], f["b"], f["prod"], f["numero"], f["de"], f["trozos"],
                        int(f["cortada"]), f["texto"]])


if __name__ == "__main__":
    main()
