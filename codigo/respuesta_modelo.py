#!/usr/bin/env python3
"""
Capítulo 8 — ¿acierta el modelo el referente, y con cuánto aplomo?

Elección forzada entre «vaso» y «cajón» para tres frases que solo se diferencian en el
adjetivo final. La tercera («caro») no tiene respuesta correcta: sirve para ver si el
modelo baja la confianza cuando la pregunta no tiene sentido. Mide la SALIDA, no las
tripas.

Uso:
    python respuesta_modelo.py
    python respuesta_modelo.py --selftest
"""

# ======================= CONSTANTES =======================

MODELOS = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"]

CASOS = {
    "alto": ("El vaso no cabía en el cajón porque era demasiado alto.", "vaso"),
    "bajo": ("El vaso no cabía en el cajón porque era demasiado bajo.", "cajón"),
    "caro": ("El vaso no cabía en el cajón porque era demasiado caro.", None),  # sin respuesta
}

# L24 (28 sep 2026): la plantilla acaba en «El » con el espacio, y las opciones van sin él
# («vaso», «cajón»). El modelo trocea «El vaso» como «El» + « vaso»: detrás de un espacio ya
# escrito, «vaso» sin espacio es un trozo que casi nunca viene, y las dos opciones salían con unas
# pocas millonésimas. Desde el 28 se mide con el espacio en su sitio (ESPACIO_EN_SU_SITIO): el texto
# acaba en «El» y las opciones son « vaso» y « cajón». El montaje viejo se conserva como
# eleccion_forzada(..., espacio_en_su_sitio=False), porque la_seguridad_por_dentro.py lo enseña.
# La salida del 20 de septiembre está en datos/salidas/_antiguos/.
ESPACIO_EN_SU_SITIO = True
PLANTILLA = "{frase}\nPregunta: ¿Qué era demasiado {adj}?\nRespuesta: El "
OPCIONES = ("vaso", "cajón")

# Control: misma pregunta con un referente único y explícito. El modelo debe acertarlo;
# si falla aquí, el montaje no mide lo que dice medir.
CONTROL = ("El vaso era demasiado alto para el estante.", "alto", "vaso")

# bfloat16 para que los modelos de más de 1.000 millones quepan en memoria. La elección
# forzada compara dos log-probabilidades con márgenes grandes; la precisión no la decide.
DTYPE = "bfloat16"

SEMILLA = 20260914
SALIDA_CSV = "../datos/salidas/respuesta_modelo.csv"
UMBRAL_CONTROL = 0.60      # confianza mínima exigida en el control
TOL_SUMA = 1e-6

# ==========================================================

import argparse
import csv
import random
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from formato import ANCHO_CAJA_CITA, coma, comprobar_ancho


def fijar_semilla(semilla):
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)


def validar_entrada(casos):
    assert isinstance(casos, dict) and casos, \
        f"Se esperaba un dict no vacío de casos; se encontró {type(casos).__name__}"
    for clave, valor in casos.items():
        assert isinstance(valor, tuple) and len(valor) == 2, \
            f"Se esperaba (frase, respuesta) en el caso «{clave}»; se encontró {valor!r}"
        frase, esperada = valor
        assert frase.rstrip(".").split()[-1] == clave, (
            f"Se esperaba que la última palabra de la frase «{clave}» fuera «{clave}»; "
            f"se encontró «{frase.rstrip('.').split()[-1]}»"
        )
        assert esperada in OPCIONES or esperada is None, \
            f"Se esperaba que la respuesta del caso «{clave}» fuera una de {OPCIONES} o None; se encontró {esperada!r}"


def log_prob_continuacion(tok, modelo, prompt, continuacion):
    """Suma de log-probabilidades de la continuación dada el prompt."""
    ids_prompt = tok(prompt, return_tensors="pt")["input_ids"]
    ids_cont = tok(continuacion, return_tensors="pt", add_special_tokens=False)["input_ids"]
    assert ids_cont.shape[1] > 0, \
        f"Se esperaba al menos una unidad en la continuación «{continuacion}»; se encontraron 0"
    completo = torch.cat([ids_prompt, ids_cont], dim=1)
    with torch.no_grad():
        logits = modelo(completo).logits
    log_probs = torch.log_softmax(logits[0, :-1].float(), dim=-1)
    objetivo = completo[0, 1:]
    inicio = ids_prompt.shape[1] - 1
    tomados = log_probs[inicio:, :].gather(1, objetivo[inicio:].unsqueeze(1)).squeeze(1)
    valor = float(tomados.sum())
    assert not np.isnan(valor), \
        f"Se esperaba una log-probabilidad finita para «{continuacion}»; se encontró NaN"
    return valor


def eleccion_forzada(tok, modelo, frase, adj, espacio_en_su_sitio=False):
    prompt = PLANTILLA.format(frase=frase, adj=adj)
    if espacio_en_su_sitio:
        prompt = prompt.rstrip(" ")
    conts = [(" " + o) if espacio_en_su_sitio else o for o in OPCIONES]
    lp = np.array([log_prob_continuacion(tok, modelo, prompt, o) for o in conts])
    exp = np.exp(lp - lp.max())
    probs = exp / exp.sum()
    assert abs(probs.sum() - 1.0) < TOL_SUMA, \
        f"Invariante roto: se esperaba que las probabilidades sumaran 1; se encontró {probs.sum()}"
    ganadora = OPCIONES[int(np.argmax(probs))]
    return ganadora, float(probs.max()), dict(zip(OPCIONES, probs))


def medir(nombre):
    tok = AutoTokenizer.from_pretrained(nombre)
    modelo = AutoModelForCausalLM.from_pretrained(nombre, dtype=getattr(torch, DTYPE))
    modelo.eval()
    filas = []
    for clave, (frase, esperada) in CASOS.items():
        ganadora, confianza, _ = eleccion_forzada(tok, modelo, frase, clave, ESPACIO_EN_SU_SITIO)
        acierto = "-" if esperada is None else ("sí" if ganadora == esperada else "NO")
        filas.append((nombre, clave, esperada or "(ninguna)", ganadora, confianza, acierto))
    return tok, modelo, filas


def informe(filas):
    """La tabla, en castellano y dentro de la caja.

    La confianza se escribe con coma decimal: el libro está en castellano y un «0.977»
    impreso obliga a castellanizarlo a mano al copiarlo, que es justo lo que la regla 6
    prohíbe. Y las columnas se estrechan hasta caber en la caja del libro, medida en
    formato.py, para que el bloque no haya que retocarlo nunca al llevarlo a la página.
    """
    lineas = [
        f"{'modelo':<14}{'caso':<7}{'correcta':<11}{'responde':<9}{'confianza':>10}  acierta",
        "-" * 60,
    ]
    for m, c, e, g, conf, a in filas:
        lineas.append(
            f"{m.split('/')[-1]:<14}{c:<7}{e:<11}{g:<9}{coma(conf, 3):>10}  {a}"
        )
    print()
    print("\n".join(comprobar_ancho(lineas, ANCHO_CAJA_CITA)))


def selftest(tok, modelo):
    fallos = []
    frase_c, adj_c, esperada_c = CONTROL

    # 1. TEST NULO — frase barajada: sin estructura no debe haber una preferencia fuerte.
    rng = random.Random(SEMILLA)
    palabras = CASOS["alto"][0].rstrip(".").split()
    barajada = palabras[:]
    rng.shuffle(barajada)
    _, conf_nula, _ = eleccion_forzada(tok, modelo, " ".join(barajada) + ".", "alto",
                                       ESPACIO_EN_SU_SITIO)
    print(f"[1] test nulo         confianza con la frase barajada = {conf_nula:.3f}")
    if conf_nula > 0.95:
        fallos.append(f"test nulo: confianza {conf_nula:.3f} con una frase sin estructura; el montaje decide por sí solo")

    # 2. SEÑAL IMPLANTADA — referente único y explícito, debe recuperarse.
    ganadora, conf, _ = eleccion_forzada(tok, modelo, frase_c, adj_c, ESPACIO_EN_SU_SITIO)
    print(f"[2] señal implantada  «{frase_c}» -> {ganadora} ({conf:.3f}), esperada {esperada_c}")
    if ganadora != esperada_c or conf < UMBRAL_CONTROL:
        fallos.append(
            f"señal implantada: se esperaba «{esperada_c}» con confianza >= {UMBRAL_CONTROL}; "
            f"se obtuvo «{ganadora}» con {conf:.3f}"
        )

    # 3. INVARIANTE DEL DOMINIO — las dos opciones suman uno en los tres casos.
    ok = True
    for clave, (frase, _) in CASOS.items():
        _, _, probs = eleccion_forzada(tok, modelo, frase, clave, ESPACIO_EN_SU_SITIO)
        if abs(sum(probs.values()) - 1.0) >= TOL_SUMA:
            fallos.append(f"invariante: en «{clave}» las probabilidades suman {sum(probs.values())}")
            ok = False
    print(f"[3] invariante        las dos opciones suman 1 en los tres casos: {'sí' if ok else 'NO'}")

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

    fijar_semilla(SEMILLA)
    validar_entrada(CASOS)

    if args.selftest:
        tok = AutoTokenizer.from_pretrained(MODELOS[0])
        modelo = AutoModelForCausalLM.from_pretrained(MODELOS[0], dtype=getattr(torch, DTYPE))
        modelo.eval()
        sys.exit(selftest(tok, modelo))

    todas = []
    for nombre in MODELOS:
        _, _, filas = medir(nombre)
        todas.extend(filas)
    informe(todas)

    with open(SALIDA_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["modelo", "caso", "respuesta_correcta", "responde", "confianza", "acierta"])
        w.writerows(todas)
    print(f"\nEscrito {SALIDA_CSV}")


if __name__ == "__main__":
    main()
