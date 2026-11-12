#!/usr/bin/env python3
"""
Capítulo 12 — lo que no viene del entrenamiento.

Compara el MISMO modelo antes y después de la fase de adiestramiento que lo convierte en algo
que conversa. Mismo tamaño, mismo material de partida: lo único que cambia es ese añadido.

Tres columnas:
  - en crudo: el modelo base, con el enunciado tal cual
  - adiestrado, enunciado tal cual: para aislar el efecto del adiestramiento
  - adiestrado, con su formato de conversación: como se usa de verdad

Uso:
    python crudo_o_adiestrado.py
    python crudo_o_adiestrado.py --selftest
"""

# ======================= CONSTANTES =======================

PAREJAS = [("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"),
           ("Qwen/Qwen2.5-7B",   "Qwen/Qwen2.5-7B-Instruct")]
DTYPE = "bfloat16"
SEMILLA = 20260914
TOP_N = 6
MAX_NUEVOS = 40

# 1. La misma frase del capítulo 7, para cobrar aquella deuda
FRASE_CAP7 = "La capital de Francia es"

# 2. Una pregunta directa, sin ejemplos delante
PREGUNTA = "¿Cuál es la capital de Francia?"

# 3. La batería del capítulo 11, tal cual
from crecer import ETIQUETA_TAMANO, TAREAS, TAREA_CONTROL, acierta, evaluar   # misma batería

UMBRAL_CONTROL = 0.99
SALIDA_CSV = "../datos/salidas/crudo_o_adiestrado.csv"

# ==========================================================

import argparse
import csv
import datetime
import platform
import sys

from formato import comprobar_ancho, pct, tabla_de_probabilidades
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def cargar(nombre):
    tok = AutoTokenizer.from_pretrained(nombre)
    modelo = AutoModelForCausalLM.from_pretrained(nombre, dtype=getattr(torch, DTYPE))
    modelo.eval()
    return tok, modelo


def con_formato(tok, texto):
    """Envuelve el enunciado en el formato de conversación del modelo adiestrado."""
    assert tok.chat_template is not None, \
        "Se esperaba un modelo con formato de conversación; este no lo tiene"
    return tok.apply_chat_template([{"role": "user", "content": texto}],
                                   tokenize=False, add_generation_prompt=True)


def siguientes(tok, modelo, texto, n=TOP_N):
    ids = tok(texto, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        logits = modelo(ids).logits[0, -1]
    p = torch.softmax(logits.float(), dim=-1)
    suma = float(p.sum())
    assert abs(suma - 1.0) < 1e-3, \
        f"Invariante roto: se esperaba que las probabilidades sumaran 1; se encontró {suma:.6f}"
    top = torch.topk(p, n)
    return [(tok.decode([int(i)]), float(v)) for v, i in zip(top.values, top.indices)]


def continuar(tok, modelo, texto, maximo=MAX_NUEVOS):
    ids = tok(texto, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        salida = modelo.generate(ids, max_new_tokens=maximo, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
    return tok.decode(salida[0, ids.shape[1]:], skip_special_tokens=True).strip()


def evaluar_con_formato(tok, modelo):
    """La batería, pero envolviendo cada enunciado en el formato de conversación."""
    res = {}
    for nombre, items in TAREAS.items():
        aciertos = sum(acierta(continuar(tok, modelo, con_formato(tok, p), 12), e)
                       for p, e in items)
        res[nombre] = aciertos / len(items)
    return res


def selftest():
    fallos = []
    base, instruido = PAREJAS[0]
    tok_b, mod_b = cargar(base)
    tok_i, mod_i = cargar(instruido)

    # 1. TEST NULO — un enunciado sin sentido no debe producir una respuesta con sentido en
    #    ninguno de los dos. Sirve para comprobar que la diferencia que midamos después viene
    #    del adiestramiento y no de que el montaje regale respuestas.
    basura = "qx zr vb kk pl ñt"
    r_b = continuar(tok_b, mod_b, basura, 12)
    r_i = continuar(tok_i, mod_i, con_formato(tok_i, basura), 12)
    print(f"[1] test nulo         en crudo: {r_b[:40]!r}")
    print(f"                      adiestrado: {r_i[:40]!r}")
    if acierta(r_b, "París") or acierta(r_i, "París"):
        fallos.append("test nulo: un enunciado sin sentido produce la respuesta esperada; "
                      "el montaje filtra la solución")

    # 2. SEÑAL IMPLANTADA — copiar la palabra anterior. Los dos tienen que poder.
    ok_b = sum(acierta(continuar(tok_b, mod_b, p, 8), e) for p, e in TAREA_CONTROL) / len(TAREA_CONTROL)
    print(f"[2] señal implantada  copiar la palabra anterior, en crudo: {ok_b:.3f}")
    if ok_b < UMBRAL_CONTROL:
        fallos.append(f"señal implantada: el modelo en crudo no copia una palabra ({ok_b:.3f})")

    # 3. INVARIANTE DEL DOMINIO — se elige siempre lo más probable: dos ejecuciones iguales.
    a = continuar(tok_i, mod_i, con_formato(tok_i, PREGUNTA), 12)
    b = continuar(tok_i, mod_i, con_formato(tok_i, PREGUNTA), 12)
    print(f"[3] invariante        dos ejecuciones idénticas: {'sí' if a == b else 'NO'}")
    if a != b:
        fallos.append(f"invariante: dos ejecuciones distintas: {a!r} y {b!r}")

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

    # Cabecera con máquina y fecha (fallo 4.6): la salida dice dónde y cuándo se midió. La
    # fecha solo aquí, nunca dentro del cálculo.
    print(f"Medido el {datetime.date.today()} en {platform.platform()}; "
          f"generación determinista (do_sample=False).")
    filas = []
    for base, instruido in PAREJAS:
        # "0.5B" (nombre corto tras quitar "Qwen2.5-") tiene un punto decimal inglés y ya
        # no lleva la barra "Qwen/" al lado que lo identificaba como nombre, así que
        # verificar_cifras.py lo marca como cifra suelta. Mismo rótulo corto que crecer.py.
        etiqueta = ETIQUETA_TAMANO[base]
        print(f"\n{'='*74}\nTAMAÑO {etiqueta}\n{'='*74}")

        try:
            tok_b, mod_b = cargar(base)
            tok_i, mod_i = cargar(instruido)
        except Exception as e:
            print(f"SALTADO {etiqueta}: {type(e).__name__}: {str(e)[:120]}")
            continue

        print(f"\n--- 1. «{FRASE_CAP7}» -> ¿qué viene después? ---")
        for titulo, tok, mod, texto in (
                ("en crudo", tok_b, mod_b, FRASE_CAP7),
                ("adiestrado, enunciado tal cual", tok_i, mod_i, FRASE_CAP7),
                ("adiestrado, con su formato", tok_i, mod_i, None)):
            t = con_formato(tok_i, FRASE_CAP7) if texto is None else texto
            top = siguientes(tok, mod, t)
            # La misma tabla del capítulo 7, con los mismos rótulos y la misma barra (formato.py).
            print(f"  {titulo}:")
            pares = [(w.replace(" ", "_"), v) for w, v in top]
            for l in tabla_de_probabilidades(pares, decimales=1):
                print("  " + l)
            filas += [[etiqueta, titulo, "siguiente_palabra", w, f"{v:.4f}"] for w, v in top]

        print(f"\n--- 2. Pregunta directa: «{PREGUNTA}» ---")
        print(f"  en crudo -> {continuar(tok_b, mod_b, PREGUNTA)!r}")
        print(f"  adiestrado, enunciado tal cual -> {continuar(tok_i, mod_i, PREGUNTA)!r}")
        print(f"  adiestrado, con su formato -> {continuar(tok_i, mod_i, con_formato(tok_i, PREGUNTA))!r}")

        print("\n--- 3. La batería del capítulo 11 ---")
        r_b = evaluar(tok_b, mod_b)
        r_i = evaluar(tok_i, mod_i)
        r_f = evaluar_con_formato(tok_i, mod_i)
        print(f"  {'tarea':<28}{'crudo':>9}{'adiestrado':>12}{'con su formato':>16}")
        for t in TAREAS:
            print(f"  {t:<28}{pct(r_b[t], 0):>9}{pct(r_i[t], 0):>12}{pct(r_f[t], 0):>16}")
            filas.append([etiqueta, "bateria", t, f"{r_b[t]:.3f}",
                          f"{r_i[t]:.3f}", f"{r_f[t]:.3f}"])
        m = [sum(r.values()) / len(TAREAS) for r in (r_b, r_i, r_f)]
        print(f"  {'media':<28}{pct(m[0], 0):>9}{pct(m[1], 0):>12}{pct(m[2], 0):>16}")
        print()
        # La clave debajo de la tabla la imprime el programa (regla 9) y el libro la copia (regla 6).
        for l in comprobar_ancho([
                "  las cifras de la tabla son aciertos sobre 5 preguntas por tarea.",
                f"  tamaño del modelo en las tres columnas: {etiqueta} (nominal).",
                "  «crudo»: el modelo base, con el enunciado tal cual.",
                "  «adiestrado»: el mismo modelo ya adiestrado, enunciado tal cual.",
                "  «con su formato»: adiestrado, con su formato de conversación real."]):
            print(l)

        del mod_b, mod_i

    with open(SALIDA_CSV, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([["tamaño", "columna", "clave", "a", "b", "c"]] + filas)
    print(f"\nEscrito {SALIDA_CSV}")


if __name__ == "__main__":
    main()
