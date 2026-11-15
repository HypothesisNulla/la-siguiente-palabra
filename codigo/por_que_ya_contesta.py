#!/usr/bin/env python3
"""
Capítulo 12 — ¿por qué la máquina en crudo de hoy ya contesta? (Orden 172, 15 de noviembre de 2026).

Las órdenes 168 a 170 midieron que `Qwen/Qwen2.5-7B`, en crudo, ya contesta a las peticiones, cuando en 2022 GPT-3 en
crudo seguía la lista. La hipótesis es que el texto con que se entrena una máquina de hoy ya lleva preguntas con su
respuesta y textos escritos por asistentes. Este programa hace dos pruebas que APUNTAN a eso (no lo demuestran):

  PARTE 1 (172.2) — una máquina en crudo de antes de ChatGPT. `EleutherAI/pythia-6.9b`, entrenada con The Pile (un texto
  reunido antes de 2021), al lado de `Qwen/Qwen2.5-7B`, en crudo, con las cinco peticiones de la 168 y sus preguntas de
  la 169, en inglés (The Pile es casi todo inglés). La petición sola y un salto de línea; el trozo más probable, sin
  sorteo, hasta 80 trozos; la misma regla escrita de «sigue la lista o contesta», con sus verbos de petición en inglés.

  PARTE 2 (172.3) — la prueba de las marcas. Las marcas del formato de conversación de Qwen (`<|im_start|>`, `<|im_end|>`
  y los nombres «system», «user» y «assistant») son de Qwen: no salen en textos corrientes. Si la máquina EN CRUDO las
  usa bien, es que en su entrenamiento ya había textos con esas marcas. A `Qwen/Qwen2.5-7B` y `Qwen/Qwen2.5-0.5B` en
  crudo se les dan las cinco peticiones en castellano dentro del formato de conversación, igual que a la adiestrada en la
  168, y se mira (a) qué escribe, (b) si cierra su turno con `<|im_end|>` y en qué trozo, (c) la probabilidad que le da
  a `<|im_end|>` justo donde la adiestrada cierra su turno, y la que le da al final de su propia respuesta, y (d) si
  después escribe otra marca (`<|im_start|>` y un nombre), inventándose el siguiente turno. PRUEBA NULA: lo mismo, con
  marcas que nadie usa (`<|abc|>`, `<|xyz|>`, «persona1», «persona2», «persona0»): si la máquina las trata igual de bien,
  la prueba no vale.

ADVERTENCIA (también va en la salida): las dos máquinas de la parte 1 se diferencian en mucho más que el texto de su
entrenamiento (tamaño, fecha, cómo se partió el texto, cuánto entrenaron). Esta prueba apunta, no demuestra.

Los textos se imprimen enteros y tal cual (con las marcas visibles). Los tiempos son de esta máquina.

Uso:
    python por_que_ya_contesta.py --selftest
    python por_que_ya_contesta.py > ../datos/salidas/por_que_ya_contesta.txt
"""

# ======================= CONSTANTES =======================

PYTHIA = "EleutherAI/pythia-6.9b"               # si no cupiera o tardara demasiado: "EleutherAI/pythia-2.8b"
QWEN_7B, QWEN_7B_INSTRUCT = "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct"
QWEN_05B, QWEN_05B_INSTRUCT = "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"
ETIQUETA = {QWEN_7B: "7.000M", QWEN_05B: "500M", PYTHIA: "Pythia 6.900M"}   # como los otros rótulos, en millones: «6.9B» sería un decimal a la inglesa
DTYPE = "bfloat16"
SEMILLA = 20260914
MAX_NUEVOS = 80          # trozos como mucho en la parte 1 (como la 168)
MAX_MARCAS = 160         # trozos como mucho en la parte 2: hay que ver qué pasa después de cerrar el turno
MAX_ADIESTRADA = 400     # trozos como mucho que se le dejan escribir a la adiestrada hasta que cierra su turno

# Las cinco peticiones en inglés, tal cual: (orden, pregunta)
PETICIONES_EN = [
    ("Explain to a 6 year old, in a few sentences, why the sky is blue.", "Why is the sky blue?"),
    ("Write a very short story in which a bear goes to the beach, makes friends with a seal and goes back home.",
     "Can you write a very short story in which a bear goes to the beach, makes friends with a seal and goes back home?"),
    ("Give me three tips to sleep better.", "How can I sleep better?"),
    ("Translate into Spanish: The cat sleeps on the chair.", "How do you say \"The cat sleeps on the chair\" in Spanish?"),
    ("Explain the moon landing to a 6 year old in a few sentences.", "How would you explain the moon landing to a 6 year old?"),
]

# REGLA (en inglés): la primera línea no vacía «sigue la lista» si es otra petición: acaba en «?» o empieza por un verbo
# de petición en imperativo; si no, «contesta»; si no escribe nada, «no escribe nada».
VERBOS_EN = ("explain", "write", "give", "translate", "describe", "tell", "list", "summarize", "summarise", "provide",
             "make", "create", "show", "say", "name", "define", "compare", "answer", "complete", "calculate", "find",
             "help", "suggest", "generate", "come", "draft", "outline", "state", "give", "please", "convert", "rewrite")

# Las marcas de la prueba nula: lo que sustituye a cada marca de Qwen
MARCAS_REALES = {"inicio": "<|im_start|>", "fin": "<|im_end|>", "system": "system", "user": "user", "assistant": "assistant"}
MARCAS_NULAS = {"inicio": "<|abc|>", "fin": "<|xyz|>", "system": "persona0", "user": "persona1", "assistant": "persona2"}
INSTRUCCION_DE_SERIE = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

# ==========================================================

import argparse
import datetime
import gc
import platform
import re
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from formato import coma
from la_misma_peticion import PETICIONES as PETICIONES_ES_3, repite

PETICIONES_ES = [(o, p) for o, p, _ in PETICIONES_ES_3]      # las cinco de la 168, con su pregunta de la 169

REGLA = ("La regla, escrita: se mira solo la primera línea no vacía de lo que escribe. «sigue la lista» si esa línea acaba en "
         "«?» o empieza por un verbo de petición en imperativo (" + ", ".join(sorted(set(VERBOS_EN))) +
         "). Si no, «contesta». Si no escribe nada, «no escribe nada». La regla no lee el sentido.")


def cargar(nombre):
    tok = AutoTokenizer.from_pretrained(nombre)
    modelo = AutoModelForCausalLM.from_pretrained(nombre, dtype=getattr(torch, DTYPE))
    modelo.eval()
    return tok, modelo


def liberar(*cosas):
    for c in cosas:
        del c
    gc.collect()


def clasificar_en(texto):
    """(clase, por qué) según la regla en inglés."""
    linea = next((l.strip() for l in texto.split("\n") if l.strip()), "")
    if not linea:
        return "no escribe nada", "no hay ninguna línea"
    if linea.endswith("?"):
        return "sigue la lista", "la primera línea es una pregunta"
    m = re.match(r"[A-Za-z']+", linea)
    palabra = m.group(0).lower() if m else ""
    if palabra in VERBOS_EN:
        return "sigue la lista", f"empieza por el verbo de petición «{palabra}»"
    return "contesta", "la primera línea no es una petición"


def escribir(tok, modelo, texto, maximo, parar=None, saltar_especiales=True):
    """(texto, ids de lo escrito): el trozo más probable, hasta `maximo`, parando en `parar` (ids) si se da."""
    ids = tok(texto, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        salida = modelo.generate(ids, max_new_tokens=maximo, do_sample=False,
                                 pad_token_id=tok.eos_token_id if tok.eos_token_id is not None else 0,
                                 **({"eos_token_id": parar} if parar else {}))
    nuevos = salida[0, ids.shape[1]:]
    return tok.decode(nuevos, skip_special_tokens=saltar_especiales), [int(i) for i in nuevos]


# ---- PARTE 1: Pythia y Qwen en crudo, en inglés ----------------------------------------------------------------

def parte_1():
    res, tiempos = {}, {}
    for nombre in (PYTHIA, QWEN_7B):
        t0 = time.time()
        tok, modelo = cargar(nombre)
        for k, (orden, pregunta) in enumerate(PETICIONES_EN):
            for forma, pet in (("orden", orden), ("pregunta", pregunta)):
                texto, ids = escribir(tok, modelo, pet + "\n", MAX_NUEVOS)
                res[(nombre, k, forma)] = (texto, len(ids))
        tiempos[nombre] = time.time() - t0
        liberar(modelo, tok)
    return res, tiempos


# ---- PARTE 2: las marcas -----------------------------------------------------------------------------------------

def prompt_de_conversacion(peticion, marcas):
    """El formato de conversación de Qwen, con las marcas que se den (las reales o las nulas)."""
    m = marcas
    return (f"{m['inicio']}{m['system']}\n{INSTRUCCION_DE_SERIE}{m['fin']}\n"
            f"{m['inicio']}{m['user']}\n{peticion}{m['fin']}\n"
            f"{m['inicio']}{m['assistant']}\n")


def analizar_marcas(texto, marcas):
    """(cierra, cuántos caracteres antes de cerrar, abre otro turno, qué nombre le pone): mira el texto escrito, con las
    marcas visibles. `abre otro turno`: tras la marca de cierre, la de inicio; el nombre, lo que sigue hasta el salto
    de línea."""
    fin, ini = marcas["fin"], marcas["inicio"]
    i = texto.find(fin)
    if i < 0:
        return False, None, False, None
    resto = texto[i + len(fin):].lstrip("\n")
    if resto.startswith(ini):
        nombre = resto[len(ini):].split("\n")[0].strip()
        return True, i, True, nombre
    return True, i, False, None


def logp_secuencia(modelo, ids_contexto, ids_secuencia):
    """Probabilidad de que, tras el contexto, venga exactamente esa secuencia de trozos (el producto de sus
    probabilidades una a una)."""
    ids = torch.tensor([ids_contexto + ids_secuencia])
    with torch.no_grad():
        p = torch.softmax(modelo(ids).logits[0].float(), dim=-1)
    base = len(ids_contexto)
    total = 1.0
    for j, t in enumerate(ids_secuencia):
        total *= float(p[base + j - 1, t])
    return total


def parte_2():
    """Por cada tamaño y cada juego de marcas (reales o nulas): lo que escribe en crudo con el formato de conversación,
    y las probabilidades de cierre."""
    todo, tiempos = {}, {}
    for base, instruido in ((QWEN_7B, QWEN_7B_INSTRUCT), (QWEN_05B, QWEN_05B_INSTRUCT)):
        t0 = time.time()
        # 1. La adiestrada, para saber dónde cierra su turno: su respuesta, y cuántos trozos escribe hasta cerrar.
        tok_i, mod_i = cargar(instruido)
        fin_id = tok_i.convert_tokens_to_ids(MARCAS_REALES["fin"])
        cierres = {}
        for k, (orden, _) in enumerate(PETICIONES_ES):
            texto = tok_i.apply_chat_template([{"role": "user", "content": orden}], tokenize=False, add_generation_prompt=True)
            assert texto == prompt_de_conversacion(orden, MARCAS_REALES), \
                "El formato de la plantilla no es el que la prueba monta a mano"
            t, ids = escribir(tok_i, mod_i, texto, MAX_ADIESTRADA, parar=[fin_id, tok_i.eos_token_id])
            cierres[k] = (ids[:-1] if ids and ids[-1] == fin_id else ids, bool(ids and ids[-1] == fin_id))
        liberar(mod_i, tok_i)
        # 2. La máquina en crudo, con las marcas reales y con las nulas.
        tok, modelo = cargar(base)
        fin_real = tok.convert_tokens_to_ids(MARCAS_REALES["fin"])
        for juego, marcas in (("reales", MARCAS_REALES), ("nulas", MARCAS_NULAS)):
            ids_fin = [fin_real] if juego == "reales" else tok(marcas["fin"], add_special_tokens=False)["input_ids"]
            for k, (orden, _) in enumerate(PETICIONES_ES):
                prompt = prompt_de_conversacion(orden, marcas)
                ctx = tok(prompt, return_tensors="pt")["input_ids"][0].tolist()
                texto, ids = escribir(tok, modelo, prompt, MAX_MARCAS, parar=[tok.eos_token_id], saltar_especiales=False)
                cierra, pos, abre, nombre = analizar_marcas(texto, marcas)
                # trozo en que cierra: cuántos trozos hay antes de la marca de cierre, más uno (el primero de la marca)
                trozo = (len(tok(texto[:pos], add_special_tokens=False)["input_ids"]) + 1) if cierra else None
                # y si termina su texto con su propio fin de texto (`<|endoftext|>`): en qué trozo
                fin_texto = tok.decode([tok.eos_token_id], skip_special_tokens=False)
                j_eot = texto.find(fin_texto)
                trozo_eot = (len(tok(texto[:j_eot], add_special_tokens=False)["input_ids"]) + 1) if j_eot >= 0 else None
                # (c1) donde la adiestrada cierra su turno: su respuesta es el contexto
                resp_i, cerro_i = cierres[k]
                p_en_su_cierre = logp_secuencia(modelo, ctx + resp_i, ids_fin) if cerro_i else None
                # (c2) al final de su propia respuesta: hasta justo antes de su cierre, o al final de lo escrito
                propia = ids[:trozo - 1] if trozo else (ids[:-1] if (ids and ids[-1] == tok.eos_token_id) else ids)
                p_propia = logp_secuencia(modelo, ctx + propia, ids_fin)
                # y, de comparación, la probabilidad media de cerrar en las demás posiciones de la respuesta de la adiestrada
                medias = []
                if resp_i:
                    ids_t = torch.tensor([ctx + resp_i])
                    with torch.no_grad():
                        p = torch.softmax(modelo(ids_t).logits[0].float(), dim=-1)
                    for j in range(len(resp_i)):
                        tot = float(p[len(ctx) + j - 1, ids_fin[0]])     # el primer trozo de la marca de cierre
                        medias.append(tot)
                # (c3) donde la adiestrada cierra su turno: la probabilidad de que la máquina en crudo escriba su propio fin de texto
                p_eot = logp_secuencia(modelo, ctx + resp_i, [tok.eos_token_id]) if cerro_i else None
                todo[(base, juego, k)] = dict(
                    texto=texto, cierra=cierra, trozo=trozo, abre=abre, nombre=nombre, trozo_eot=trozo_eot, p_eot=p_eot,
                    p_su_cierre=p_en_su_cierre, p_propia=p_propia,
                    p_resto=(sum(medias) / len(medias)) if medias else None,
                    trozos_adiestrada=len(resp_i), cerro_adiestrada=cerro_i)
        tiempos[base] = time.time() - t0
        liberar(modelo, tok)
    return todo, tiempos


# ---- selftest ---------------------------------------------------------------------------------------------------

def selftest():
    fallos = []

    # 1. TEST NULO — (a) la regla en inglés no regala «contesta» a una petición copiada, y (b) el analizador de marcas no
    #    ve cierre ni turno nuevo en un texto sin marcas, y el prompt de la prueba nula no lleva ninguna marca de Qwen.
    copiadas = [clasificar_en(p)[0] for o, q in PETICIONES_EN for p in (o, q)]
    sin = analizar_marcas("Just a plain text with no marks, as in any book.", MARCAS_REALES)
    nulo = prompt_de_conversacion("hola", MARCAS_NULAS)
    ok1 = (all(c == "sigue la lista" for c in copiadas) and sin == (False, None, False, None)
           and "im_start" not in nulo and "im_end" not in nulo and clasificar_en("  \n ")[0] == "no escribe nada")
    print(f"[1] test nulo         peticiones copiadas: {sum(c == 'sigue la lista' for c in copiadas)} de {len(copiadas)} «sigue la lista»; "
          f"un texto sin marcas no cierra ni abre turno; el prompt nulo no lleva ninguna marca de Qwen: {'sí' if ok1 else 'NO'}")
    if not ok1:
        fallos.append("test nulo: la regla o el analizador regalan un resultado")
    # 2. SEÑAL IMPLANTADA — un texto escrito a mano con las marcas: el analizador ve el cierre y el turno inventado, con
    #    las reales y con las nulas, y la regla separa una respuesta de una lista.
    a = analizar_marcas(f"Hello!{MARCAS_REALES['fin']}\n{MARCAS_REALES['inicio']}user\nthanks", MARCAS_REALES)
    b = analizar_marcas(f"Hello!{MARCAS_NULAS['fin']}\n{MARCAS_NULAS['inicio']}persona1\nthanks", MARCAS_NULAS)
    ok2 = (a == (True, 6, True, "user") and b == (True, 6, True, "persona1")
           and clasificar_en("The sky is blue because of Rayleigh scattering.")[0] == "contesta"
           and clasificar_en("Why is the grass green?")[0] == "sigue la lista")
    print(f"[2] señal implantada  un texto con marcas a mano: cierra y se inventa el turno de «{a[3]}» (reales) y de «{b[3]}» (nulas): "
          f"{'sí' if ok2 else 'NO'}")
    if not ok2:
        fallos.append(f"señal implantada: {a} y {b}")
    # 3. INVARIANTE — se escribe siempre lo más probable (dos ejecuciones iguales, con el pequeño), el formato montado a mano
    #    es el de la plantilla del adiestrado y la probabilidad de una secuencia está entre 0 y 1.
    tok, modelo = cargar(QWEN_05B)
    p1 = prompt_de_conversacion(PETICIONES_ES[2][0], MARCAS_REALES)
    t1, _ = escribir(tok, modelo, p1, 20, saltar_especiales=False)
    t2, _ = escribir(tok, modelo, p1, 20, saltar_especiales=False)
    tok_i = AutoTokenizer.from_pretrained(QWEN_05B_INSTRUCT)
    plantilla = tok_i.apply_chat_template([{"role": "user", "content": PETICIONES_ES[2][0]}], tokenize=False,
                                          add_generation_prompt=True)
    ctx = tok(p1, return_tensors="pt")["input_ids"][0].tolist()
    p = logp_secuencia(modelo, ctx, [tok.convert_tokens_to_ids(MARCAS_REALES["fin"])])
    ok3 = t1 == t2 and plantilla == p1 and 0.0 <= p <= 1.0
    print(f"[3] invariante        dos ejecuciones iguales: {'sí' if t1 == t2 else 'NO'}; el formato a mano es el de la plantilla: "
          f"{'sí' if plantilla == p1 else 'NO'}; la probabilidad de cerrar, entre 0 y 1: {'sí' if 0 <= p <= 1 else 'NO'}")
    if not ok3:
        fallos.append("invariante: dos ejecuciones distintas, el formato no es el de la plantilla, o la probabilidad no vale")
    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


# ---- main -------------------------------------------------------------------------------------------------------

def pct1(x):
    """Un porcentaje con coma decimal; si es menor que 0,01 %, en notación científica (la de la probabilidad, no el %)."""
    if x is None:
        return "—"
    if x == 0:
        return "0 %"
    if 100 * x < 0.01:
        mant, exp = f"{100 * x:.1e}".split("e")
        return f"{mant.replace('.', ',')}e{exp} %"
    return coma(100 * x, 2) + " %"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())

    torch.manual_seed(SEMILLA)
    print(f"Medido el {datetime.date.today()} en {platform.platform()}; el trozo más probable, sin sorteo (do_sample=False), "
          f"en procesador, en {DTYPE}.")
    print(f"Bibliotecas: torch {torch.__version__}. Modelos: {PYTHIA} (The Pile, de antes de 2021), {QWEN_7B}, {QWEN_05B} "
          "(y sus -Instruct, solo para saber dónde cierran su turno).")
    print("\nADVERTENCIA: en la parte 1 las dos máquinas se diferencian en mucho más que el texto de su entrenamiento (tamaño, "
          "fecha, cómo se partió el texto, cuánto entrenaron, qué tokenizador). Esta prueba apunta, no demuestra.")

    # ---------- parte 1
    res1, t1 = parte_1()
    print(f"\n{'=' * 74}\nPARTE 1 — UNA MÁQUINA EN CRUDO DE ANTES DE ChatGPT, EN INGLÉS\n{'=' * 74}")
    print("La petición sola y un salto de línea; el trozo más probable, hasta 80 trozos.")
    print(REGLA)
    print(f"Tiempo en esta máquina: " + "; ".join(f"{ETIQUETA[n]} {coma(t / 60)} min" for n, t in t1.items()) + " (carga incluida).")
    resumen1 = {}
    for k, (orden, pregunta) in enumerate(PETICIONES_EN, 1):
        print(f"\n--- petición {k} ---\n  orden:    {orden}\n  pregunta: {pregunta}")
        for forma, pet in (("orden", orden), ("pregunta", pregunta)):
            for nombre in (PYTHIA, QWEN_7B):
                texto, n = res1[(nombre, k - 1, forma)]
                clase, porque = clasificar_en(texto)
                extra = repite(texto, pet)
                resumen1[(nombre, k, forma)] = clase + (" (repite la petición)" if extra else "")
                print(f"\n[{ETIQUETA[nombre]}, {forma}, en crudo] ({n} trozos)")
                print(texto)
                print(f"-> {clase} ({porque}){extra}")
    print(f"\n{'-' * 74}\nTABLA, parte 1: en crudo, ¿contesta o sigue la lista?\n{'-' * 74}")
    print(f"{'petición':<12}{'forma':<10}{'Pythia 6.900M':<38}{'Qwen2.5 7.000M'}")
    for k in range(1, 6):
        for forma in ("orden", "pregunta"):
            print(f"{k:<12}{forma:<10}{resumen1[(PYTHIA, k, forma)]:<38}{resumen1[(QWEN_7B, k, forma)]}")

    # ---------- parte 2
    res2, t2 = parte_2()
    print(f"\n{'=' * 74}\nPARTE 2 — LA PRUEBA DE LAS MARCAS (Qwen en crudo, con el formato de conversación)\n{'=' * 74}")
    print("Las cinco peticiones en castellano de la 168 (la orden), dentro del formato de conversación, con la instrucción de "
          "serie. Se deja escribir hasta 160 trozos (no se para en el cierre, para ver si inventa otro turno).")
    print(f"Marcas reales: {MARCAS_REALES}. Marcas nulas: {MARCAS_NULAS}.")
    print("(c1) probabilidad de la marca de cierre justo donde la adiestrada (-Instruct) cierra su turno (si lo cierra antes "
          f"de {MAX_ADIESTRADA} trozos); (c2) la probabilidad de la marca de cierre al final de su propia respuesta; "
          "«resto» es la probabilidad media de empezar la marca de cierre en las demás posiciones de la respuesta de la adiestrada.")
    print("Tiempo en esta máquina: " + "; ".join(f"{ETIQUETA[n]} {coma(t / 60)} min" for n, t in t2.items()) + " (carga incluida).")
    for base in (QWEN_7B, QWEN_05B):
        for juego in ("reales", "nulas"):
            print(f"\n{'-' * 74}\n{ETIQUETA[base]}, marcas {juego}\n{'-' * 74}")
            for k, (orden, _) in enumerate(PETICIONES_ES):
                d = res2[(base, juego, k)]
                print(f"\n[petición {k + 1}: {orden}]")
                print(f"(a) lo que escribe, con las marcas visibles:\n{d['texto']}")
                print(f"(b) cierra su turno con la marca de cierre: {'sí, en el trozo ' + str(d['trozo']) if d['cierra'] else 'no'}"
                      + (f"; termina su texto con su propio fin de texto en el trozo {d['trozo_eot']}" if d["trozo_eot"] else ""))
                print(f"(c) probabilidad de la marca de cierre: donde cierra la adiestrada (c1) {pct1(d['p_su_cierre'])}"
                      + ("" if d["cerro_adiestrada"] else " (la adiestrada no cerró antes de %d trozos)" % MAX_ADIESTRADA)
                      + f" ({d['trozos_adiestrada']} trozos); al final de su propia respuesta (c2) {pct1(d['p_propia'])}; "
                      f"resto {pct1(d['p_resto'])}; probabilidad de su propio fin de texto donde cierra la adiestrada (c3) {pct1(d['p_eot'])}")
                print(f"(d) después de cerrar escribe otra marca de inicio: "
                      f"{'sí, con el nombre «' + d['nombre'] + '»' if d['abre'] else 'no'}")
        print(f"\n{'-' * 74}\nTABLA, parte 2, {ETIQUETA[base]}: de las 5 peticiones\n{'-' * 74}")
        for juego in ("reales", "nulas"):
            ds = [res2[(base, juego, k)] for k in range(5)]
            print(f"marcas {juego:<7}: cierra su turno {sum(d['cierra'] for d in ds)} de 5; "
                  f"escribe otro turno {sum(d['abre'] for d in ds)} de 5; "
                  f"c1 media {pct1(_media([d['p_su_cierre'] for d in ds]))}; c2 media {pct1(_media([d['p_propia'] for d in ds]))}; "
                  f"resto, media {pct1(_media([d['p_resto'] for d in ds]))}; "
                  f"termina con su fin de texto {sum(1 for d in ds if d['trozo_eot'])} de 5; c3 media {pct1(_media([d['p_eot'] for d in ds]))}")


def _media(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


if __name__ == "__main__":
    main()
