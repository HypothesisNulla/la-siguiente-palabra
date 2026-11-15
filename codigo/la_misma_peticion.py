#!/usr/bin/env python3
"""
Capítulo 12 — la misma petición, en crudo y adiestrada; y cada petición, en forma de orden y de pregunta
(Órdenes 168, 169 y 170, 15 de noviembre de 2026).

Para cinco peticiones en castellano, el programa le da a dos máquinas el MISMO modelo —el de 7.000 millones de
pesos y el de 500 millones, de la familia Qwen2.5— antes y después del adiestramiento, y escribe lo que sale:

  - en crudo (`Qwen/Qwen2.5-7B`, `Qwen/Qwen2.5-0.5B`): la petición sola, seguida de un salto de línea, sin nada más;
  - adiestrada (`-Instruct`): la petición dentro de su formato de conversación, con la instrucción de serie que
    pone su plantilla (como en la tercera columna de la batería del capítulo 12).

Cada petición se da de tres formas: como orden, como pregunta y como pregunta sin los signos «¿» y «?» (Orden 170).
Son seis textos por petición y por tamaño: orden en crudo, pregunta en crudo, pregunta sin signos en crudo, orden
adiestrada, pregunta adiestrada, pregunta sin signos adiestrada.

Se escribe siempre el trozo más probable, sin sorteo, hasta 80 trozos o hasta que la máquina cierra su turno.
Debajo de cada texto va una línea que dice si lo primero que escribe PARECE SEGUIR LA LISTA (otra petición del mismo
estilo) o CONTESTA, con una regla escrita y sencilla que está más abajo (REGLA) y que se imprime en la salida.
La regla no lee el sentido: lee solo la primera línea. La tabla del libro, si la hay, la decide una persona leyendo
los textos.

Los textos se imprimen enteros y tal cual. Los tiempos son de esta máquina y van etiquetados como tales.

Uso:
    python la_misma_peticion.py --selftest
    python la_misma_peticion.py > ../datos/salidas/la_misma_peticion.txt
"""

# ======================= CONSTANTES =======================

PAREJAS = [("Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct"),
           ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct")]
FORMAS = ("orden", "pregunta", "pregunta sin signos")
ETIQUETA = {"Qwen/Qwen2.5-0.5B": "500M", "Qwen/Qwen2.5-7B": "7.000M"}   # los rótulos de la batería del capítulo 11
DTYPE = "bfloat16"
SEMILLA = 20260914
MAX_NUEVOS = 80          # trozos como mucho

# (la petición como orden, la misma como pregunta, la pregunta sin los signos «¿» y «?»), tal cual
PETICIONES = [
    ("Explica a un niño de seis años, en unas pocas frases, por qué el cielo es azul.",
     "¿Por qué el cielo es azul?",
     "Por qué el cielo es azul"),
    ("Escribe un cuento muy corto en el que un oso va a la playa, se hace amigo de una foca y vuelve a casa.",
     "¿Me escribes un cuento muy corto en el que un oso va a la playa, se hace amigo de una foca y vuelve a casa?",
     "Me escribes un cuento muy corto en el que un oso va a la playa, se hace amigo de una foca y vuelve a casa"),
    ("Dame tres consejos para dormir mejor.",
     "¿Cómo puedo dormir mejor?",
     "Cómo puedo dormir mejor"),
    ("Traduce al inglés: El gato duerme encima de la silla.",
     "¿Cómo se dice en inglés «El gato duerme encima de la silla»?",
     "Cómo se dice en inglés «El gato duerme encima de la silla»"),
    ("Explica la llegada a la Luna a un niño de seis años, en unas pocas frases.",
     "¿Cómo le explicarías a un niño de seis años la llegada a la Luna?",
     "Cómo le explicarías a un niño de seis años la llegada a la Luna"),     # la de OpenAI (2022), en castellano
]

# REGLA: la primera línea no vacía de lo que escribe «sigue la lista» si es otra petición, y si no, «contesta».
# Es otra petición si (a) empieza por «¿» o termina en «?», o (b) empieza por uno de estos verbos de petición, en
# imperativo, tal cual o con «me» o «te» pegado (por ejemplo «Cuéntame»). Si no escribe nada, «no escribe nada».
VERBOS_DE_PETICION = ("explica", "escribe", "dame", "traduce", "describe", "cuenta", "cuéntame", "dime", "redacta",
                      "resume", "enumera", "menciona", "haz", "hazme", "da", "pon", "indica", "responde", "completa",
                      "calcula", "busca", "ayúdame", "ayuda", "inventa", "compara", "ponme", "muéstrame", "enséñame",
                      "traduceme", "tradúceme", "escríbeme", "explícame", "sugiere", "propón", "enumera", "lista")

# ==========================================================

import argparse
import datetime
import gc
import platform
import re
import sys
import time

import torch
from formato import coma
from transformers import AutoModelForCausalLM, AutoTokenizer

REGLA = ("La regla, escrita: se mira solo la primera línea no vacía de lo que escribe. «sigue la lista» si esa línea "
         "es otra petición: empieza por «¿» o acaba en «?», o empieza por un verbo de petición en imperativo "
         "(" + ", ".join(sorted(set(VERBOS_DE_PETICION))) + "). Si no, «contesta». Si no escribe nada, «no escribe nada». "
         "La regla no lee el sentido.")


def cargar(nombre):
    tok = AutoTokenizer.from_pretrained(nombre)
    modelo = AutoModelForCausalLM.from_pretrained(nombre, dtype=getattr(torch, DTYPE))
    modelo.eval()
    return tok, modelo


def en_crudo(peticion):
    """La petición sola, seguida de un salto de línea."""
    return peticion + "\n"


def con_formato(tok, peticion):
    """La petición en el formato de conversación del modelo adiestrado, con la instrucción de serie de su plantilla."""
    assert tok.chat_template is not None, "Se esperaba un modelo con formato de conversación; este no lo tiene"
    return tok.apply_chat_template([{"role": "user", "content": peticion}], tokenize=False, add_generation_prompt=True)


def escribir(tok, modelo, texto):
    """(lo que escribe, cuántos trozos, si cerró su turno): el trozo más probable, hasta MAX_NUEVOS."""
    ids = tok(texto, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        salida = modelo.generate(ids, max_new_tokens=MAX_NUEVOS, do_sample=False, pad_token_id=tok.eos_token_id)
    nuevos = salida[0, ids.shape[1]:]
    cerro = bool(len(nuevos) and int(nuevos[-1]) in _fines(tok, modelo))
    return tok.decode(nuevos, skip_special_tokens=True), len(nuevos), cerro


def _fines(tok, modelo):
    """Los trozos con que la máquina cierra su turno o su texto (de su configuración de generación)."""
    fines = modelo.generation_config.eos_token_id
    fines = fines if isinstance(fines, (list, tuple)) else [fines]
    return {int(f) for f in fines if f is not None} | ({int(tok.eos_token_id)} if tok.eos_token_id is not None else set())


def clasificar(texto):
    """(clase, por qué) según la REGLA: miran solo la primera línea no vacía."""
    linea = next((l.strip() for l in texto.split("\n") if l.strip()), "")
    if not linea:
        return "no escribe nada", "no hay ninguna línea"
    if linea.startswith("¿") or linea.endswith("?"):
        return "sigue la lista", "la primera línea es una pregunta"
    primera = re.match(r"[\wáéíóúüñ]+", linea.lower())
    palabra = primera.group(0) if primera else ""
    for v in VERBOS_DE_PETICION:
        if palabra == v or (palabra.startswith(v) and palabra[len(v):] in ("me", "te", "nos", "lo", "la")):
            return "sigue la lista", f"empieza por el verbo de petición «{v}»"
    return "contesta", "la primera línea no es una petición"


def repite(texto, peticion):
    """« Además: repite su petición.» si la primera línea no vacía es la petición misma (sin signos ni mayúsculas ni
    espacios de más); si no, nada. Es una línea aparte de la regla, para las preguntas sin signos."""
    def limpia(s):
        return re.sub(r"[^\wáéíóúüñ]+", " ", s.lower()).strip()
    linea = next((l for l in texto.split("\n") if l.strip()), "")
    return " Además: repite su petición." if linea.strip() and limpia(linea) == limpia(peticion) else ""


def medir(base, instruido):
    """{(indice de petición, forma, version): (texto, trozos, cerró)} para un tamaño; carga de a uno (7.000M ocupa unos
    quince gigas en bfloat16 y dos a la vez no caben cómodos)."""
    res, tiempos = {}, {}
    for version, nombre in (("en crudo", base), ("adiestrada", instruido)):
        t0 = time.time()
        tok, modelo = cargar(nombre)
        for k, formas in enumerate(PETICIONES):
            for forma, peticion in zip(FORMAS, formas):
                texto = en_crudo(peticion) if version == "en crudo" else con_formato(tok, peticion)
                res[(k, forma, version)] = escribir(tok, modelo, texto)
        tiempos[version] = time.time() - t0
        del modelo, tok
        gc.collect()
    return res, tiempos


def selftest():
    fallos = []
    base, instruido = PAREJAS[1]       # el pequeño: es el que se carga deprisa
    tok_b, mod_b = cargar(base)
    tok_i, mod_i = cargar(instruido)

    # 1. TEST NULO — la regla no regala el resultado: si la máquina se limitara a repetir la petición (que es
    #    seguir la lista, no contestar), cada una de las diez formas se clasifica «sigue la lista»; y nada es nada.
    #    (Las preguntas sin signos no se prueban aquí: la regla mira «¿», «?» y los verbos, y una pregunta sin signos
    #    copiada, sin ninguno de los tres, saldría «contesta»; por eso la salida dice además si repite su petición.)
    copiadas = [clasificar(p)[0] for f in PETICIONES for p in f[:2]]
    nada = clasificar("  \n\n ")[0]
    ok1 = all(c == "sigue la lista" for c in copiadas) and nada == "no escribe nada"
    print(f"[1] test nulo         repetir la orden o la pregunta no cuenta como contestar: "
          f"{sum(c == 'sigue la lista' for c in copiadas)} de {len(copiadas)} «sigue la lista»; un texto vacío, «{nada}»")
    if not ok1:
        fallos.append(f"test nulo: la regla regala «contesta» a una petición copiada: {copiadas}, {nada}")

    # 2. SEÑAL IMPLANTADA — respuestas escritas a mano se clasifican «contesta», y listas escritas a mano,
    #    «sigue la lista»; y la máquina del tamaño pequeño, adiestrada, escribe algo con la forma esperada.
    respuestas = ["El cielo es azul porque la luz del sol se reparte en el aire.",
                  "Para dormir mejor, acuéstate siempre a la misma hora.",
                  "The cat sleeps on the chair.", "Había una vez un oso que fue a la playa."]
    listas = ["Explica a un niño de seis años por qué llueve.", "¿Por qué la noche es oscura?",
              "Cuéntame un cuento corto.", "Dame tres consejos para estudiar."]
    ok2 = all(clasificar(r)[0] == "contesta" for r in respuestas) and all(clasificar(l)[0] == "sigue la lista" for l in listas)
    texto, _, _ = escribir(tok_i, mod_i, con_formato(tok_i, "Dame tres consejos para dormir mejor."))
    print(f"[2] señal implantada  {len(respuestas)} respuestas a mano «contesta» y {len(listas)} listas a mano "
          f"«sigue la lista»: {'bien' if ok2 else 'MAL'}; adiestrada, el pequeño escribe {len(texto)} letras")
    if not ok2 or not texto.strip():
        fallos.append("señal implantada: la regla no separa las respuestas de las listas, o el adiestrado no escribe")

    # 3. INVARIANTE — se escribe siempre lo más probable: dos ejecuciones dan lo mismo; y el formato adiestrado lleva
    #    la instrucción de serie y el crudo, nada más que el salto de línea.
    sin_signos = all(f[2] == f[1].replace("¿", "").replace("?", "") for f in PETICIONES)
    a = escribir(tok_b, mod_b, en_crudo(PETICIONES[2][0]))
    b = escribir(tok_b, mod_b, en_crudo(PETICIONES[2][0]))
    formato = con_formato(tok_i, PETICIONES[2][0])
    ok3 = (a == b and en_crudo("x") == "x\n" and "system" in formato and formato.rstrip().endswith("assistant")
           and sin_signos)
    print(f"[3] invariante        dos ejecuciones iguales: {'sí' if a == b else 'NO'}; crudo = petición + salto de línea; "
          f"adiestrada con su instrucción de serie: {'sí' if ('system' in formato) else 'NO'}; "
          f"cada pregunta sin signos es la pregunta sin «¿» ni «?»: {'sí' if sin_signos else 'NO'}")
    if not ok3:
        fallos.append("invariante: dos ejecuciones distintas, o el montaje de los dos formatos no es el esperado")
    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())

    torch.manual_seed(SEMILLA)
    print(f"Medido el {datetime.date.today()} en {platform.platform()}; el trozo más probable, sin sorteo "
          f"(do_sample=False), en procesador, hasta {MAX_NUEVOS} trozos o hasta que cierra su turno.")
    print(f"Bibliotecas: torch {torch.__version__}.")
    print("En crudo: la petición sola, seguida de un salto de línea. Adiestrada: la petición dentro de su formato "
          "de conversación, con la instrucción de serie de su plantilla.\n")
    print(REGLA)
    todo = {}
    for base, instruido in PAREJAS:
        etiqueta = ETIQUETA[base]
        try:
            res, tiempos = medir(base, instruido)
        except Exception as e:
            print(f"\nSALTADO {etiqueta}: {type(e).__name__}: {str(e)[:160]}")
            continue
        todo[etiqueta] = (res, tiempos)
    for etiqueta, (res, tiempos) in todo.items():
        print(f"\n{'=' * 74}\nTAMAÑO {etiqueta}\n{'=' * 74}")
        print(f"tiempo en esta máquina: en crudo {coma(tiempos['en crudo'] / 60)} min; "
              f"adiestrada {coma(tiempos['adiestrada'] / 60)} min (carga del modelo incluida).")
        for k, (orden, pregunta, sin_signos) in enumerate(PETICIONES, 1):
            print(f"\n--- petición {k} ---\n  orden:               {orden}\n  pregunta:            {pregunta}\n"
                  f"  pregunta sin signos: {sin_signos}")
            for version in ("en crudo", "adiestrada"):
              for forma in FORMAS:
                texto, n, cerro = res[(k - 1, forma, version)]
                clase, porque = clasificar(texto)
                fin = "cerró su turno" if cerro else f"cortado a los {MAX_NUEVOS} trozos"
                print(f"\n[{forma}, {version}] ({n} trozos; {fin})")
                print(texto)
                print(f"-> {clase} ({porque}){repite(texto, PETICIONES[k - 1][FORMAS.index(forma)])}")
    resumen = {}
    for etiqueta, (res, _) in todo.items():
        for (k, forma, version), (texto, _, _) in res.items():
            resumen.setdefault((etiqueta, forma, version), []).append(clasificar(texto)[0])
    print(f"\n{'=' * 74}\nRESUMEN (según la regla; de las 5 peticiones, cuántas «contestan»)\n{'=' * 74}")
    print("Las preguntas sin signos: la regla no mira sus palabras, así que una que se repite a sí misma sale «contesta»;")
    print("la línea «Además: repite su petición.» de cada texto lo dice.")
    for (etiqueta, forma, version), clases in sorted(resumen.items()):
        print(f"{etiqueta:>7}  {forma:<9} {version:<11} contesta {clases.count('contesta')} de {len(clases)}")


if __name__ == "__main__":
    main()
