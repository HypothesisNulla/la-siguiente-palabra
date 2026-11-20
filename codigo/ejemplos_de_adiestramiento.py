#!/usr/bin/env python3
"""
Capítulo 12 — ejemplos reales del material del adiestramiento, en castellano (Orden 171, 15 de noviembre de 2026).

El adiestramiento tiene dos fases que el capítulo cuenta: en la primera se le enseñan a la máquina conversaciones
escritas por personas (una petición y la respuesta de otra persona); en la segunda, un «juez» aprende de lo que las
personas prefieren cuando se les enseñan varias respuestas a la misma petición. Este programa saca ejemplos REALES de
las dos fases, no inventados, de OpenAssistant (oasst1): conversaciones escritas a mano por voluntarios en 2023,
muchas en castellano, con varias respuestas a la misma petición ORDENADAS por personas (el campo `rank`).
Fuente: https://huggingface.co/datasets/OpenAssistant/oasst1, licencia Apache 2.0.

Qué hace:
  - lee la copia guardada en `datos/entrada/oasst1/` (si no está, la descarga y anota su md5);
  - se queda con mensajes en castellano (`lang == "es"`), no borrados, con la revisión aprobada, escritos por
    personas (no `synthetic`) y con la toxicidad automática (`detoxify`) que trae el propio conjunto por debajo de un
    umbral bajo;
  - FASE 1, el ejemplo: parejas de petición (de quien pregunta) y respuesta (de un voluntario: la mejor puesta por las
    personas, `rank` 0), cortas; oasst1 no trae etiqueta de clase, así que saca al azar, con la semilla, una lista
    de candidatas y la persona elige;
  - FASE 2, el juez: peticiones con al menos tres respuestas y su puesto (`rank`), puesto por las personas;
  - imprime cada texto tal cual, con su identificador de mensaje, para encontrarlo en el conjunto.

Orden 173 (16 de noviembre de 2026): al final de la salida, «LO QUE EL LIBRO CITA DE AQUÍ», el programa imprime la
muestra «Tres ejemplos del libro de conversaciones» del capítulo 12: las parejas EN_EL_LIBRO de la fase 1, con
«Usuario:» delante de la petición y «Asistente:» delante de la respuesta, partidas en renglones de ANCHO_MUESTRA
(formato.muestra_editorial), como las enseña el libro.

POR QUÉ ESTA MEDICIÓN NO DEPENDE DE LA MÁQUINA. Se leen ficheros de texto.

Uso:
    python ejemplos_de_adiestramiento.py --selftest
    python ejemplos_de_adiestramiento.py > ../datos/salidas/ejemplos_de_adiestramiento.txt
"""

# ======================= CONSTANTES =======================

URL = "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/2023-04-12_oasst_ready.trees.jsonl.gz"
VERSION = "2023-04-12_oasst_ready.trees.jsonl.gz"     # el árbol de mensajes de oasst1, versión del 12 de abril de 2023
RUTA = "../datos/entrada/oasst1/2023-04-12_oasst_ready.trees.jsonl.gz"
MD5 = "bd119c17938362c3ea25bc1523e99551"                # el de la copia que se bajó el 15 de noviembre de 2026
LICENCIA = "Apache 2.0"
LENGUA = "es"
UMBRAL_TOXICIDAD = 0.02          # toxicidad automática (`detoxify`, de 0 a 1) por debajo de la cual se queda el mensaje
SEMILLA = 20260914
FASE_1_CANDIDATAS = 20           # parejas al azar de las que la persona elige
FASE_1_PETICION_MAX = 30         # palabras: la petición, de menos de
FASE_1_RESPUESTA_MAX = 120       # palabras: la respuesta, de menos de
FASE_2_PETICIONES = 3            # peticiones con varias respuestas ordenadas
FASE_2_RESPUESTAS_MIN = 3        # al menos tres respuestas, todas con su puesto
EN_EL_LIBRO = (1, 17, 19)        # Orden 173: las parejas de la fase 1 (de 1 a 20) que enseña el capítulo 12
ANCHO_MUESTRA = 57               # caracteres por renglón de la muestra del libro; el que se parte sigue con dos espacios

# ==========================================================

import argparse
import datetime
import gzip
import hashlib
import json
import os
import platform
import random
import sys
import urllib.request

from formato import muestra_editorial, partir, tabla_editorial

AQUI = os.path.dirname(os.path.abspath(__file__))


def md5_de(ruta):
    h = hashlib.md5()
    with open(ruta, "rb") as fh:
        for trozo in iter(lambda: fh.read(1 << 20), b""):
            h.update(trozo)
    return h.hexdigest()


def conseguir(ruta=None):
    """La ruta de la copia guardada; si no está, la baja y comprueba su md5."""
    ruta = os.path.join(AQUI, RUTA) if ruta is None else ruta
    if not os.path.exists(ruta):
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        urllib.request.urlretrieve(URL, ruta)
    assert md5_de(ruta) == MD5, f"el md5 de {ruta} no es el anotado ({MD5}): otra versión del conjunto"
    return ruta


def arboles(ruta):
    with gzip.open(ruta, "rt", encoding="utf-8") as fh:
        for linea in fh:
            yield json.loads(linea)


def vale(m, lengua=LENGUA, umbral=UMBRAL_TOXICIDAD):
    """El mensaje se queda si es de esa lengua, no está borrado, tiene la revisión aprobada, lo escribió una persona
    y su toxicidad automática está por debajo del umbral."""
    return (m.get("lang") == lengua and not m.get("deleted") and m.get("review_result") is True
            and not m.get("synthetic") and (m.get("detoxify") or {}).get("toxicity", 1.0) < umbral)


def palabras(texto):
    return len(texto.split())


def fase_1(arbs):
    """Las parejas (petición, respuesta mejor puesta) cortas, en el orden del conjunto."""
    out = []
    for t in arbs:
        p = t["prompt"]
        if p.get("role") != "prompter" or not vale(p) or palabras(p["text"]) >= FASE_1_PETICION_MAX:
            continue
        buenas = [r for r in p.get("replies", []) if vale(r) and r.get("rank") == 0
                  and palabras(r["text"]) < FASE_1_RESPUESTA_MAX]
        if buenas:
            out.append((p, buenas[0]))
    return out


def fase_2(arbs):
    """Las peticiones cortas con al menos FASE_2_RESPUESTAS_MIN respuestas buenas, todas con su puesto, y cada
    respuesta de menos de FASE_1_RESPUESTA_MAX palabras; las respuestas, por puesto."""
    out = []
    for t in arbs:
        p = t["prompt"]
        if p.get("role") != "prompter" or not vale(p) or palabras(p["text"]) >= FASE_1_PETICION_MAX:
            continue
        rs = [r for r in p.get("replies", []) if vale(r)]
        if len(rs) >= FASE_2_RESPUESTAS_MIN and all(r.get("rank") is not None for r in rs) \
                and all(palabras(r["text"]) < FASE_1_RESPUESTA_MAX for r in rs):
            out.append((p, sorted(rs, key=lambda r: r["rank"])))
    return out


def elegir(candidatas, n, semilla=SEMILLA):
    """n al azar, con la semilla, en un orden que no depende del orden de entrada."""
    ordenadas = sorted(candidatas, key=lambda c: c[0]["message_id"])
    return random.Random(semilla).sample(ordenadas, min(n, len(ordenadas)))


def selftest():
    fallos = []

    def msg(id_, texto, **k):
        base = {"message_id": id_, "text": texto, "role": "prompter", "lang": "es", "deleted": False,
                "review_result": True, "synthetic": False, "detoxify": {"toxicity": 0.001}, "rank": None, "replies": []}
        base.update(k)
        return base

    # 1. TEST NULO — mensajes que NO tienen que pasar: en otra lengua, borrado, sin revisar, sintético, tóxico.
    malos = [msg("a", "x", lang="en"), msg("b", "x", deleted=True), msg("c", "x", review_result=False),
             msg("d", "x", synthetic=True), msg("e", "x", detoxify={"toxicity": 0.9}), msg("f", "x", detoxify=None)]
    ok1 = not any(vale(m) for m in malos)
    print(f"[1] test nulo         seis mensajes que no valen (otra lengua, borrado, sin revisar, sintético, tóxico, sin "
          f"toxicidad): {sum(vale(m) for m in malos)} pasan")
    if not ok1:
        fallos.append("test nulo: un mensaje que no valía pasa el filtro")
    # 2. SEÑAL IMPLANTADA — un árbol de juguete con tres respuestas ordenadas: sale en la fase 2, con las respuestas por
    #    puesto, y la mejor es la de la fase 1; y una petición larga o con una respuesta larga no sale.
    resp = [msg(f"r{k}", f"respuesta {k}", role="assistant", rank=k) for k in (2, 0, 1)]
    arb = {"prompt": msg("p", "una petición corta", replies=resp)}
    largo = {"prompt": msg("q", "palabra " * 40, replies=[msg("s", "ok", role="assistant", rank=0)])}
    larga_r = {"prompt": msg("u", "corta", replies=[msg("v", "palabra " * 200, role="assistant", rank=0)])}
    f2, f1 = fase_2([arb, largo, larga_r]), fase_1([arb, largo, larga_r])
    ok2 = (len(f2) == 1 and [r["rank"] for r in f2[0][1]] == [0, 1, 2] and len(f1) == 1 and f1[0][1]["message_id"] == "r0")
    print(f"[2] señal implantada  árbol de juguete: fase 2 con {len(f2)} petición y respuestas por puesto "
          f"{[r['rank'] for r in f2[0][1]] if f2 else []}; fase 1, la respuesta de puesto 0: {'sí' if f1 and f1[0][1]['message_id'] == 'r0' else 'NO'}")
    if not ok2:
        fallos.append("señal implantada: el árbol de juguete no sale como se esperaba")
    # 3. INVARIANTE — la elección es reproducible (misma semilla, mismo orden aunque cambie el orden de entrada), y la
    #    copia guardada, si está, es la del md5 anotado.
    cands = [(msg(f"m{k:02d}", "t"), msg(f"n{k:02d}", "r")) for k in range(50)]
    a = [c[0]["message_id"] for c in elegir(cands, 8)]
    b = [c[0]["message_id"] for c in elegir(list(reversed(cands)), 8)]
    ruta = os.path.join(AQUI, RUTA)
    md5_ok = (md5_de(ruta) == MD5) if os.path.exists(ruta) else None
    print(f"[3] invariante        la elección con la semilla {SEMILLA} es la misma aunque cambie el orden de entrada: "
          f"{'sí' if a == b else 'NO'}; md5 de la copia guardada: "
          f"{'el anotado' if md5_ok else ('NO es el anotado' if md5_ok is False else 'no hay copia (se baja al ejecutar)')}")
    # (Orden 173) la muestra del libro: «Usuario:» y «Asistente:» delante, ningún renglón de más de ANCHO_MUESTRA (el
    # que se parte sigue con dos espacios) y, si se juntan los renglones, vuelve a salir el texto de cada mensaje.
    largas = [(msg(f"x{k}", "pregunta " * 9 + f"{k}"), msg(f"y{k}", "respuesta " * 14 + f"{k}")) for k in range(20)]
    sal = muestra_del_libro(largas)
    cuerpo = [l[4:] for l in sal[3:sal.index(":::", 1) - 2] if l.strip()]
    renglones = [l for l in cuerpo if len(l) <= ANCHO_MUESTRA]
    unidos = " ".join(l.strip() for l in cuerpo)
    ok_muestra = (len(renglones) == len(cuerpo) and sal[0] == "::: muestra" and sal[-1] == ":::"
                  and all(f"Usuario: {largas[k - 1][0]['text']}" in unidos and f"Asistente: {largas[k - 1][1]['text']}" in unidos
                          for k in EN_EL_LIBRO))
    print(f"    la muestra del libro con parejas largas: renglones de {ANCHO_MUESTRA} como mucho y el texto entero al juntarlos: "
          f"{'sí' if ok_muestra else 'NO'}")
    if a != b or md5_ok is False:
        fallos.append("invariante: la elección no es reproducible, o el md5 de la copia no es el anotado")
    if not ok_muestra:
        fallos.append("invariante: la muestra del libro parte mal los renglones o pierde texto")
    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def muestra_del_libro(elegidas):
    """La muestra del capítulo 12: las parejas EN_EL_LIBRO de `elegidas` (la lista de la fase 1, en su orden), cada
    una como «Usuario: …» y «Asistente: …», con una línea en blanco entre pareja y pareja."""
    lineas = []
    for k in EN_EL_LIBRO:
        p, r = elegidas[k - 1]
        for quien, m in (("Usuario", p), ("Asistente", r)):
            assert "\n" not in m["text"].strip(), f"Se esperaba un mensaje de un solo renglón; la pareja {k} trae más"
            partes = partir(f"{quien}: {m['text'].strip()}", ANCHO_MUESTRA)
            seguido = partir(" ".join(partes[1:]), ANCHO_MUESTRA - 2) if len(partes) > 1 else []
            lineas += [partes[0]] + ["  " + x for x in seguido]       # el renglón que sigue, con sus dos espacios, cabe
        lineas.append("")
    return muestra_editorial(
        "Tres ejemplos del libro de conversaciones", lineas[:-1],
        ["Tres conversaciones distintas, tal como las escribieron. El mensaje lo escribió una persona, y la "
         "respuesta, otra que hacía de asistente."])


def tabla_quien_hace_que():
    """La tabla «Quién hace qué en cada fase» del capítulo 12 (Orden 182, 20 de noviembre de 2026): lo que el libro cuenta de las
    tres fases del adiestramiento de OpenAI de 2022 (el artículo de InstructGPT) y de cómo las hizo OpenAssistant. No es una
    medición: es una descripción, y la imprime el programa para que el libro la copie y la regla 6 se cumpla."""
    return tabla_editorial(
        "Quién hace qué en cada fase",
        ["fase", "quién escribe el mensaje", "quién escribe la respuesta", "quién decide si es buena"],
        [["1. el libro de conversaciones", "una persona", "una persona", "nadie: se copia tal cual"],
         ["2. el juez", "una persona", "la máquina, varias", "personas, que las ordenan"],
         ["3. estudiar para el juez", "mensajes ya guardados", "la máquina", "el juez, con una nota"]],
        "iiii",
        ["Las tres fases del adiestramiento, en el trabajo de OpenAI de 2022. En OpenAssistant, también las respuestas de la "
         "segunda fase las escribieron personas."])


def imprimir(m, etiqueta):
    print(f"[{etiqueta}] mensaje {m['message_id']}" + (f" (puesto {m['rank']})" if m.get("rank") is not None else ""))
    print(m["text"])
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())

    ruta = conseguir()
    arbs = list(arboles(ruta))
    raices_es = [t for t in arbs if t["prompt"].get("lang") == LENGUA]
    print(f"Sacado el {datetime.date.today()} en {platform.platform()}. La salida no depende de la máquina: lee un fichero guardado.")
    print(f"Fuente: OpenAssistant/oasst1 (https://huggingface.co/datasets/OpenAssistant/oasst1), licencia {LICENCIA}.")
    print(f"Versión: {VERSION}; md5 de la copia: {MD5}.")
    print("Los mensajes de abajo los escribieron voluntarios de OpenAssistant, no el autor de este repositorio: van tal cual, "
          f"con sus faltas, y se publican con la licencia del conjunto, que pide atribución: licencia {LICENCIA}.")
    print(f"El conjunto trae {len(arbs):,} árboles de conversación; {len(raices_es):,} empiezan en castellano.".replace(",", "."))
    print(f"Filtro: lang == «{LENGUA}», no borrado, revisión aprobada, escrito por una persona (no `synthetic`) y toxicidad "
          f"automática (`detoxify`) por debajo de {str(UMBRAL_TOXICIDAD).replace('.', ',')}.")
    f1, f2 = fase_1(arbs), fase_2(arbs)
    print(f"Pasan el filtro: {len(f1)} parejas cortas para la fase 1 (petición de menos de {FASE_1_PETICION_MAX} palabras, "
          f"mejor respuesta de menos de {FASE_1_RESPUESTA_MAX}) y {len(f2)} peticiones con al menos {FASE_2_RESPUESTAS_MIN} "
          f"respuestas ordenadas para la fase 2.")
    print(f"Semilla: {SEMILLA}.")

    print(f"\n{'=' * 74}\nFASE 1, EL EJEMPLO: {FASE_1_CANDIDATAS} parejas al azar (petición y la respuesta que las personas "
          f"pusieron la primera)\n{'=' * 74}")
    for k, (p, r) in enumerate(elegir(f1, FASE_1_CANDIDATAS), 1):
        print(f"\n--- pareja {k} ---")
        imprimir(p, "petición")
        imprimir(r, "respuesta")

    print(f"{'=' * 74}\nFASE 2, EL JUEZ: {FASE_2_PETICIONES} peticiones con sus respuestas, por el puesto que les dieron las "
          f"personas (0 es la mejor)\n{'=' * 74}")
    for k, (p, rs) in enumerate(elegir(f2, FASE_2_PETICIONES), 1):
        print(f"\n--- petición {k} ---")
        imprimir(p, "petición")
        for r in rs:
            imprimir(r, "respuesta")

    print(f"{'=' * 74}\nLO QUE EL LIBRO CITA DE AQUÍ (capítulo 12)\n{'=' * 74}\n")
    print("\n".join(muestra_del_libro(elegir(f1, FASE_1_CANDIDATAS))))
    print()
    print("\n".join(tabla_quien_hace_que()))


if __name__ == "__main__":
    main()
