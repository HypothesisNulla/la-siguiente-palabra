#!/usr/bin/env python3
"""
Capítulo 13 — la sorpresa, hecha una vez entera antes de dar la media (L24, E05).

El capítulo daba «sorpresa media: 2,754» y «0,772» sin haber dicho nunca qué es la sorpresa ni en
qué se mide. Este programa lo enseña en tres pasos, con la misma máquina, los mismos párrafos y la
misma cuenta que `romper_la_maquina.py` (medición B):

  1. QUÉ ES LA SORPRESA DE UN TROZO. La moneda del libro es «de cada cien papeletas, cuántas». La
     sorpresa de un trozo sale de las papeletas que la máquina le daba al trozo que vino de verdad:
     con cien, cero; con menos, más. La tabla dice cuánto.
  2. UN PÁRRAFO, TROZO A TROZO. El primero de los ocho párrafos: el arranque común, y después, trozo
     a trozo, las papeletas que la máquina daba a lo que escribió la persona y a lo que escribió
     ella misma.
  3. LA MEDIA DE LOS OCHO PÁRRAFOS. La sorpresa media de cada texto, como en el capítulo, y la
     misma media pasada otra vez a papeletas con la tabla del paso 1. Se comprueba que salen las
     cifras de `romper_la_maquina.py`.

Uso (en el Mac, con la tarjeta gráfica):
    python sorpresa_a_mano.py --selftest
    python sorpresa_a_mano.py > ../datos/salidas/sorpresa_a_mano.txt
"""

# ======================= CONSTANTES =======================

PAPELETAS_ESCALA = (100, 50, 25, 10, 5, 1, 0.1)   # la tabla del paso 1 (segunda vuelta: más filas,
                                          # para que un trozo cualquiera se pueda situar entre dos)
CASO_A_MANO = 23.7                        # el trozo «f» del párrafo de la persona, convertido a mano
TROZOS_VISTOS = 12                        # cuántos trozos del paso 2 se enseñan
PARRAFO = 0                               # cuál de los ocho párrafos se hace a mano (el primero)
# Orden 97: cifra que la nota escribe con letra; si cambia el parámetro, falla aquí y no en la página.
EN_LETRA = {12: "doce"}
SALIDA_CSV_ROMPER = "../datos/salidas/romper_la_maquina.csv"   # las medias del capítulo
TOL_MEDIA = 0.0015                        # la media recalculada tiene que dar la del capítulo
TOL_CUENTA = 1e-4                         # la cuenta trozo a trozo es la del programa del capítulo
MINIMO_FAVORITOS = 0.95                   # a temperatura 0, casi todos los trozos de la máquina tienen
                                          # que salir favoritos al medirlos de una pasada; no todos a la
                                          # fuerza: generar trozo a trozo y medir de una vez redondean
                                          # distinto, y en un empate casi exacto puede ganar el otro

# ==========================================================

import argparse
import csv
import datetime
import math
import platform
import random
import sys
from pathlib import Path

from formato import coma, muestra_editorial, tabla_editorial, trozo

# Desde el 8 oct 2026 los tres pasos salen como tablas de libro (formato.py; REGLAS 6 ter).
# Las cuentas no cambian.

AQUI = Path(__file__).resolve().parent


def sorpresa_de(papeletas):
    """La sorpresa de un trozo al que la máquina daba esas papeletas de cada cien. (El «+ 0.0»
    quita el cero negativo, que se imprimía «-0,00».)"""
    return -math.log(papeletas / 100) + 0.0


def papeletas_de(sorpresa):
    """Lo contrario: las papeletas de cada cien que corresponden a esa sorpresa."""
    return 100 * math.exp(-sorpresa)


def trozo_a_trozo(R, tok, modelo, ids_p, ids_c):
    """Para cada trozo de la continuación, lo que la máquina le daba: (trozo, probabilidad).

    Trabaja con los trozos tal cual, sin pasarlos a texto y volver a trocearlos. Es la corrección
    de una cosa que se vio al hacer el párrafo a mano (28 de septiembre de 2026):
    `romper_la_maquina.py` copia el texto de la máquina con `.strip()`, que le quita el espacio
    del principio, y al volver a trocearlo el primer trozo («la» pegado a la palabra anterior)
    le parece a la máquina casi imposible. Con los trozos tal cual, eso no pasa."""
    import torch
    ids = torch.cat([ids_p, ids_c], dim=1).to(R.dispositivo())
    with torch.no_grad():
        logits = modelo(ids).logits[0].float()
    prob = torch.softmax(logits, dim=-1)
    inicio = ids_p.shape[1]
    return [(tok.decode(ids[0, k:k + 1]), float(prob[k - 1, ids[0, k]]))
            for k in range(inicio, ids.shape[1])]


def ids_h_largo(ids):
    return ids.shape[1]


def media(lista):
    return sum(-math.log(p) for _, p in lista) / len(lista)


def visible(trozo):
    return trozo.replace(" ", "_").replace("\n", "¶")


def medias_del_capitulo():
    with open(AQUI / SALIDA_CSV_ROMPER, encoding="utf-8") as fh:
        f = {r["clave"]: float(r["a"]) for r in csv.DictReader(fh) if r["medicion"] == "B_sorpresa"}
    return f["humano"], f["maquina"]


def continuar(R, tok, modelo, ids_p):
    """Lo que escribe la máquina a temperatura 0 a partir del arranque, como trozos."""
    import torch
    with torch.no_grad():
        salida = modelo.generate(ids_p.to(R.dispositivo()), max_new_tokens=R.LARGO_B,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
    return salida[:, ids_p.shape[1]:].cpu()


def pares(R, tok, modelo, n=None):
    """Los párrafos de la medición B, como trozos: (arranque, lo humano, lo de la máquina)."""
    out = []
    for p in R.parrafos_humanos(n or R.MUESTRAS_B):
        ids = tok(p, return_tensors="pt")["input_ids"]
        ids_p = ids[:, :R.ARRANQUE]
        out.append((ids_p, ids[:, R.ARRANQUE:R.ARRANQUE + R.LARGO_B], continuar(R, tok, modelo, ids_p)))
    return out


def informe(R, tok, modelo):
    print(f"Medido el {datetime.date.today()} en {platform.platform()}.")
    print(f"modelo: {R.MODELO}   dispositivo: {R.dispositivo()}")

    print("\n--- 1. QUÉ ES LA SORPRESA DE UN TROZO ---")
    filas = [[coma(p, 1) if p < 1 else str(int(p)), coma(sorpresa_de(p), 2)]
             for p in PAPELETAS_ESCALA]
    print()
    print("\n".join(tabla_editorial(
        "La sorpresa de un trozo",
        ["papeletas de cada cien que le daba al trozo que vino", "sorpresa"], filas, "dd",
        [f"Con cien papeletas, ninguna sorpresa. Cada vez que las papeletas se dividen entre dos, "
         f"la sorpresa sube {coma(sorpresa_de(50), 2)}; entre diez, sube "
         f"{coma(sorpresa_de(10), 2)}.",
         f"Un caso: {coma(CASO_A_MANO, 1)} papeletas, entre las filas de 25 y de 10, muy cerca de "
         f"25: sorpresa {coma(sorpresa_de(CASO_A_MANO), 2)}."])))

    todos = pares(R, tok, modelo)
    prefijo, humano, maquina = todos[PARRAFO]
    h = trozo_a_trozo(R, tok, modelo, prefijo, humano)
    m = trozo_a_trozo(R, tok, modelo, prefijo, maquina)
    print("\n--- 2. UN PÁRRAFO, TROZO A TROZO ---")
    import textwrap
    print()
    print("\n".join(muestra_editorial(
        f"El arranque, igual para los dos ({R.ARRANQUE} trozos)",
        textwrap.wrap("«" + " ".join(tok.decode(prefijo[0]).split()) + "»", 60),
        ["El principio del primero de los ocho párrafos, de los trescientos libros."])))
    filas = [[trozo(visible(th)), coma(100 * ph, 1), trozo(visible(tm)), coma(100 * pm, 1)]
             for (th, ph), (tm, pm) in list(zip(h, m))[:TROZOS_VISTOS]]
    print()
    print("\n".join(tabla_editorial(
        "Un párrafo, trozo a trozo",
        ["una persona: trozo", "una persona: papeletas", "la máquina: trozo",
         "la máquina: papeletas"], filas, "idid",
        [# Orden 97: nota para el lector; el modelo (7.000 millones, adiestrado y sin comprimir, a
         # temperatura 0) sale arriba y los trozos de cada texto, en `ids_h_largo`.
         f"«_» es un espacio. Los {EN_LETRA[TROZOS_VISTOS]} primeros trozos de cada texto, cada uno "
         "con las papeletas de cada cien que la máquina le daba justo antes de que viniera. La "
         f"sorpresa media del texto entero es {coma(media(h), 3)} en el de la persona y "
         f"{coma(media(m), 3)} en el de la máquina."])))

    print("\n--- 3. LA MEDIA DE LOS OCHO PÁRRAFOS ---")
    sh = sum(media(trozo_a_trozo(R, tok, modelo, a, b)) for a, b, _ in todos) / len(todos)
    sm = sum(media(trozo_a_trozo(R, tok, modelo, a, c)) for a, _, c in todos) / len(todos)
    filas = [["párrafo humano", coma(sh, 3), coma(papeletas_de(sh), 1)],
             ["párrafo de la máquina", coma(sm, 3), coma(papeletas_de(sm), 1)],
             ["humano entre máquina", f"{coma(sh / sm, 1)} veces", ""],
             ["máquina entre humano", "", f"{coma(papeletas_de(sm) / papeletas_de(sh), 1)} veces"]]
    print()
    print("\n".join(tabla_editorial(
        f"La media de los {len(todos)} párrafos",
        ["", "sorpresa media", "papeletas por trozo"], filas, "idd",
        ["«Papeletas por trozo» es la sorpresa media vuelta a papeletas con la tabla de la "
         "sorpresa, no la media de las papeletas. Los textos de la máquina son a veces más cortos, "
         "porque da el suyo por terminado antes."])))
    assert min(c.shape[1] for _, _, c in todos) < R.LARGO_B, (
        "la nota dice que los textos de la máquina son a veces más cortos")
    ch, cm = medias_del_capitulo()
    # Desde el 28 de septiembre, romper_la_maquina.py también cuenta con los trozos tal cual, así
    # que las dos medias tienen que coincidir; si no, revienta.
    assert abs(sh - ch) < TOL_MEDIA and abs(sm - cm) < TOL_MEDIA, \
        f"se esperaban las medias de romper_la_maquina.py ({ch}, {cm}); salen {sh:.3f} y {sm:.3f}"
    print(f"  las medias de romper_la_maquina.py: {coma(ch, 3)} y {coma(cm, 3)}; coinciden: sí")
    print(f"  trozos que escribió la máquina en cada párrafo: "
          + ", ".join(str(c.shape[1]) for _, _, c in todos))
    print(f"  la sorpresa, humano entre máquina: {coma(sh / sm, 1)} veces; "
          f"las papeletas, máquina entre humano: {coma(papeletas_de(sm) / papeletas_de(sh), 1)} veces")


def selftest():
    import romper_la_maquina as R
    fallos = []

    # 1. TEST NULO — la escala no se inventa nada: cien papeletas son sorpresa cero, e ir y
    #    volver de papeletas a sorpresa deja las papeletas como estaban.
    ida_vuelta = all(abs(papeletas_de(sorpresa_de(p)) - p) < 1e-9 for p in (100, 73, 6.4, 0.01))
    print(f"[1] test nulo         sorpresa de cien papeletas: {coma(sorpresa_de(100), 3)}; "
          f"ida y vuelta: {'sin pérdida' if ida_vuelta else 'CON PÉRDIDA'}")
    if sorpresa_de(100) != 0 or not ida_vuelta:
        fallos.append("test nulo: la escala de la sorpresa no es la que se dice")

    tok, modelo = R.cargar()
    ids_p, ids_h, ids_m = pares(R, tok, modelo, n=1)[PARRAFO]

    # 2. SEÑAL IMPLANTADA — las mismas palabras de la persona, barajadas, tienen que sorprender
    #    más que en su orden.
    humano = tok.decode(ids_h[0])
    palabras = humano.split()
    random.Random(R.SEMILLA).shuffle(palabras)
    barajado = tok(" " + " ".join(palabras), return_tensors="pt", add_special_tokens=False)["input_ids"]
    s_h = media(trozo_a_trozo(R, tok, modelo, ids_p, ids_h))
    s_b = media(trozo_a_trozo(R, tok, modelo, ids_p, barajado))
    print(f"[2] señal implantada  sorpresa en su orden {coma(s_h, 3)}; barajado {coma(s_b, 3)}")
    if not s_b > s_h:
        fallos.append("señal: barajar las palabras no aumenta la sorpresa")

    # 3. INVARIANTE DEL DOMINIO — para el texto de la persona, la cuenta trozo a trozo es la de
    #    `romper_la_maquina.sorpresa`, que da las cifras del capítulo; y a temperatura 0 cada
    #    trozo de la máquina es el favorito, así que ninguno tiene menos papeletas que cualquier
    #    otro que pudiera haber ido ahí.
    s_r = R.sorpresa(tok, modelo, tok.decode(ids_p[0]), humano)
    fav = favoritos(R, tok, modelo, ids_p, ids_m)
    igual = abs(s_r - s_h) < TOL_CUENTA
    print(f"[3] invariante        persona, trozo a trozo {coma(s_h, 4)}; la del capítulo "
          f"{coma(s_r, 4)}: {'iguales' if igual else 'DISTINTAS'}; trozos de la máquina que "
          f"son el favorito: {fav} de {ids_m.shape[1]}")
    if not igual:
        fallos.append("invariante: la cuenta trozo a trozo no es la del programa del capítulo")
    if fav < MINIMO_FAVORITOS * ids_m.shape[1]:
        fallos.append("invariante: la máquina a temperatura 0 no escribió su favorito")
    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def favoritos(R, tok, modelo, ids_p, ids_c):
    """Cuántos trozos de la continuación son el favorito de la máquina en su sitio."""
    import torch
    ids = torch.cat([ids_p, ids_c], dim=1).to(R.dispositivo())
    with torch.no_grad():
        logits = modelo(ids).logits[0].float()
    inicio = ids_p.shape[1]
    return sum(int(logits[k - 1].argmax()) == int(ids[0, k]) for k in range(inicio, ids.shape[1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        sys.exit(selftest())
    import romper_la_maquina as R
    tok, modelo = R.cargar()
    informe(R, tok, modelo)


if __name__ == "__main__":
    main()
