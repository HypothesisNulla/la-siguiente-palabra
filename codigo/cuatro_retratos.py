#!/usr/bin/env python3
"""
Capítulo 10 — los cuatro retratos, mirados de cerca (L24, hallazgos D03, D04, D05, D06 y D09).

`en_que_orden_aprende.py` entrena la máquina del capítulo e imprime lo que escribe en cuatro
momentos. Este programa NO entrena: lee esas cuatro muestras y el mismo texto de entrenamiento
(con el cargador del capítulo, no con una copia) y cuenta lo que el lector no debería tener que
contar él solo:

  1. qué lee la máquina en un paso de aprendizaje, y cuánto lleva leído en cada retrato;
  2. qué hay de palabras de verdad en cada muestra, con el mismo criterio que la escalera del
     capítulo 1 (palabras del Quijote), para subrayarlas en la figura;
  3. el trío de letras «ten», dibujado dentro de las palabras que lo llevan (el capítulo 6 ya
     usa «racha» para otra cosa: palabras seguidas copiadas de un libro);
  4. cuántas veces se encuentra la máquina, en un paso, cada peldaño de la escalera;
  5. cuáles son los 43 símbolos;
  6. por qué la muestra de 30 pasos y la de 300 se parecen tanto.

Da lo mismo en cualquier ordenador: cuenta, no entrena ni cronometra.

Uso:
    python cuatro_retratos.py --selftest
    python cuatro_retratos.py
"""

# ======================= CONSTANTES =======================

SALIDA_CAP10 = "../datos/salidas/en_que_orden_aprende.txt"

# Criterio de «palabra de verdad»: el de la escalera del capítulo 1 (casillas_vacias.py), que
# es una palabra del Quijote, con dos cautelas escritas aquí y en la clave impresa: que salga
# en el Quijote al menos VECES_QUIJOTE veces (una sola vez deja pasar restos como «mo»), y que,
# si es de una sola letra, sea una de las que son palabra por sí solas.
VECES_QUIJOTE = 2
UNA_LETRA = {"a", "e", "o", "u", "y"}

# Las palabras que el capítulo pone como ejemplo de la racha «ten».
RACHA = "ten"
# Orden 97: cifra que la nota escribe con letra (las palabras del ejemplo); si cambia, falla aquí y no en la página.
EN_LETRA = {5: "cinco"}
PALABRAS_RACHA = ["tener", "tenía", "intento", "atención", "contento"]

# Los peldaños de la escalera, con lo que se cuenta de cada uno en el texto de entrenamiento.
# El cuarto es un ejemplo concreto de concordancia entre palabras separadas por otra, sacado de
# la muestra de 3.000 pasos («de los ansitos ojos»).
PELDANOS = [
    ("el aspecto", "un espacio", r" "),
    ("los tríos de letras más frecuentes", "el trío «que», suelto o dentro de una palabra", r"que"),
    ("las palabras y sus terminaciones", "una palabra acabada en «-aba»",
     r"[a-zñáéíóúü]{1,}aba(?![a-zñáéíóúü])"),
    ("la sintaxis", "«los», otra palabra y «ojos»",
     r"(?<![a-zñáéíóúü])los [a-zñáéíóúü]+ ojos(?![a-zñáéíóúü])"),
]
# Los nombres de los peldaños son los mismos, letra a letra, que los de la figura de los cuatro
# retratos (figura_cuatro_retratos.PELDANO) y los del texto del capítulo: regla 5 ter.
MUESTRAS_ESPERADAS = 4

# ==========================================================

import argparse
import collections
import re
import sys

import en_que_orden_aprende as cap10
import ngrama as N
from formato import (ANCHO_CAJA_CITA, coma, comprobar_ancho, miles, muestra_editorial,
                     partir, tabla_editorial)


def leer_muestras(ruta=SALIDA_CAP10):
    texto = open(ruta, encoding="utf-8").read()
    pares = re.findall(r"\[tras ([\d.]+) pasos[^\]]*\]\n(.*)\n", texto)
    assert len(pares) == MUESTRAS_ESPERADAS, f"Se esperaban {MUESTRAS_ESPERADAS} muestras; hay {len(pares)}"
    return [(int(p.replace(".", "")), m.rstrip()) for p, m in pares]


def vocabulario_quijote():
    q = N.normalizar(N.cargar_corpus(N.CORPUS), N.ALFABETO)
    return collections.Counter(N.solo_palabras(N.trocear(q)))


def es_palabra(w, voc):
    if len(w) == 1 and w not in UNA_LETRA:
        return False
    return voc[w] >= VECES_QUIJOTE


def marcar(muestra, voc):
    """La muestra con cada palabra de verdad entre corchetes. El «el » del principio no lo
    escribió la máquina: se le dio para arrancar, y va entre llaves."""
    assert muestra.startswith(cap10.ARRANQUE), "Se esperaba que la muestra empezara por el arranque"
    resto = muestra[len(cap10.ARRANQUE):]
    trozos = re.split(r"([^\s" + re.escape(N.SIGNOS) + r"]+)", resto)
    fuera, n, reales = [], 0, 0
    for i, t in enumerate(trozos):
        if i % 2 == 1:
            n += 1
            if es_palabra(t, voc):
                reales += 1
                t = f"[{t}]"
        fuera.append(t)
    return "{" + cap10.ARRANQUE.strip() + "} " + "".join(fuera), n, reales


def rachas(palabra, largo=len(RACHA)):
    return [palabra[i:i + largo] for i in range(len(palabra) - largo + 1)]


def bloque_paso(muestras):
    por_paso = cap10.LOTE * cap10.CONTEXTO
    lin = ["--- 1. LO QUE LEE UN PASO DE APRENDIZAJE ---",
           f"en cada paso: {cap10.LOTE} fragmentos de texto de {cap10.CONTEXTO} letras,",
           f"sacados al azar de los libros: {miles(por_paso)} letras. En cada",
           "una adivina la siguiente, mira cuánto ha fallado, y al",
           "final del paso se corrigen todos sus números una vez.", "",
           f"{'retrato':>16}{'letras leídas':>16}{'veces el texto entero':>24}"]
    for pasos, _ in muestras:
        leidas = pasos * por_paso
        lin.append(f"{miles(pasos) + ' pasos':>16}{miles(leidas):>16}"
                   f"{coma(leidas / cap10.MAX_CARACTERES, 2):>24}")
    lin.append(f"(el texto entero: {miles(cap10.MAX_CARACTERES)} letras)")
    return comprobar_ancho(lin, ANCHO_CAJA_CITA)


def bloque_palabras(muestras, voc):
    lin = ["--- 2. PALABRAS DE VERDAD EN CADA RETRATO ---",
           "entre corchetes, las palabras que salen en el Quijote al",
           f"menos {VECES_QUIJOTE} veces (de una letra, solo a, e, o, u, y); entre",
           "llaves, el «el» que se le dio para arrancar."]
    for pasos, m in muestras:
        marcada, n, reales = marcar(m, voc)
        lin += ["", f"[tras {miles(pasos)} pasos: {reales} de {n} palabras]"]
        lin.append(marcada)
    return lin      # la muestra no se parte: la lee la figura


def bloque_racha():
    # L24 (9 de octubre de 2026): en tabla editorial, con el trío en negrita en vez de entre
    # corchetes. Ninguna figura lee este bloque.
    assert all(RACHA in rachas(w) for w in PALABRAS_RACHA), "una palabra del ejemplo no lleva la racha"
    return [f"--- 3. EL TRÍO «{RACHA}», DENTRO DE LAS PALABRAS ---", ""] + tabla_editorial(
        f"El trío «{RACHA}», dentro de cinco palabras",
        ["palabra", "sus tríos de letras"],
        [[w, ", ".join(f"**{r}**" if r == RACHA else r for r in rachas(w))]
         for w in PALABRAS_RACHA],
        "ii",
        ["Cada palabra, partida en tríos: tres letras seguidas, corriendo una letra cada vez. "
         f"En negrita, «{RACHA}», que sale en las {EN_LETRA[len(PALABRAS_RACHA)]}."])


def bloque_muestras(muestras):
    """Los cuatro retratos tal como los escribe la máquina, en muestras editoriales (L24, 9 de
    octubre de 2026: el capítulo los copiaba de en_que_orden_aprende.txt a mano, partidos en
    renglones a ojo)."""
    lin = ["--- 7. LOS CUATRO RETRATOS, TAL CUAL ---", ""]
    for pasos, m in muestras:
        lin += muestra_editorial(f"Lo que escribe tras {miles(pasos)} pasos", partir(m),
                                 ["El «el» del principio se le da para arrancar."]) + [""]
        junta = " ".join(partir(m))
        assert junta == m, f"partir() ha cambiado la muestra de {miles(pasos)} pasos"
    return lin


def cuenta_peldanos(texto):
    por_paso = cap10.LOTE * cap10.CONTEXTO
    return [(p, q, len(re.findall(rx, texto)) * por_paso / len(texto)) for p, q, rx in PELDANOS]


def bloque_peldanos(texto):
    lin = ["--- 4. LO QUE SE ENCUENTRA EN UN PASO, DE MEDIA ---",
           f"{'peldaño, y lo que se cuenta de él':<50}{'veces':>10}"]
    for p, q, v in cuenta_peldanos(texto):
        lin.append(f"{p:<50}{coma(v, 3):>10}")
        lin.append(f"  {q}")
    ultimo = cuenta_peldanos(texto)[-1][2]
    lin += ["", f"el último, una vez cada {round(1 / ultimo)} pasos."]
    return comprobar_ancho(lin, ANCHO_CAJA_CITA)


def bloque_simbolos(texto):
    s = sorted(set(texto))
    letras = [c for c in s if c.isalpha() and c in "abcdefghijklmnñopqrstuvwxyz"]
    tildes = [c for c in s if c.isalpha() and c not in letras]
    otros = [c for c in s if not c.isalpha()]
    nombres = {" ": "espacio", "\n": "salto de línea"}
    lin = ["--- 5. LOS SÍMBOLOS QUE PUEDE ESCRIBIR ---",
           f"{len(letras)} letras: {''.join(letras)}",
           f"{len(tildes)} con tilde o diéresis: {' '.join(tildes)}",
           f"{len(otros)} más: " + ", ".join(nombres.get(c, c) for c in otros),
           f"en total, {len(s)}: al azar, acertaría uno de cada {len(s)}."]
    return comprobar_ancho(lin, ANCHO_CAJA_CITA)


def bloque_parecido(muestras):
    (p1, a), (p2, b) = muestras[0], muestras[1]
    iguales = sum(x == y for x, y in zip(a, b))
    lin = ["--- 6. POR QUÉ SE PARECEN LAS DOS PRIMERAS ---",
           "cada retrato se escribe con los mismos sorteos: la misma",
           f"semilla ({cap10.SEMILLA}) en cada uno. Donde la máquina apenas",
           "ha cambiado, el mismo sorteo da la misma letra.",
           f"letras iguales y en el mismo sitio, entre la de {miles(p1)} y la",
           f"de {miles(p2)} pasos: {iguales} de {min(len(a), len(b))}"]
    for (pa, x), (pb, y) in zip(muestras[1:], muestras[2:]):
        lin.append(f"entre la de {miles(pa)} y la de {miles(pb)}: "
                   f"{sum(u == v for u, v in zip(x, y))} de {min(len(x), len(y))}")
    return comprobar_ancho(lin, ANCHO_CAJA_CITA)


def selftest():
    fallos = []
    voc = vocabulario_quijote()
    muestras = leer_muestras()
    import random

    # 1. TEST NULO — la muestra del final con las letras barajadas: casi no quedan palabras.
    ultima = muestras[-1][1]
    letras = list(ultima[len(cap10.ARRANQUE):])
    random.Random(cap10.SEMILLA).shuffle(letras)
    _, n0, r0 = marcar(cap10.ARRANQUE + "".join(letras), voc)
    _, n1, r1 = marcar(ultima, voc)
    print(f"[1] test nulo         palabras de verdad: barajada {r0} de {n0}; tal cual {r1} de {n1}")
    if r0 >= r1 / 3:
        fallos.append("test nulo: con las letras barajadas siguen saliendo palabras")

    # 2. SEÑAL IMPLANTADA — una frase del Quijote: todas sus palabras cuentan.
    frase = "el hidalgo de la mancha salió de su casa una mañana"
    _, n, r = marcar(frase, voc)
    print(f"[2] señal implantada  una frase del Quijote: {r} de {n}")
    if r != n:
        fallos.append("señal implantada: una frase del Quijote no sale entera")
    if RACHA not in rachas("tener") or RACHA in rachas("tiene"):
        fallos.append("señal implantada: la racha no se encuentra donde está o sale donde no está")

    # 3. INVARIANTE — los símbolos del texto son los del capítulo (43) y el paso lee lo que
    #    dicen las constantes del capítulo.
    texto = cap10.cargar_texto(600_000)
    ok = set(texto) <= set(cap10.ALFABETO) and len(cap10.ALFABETO) == 43
    print(f"[3] invariante        símbolos del capítulo: {len(cap10.ALFABETO)}; el texto no tiene "
          f"otros: {'sí' if ok else 'NO'}")
    if not ok:
        fallos.append("invariante: el texto tiene símbolos fuera del alfabeto del capítulo")
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
    codigo = selftest()
    if codigo or args.selftest:
        sys.exit(codigo)
    print()
    texto = cap10.cargar_texto()
    muestras = leer_muestras()
    voc = vocabulario_quijote()
    for b in (bloque_paso(muestras), bloque_palabras(muestras, voc), bloque_racha(),
              bloque_peldanos(texto), bloque_simbolos(texto), bloque_parecido(muestras),
              bloque_muestras(muestras)):
        print("\n".join(b))
        print()


if __name__ == "__main__":
    main()
