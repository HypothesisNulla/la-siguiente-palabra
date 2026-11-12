#!/usr/bin/env python3
"""
Capítulo 5 — palabras convertidas en números.

Entrena representaciones de palabras sobre dos corpus de dominio público y mide:
  1. Qué palabras quedan cerca de cuáles, sin diccionario ni ayuda humana.
  2. Si la aritmética famosa (rey - hombre + mujer) funciona, y con cuánto texto.

Uso:
    python palabras_numeros.py
    python palabras_numeros.py --selftest
"""

# ======================= CONSTANTES =======================

CORPUS_LIBRO = "../datos/entrada/quijote.txt"
CORPUS_BIBLIOTECA = "../datos/entrada/corpus_es"      # 300 libros en español de dominio público

DIMENSION = 100            # números por palabra
VENTANA = 5                # palabras de contexto a cada lado
MIN_APARICIONES = 10
EPOCAS = 5
SEMILLA = 20260914
HILOS = 1                  # 1 hilo: reproducible. Con más, el resultado varía entre ejecuciones.

SONDAS = ["caballo", "rey", "espada", "noche", "mujer", "dinero", "banco", "cura"]
VECINOS = 5
LETRA_VECINOS = "cinco"    # Orden 150: los títulos de las tablas lo dicen con letra, como el libro

ANALOGIAS = [
    ("rey", "hombre", "mujer", "reina"),
    ("padre", "hombre", "mujer", "madre"),
    ("hijo", "hombre", "mujer", "hija"),
    ("caballo", "macho", "hembra", "yegua"),
    ("hermano", "hombre", "mujer", "hermana"),
]

PAREJAS_RELACIONADAS = [("rey", "reina"), ("padre", "madre"), ("noche", "mañana"),
                        ("caballo", "galope"), ("mano", "brazo")]

# El selftest necesita un corpus donde el método FUNCIONE: con un solo libro no funciona,
# como demuestra la medición principal. Se usa un trozo de la biblioteca.
LIBROS_SELFTEST = 60
PAREJAS_AL_AZAR = 200

TOL_INVARIANTE = 1e-5

# ==========================================================

import argparse
import glob
import os
import random
import re
import sys
import unicodedata

from formato import miles, tabla_editorial
import numpy as np

ALFABETO = set("abcdefghijklmnñopqrstuvwxyzáéíóúü ")


def limpiar(texto):
    if "*** START OF" in texto:
        texto = texto.split("*** START OF", 1)[1].split("\n", 1)[-1]
    if "*** END OF" in texto:
        texto = texto.split("*** END OF", 1)[0]
    texto = unicodedata.normalize("NFC", texto.lower())
    texto = "".join(c if c in ALFABETO else " " for c in texto)
    return re.sub(r" {2,}", " ", texto)


def frases_de_fichero(ruta):
    with open(ruta, encoding="utf-8", errors="replace") as fh:
        for linea in limpiar(fh.read()).split("\n"):
            palabras = linea.split()
            if len(palabras) >= 5:
                yield palabras


def cargar_corpus(origen):
    if os.path.isdir(origen):
        ficheros = sorted(glob.glob(os.path.join(origen, "*.txt")))
        assert ficheros, f"Se esperaban ficheros .txt en «{origen}»; no se encontró ninguno"
        frases = [f for ruta in ficheros for f in frases_de_fichero(ruta)]
    else:
        assert os.path.exists(origen), f"Se esperaba el fichero «{origen}»; no existe"
        frases = list(frases_de_fichero(origen))
    total = sum(len(f) for f in frases)
    assert total > 100_000, \
        f"Se esperaban más de 100.000 palabras en «{origen}»; se encontraron {miles(total)}"
    return frases, total


def entrenar(frases, semilla=SEMILLA, epocas=EPOCAS):
    from gensim.models import Word2Vec
    return Word2Vec(sentences=frases, vector_size=DIMENSION, window=VENTANA,
                    min_count=MIN_APARICIONES, sg=1, workers=HILOS,
                    seed=semilla, epochs=epocas)


def vecinos(modelo, palabra, n=VECINOS):
    if palabra not in modelo.wv:
        return None
    return [p for p, _ in modelo.wv.most_similar(palabra, topn=n)]


def probar_analogias(modelo, analogias=ANALOGIAS, topn=5):
    aciertos, evaluadas, detalle = 0, 0, []
    for a, b, c, esperada in analogias:
        if not all(w in modelo.wv for w in (a, b, c, esperada)):
            detalle.append((a, b, c, esperada, "(alguna palabra no está en el vocabulario)"))
            continue
        evaluadas += 1
        propuestas = [p for p, _ in modelo.wv.most_similar(positive=[a, c], negative=[b], topn=topn)]
        ok = esperada in propuestas
        aciertos += ok
        detalle.append((a, b, c, esperada, ("ACIERTA: " if ok else "falla:  ") + ", ".join(propuestas[:3])))
    return aciertos, evaluadas, detalle


def similitud_media(modelo, parejas):
    vals = [modelo.wv.similarity(a, b) for a, b in parejas
            if a in modelo.wv and b in modelo.wv]
    return float(np.mean(vals)) if vals else float("nan")


def contraste(modelo, parejas, semilla=SEMILLA, n_azar=PAREJAS_AL_AZAR):
    """Diferencia entre lo próximas que quedan las parejas relacionadas y lo próximas que
    quedan dos palabras cualesquiera. Mirar la semejanza a secas NO sirve: cuando no hay
    estructura que aprender, todos los vectores se apelotonan y TODO se parece a todo."""
    rng = random.Random(semilla)
    vocab = list(modelo.wv.index_to_key)
    azar = [(rng.choice(vocab), rng.choice(vocab)) for _ in range(n_azar)]
    return similitud_media(modelo, parejas) - similitud_media(modelo, azar)


def selftest():
    fallos = []
    ficheros = sorted(glob.glob(os.path.join(CORPUS_BIBLIOTECA, "*.txt")))[:LIBROS_SELFTEST]
    assert ficheros, f"Se esperaban libros en «{CORPUS_BIBLIOTECA}»; no se encontró ninguno"
    frases = [f for ruta in ficheros for f in frases_de_fichero(ruta)]
    total = sum(len(f) for f in frases)
    print(f"Corpus de prueba: {len(ficheros)} libros, {miles(total)} palabras.\n")

    modelo = entrenar(frases, epocas=3)

    # 1. TEST NULO — barajar TODAS las palabras del corpus destruye el contexto. Si el
    #    método captura significado por compañía, las parejas relacionadas deben dejar de
    #    estar más cerca entre sí que dos palabras cualesquiera.
    rng = random.Random(SEMILLA)
    todas = [p for f in frases for p in f]
    rng.shuffle(todas)
    barajado = [todas[i:i + 20] for i in range(0, len(todas), 20)]
    modelo_nulo = entrenar(barajado, epocas=3)
    real = contraste(modelo, PAREJAS_RELACIONADAS)
    nulo = contraste(modelo_nulo, PAREJAS_RELACIONADAS)
    print(f"[1] test nulo         ventaja de las parejas relacionadas sobre dos palabras "
          f"cualesquiera: corpus real {real:+.3f}  barajado {nulo:+.3f}")
    if not (real > 0.05 and nulo < real / 2):
        fallos.append(f"test nulo: ventaja {real:+.3f} en el corpus real y {nulo:+.3f} en el barajado; "
                      "se esperaba una ventaja clara solo en el real")

    # 2. SEÑAL IMPLANTADA — una palabra inventada, colocada siempre junto a «caballo», debe
    #    acabar entre sus vecinas más próximas.
    marca = "zzqx"
    implantadas = [[marca if p == "caballo" and rng.random() < 0.7 else p for p in f] for f in frases]
    modelo_imp = entrenar(implantadas, epocas=3)
    cercanas = vecinos(modelo_imp, marca, n=10) or []
    print(f"[2] señal implantada  «{marca}» -> vecinas: {', '.join(cercanas[:5])}")
    if "caballo" not in cercanas:
        fallos.append(f"señal implantada: «caballo» no aparece entre las vecinas de «{marca}»")

    # 3. INVARIANTE DEL DOMINIO — la semejanza de una palabra consigo misma es 1, y ninguna
    #    semejanza se sale del intervalo de -1 a 1.
    palabras = list(modelo.wv.index_to_key)[:300]
    propias = [modelo.wv.similarity(p, p) for p in palabras]
    matriz = [modelo.wv.similarity(a, b) for a in palabras[:60] for b in palabras[:60]]
    peor = max(abs(v - 1.0) for v in propias)
    fuera = [v for v in matriz if v < -1 - TOL_INVARIANTE or v > 1 + TOL_INVARIANTE]
    print(f"[3] invariante        semejanza consigo misma: error máximo {peor:.2e}; "
          f"valores fuera de [-1, 1]: {len(fuera)}")
    if peor > 1e-3 or fuera:
        fallos.append(f"invariante: error {peor:.2e} en la semejanza consigo misma, {len(fuera)} fuera de rango")

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

    # L24 (9 de octubre): tablas editoriales (regla 6 ter). Las cuentas no cambian.
    for etiqueta, origen, aprendio in (("con un libro (el Quijote)", CORPUS_LIBRO,
                                        "el Quijote entero"),
                                       ("con una biblioteca (300 libros)", CORPUS_BIBLIOTECA,
                                        "los trescientos libros")):
        frases, total = cargar_corpus(origen)
        modelo = entrenar(frases)
        cuantas = (f"El modelo aprendió con {aprendio}, {miles(total)} palabras de texto, y le dio "
                   f"una lista de números a {miles(len(modelo.wv.index_to_key))} palabras distintas.")
        filas = []
        for sonda in SONDAS:
            v = vecinos(modelo, sonda)
            filas.append([sonda, ", ".join(v) if v else "no aparece bastante"])
        print()
        print("\n".join(tabla_editorial(
            f"Las {LETRA_VECINOS} vecinas más próximas, {etiqueta}",
            ["palabra", f"sus {VECINOS} vecinas más próximas, de la más parecida a la menos"],
            filas, "ii",
            [cuantas,
             f"Una palabra que sale menos de {MIN_APARICIONES} veces en el texto no tiene lista de "
             "números propia, y la tabla lo dice con «no aparece bastante»."])))

        # Orden 150: la fila de «mujer» sola, con su título, para la tabla suelta del capítulo (la del texto)
        for sonda in SONDAS:
            if sonda == "mujer" and "biblioteca" in etiqueta:       # la del cuerpo sale de los trescientos libros
                v = vecinos(modelo, sonda)
                print()
                print("\n".join(tabla_editorial(
                    f"Las {LETRA_VECINOS} vecinas más próximas de «{sonda}»",
                    ["palabra", f"sus {VECINOS} vecinas más próximas, de la más parecida a la menos"],
                    [[sonda, ", ".join(v) if v else "no aparece bastante"]], "ii")))

        aciertos, evaluadas, detalle = probar_analogias(modelo)
        filas = []
        for a, b, c, esp, res in detalle:
            if res.startswith("("):
                filas.append([f"{a}, menos {b}, más {c}", esp, "no se puede: falta alguna palabra", ""])
            else:
                sale = res.startswith("ACIERTA")
                filas.append([f"{a}, menos {b}, más {c}", esp, "sí" if sale else "no",
                              res.split(":", 1)[1].strip()])
        print()
        print("\n".join(tabla_editorial(
            f"Aritmética con palabras, {etiqueta}",
            ["la cuenta", "la buscada", "¿sale entre las 5 primeras?", "las 3 primeras"],
            filas, "iici",
            [f"Aciertos: {aciertos} de {evaluadas} evaluadas."])))

if __name__ == "__main__":
    main()
