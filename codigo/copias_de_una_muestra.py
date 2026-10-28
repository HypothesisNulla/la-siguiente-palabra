#!/usr/bin/env python3
"""
Capítulo 6 — el «% copiado», hecho a mano sobre una muestra (L24).

El capítulo 6 da, para cada muestra de la red, un «copiado»: qué parte de sus cuarenta y cinco
palabras está, tal cual, en los libros con que se entrenó. La medida la hace
`lo_habia_visto_ya.py` y el libro solo la describía con palabras. Carlos, el 27 de septiembre:
«No tiene que imaginar: tiene que verlo escrito y dibujado». Esto la hace una vez, entera, con
las mismas funciones de `lo_habia_visto_ya.py` (importadas, no copiadas):

  1. LAS TRES MUESTRAS DEL CAPÍTULO, con las tres palabras que se le dieron separadas de lo que
     escribió la red, y su número entre las doscientas.
  2. LA MUESTRA 36, CON SUS TRAMOS COPIADOS: cada tramo que está tal cual en los libros de
     entrenamiento (siete palabras o más, el mismo listón de la tabla), su largo, de qué libro
     sale, y la cuenta: palabras dentro de algún tramo, entre cuarenta y cinco.
  3. LA 173, IGUAL, en una línea.
  4. EL SUELO: cuántas de las doscientas muestras de los libros que no entraron en el
     entrenamiento llegan al listón, y cuánto «copian».

El listón no se escribe aquí: se recalcula con `lo_habia_visto_ya._umbral_45`, que es el mismo
número que imprime la tabla. La figura `figura_copias_muestra.py` lee el bloque 2.

Uso:
    python copias_de_una_muestra.py --selftest
    python copias_de_una_muestra.py
"""

# ======================= CONSTANTES =======================

FICHERO_MUESTRAS = "../datos/salidas/muestras/red_s20260914_t1.txt"   # la fila de la tabla
LARGO = 45
MUESTRAS_DEL_CAPITULO = [1, 36, 173]
MUESTRA_A_MANO = 36
OTRA_A_MANO = 173
PALABRAS_ARRANQUE = 3

# Cómo se imprime la tabla del capítulo (Orden 57): el orden de las filas, de quien menos copia
# a quien más, y cómo se llama cada una. Los nombres de dentro (techo, contar_3, red_s20260914_t1…)
# no cambian: solo lo que se ve.
ORDEN_IMPRESO = ["suelo", "contar_1", "red_s20260914_t1", "contar_2", "contar_3", "techo"]
ETIQUETAS = {"suelo": "libros que no vio", "contar_1": "contar, 1 palabra", "red_s20260914_t1": "la red",
             "contar_2": "contar, 2 palabras", "contar_3": "contar, 3 palabras",
             "techo": "el texto real"}

# ==========================================================

import argparse
import glob
import os
import re
import sys
import unicodedata
from collections import Counter

import numpy as np

import lo_habia_visto_ya as L
import memoria_recurrente as mr
# Orden 97: cifras que las notas escriben con letra; si cambia el parámetro, falla aquí y no en la página.
EN_LETRA = {200: "doscientas", 7: "siete"}

from formato import ANCHO_CAJA, coma, miles, muestra_editorial, pct, tabla_editorial

AQUI = os.path.dirname(os.path.abspath(__file__))


def tramos(cubiertas):
    """Los tramos seguidos de palabras cubiertas: (desde, hasta) con las dos incluidas."""
    out, i = [], 0
    while i < len(cubiertas):
        if cubiertas[i]:
            j = i
            while j + 1 < len(cubiertas) and cubiertas[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def libros_de_entrenamiento():
    """(título, texto limpio con bordes) de cada libro que entró en el entrenamiento, con la
    misma limpieza y el mismo tope que `memoria_recurrente.cargar_texto()`."""
    ficheros = sorted(glob.glob(os.path.join(AQUI, mr.CORPUS, "*.txt")))
    out, total = [], 0
    for ruta in ficheros:
        crudo = open(ruta, encoding="utf-8", errors="replace").read()
        m = re.search(r"^Title:\s*(.+)$", crudo, re.M)
        titulo = m.group(1).strip() if m else os.path.basename(ruta)
        t = crudo
        if "*** START OF" in t:
            t = t.split("*** START OF", 1)[1].split("\n", 1)[-1]
        if "*** END OF" in t:
            t = t.split("*** END OF", 1)[0]
        t = unicodedata.normalize("NFC", t.lower())
        t = "".join(c if c in mr.ALFABETO else " " for c in t)
        t = re.sub(r" {2,}", " ", t)
        out.append((titulo, L._con_bordes(t)))
        total += len(t)
        if total >= mr.MAX_CARACTERES:
            break
    return out


def de_que_libro(trozo, libros):
    t = " " + trozo + " "
    return [titulo for titulo, texto in libros if t in texto]


def muestras_45():
    _, cuerpo = L._leer_fichero_muestras(os.path.join(AQUI, FICHERO_MUESTRAS))
    m = [(a, s) for (l, a, s) in cuerpo if l == LARGO]
    assert len(m) == L.VENTANAS, f"se esperaban {L.VENTANAS} muestras de {LARGO}; hay {len(m)}"
    return m


def partir(palabras, ancho, sangria="   "):
    out, linea = [], sangria
    for w in palabras:
        if len(linea) + len(w) + 1 > ancho and linea.strip():
            out.append(linea.rstrip())
            linea = sangria
        linea += w + " "
    out.append(linea.rstrip())
    return out


def marcar(palabras, cubiertas):
    """Las palabras con cada tramo copiado entre corchetes y su largo detrás: [a b c]·3."""
    out = []
    tr = {d: h for d, h in tramos(cubiertas)}
    i = 0
    while i < len(palabras):
        if i in tr:
            h = tr[i]
            grupo = palabras[i:h + 1]
            grupo[0] = "[" + grupo[0]
            grupo[-1] = grupo[-1] + f"]{h - i + 1}"
            out += grupo
            i = h + 1
        else:
            out.append(palabras[i])
            i += 1
    return out


# L24 (9 de octubre): los bloques salen como tablas y muestras editoriales (regla 6 ter). Las
# cuentas no cambian.

def bloque_tres(muestras):
    out = ["--- 1. LAS TRES MUESTRAS DEL CAPÍTULO ---", ""]
    for n in MUESTRAS_DEL_CAPITULO:
        arranque, s = muestras[n - 1]
        palabras = s.split()
        assert " ".join(palabras[:PALABRAS_ARRANQUE]) == arranque, \
            f"la muestra {n} no empieza por su arranque «{arranque}»"
        out += muestra_editorial(
            f"Muestra {n} de {L.VENTANAS}", partir(palabras[PALABRAS_ARRANQUE:], ANCHO_CAJA, ""),
            [f"Le di las {PALABRAS_ARRANQUE} palabras del comienzo, «{arranque}»; lo de arriba es "
             "lo que la red escribió detrás."]) + [""]
    return out[:-1]


def tabla_tramos(n_muestra, palabras, cubiertas, valores, libros):
    """Los tramos de una muestra: cuántas palabras tiene cada uno, dónde está y de qué libro
    sale; debajo, el copiado."""
    tr = tramos(cubiertas)
    n_cub = sum(cubiertas)
    filas = []
    for k, (d, h) in enumerate(tr, 1):
        fuentes = de_que_libro(" ".join(palabras[d:h + 1]), libros)
        fuente = fuentes[0] if len(fuentes) == 1 else (
            f"{len(fuentes)} libros" if fuentes else "de varios trozos")
        filas.append([str(k), str(h - d + 1), str(d + 1), str(h + 1), fuente])
    return tabla_editorial(
        f"Los tramos de la muestra {n_muestra}",
        ["tramo", "palabras", "desde la palabra", "hasta la", "de qué libro sale"], filas, "cdddi",
        [f"Palabras dentro de algún tramo: {n_cub} de {len(palabras)}; {n_cub} entre "
         f"{len(palabras)}: {pct(n_cub / len(palabras), 1)} (el «copiado»).",
         f"El tramo más largo que empieza en una palabra: {max(valores)}."])


def bloque_a_mano(muestras, corpus, umbral, libros):
    arranque, s = muestras[MUESTRA_A_MANO - 1]
    palabras = s.split()
    valores = L.rachas(s, corpus)
    cubiertas = L._cobertura(valores, umbral)
    tr = tramos(cubiertas)
    n_cub = sum(cubiertas)
    out = [f"--- 2. LA MUESTRA {MUESTRA_A_MANO}, TRAMO A TRAMO ---", ""]
    out += muestra_editorial(
        f"La muestra {MUESTRA_A_MANO}, tramo a tramo", partir(marcar(palabras, cubiertas), ANCHO_CAJA, ""),
        [f"Entre corchetes, cada tramo que está tal cual en los libros de entrenamiento con "
         f"{umbral} palabras seguidas o más; detrás del corchete, cuántas palabras tiene el tramo."])
    out += [""] + tabla_tramos(MUESTRA_A_MANO, palabras, cubiertas, valores, libros)
    return out, palabras, tr, n_cub, max(valores)


def bloque_otra(muestras, corpus, umbral, libros):
    _, s = muestras[OTRA_A_MANO - 1]
    valores = L.rachas(s, corpus)
    cub = L._cobertura(valores, umbral)
    return ([f"--- 3. LA MUESTRA {OTRA_A_MANO} ---", ""] +
            tabla_tramos(OTRA_A_MANO, s.split(), cub, valores, libros))


def bloque_suelo(corpus, umbral):
    ajenas = L._palabras_libros_ajenos()
    vocab_ajenas = set(w for w, _ in Counter(ajenas).most_common(20_000))
    primera = L._primera_aparicion(ajenas)
    puntos = L.arranques(ajenas, vocab_ajenas, n=L.VENTANAS)
    suelo = [L._ventana_literal(primera[t], ajenas, LARGO) for t in puntos]
    rmax, cop = L.medir(suelo, corpus, umbral)
    llegan = [i for i, r in enumerate(rmax) if r >= umbral]
    out = ["--- 4. EL SUELO: TEXTO QUE LA RED NO VIO ---", ""] + tabla_editorial(
        "El suelo: texto que la red no vio", ["", "cuánto"],
        [[f"trozos de {LARGO} palabras de seis libros que no entraron", str(L.VENTANAS)],
         [f"cuántos tienen algún tramo de {umbral} palabras o más", str(len(llegan))],
         ["en esos, la parte de la muestra dentro de un tramo",
          ", ".join(pct(cop[i], 1, de_uno=False) for i in llegan)],
         [f"el tramo más largo de los {L.VENTANAS}", f"{max(rmax)} palabras"]], "id")
    return out, len(llegan), max(rmax)


FILAS_TABLA = ["techo", "contar_3", "contar_2", "red_s20260914_t1", "contar_1", "suelo"]


def bloque_tabla(puntos, palabras, corpus, umbral, cache=None):
    """La tabla del capítulo (la de `lo_habia_visto_ya.txt`, ventana de 45) solo con las filas y
    columnas que usa el libro, y con cabeceras que no se pegan (tercera vuelta de L24). Los
    números son los mismos: se generan las mismas muestras, en el mismo orden y con la misma
    semilla que `lo_habia_visto_ya._medir_largo`, y se miden con la misma función. Con `cache`,
    cada fila se guarda al terminar (medirlas todas no cabe en una sola orden del portátil)."""
    import json
    largo = LARGO
    primera = L._primera_aparicion(palabras)
    rng = np.random.default_rng(L.SEMILLA)
    gen = {"techo": lambda: [L._ventana_literal(primera[t], palabras, largo) for t in puntos]}
    c3 = L._muestras_contar(3, puntos, palabras, largo, rng)
    c2 = L._muestras_contar(2, puntos, palabras, largo, rng)
    c1 = L._muestras_contar(1, puntos, palabras, largo, rng)
    gen.update({"contar_3": lambda: c3, "contar_2": lambda: c2, "contar_1": lambda: c1})
    gen["red_s20260914_t1"] = lambda: [m for _, m in muestras_45()]

    def suelo():
        ajenas = L._palabras_libros_ajenos()
        voc = set(w for w, _ in Counter(ajenas).most_common(20_000))
        pa = L.arranques(ajenas, voc, n=L.VENTANAS)
        pr = L._primera_aparicion(ajenas)
        return [L._ventana_literal(pr[t], ajenas, largo) for t in pa]
    gen["suelo"] = suelo
    res = {}
    for nombre in FILAS_TABLA:
        ruta = os.path.join(cache, nombre + ".json") if cache else None
        if ruta and os.path.exists(ruta):
            res[nombre] = json.load(open(ruta))
            continue
        rmax, cop = L.medir(gen[nombre](), corpus, umbral)
        res[nombre] = [float(np.median(rmax)), float(np.max(rmax)), float(np.median(cop))]
        if ruta:
            os.makedirs(cache, exist_ok=True)
            json.dump(res[nombre], open(ruta, "w"))
    assert sorted(ORDEN_IMPRESO) == sorted(FILAS_TABLA) == sorted(ETIQUETAS)
    filas = []
    for nombre in ORDEN_IMPRESO:
        med, mx, cop = res[nombre]
        filas.append([ETIQUETAS[nombre], coma(med, 1), coma(mx, 0), pct(cop, 1, de_uno=False)])
    lin = ["--- 5. LA TABLA DEL CAPÍTULO ---", ""] + tabla_editorial(
        "Rachas y copiado, según quién escribe",
        ["quién escribe", "racha, en palabras: mediana", "racha, en palabras: máximo",
         "copiado: mediana"], filas, "iddd",
        [# Orden 97: las notas dicen al lector qué es cada fila, no cómo se obtuvo; lo técnico
         # (los seis libros ajenos, el umbral y su margen) está en las constantes y en el código.
         f"Cada fila son {EN_LETRA[L.VENTANAS]} muestras de {largo} palabras. «Contar» es la "
         "máquina del capítulo 1, mirando una, dos o tres palabras atrás, y «el texto real», lo que "
         f"sigue a cada comienzo en su libro. Una racha cuenta como copia desde {EN_LETRA[umbral]} "
         "palabras seguidas."])
    return lin, res


def selftest():
    fallos = []
    texto = mr.cargar_texto()
    corpus = L._con_bordes(texto)
    palabras = texto.split()

    # 1. TEST NULO — una muestra de palabras barajadas no tiene ningún tramo de 7 o más: nada
    #    entre corchetes, 0 de 45.
    import random
    base = palabras[1000:1000 + LARGO]
    bar = list(base)
    random.Random(1).shuffle(bar)
    cub = L._cobertura(L.rachas(" ".join(bar), corpus), 7)
    print(f"[1] test nulo         45 palabras barajadas: {sum(cub)} dentro de un tramo, "
          f"{len(tramos(cub))} tramos")
    if sum(cub) != 0:
        fallos.append(f"test nulo: una muestra barajada tiene {sum(cub)} palabras copiadas")

    # 2. SEÑAL IMPLANTADA — se meten dos tramos de 10 y de 8 palabras del corpus en esa muestra
    #    barajada, separados: tienen que salir dos tramos, en su sitio, y 18 de 45.
    impl = bar[:5] + palabras[5000:5010] + bar[15:25] + palabras[9000:9008] + bar[33:]
    impl = impl[:LARGO]
    cub = L._cobertura(L.rachas(" ".join(impl), corpus), 7)
    tr = tramos(cub)
    print(f"[2] señal implantada  tramos {[(d + 1, h + 1) for d, h in tr]}, {sum(cub)} de {len(impl)} "
          f"(se esperaban (6, 15) y (26, 33), 18)")
    if not (tr[:1] and tr[0][0] <= 5 and tr[0][1] >= 14 and sum(cub) >= 18
            and any(d <= 25 and h >= 32 for d, h in tr)):
        fallos.append(f"señal implantada: tramos {tr}, {sum(cub)} palabras")

    # 3. INVARIANTE DEL DOMINIO — un trozo literal del corpus sale entero en un solo tramo, y
    #    la suma de los largos de los tramos es siempre el número de palabras cubiertas.
    cub = L._cobertura(L.rachas(" ".join(base), corpus), 7)
    tr = tramos(cub)
    suma = sum(h - d + 1 for d, h in tr)
    print(f"[3] invariante        trozo literal: {len(tr)} tramo de {suma} palabras; "
          f"suma de largos = cubiertas: {suma == sum(cub)}")
    if len(tr) != 1 or suma != LARGO or suma != sum(cub):
        fallos.append(f"invariante: trozo literal en {len(tr)} tramos, {suma} palabras")
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
    ap.add_argument("--cache", metavar="DIR")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    if selftest():
        sys.exit("El selftest falla: los números no valen.")
    from datetime import date
    print(f"\nMedido el {date.today().isoformat()}. No entrena nada: busca texto en texto.")
    texto = mr.cargar_texto()
    palabras = texto.split()
    vocab, _, _, _ = mr.vocabulario_y_datos(texto)
    corpus = L._con_bordes(texto)
    puntos = L.arranques(palabras, set(vocab), n=L.VENTANAS)
    umbral = L._umbral_45(puntos, palabras, corpus, np.random.default_rng(L.SEMILLA))
    muestras = muestras_45()
    libros = libros_de_entrenamiento()
    print(f"listón: {umbral} palabras (el de la tabla de lo_habia_visto_ya.txt)\n")
    print("\n".join(bloque_tres(muestras)) + "\n")
    lin, *_ = bloque_a_mano(muestras, corpus, umbral, libros)
    print("\n".join(lin) + "\n")
    print("\n".join(bloque_otra(muestras, corpus, umbral, libros)) + "\n")
    lin, *_ = bloque_suelo(corpus, umbral)
    print("\n".join(lin) + "\n")
    cache = sys.argv[sys.argv.index("--cache") + 1] if "--cache" in sys.argv else None
    lin, _ = bloque_tabla(puntos, palabras, corpus, umbral, cache)
    print("\n".join(lin))


if __name__ == "__main__":
    main()
