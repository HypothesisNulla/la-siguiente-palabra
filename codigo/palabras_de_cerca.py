#!/usr/bin/env python3
"""
Capítulo 5 — qué es que dos palabras estén «cerca», hecho con los números delante (L24).

El capítulo 5 dice que el método deja «cerca» a las palabras que se parecen, que «rey» menos
«hombre» más «mujer» cae junto a «reina», y que los caminos del género «apuntan hacia el mismo
lado». Todo eso se contaba con palabras. Carlos, el 27 de septiembre: «No tiene que imaginar:
tiene que verlo escrito y dibujado». Este programa entrena el mismo modelo que
`palabras_numeros.py` (mismo corpus, mismos ajustes, misma semilla, importados de allí) y
enseña, con sus números:

  1. LA TAREA, HECHA UNA VEZ. Una frase del texto con «caballo» en medio, sus vecinas y cinco
     palabras al azar, y lo que contesta la máquina a «¿van juntas?» antes y después de entrenar.
  2. LOS NÚMEROS DE TRES PALABRAS, antes y después: los primeros de los cien de «caballo»,
     «corcel» y «lunes».
  3. CÓMO SE MIDE EL PARECIDO: en cuántas de las cien posiciones coinciden de signo, y el
     parecido que da la cuenta entera (la misma de `la_misma_direccion.coseno`).
  4. LAS VECINAS DE «CABALLO», CON SU PARECIDO: «más cerca» quiere decir «más parecido».
  5. LA AVERÍA DEL CAPÍTULO 1: «hidalgo» y «caballero», «rocín» y «caballo», «dijo» y
     «respondió»..., con su parecido y en qué puesto sale una entre las vecinas de la otra.
  6. LA RESTA, NÚMERO A NÚMERO, con los tres primeros de los cien, y en qué puesto sale cada
     palabra buscada.
  7. LOS CAMINOS DEL GÉNERO: el parecido de cada pareja de caminos y el largo de cada camino,
     para dibujarlos (`figura_misma_direccion.py`).
  8. EL MAPA: unas pocas palabras de los bloques del capítulo aplanadas de cien números a dos,
     y cuántas conservan en el papel su vecina más parecida (`figura_mapa_de_palabras.py`).
  (9. Quitado en la segunda vuelta de L24: usaba las listas de `cuantas_veces_sale.py`, que se
     escribieron después de ver la primera tirada. Lo sustituye `banco_segun_el_diccionario.py`.)
 10. LA CUENTA DEL PARECIDO, hecha a mano con los tres primeros números de dos palabras.

Comprueba antes de nada que el modelo es el del capítulo: las vecinas de las ocho palabras de
`palabras_numeros.py` tienen que ser las de su salida, `datos/salidas/palabras_numeros.txt`,
medida en esta misma máquina. Si no lo son, revienta: las cifras de aquí no casarían con los
bloques del libro.

Uso:
    python palabras_de_cerca.py --selftest
    python palabras_de_cerca.py
"""

# ======================= CONSTANTES =======================

SALIDA_PALABRAS_NUMEROS = "../datos/salidas/palabras_numeros.txt"

PALABRA_TAREA = "caballo"
TRES_PALABRAS = ["caballo", "corcel", "lunes"]     # «lunes»: la del capítulo 1
NUMEROS_A_ENSENAR = 6                               # de los cien
AL_AZAR = 5                                         # palabras sacadas al azar para la tarea
NUMEROS_CUENTA = 3                                  # la cuenta del parecido, hecha a mano

PAREJAS_PARECIDO = [("caballo", "corcel"), ("caballo", "galope"), ("caballo", "lunes")]

# Las del capítulo 1, con la palabra más frecuente delante: el puesto se mira entre las vecinas
# de la primera (el de una palabra rara entre las vecinas de otra no dice nada: las raras
# tienen de vecinas otras raras).
PAREJAS_AVERIA = [("caballo", "rocín"), ("caballo", "rocinante"), ("caballo", "corcel"),
                  ("caballero", "hidalgo"), ("dijo", "respondió"), ("dijo", "preguntó"),
                  ("dijo", "replicó"), ("caballo", "lunes")]

RESTAS = [("rey", "hombre", "mujer", "reina"), ("padre", "hombre", "mujer", "madre"),
          ("hijo", "hombre", "mujer", "hija"), ("hermano", "hombre", "mujer", "hermana"),
          ("caballo", "macho", "hembra", "yegua")]
NUMEROS_RESTA = 6
PUESTOS_RESULTADO = 5

CAMINOS = [("rey", "reina"), ("padre", "madre"), ("hermano", "hermana"), ("hijo", "hija")]

# El mapa: palabras de los bloques del capítulo, escogidas antes de mirar el mapa. Los grupos
# solo sirven para la clave de la figura; el aplanamiento no los conoce.
MAPA = [("montar a caballo", ["caballo", "galope", "estribo", "montar"]),
        ("reyes", ["rey", "emperador", "soberano"]),
        ("armas", ["espada", "pistola", "carabina"]),
        ("horas", ["noche", "madrugada", "tarde"]),
        ("dinero", ["dinero", "pagar", "negocio"]),
        ("banco", ["banco"]),
        ("las cinco vecinas de «banco»", ["escritorio", "armario", "baúl", "almacén", "ferrocarril"])]
VUELTAS_APLANADO = 500


TOL = 1e-5

# Los números que las notas de las tablas dicen con letra (Orden 55: las notas se escriben como las
# diría una persona); el número sale de la variable, la palabra de esta tabla.
EN_LETRA = {3: "tres", 5: "cinco", 6: "seis", 100: "cien"}

# ==========================================================

import argparse
import os
import platform
import random
import re
import sys
from collections import Counter
from datetime import date

import numpy as np

from formato import ANCHO_CAJA, coma, leer_tablas, miles, muestra_editorial, tabla_editorial
from palabras_numeros import (CORPUS_BIBLIOTECA, DIMENSION, EPOCAS, HILOS, MIN_APARICIONES,
                              SEMILLA, SONDAS, VECINOS, VENTANA, cargar_corpus)

AQUI = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- las cuentas

def letra(n):
    return EN_LETRA[n]


def parecido(u, v):
    """La cuenta del capítulo: 1 si las dos listas suben y bajan a la par, 0 si no se siguen
    en nada (perpendiculares), -1 si van al revés. Es la de `la_misma_direccion.coseno` y la de gensim."""
    u, v = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))


def mismo_signo(u, v):
    """En cuántas posiciones las dos listas tienen el mismo signo."""
    return int(np.sum(np.sign(u) == np.sign(v)))


def si_van_juntas(vec_otra, vec_salida):
    """Lo que contesta el método a «¿van juntas estas dos palabras?», de 0 (no) a 100 (sí).
    Es la cuenta que se corrige al entrenar (muestreo negativo de gensim): la lista de la otra
    palabra contra la segunda lista de la palabra del centro."""
    x = float(np.dot(vec_otra, vec_salida))
    return 100.0 / (1.0 + np.exp(-x))


def unidad(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def resta(vectores, a, b, c, n, excluir=True):
    """a - b + c con cada lista antes puesta a la misma escala (lo que hace gensim), y las n
    palabras más parecidas al resultado. Con excluir, las tres de la pregunta no cuentan."""
    r = unidad(vectores[a]) - unidad(vectores[b]) + unidad(vectores[c])
    todas = vectores.get_normed_vectors()
    s = todas @ (r / np.linalg.norm(r))
    orden = np.argsort(-s)
    salida = []
    for i in orden:
        p = vectores.index_to_key[i]
        if excluir and p in (a, b, c):
            continue
        salida.append((p, float(s[i])))
        if len(salida) == n:
            break
    return r, salida


def puesto_en_resta(vectores, a, b, c, buscada):
    r = unidad(vectores[a]) - unidad(vectores[b]) + unidad(vectores[c])
    s = vectores.get_normed_vectors() @ (r / np.linalg.norm(r))
    orden = [vectores.index_to_key[i] for i in np.argsort(-s) if vectores.index_to_key[i] not in (a, b, c)]
    return orden.index(buscada) + 1


def puesto_vecina(vectores, palabra, otra):
    """En qué puesto sale `otra` entre las vecinas de `palabra` (1 = la más parecida)."""
    s = vectores.get_normed_vectors() @ unidad(vectores[palabra])
    orden = [vectores.index_to_key[i] for i in np.argsort(-s) if vectores.index_to_key[i] != palabra]
    return orden.index(otra) + 1


def aplanar(distancias, vueltas=VUELTAS_APLANADO):
    """De una tabla de distancias a puntos en un plano: primero el aplanado clásico, luego se
    corrige vuelta a vuelta para que las distancias del papel se parezcan a las de verdad
    (SMACOF). Sin azar: siempre sale el mismo mapa."""
    d = np.asarray(distancias, dtype=np.float64)
    n = len(d)
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ (d ** 2) @ j
    val, vec = np.linalg.eigh(b)
    orden = np.argsort(-val)[:2]
    x = vec[:, orden] * np.sqrt(np.maximum(val[orden], 0))
    for _ in range(vueltas):
        dx = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(dx > 0, d / dx, 0.0)
        bm = -r
        np.fill_diagonal(bm, 0.0)
        np.fill_diagonal(bm, -bm.sum(axis=1))
        x = bm @ x / n
    # centrado, y con el primer punto siempre a la izquierda y arriba: el signo de un
    # aplanado es arbitrario, y así el mapa no se da la vuelta de una ejecución a otra
    x -= x.mean(axis=0)
    x *= np.where(x[0] > 0, -1.0, 1.0) * np.array([1.0, -1.0])
    return x


def vecina_mas_cercana(dist):
    d = np.array(dist, dtype=np.float64)
    np.fill_diagonal(d, np.inf)
    return d.argmin(axis=1)


# --------------------------------------------------------------------------- el modelo

def entrenar_guardando_el_arranque(frases, palabras):
    """El mismo entrenamiento que `palabras_numeros.entrenar()`, en sus dos mitades (lo que
    hace gensim por dentro al darle las frases de golpe): preparar el vocabulario, que pone los
    números al azar, y entrenar. Entre las dos se guardan los números de arranque."""
    from gensim.models import Word2Vec
    m = Word2Vec(vector_size=DIMENSION, window=VENTANA, min_count=MIN_APARICIONES, sg=1,
                 workers=HILOS, seed=SEMILLA, epochs=EPOCAS)
    m.build_vocab(frases)
    antes = {p: m.wv[p].copy() for p in palabras if p in m.wv}
    salida_antes = {p: m.syn1neg[m.wv.key_to_index[p]].copy() for p in palabras if p in m.wv}
    m.train(frases, total_examples=m.corpus_count, total_words=m.corpus_total_words,
            epochs=m.epochs, start_alpha=m.alpha, end_alpha=m.min_alpha)
    return m, antes, salida_antes


def vecinas_del_libro(ruta):
    """Las vecinas de la biblioteca en la salida de palabras_numeros.py (su tabla editorial)."""
    tablas = leer_tablas(open(ruta, encoding="utf-8").read())
    titulo = "Las vecinas más próximas, con una biblioteca (300 libros)"
    assert titulo in tablas, f"se esperaba la tabla «{titulo}» en {ruta}"
    out = {f[0]: [x.strip() for x in f[1].split(",")] for f in tablas[titulo][1]}
    assert set(out) == set(SONDAS), f"se esperaban las vecinas de {SONDAS}; hay {sorted(out)}"
    return out


# --------------------------------------------------------------------------- los bloques
# L24 (9 de octubre): cada bloque sale en tablas editoriales (regla 6 ter). Las cuentas no
# cambian; cambia la forma, y lo que antes eran frases con cifras pasa a casillas con rótulo.

def imprimir(*bloques):
    for b in bloques:
        print("\n".join(b))
        print()


def bloque_tarea(m, frases, antes_sal, rng):
    frase = None
    for f in frases:
        for i, p in enumerate(f):
            if p == PALABRA_TAREA and i >= VENTANA and i + VENTANA < len(f) \
                    and all(w in m.wv for w in f[i - VENTANA:i + VENTANA + 1]):
                frase, pos = f, i
                break
        if frase:
            break
    assert frase, f"no hay ninguna frase con «{PALABRA_TAREA}» y {VENTANA} palabras a cada lado"
    alrededor = frase[pos - VENTANA:pos] + frase[pos + 1:pos + VENTANA + 1]
    # cinco palabras al azar, con la misma regla con que las saca el método para corregirse:
    # más probable cuanto más frecuente (la frecuencia elevada a 0,75), sin repetir las de la frase
    vocab = m.wv.index_to_key
    frec = np.array([m.wv.get_vecattr(p, "count") for p in vocab], dtype=np.float64) ** 0.75
    frec /= frec.sum()
    azar = []
    while len(azar) < AL_AZAR:
        p = vocab[int(rng.choice(len(vocab), p=frec))]
        if p not in alrededor and p != PALABRA_TAREA and p not in azar:
            azar.append(p)
    trozo_ = frase[pos - VENTANA:pos + VENTANA + 1]
    texto = " ".join(trozo_[:VENTANA]) + " [" + PALABRA_TAREA + "] " + " ".join(trozo_[VENTANA + 1:])
    muestra = muestra_editorial(
        f"Una frase del texto, con «{PALABRA_TAREA}» en medio",
        _partir(texto, ANCHO_CAJA),
        [f"Las {letra(VENTANA)} palabras de cada lado de «{PALABRA_TAREA}», que va entre corchetes."])
    sal = m.syn1neg[m.wv.key_to_index[PALABRA_TAREA]]
    filas, tabla = [], []
    for de_verdad, lista, rotulo in ((True, alrededor, "de alrededor"),
                                     (False, azar, "sacada al azar")):
        for k, p in enumerate(lista):
            a = si_van_juntas(m.wv[p], antes_sal[PALABRA_TAREA])
            d = si_van_juntas(m.wv[p], sal)
            filas.append((p, a, d, de_verdad))
            tabla.append([rotulo if k == 0 else "", p, coma(a, 0), coma(d, 0)])
    media = lambda de_verdad: float(np.mean([d for _, _, d, v in filas if v == de_verdad]))
    tabla_ = tabla_editorial(
        f"¿Van juntas «{PALABRA_TAREA}» y esta palabra?",
        ["de dónde sale", "la palabra", "antes de entrenar", "después de entrenar"], tabla, "iidd",
        ["Cada número es lo que contesta la máquina, de 0, seguro que no, a 100, seguro que sí. "
         f"Después de entrenar, las palabras de alrededor sacan de media {coma(media(True), 1)}, "
         f"y las sacadas al azar, {coma(media(False), 1)}."])
    return [muestra, tabla_], filas


def _partir(texto, ancho):
    out, linea = [], ""
    for w in texto.split():
        if linea and len(linea) + 1 + len(w) > ancho:
            out.append(linea)
            linea = w
        else:
            linea = (linea + " " + w).strip()
    if linea:
        out.append(linea)
    return out


def bloque_tres_palabras(m, antes):
    n = NUMEROS_A_ENSENAR
    rotulos = ["palabra"] + [f"{k}.º" for k in range(1, n + 1)]
    fila = lambda p, v: [p] + [coma(float(x), 3) for x in v[:n]]
    return [tabla_editorial(f"Los {n} primeros de sus {DIMENSION} números, antes de entrenar",
                            rotulos, [fila(p, antes[p]) for p in TRES_PALABRAS], "i" + "d" * n,
                            ["Puestos al azar."]),
            tabla_editorial(f"Los {n} primeros de sus {DIMENSION} números, después de entrenar",
                            rotulos, [fila(p, m.wv[p]) for p in TRES_PALABRAS], "i" + "d" * n)]


def bloque_parecido(m, antes):
    n = len(m.wv.index_to_key)
    valores, filas = {}, []
    for a, b in PAREJAS_PARECIDO:
        sa, pa = mismo_signo(antes[a], antes[b]), parecido(antes[a], antes[b])
        sd, pd = mismo_signo(m.wv[a], m.wv[b]), parecido(m.wv[a], m.wv[b])
        pu = puesto_vecina(m.wv, a, b)
        valores[(a, b)] = (sa, pa, sd, pd, pu)
        filas.append([f"{a} y {b}", str(sa), coma(pa, 2), str(sd), coma(pd, 2), miles(pu)])
    # La referencia (tercera vuelta de L24): la vecina de «caballo» que queda a media tabla. Es
    # contra lo que se leen las cifras de «caballo»; el parecido medio de dos palabras sacadas
    # al azar de todo el vocabulario no sirve, porque casi todas son raras y las raras se
    # parecen mucho entre sí.
    a0 = PAREJAS_PARECIDO[0][0]
    s_ = m.wv.get_normed_vectors() @ unidad(m.wv[a0])
    orden = [i for i in np.argsort(-s_) if m.wv.index_to_key[i] != a0]
    medio = (len(orden) + 1) // 2
    w_medio = m.wv.index_to_key[orden[medio - 1]]
    A, D = "antes de entrenar", "después de entrenar"
    return [tabla_editorial(
        "Cómo se parecen dos listas",
        ["pareja", f"{A}: signo", f"{A}: parecido", f"{D}: signo", f"{D}: parecido", f"{D}: puesto"],
        filas, "iddddd",
        [f"«Signo» cuenta en cuántas de las {letra(DIMENSION)} posiciones coinciden de signo las "
         "dos listas; «parecido» es la cuenta entera, de -1 a 1, y «puesto», el lugar de la "
         f"segunda palabra entre las {miles(n - 1)} vecinas de la primera. Para comparar: a media "
         f"tabla de las vecinas de «{a0}», en el puesto {miles(medio)}, está «{w_medio}», con un "
         f"parecido de {coma(parecido(m.wv[a0], m.wv[w_medio]), 2)}."])], valores


def bloque_vecinas(m):
    return [tabla_editorial(
        f"Las vecinas de «{PALABRA_TAREA}», con su parecido", ["vecina", "parecido"],
        [[p, coma(s, 2)] for p, s in m.wv.most_similar(PALABRA_TAREA, topn=VECINOS)], "id",
        [f"De las {miles(len(m.wv.index_to_key))} palabras, las {VECINOS} más parecidas."])]


def bloque_averia(m):
    n = len(m.wv.index_to_key)
    filas, tabla = [], []
    for a, b in PAREJAS_AVERIA:
        assert a in m.wv and b in m.wv, f"«{a}» o «{b}» no está en el vocabulario"
        s, pu = parecido(m.wv[a], m.wv[b]), puesto_vecina(m.wv, a, b)
        filas.append((a, b, s, pu))
        tabla.append([f"{a} y {b}", coma(s, 2), miles(pu)])
    return [tabla_editorial(
        "Las palabras del capítulo 1", ["pareja", "parecido", "puesto"], tabla, "idd",
        [f"La columna «puesto» dice en qué puesto sale la segunda palabra entre las {miles(n - 1)} "
         "vecinas de la primera; el 1 es la más parecida."])], filas


def bloque_resta(m):
    a, b, c, buscada = RESTAS[0]
    r, _ = resta(m.wv, a, b, c, PUESTOS_RESULTADO)
    k = NUMEROS_RESTA
    fila = lambda nombre, v, negrita=False: [
        (f"**{x}**" if negrita else x) for x in [nombre] + [coma(float(y), 3) for y in v[:k]]]
    numeros = tabla_editorial(
        "La resta, número a número", [""] + [f"{j}.º" for j in range(1, k + 1)],
        [fila(a, unidad(m.wv[a])), fila(f"menos {b}", unidad(m.wv[b])),
         fila(f"más {c}", unidad(m.wv[c])), fila("el resultado", r, True),
         fila(f"y, para comparar, {buscada}", unidad(m.wv[buscada]))], "i" + "d" * k,
        [f"Son los {letra(k)} primeros de los {letra(DIMENSION)} números. Antes de restar, cada "
         "lista se divide por su tamaño, para que no mande la que tiene los números más grandes."])
    resultado = tabla_editorial(
        f"El resultado, con los {DIMENSION} números", ["comparado con", "signo", "parecido"],
        [[f"«{p}»", str(mismo_signo(r, m.wv[p])), coma(parecido(r, m.wv[p]), 2)]
         for p in (buscada, a, b, c)], "idd")
    _, todas = resta(m.wv, a, b, c, PUESTOS_RESULTADO, excluir=False)
    filas, i = [], 0
    for p, s in todas:
        if p in (a, b, c):
            filas.append(["no cuenta", p, coma(s, 2)])
        else:
            i += 1
            filas.append([str(i), p, coma(s, 2)])
    parecidas = tabla_editorial(
        "Las más parecidas al resultado, de todas", ["puesto", "palabra", "parecido"], filas, "cid",
        ["Las tres palabras de la pregunta no cuentan, y por eso no llevan puesto."])
    puestos, filas = [], []
    for a2, b2, c2, bus in RESTAS:
        pu = puesto_en_resta(m.wv, a2, b2, c2, bus)
        puestos.append((a2, b2, c2, bus, pu))
        filas.append([f"{a2}, menos {b2}, más {c2}", bus, miles(pu)])
    donde = tabla_editorial(
        "En qué puesto sale la palabra buscada", ["la cuenta", "la buscada", "puesto"], filas, "iid",
        ["El puesto se cuenta sin las tres palabras de la pregunta."])
    return [numeros, resultado, parecidas, donde], puestos


def nombre_camino(x, y):
    return f"de {x} a {y}"


def bloque_caminos(m, rng):
    difs = {(x, y): m.wv[y].astype(np.float64) - m.wv[x] for x, y in CAMINOS}
    largos = tabla_editorial(
        "El tamaño de cada camino", ["camino", "tamaño"],
        [[nombre_camino(x, y), coma(float(np.linalg.norm(d)), 2)] for (x, y), d in difs.items()],
        "id",
        ["Camino: la resta de las dos listas, número a número. Tamaño: la misma cuenta que el "
         "de una lista."])
    claves = list(difs)
    pares, filas = [], []
    for i in range(len(claves)):
        for j in range(i + 1, len(claves)):
            s = parecido(difs[claves[i]], difs[claves[j]])
            pares.append((claves[i], claves[j], s))
            filas.append([nombre_camino(*claves[i]), nombre_camino(*claves[j]), coma(s, 2)])
    media = float(np.mean([s for *_, s in pares]))
    from la_misma_direccion import referencia_al_azar
    azar = referencia_al_azar(m.wv)
    filas += [[f"**media de las {len(pares)}**", "", f"**{coma(media, 2)}**"],
              ["caminos entre palabras sacadas al azar", f"media de {len(azar)} comparaciones",
               coma(float(np.mean(azar)), 3)]]
    return [largos, tabla_editorial(
        "Los caminos del género, de dos en dos", ["un camino", "y otro", "parecido"], filas, "iid",
        ["Un camino es la resta de dos listas, número a número. Un parecido de 1 querría decir que "
         "los dos caminos apuntan exactamente hacia el mismo lado."])], pares


def bloque_mapa(m):
    palabras = [p for _, g in MAPA for p in g]
    for p in palabras:
        assert p in m.wv, f"«{p}» no está en el vocabulario"
    v = np.array([unidad(m.wv[p]) for p in palabras])
    dist = 1.0 - np.clip(v @ v.T, -1, 1)
    np.fill_diagonal(dist, 0.0)
    x = aplanar(dist)
    papel = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    de_verdad, en_papel = vecina_mas_cercana(dist), vecina_mas_cercana(papel)
    iguales = int(np.sum(de_verdad == en_papel))
    papel_sin = papel.copy()
    np.fill_diagonal(papel_sin, np.inf)
    entre_tres = int(sum(de_verdad[i] in np.argsort(papel_sin[i])[:3] for i in range(len(palabras))))
    filas = []
    for i, p in enumerate(palabras):
        grupo = next(k for k, (g, ps) in enumerate(MAPA, 1) if p in ps)
        filas.append([p, coma(x[i, 0], 3), coma(x[i, 1], 3), palabras[de_verdad[i]], str(grupo)])
    mapa = tabla_editorial(
        "El mapa: de cien números a dos", ["palabra", "x", "y", "su más parecida", "grupo"],
        filas, "iddic",
        ["Distancia en el papel: 1 menos el parecido. Aplanado para que las distancias del papel "
         "se acerquen a esas.",
         f"Lo que se pierde al aplanar: de las {len(palabras)} palabras, en {iguales} su más "
         f"parecida es también la más cercana en el papel, y en {entre_tres} está entre las tres "
         "más cercanas en el papel."])
    grupos = tabla_editorial(
        "Los grupos del mapa", ["grupo", "qué palabras"],
        [[str(k), g] for k, (g, _) in enumerate(MAPA, 1)], "ci",
        ["Solo para la clave de la figura: el aplanado no los conoce."])
    return [mapa, grupos], iguales


def bloque_cuenta(m):
    """La cuenta del parecido hecha a mano con los primeros números de dos palabras, para que el
    lector vea cada paso. Con los cien números es la misma cuenta, y da el parecido del bloque 3."""
    a, b = PAREJAS_PARECIDO[0]
    u = [float(x) for x in m.wv[a][:NUMEROS_CUENTA]]
    v = [float(x) for x in m.wv[b][:NUMEROS_CUENTA]]
    prod = [x * y for x, y in zip(u, v)]
    ta2, tb2 = sum(x * x for x in u), sum(y * y for y in v)
    ta, tb = ta2 ** 0.5, tb2 ** 0.5
    filas = [[f"{k + 1}.º", coma(u[k], 3), coma(v[k], 3), coma(prod[k], 3)]
             for k in range(NUMEROS_CUENTA)]
    filas.append(["**la suma**", "", "", f"**{coma(sum(prod), 3)}**"])
    productos = tabla_editorial(
        f"La cuenta del parecido, con los {NUMEROS_CUENTA} primeros números",
        ["posición", a, b, "uno por otro"], filas, "cddd",
        [f"Son los {letra(NUMEROS_CUENTA)} primeros números de «{a}» y de «{b}», después de "
         f"entrenar. Con los {letra(DIMENSION)}, la cuenta es la misma, solo que más larga."])
    tamanos = tabla_editorial(
        "El tamaño de cada lista", ["", a, b],
        [["cada número por sí mismo, sumados", coma(ta2, 3), coma(tb2, 3)],
         ["**el tamaño: la raíz cuadrada de esa suma**", f"**{coma(ta, 4)}**", f"**{coma(tb, 4)}**"]],
        "idd",
        ["El tamaño es la raíz cuadrada de la suma: el número que, multiplicado por sí mismo, da "
         f"esa suma. Por ejemplo, {coma(ta, 4)} por {coma(ta, 4)} da {coma(ta * ta, 3)}."])
    final = tabla_editorial(
        f"El parecido de «{a}» y «{b}»", ["", "cuánto"],
        [["la suma de «uno por otro»", coma(sum(prod), 3)],
         ["un tamaño por el otro", coma(ta * tb, 3)],
         ["**el parecido: la suma entre los dos tamaños multiplicados**",
          f"**{coma(sum(prod) / (ta * tb), 2)}**"],
         [f"el parecido, con los {DIMENSION} números", coma(parecido(m.wv[a], m.wv[b]), 2)]], "id")
    return [productos, tamanos, final], sum(prod) / (ta * tb)


# --------------------------------------------------------------------------- selftest

def selftest():
    fallos = []
    rng = np.random.default_rng(SEMILLA)

    # 1. TEST NULO — dos listas de cien números sacadas al azar, sin relación: coinciden de
    #    signo más o menos en la mitad de las posiciones y su parecido anda cerca de 0. Si la
    #    cuenta diera parecidos altos sin relación, «cerca» no querría decir nada.
    signos, parecidos_ = [], []
    for _ in range(500):
        u, v = rng.normal(size=DIMENSION), rng.normal(size=DIMENSION)
        signos.append(mismo_signo(u, v))
        parecidos_.append(parecido(u, v))
    ms, mp = float(np.mean(signos)), float(np.mean(np.abs(parecidos_)))
    print(f"[1] test nulo         listas al azar: mismo signo {coma(ms, 1)} de {DIMENSION}; "
          f"parecido medio en valor absoluto {coma(mp, 3)}")
    if not (45 <= ms <= 55 and mp < 0.15):
        fallos.append(f"test nulo: mismo signo {ms:.1f}, parecido {mp:.3f}")

    # 2. SEÑAL IMPLANTADA — (a) veinte puntos que de verdad están en un plano, metidos en cien
    #    números con un giro al azar: el aplanado tiene que devolverles su vecina más cercana a
    #    todos. (b) cuatro palabras fabricadas en que «d - c» es exactamente «b - a»: la resta
    #    tiene que encontrar la cuarta la primera.
    plano = rng.normal(size=(20, 2))
    giro, _ = np.linalg.qr(rng.normal(size=(DIMENSION, DIMENSION)))
    puntos = np.hstack([plano, np.zeros((20, DIMENSION - 2))]) @ giro
    dist = np.linalg.norm(puntos[:, None] - puntos[None], axis=2)
    x = aplanar(dist)
    papel = np.linalg.norm(x[:, None] - x[None], axis=2)
    iguales = int(np.sum(vecina_mas_cercana(dist) == vecina_mas_cercana(papel)))
    from gensim.models import KeyedVectors
    kv = KeyedVectors(DIMENSION)
    base = rng.normal(size=(50, DIMENSION))
    a, b, c = base[0], base[1], base[2]
    d = unidad(unidad(a) - unidad(b) + unidad(c))
    kv.add_vectors([f"p{i}" for i in range(50)] + ["objetivo"], np.vstack([base, d]).astype(np.float32))
    _, top = resta(kv, "p0", "p1", "p2", 1)
    print(f"[2] señal implantada  puntos en un plano: {iguales} de 20 conservan su vecina; "
          f"resta fabricada -> «{top[0][0]}»")
    if iguales != 20:
        fallos.append(f"señal implantada: el aplanado pierde vecinas de puntos que ya son planos ({iguales} de 20)")
    if top[0][0] != "objetivo":
        fallos.append(f"señal implantada: la resta no encuentra la palabra fabricada, sale {top[0][0]}")

    # 3. INVARIANTE DEL DOMINIO — una lista consigo misma: parecido 1 y cien de cien de mismo
    #    signo; con la misma cambiada de signo, -1 y cero; el parecido no cambia al estirar una
    #    lista; «¿van juntas?» con una lista de ceros da exactamente 50; y la resta propia da lo
    #    mismo que la de gensim.
    u = rng.normal(size=DIMENSION)
    ok = (abs(parecido(u, u) - 1) < TOL and abs(parecido(u, -u) + 1) < TOL
          and mismo_signo(u, u) == DIMENSION and mismo_signo(u, -u) == 0
          and abs(parecido(u, 3 * u) - 1) < TOL
          and abs(si_van_juntas(u, np.zeros(DIMENSION)) - 50) < TOL)
    propia = [p for p, _ in resta(kv, "p3", "p4", "p5", 5)[1]]
    gensim_ = [p for p, _ in kv.most_similar(positive=["p3", "p5"], negative=["p4"], topn=5)]
    print(f"[3] invariante        consigo misma, al revés, estirada y con ceros: "
          f"{'bien' if ok else 'MAL'}; resta propia = gensim: {propia == gensim_}")
    if not ok:
        fallos.append("invariante: la cuenta del parecido o de «¿van juntas?» no cumple lo básico")
    if propia != gensim_:
        fallos.append(f"invariante: la resta propia {propia} no es la de gensim {gensim_}")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


# --------------------------------------------------------------------------- principal

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--comprobar-con", default=SALIDA_PALABRAS_NUMEROS, metavar="SALIDA",
                    help="la salida de palabras_numeros.py con que se comprueba que el modelo "
                         "es el mismo (por omisión, la de datos/salidas)")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    if selftest():
        sys.exit("El selftest falla: los números no valen.")

    print(f"\nMáquina: {platform.machine()}, {platform.system()} {platform.release()}.")
    print(f"Medido el {date.today().isoformat()}. Semilla: {SEMILLA}.")
    frases, total = cargar_corpus(os.path.join(AQUI, CORPUS_BIBLIOTECA))
    necesarias = TRES_PALABRAS + [PALABRA_TAREA] + [p for par in PAREJAS_PARECIDO for p in par]
    m, antes, antes_sal = entrenar_guardando_el_arranque(frases, necesarias)
    print(f"Biblioteca: {miles(total)} palabras, {miles(len(m.wv.index_to_key))} distintas.")

    libro = vecinas_del_libro(os.path.join(AQUI, args.comprobar_con))
    for s in SONDAS:
        mias = [p for p, _ in m.wv.most_similar(s, topn=VECINOS)]
        assert mias == libro[s], (f"«{s}»: este modelo da {mias} y palabras_numeros.txt dice "
                                  f"{libro[s]}; no es el mismo modelo (¿otra máquina?)")
    print("Mismo modelo que palabras_numeros.txt: las vecinas de las ocho palabras coinciden.\n")

    rng = np.random.default_rng(SEMILLA)
    secciones = [("1. LA TAREA, HECHA UNA VEZ", bloque_tarea(m, frases, antes_sal, rng)[0]),
                 ("2. LOS NÚMEROS DE TRES PALABRAS", bloque_tres_palabras(m, antes)),
                 ("3. CÓMO SE PARECEN DOS LISTAS", bloque_parecido(m, antes)[0]),
                 ("4. LAS VECINAS DE «CABALLO», CON SU PARECIDO", bloque_vecinas(m)),
                 ("5. LAS PALABRAS DEL CAPÍTULO 1", bloque_averia(m)[0]),
                 ("6. LA RESTA, NÚMERO A NÚMERO", bloque_resta(m)[0]),
                 ("7. LOS CAMINOS DEL GÉNERO", bloque_caminos(m, rng)[0]),
                 ("8. EL MAPA: DE CIEN NÚMEROS A DOS", bloque_mapa(m)[0]),
                 ("10. LA CUENTA DEL PARECIDO, HECHA CON TRES NÚMEROS", bloque_cuenta(m)[0])]
    for titulo, bloques in secciones:
        print(f"--- {titulo} ---\n")
        imprimir(*bloques)

if __name__ == "__main__":
    main()
