#!/usr/bin/env python3
"""
"¿Lo había visto ya?" — el instrumento que mide cuánto de un texto generado es copia
literal del corpus de entrenamiento.

Sustituye a `cuanto_copia.py`, que comparaba la red del capítulo 6 (686.002 palabras, ventana
de 45) con la máquina de contar del capítulo 1 (el Quijote, ventana de 35) en un solo tiro cada
una y sin suelo: nadie había medido cuánto coincide por puro azar del castellano. Aquí las dos
máquinas se miden sobre EL MISMO corpus, la MISMA ventana y con un suelo de verdad (`suelo`,
`contar_1`, `barajada`), y el umbral de qué cuenta como copiado sale de esa propia medición, no
de un número puesto a mano.

Expone `rachas()` y `medir()`, públicas y sin efectos secundarios, para que el capítulo 1
(`ngrama.py`) las reutilice más adelante sin duplicar esta lógica.

Toda comparación de subcadena exige frontera de palabra: se compara `f" {trozo} "` contra un
corpus guardado como `f" {corpus} "`. Sin esto, el trozo «la casa» encuentra «hola casa» y
alarga rachas que no son copia (medido: sin frontera, la mediana del suelo sube de 3 a 4; las
rachas largas —15 y 25, las tres muestras impresas en el capítulo 6— no cambian).

Uso:
    python lo_habia_visto_ya.py
    python lo_habia_visto_ya.py --selftest
"""

# ======================= CONSTANTES =======================

VENTANAS = 200
VENTANAS_LARGAS = 50
LARGOS_VENTANA = (45, 200)
ORDENES_CONTAR = (1, 2, 3)      # palabras de contexto
LIBROS_AJENOS = 6               # los 6 siguientes al tope de entrenamiento
MIN_APARICIONES_ARRANQUE = 5
PALABRAS_IMPLANTADAS = 12
MARGEN_UMBRAL = 1
SEMILLA = 20260914
DIR_MUESTRAS = "../datos/salidas/muestras"

# El tope del test nulo del selftest (parte 1a) empezó en 7, tomado de una medida previa que
# daba máximo 6. Medido ahora con este instrumento: máximo real 8, y no por un fallo del
# programa. Rastreado a un fragmento concreto, "dia . a las ocho de la mañana" (el punto
# suelto es un artefacto de que ALFABETO no incluye dígitos: "Día 15." pierde el "15" y queda
# "dia ."), que aparece 4 veces en un libro de entrenamiento (Diario de la navegación
# emprendida en 1781) y 2 veces en uno de los libros ajenos (Colección de viajes y
# expediciones ... a las costas de Patagonia): dos diarios de expedición distintos que
# comparten la fórmula de apertura de entrada de diario. Es señal de género, no ruido de
# programa. Tope subido a la medida (8) más el mismo margen que se usa en el resto del
# instrumento (MARGEN_UMBRAL).
TOPE_NULO_AJENO = 8 + MARGEN_UMBRAL

# ==========================================================

import argparse
import glob
import os
import random
import re
import sys
from collections import Counter
from datetime import date

import numpy as np

import ngrama
import memoria_recurrente as mr
# Orden 97: cifras que la nota escribe con letra; si cambia el umbral, falla aquí y no en la página.
EN_LETRA = {7: "siete"}

from formato import coma, comprobar_ancho, miles, pct, tabla_editorial


# --------------------------- el corpus, con frontera de palabra ---------------------------

def _con_bordes(texto):
    """Aplana los espacios y añade uno a cada lado. Comparar contra ESTO, nunca contra el
    texto crudo, es lo que impone la frontera de palabra en toda esta medida."""
    return " " + re.sub(r"\s+", " ", texto).strip() + " "


# --------------------------- arranques, regla fija y sin azar ---------------------------

def arranques(palabras, vocab, n=VENTANAS):
    """`n` trigramas de arranque, a intervalos regulares de `len(palabras) // n`.

    En cada punto toma las 3 palabras siguientes. Las acepta si el trigrama aparece al menos
    MIN_APARICIONES_ARRANQUE veces en `palabras` y las tres palabras están en `vocab`; si no,
    avanza de una en una hasta encontrar uno que valga. Sin azar: la misma llamada da siempre
    los mismos `n` trigramas, para que todos los generadores arranquen en el mismo sitio.
    """
    cuenta_trigramas = Counter(
        tuple(palabras[i:i + 3]) for i in range(len(palabras) - 2)
    )
    paso = len(palabras) // n
    assert paso > 0, f"corpus demasiado corto para {n} ventanas: {len(palabras)} palabras"
    vocab = set(vocab)
    salida = []
    for idx in range(n):
        i = idx * paso
        while True:
            assert i + 3 <= len(palabras), (
                f"no se encontró un arranque válido para la ventana {idx} "
                f"antes del final del corpus"
            )
            tri = tuple(palabras[i:i + 3])
            if cuenta_trigramas[tri] >= MIN_APARICIONES_ARRANQUE and all(p in vocab for p in tri):
                salida.append(tri)
                break
            i += 1
    assert len(salida) == n, f"se esperaban {n} arranques; salieron {len(salida)}"
    return salida


def _primera_aparicion(palabras):
    """trigrama -> índice de su primera aparición, en un único paso por el corpus."""
    primera = {}
    for i in range(len(palabras) - 2):
        tri = tuple(palabras[i:i + 3])
        if tri not in primera:
            primera[tri] = i
    return primera


def _ventana_literal(idx, palabras, largo):
    """Los `largo` palabras del corpus que empiezan en `idx`. Si al corpus no le queda sitio
    (el arranque cae a menos de `largo` palabras del final), es una situación que el enunciado
    no contempla: un trigrama de MIN_APARICIONES_ARRANQUE >= 5 apariciones casi nunca tiene su
    ÚNICA aparición útil pegada al final de 686.002 palabras, pero si ocurre se anota aquí en
    vez de fallar en silencio con una ventana corta que rompería las comparaciones."""
    fin = idx + largo
    assert fin <= len(palabras), (
        f"arranque en la posición {idx} sin sitio para una ventana de {largo} palabras "
        f"(quedan {len(palabras) - idx})"
    )
    return " ".join(palabras[idx:fin])


# --------------------------- rachas() y medir(), públicas y puras ---------------------------

def rachas(muestra, corpus_con_bordes):
    """La racha de copia literal que EMPIEZA en cada palabra de `muestra`, exigiendo frontera
    de palabra en `corpus_con_bordes` (pásale el resultado de `_con_bordes`).

    Devuelve una lista con una entrada por palabra de `muestra`: en la posición i, cuántas
    palabras seguidas desde ahí aparecen tal cual en el corpus. Sin efectos secundarios: no
    imprime ni toca nada global, para que otro capítulo la llame sobre sus propias muestras.
    """
    palabras = muestra.split()
    n = len(palabras)
    valores = [0] * n
    for i in range(n):
        k = 0
        while i + k < n:
            trozo = " " + " ".join(palabras[i:i + k + 1]) + " "
            if trozo not in corpus_con_bordes:
                break
            k += 1
        valores[i] = k
    return valores


def _cobertura(valores, umbral):
    """Qué posiciones quedan dentro de alguna racha de copiado (>= umbral), a partir de lo que
    devuelve `rachas()`."""
    n = len(valores)
    cubiertas = [False] * n
    for i, k in enumerate(valores):
        if k >= umbral:
            for j in range(i, min(i + k, n)):
                cubiertas[j] = True
    return cubiertas


def medir(muestras, corpus_con_bordes, umbral):
    """Para cada muestra de `muestras`: la racha más larga y el % de la muestra cubierto por
    rachas de copiado (>= `umbral`). Devuelve dos listas paralelas (rachas_max, copiados_pct),
    una entrada por muestra. Pura: no decide el umbral, se lo dan hecho.
    """
    rachas_max, copiados = [], []
    for m in muestras:
        valores = rachas(m, corpus_con_bordes)
        rachas_max.append(max(valores) if valores else 0)
        cubiertas = _cobertura(valores, umbral)
        copiados.append(100 * sum(cubiertas) / len(cubiertas) if cubiertas else 0.0)
    return rachas_max, copiados


def _resumen(valores):
    """Mediana, percentil 90 y máximo de una lista de números."""
    a = np.asarray(valores, dtype=float)
    return float(np.median(a)), float(np.percentile(a, 90)), float(np.max(a))


def _barajar(muestra, rng):
    palabras = muestra.split()
    rng.shuffle(palabras)
    return " ".join(palabras)


# --------------------------- las máquinas de contar ---------------------------

def _muestras_contar(orden, arranques_lista, palabras_corpus, largo, rng):
    """`len(arranques_lista)` muestras de `largo` palabras de la máquina de contar de orden
    `orden`, construida sobre `palabras_corpus`, arrancando cada una en las últimas `orden`
    palabras del trigrama correspondiente (los tres son el mismo punto de partida para todas
    las máquinas; cada orden solo usa el trozo de contexto que necesita)."""
    tabla = ngrama.construir(palabras_corpus, orden)
    salida = []
    for tri in arranques_lista:
        arranque = list(tri[3 - orden:]) if orden > 0 else None
        generado = ngrama.generar(tabla, orden, largo, rng, arranque=arranque)
        salida.append(" ".join(generado))
    return salida


# --------------------------- lectura de las muestras de la red ---------------------------

def _leer_fichero_muestras(ruta):
    """Cabecera (clave: valor, una por línea) y cuerpo (largo, arranque, muestra) de un
    fichero escrito por `escribir_muestras.py`."""
    cabecera = {}
    cuerpo = []  # (largo, arranque, muestra)
    with open(ruta, encoding="utf-8") as fh:
        lineas = fh.read().splitlines()
    i = 0
    while i < len(lineas) and ":" in lineas[i] and "\t" not in lineas[i]:
        clave, valor = lineas[i].split(":", 1)
        cabecera[clave.strip()] = valor.strip()
        i += 1
    for linea in lineas[i:]:
        if not linea.strip():
            continue
        largo_txt, arranque, muestra = linea.split("\t", 2)
        cuerpo.append((int(largo_txt), arranque, muestra))
    return cabecera, cuerpo


def _ficheros_de_muestras():
    return sorted(glob.glob(os.path.join(DIR_MUESTRAS, "*.txt")))


# --------------------------- el libro ajeno (suelo) ---------------------------

def _palabras_libros_ajenos():
    """Las palabras de los `LIBROS_AJENOS` libros siguientes al tope de entrenamiento, con la
    MISMA limpieza que `cargar_texto()` (si se limpiara distinto, el suelo mediría otro
    corpus, no el mismo idioma con el que se entrena)."""
    ficheros = sorted(glob.glob(os.path.join(mr.CORPUS, "*.txt")))
    entrenados = 0
    total = 0
    for ruta in ficheros:
        with open(ruta, encoding="utf-8", errors="replace") as fh:
            t = fh.read()
        if "*** START OF" in t:
            t = t.split("*** START OF", 1)[1].split("\n", 1)[-1]
        if "*** END OF" in t:
            t = t.split("*** END OF", 1)[0]
        import unicodedata
        t = unicodedata.normalize("NFC", t.lower())
        t = "".join(c if c in mr.ALFABETO else " " for c in t)
        t = re.sub(r" {2,}", " ", t)
        total += len(t)
        entrenados += 1
        if total >= mr.MAX_CARACTERES:
            break
    ajenos = ficheros[entrenados:entrenados + LIBROS_AJENOS]
    assert len(ajenos) == LIBROS_AJENOS, (
        f"se esperaban {LIBROS_AJENOS} libros ajenos tras los {entrenados} de entrenamiento; "
        f"solo hay {len(ajenos)} disponibles en «{mr.CORPUS}»"
    )
    trozos = []
    for ruta in ajenos:
        with open(ruta, encoding="utf-8", errors="replace") as fh:
            t = fh.read()
        if "*** START OF" in t:
            t = t.split("*** START OF", 1)[1].split("\n", 1)[-1]
        if "*** END OF" in t:
            t = t.split("*** END OF", 1)[0]
        import unicodedata
        t = unicodedata.normalize("NFC", t.lower())
        t = "".join(c if c in mr.ALFABETO else " " for c in t)
        t = re.sub(r" {2,}", " ", t)
        trozos.append(t)
    return "".join(trozos).split()


# --------------------------- pruebas ---------------------------

def selftest():
    fallos = []
    texto = mr.cargar_texto()
    palabras = texto.split()
    corpus = _con_bordes(texto)
    largo = LARGOS_VENTANA[0]

    vocab, _, _, _ = mr.vocabulario_y_datos(texto)
    puntos = arranques(palabras, set(vocab), n=VENTANAS)
    primera = _primera_aparicion(palabras)

    # 1. TEST NULO, EN LOS DOS SENTIDOS.
    ajenas = _palabras_libros_ajenos()
    puntos_ajenos = arranques(ajenas, set(w for w, _ in Counter(ajenas).most_common(20_000)),
                               n=VENTANAS)
    primera_ajena = _primera_aparicion(ajenas)
    ventanas_ajenas = [_ventana_literal(primera_ajena[t], ajenas, largo) for t in puntos_ajenos]
    racha_ajena, _ = medir(ventanas_ajenas, corpus, umbral=largo + 1)  # umbral inalcanzable:
    # aquí solo se quiere la racha máxima real, no el copiado a un umbral concreto.
    maximo_ajeno = max(racha_ajena)
    print(f"[1a] test nulo (libro ajeno)    racha máxima en {VENTANAS} ventanas: "
          f"{maximo_ajeno} (tope {TOPE_NULO_AJENO})")
    if maximo_ajeno > TOPE_NULO_AJENO:
        fallos.append(f"libro ajeno: racha máxima {maximo_ajeno} > {TOPE_NULO_AJENO}; la "
                       f"medida cuenta coincidencia como copia")

    ventanas_propias = [_ventana_literal(primera[t], palabras, largo) for t in puntos]
    racha_propia, _ = medir(ventanas_propias, corpus, umbral=largo + 1)
    minimo_propio = min(racha_propia)
    print(f"[1b] test nulo (propio corpus)  racha mínima en {VENTANAS} ventanas: "
          f"{minimo_propio} (se espera {largo}, el largo de la ventana)")
    if minimo_propio < largo:
        fallos.append(f"propio corpus: racha mínima {minimo_propio} < {largo}; "
                       f"la medida deja copias sin ver")

    # 2. SEÑAL IMPLANTADA.
    base = _barajar(ventanas_propias[0], random.Random(SEMILLA))
    palabras_base = base.split()
    injerto = palabras[primera[puntos[1]]:primera[puntos[1]] + PALABRAS_IMPLANTADAS]
    pos = 15
    injertada = palabras_base[:pos] + injerto + palabras_base[pos + PALABRAS_IMPLANTADAS:]
    muestra_injertada = " ".join(injertada)
    valores = rachas(muestra_injertada, corpus)
    racha_injerto = max(valores)
    # Umbral = el propio tamaño del injerto, no el umbral derivado de las tablas: aquí se
    # comprueba que el instrumento marca el sitio y el ancho correctos, no cuánto vale ese
    # umbral en la medida real (eso lo comprueba la parte 1).
    cubiertas = _cobertura(valores, PALABRAS_IMPLANTADAS)
    marcadas = [i for i, c in enumerate(cubiertas) if c]
    esperadas = list(range(pos, pos + PALABRAS_IMPLANTADAS))
    print(f"[2] señal implantada   racha {racha_injerto} (se esperaban {PALABRAS_IMPLANTADAS}); "
          f"posiciones cubiertas {marcadas[:1] and (marcadas[0], marcadas[-1])} "
          f"(se esperaban {(esperadas[0], esperadas[-1])})")
    if racha_injerto != PALABRAS_IMPLANTADAS:
        fallos.append(f"señal implantada: racha {racha_injerto}, se esperaban "
                       f"{PALABRAS_IMPLANTADAS}")
    if marcadas != esperadas:
        fallos.append(f"señal implantada: cubre {marcadas}, se esperaba {esperadas}")

    # 3. INVARIANTE DEL DOMINIO.
    rng = np.random.default_rng(SEMILLA)
    minimos = {}
    for orden in ORDENES_CONTAR:
        muestras_k = _muestras_contar(orden, puntos[:50], palabras, largo, rng)
        racha_k, _ = medir(muestras_k, corpus, umbral=largo + 1)
        minimos[orden] = min(racha_k)
    print("[3a] invariante        racha mínima de una máquina de orden k, sobre 50 muestras:")
    for orden in ORDENES_CONTAR:
        print(f"     orden {orden}: mínimo {minimos[orden]} (se espera >= {orden + 1})")
        if minimos[orden] < orden + 1:
            fallos.append(f"invariante: orden {orden} da racha mínima {minimos[orden]}, "
                           f"se esperaba >= {orden + 1}")

    rng_baraja = random.Random(SEMILLA)
    barajadas = [_barajar(v, rng_baraja) for v in ventanas_propias[:50]]
    racha_baraj, _ = medir(barajadas, corpus, umbral=largo + 1)
    mediana_baraj = float(np.median(racha_baraj))
    print(f"[3b] invariante        mediana de racha tras barajar 50 ventanas propias: "
          f"{coma(mediana_baraj, 1)} (se espera <= 4)")
    if mediana_baraj > 4:
        fallos.append(f"invariante: barajar deja mediana {coma(mediana_baraj, 1)} > 4; "
                       f"barajar no está destruyendo la racha")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


# --------------------------- medición real ---------------------------

def _fila(nombre, racha_max, copiados):
    med_r, p90_r, max_r = _resumen(racha_max)
    med_c, p90_c, _ = _resumen(copiados)
    return {
        "nombre": nombre,
        "racha_med": med_r, "racha_p90": p90_r, "racha_max": max_r,
        "copiado_med": med_c, "copiado_p90": p90_c,
    }


def _tabla(filas, ancho_nombre):
    cab = [
        f"{'generador':<{ancho_nombre}}{'racha, mediana':>15}{'racha, p90':>12}"
        f"{'racha, máximo':>14}",
        f"{'':<{ancho_nombre}}{'en palabras':>15}{'en palabras':>12}{'en palabras':>14}",
        f"{'-' * ancho_nombre}{'-' * 15}{'-' * 12}{'-' * 14}",
    ]
    for l in comprobar_ancho(cab):
        print(l)
    for f in filas:
        linea = (f"{f['nombre']:<{ancho_nombre}}{coma(f['racha_med'], 1):>15}"
                 f"{coma(f['racha_p90'], 1):>12}{coma(f['racha_max'], 0):>14}")
        print(*comprobar_ancho([linea]), sep="")
    print()
    cab2 = [
        f"{'generador':<{ancho_nombre}}{'copiado, mediana':>17}{'copiado, p90':>13}",
        f"{'':<{ancho_nombre}}{'% de la muestra':>17}{'% de la muestra':>13}",
        f"{'-' * ancho_nombre}{'-' * 17}{'-' * 13}",
    ]
    for l in comprobar_ancho(cab2):
        print(l)
    for f in filas:
        linea = (f"{f['nombre']:<{ancho_nombre}}{pct(f['copiado_med'], 1, de_uno=False):>17}"
                 f"{pct(f['copiado_p90'], 1, de_uno=False):>13}")
        print(*comprobar_ancho([linea]), sep="")


def _clave_generadores():
    """La fila de cada tabla no se explica sola (regla 9): esto es lo que significa cada
    nombre de generador, para quien llega a la tabla sin haber leído el párrafo de antes."""
    return comprobar_ancho([
        "clave de los generadores:",
        "  techo: el texto real que sigue a cada arranque",
        "  contar_k: máquina de contar con k palabras de contexto",
        "  red_sX_tY: la red, semilla X, tirada Y",
        "  suelo: seis libros que no entraron en el entrenamiento",
        "  barajada: cada muestra con sus palabras desordenadas",
    ])


def _linea_umbral(umbral, margen=MARGEN_UMBRAL):
    """La línea del umbral, sin «>=» (se lee peor que la palabra) y dentro de los 68
    caracteres de la caja."""
    return comprobar_ancho([
        f"copiado: racha de {int(umbral)} palabras o más "
        f"(máximo de contar_1 + {margen})"
    ])[0]


def _umbral_45(puntos, palabras, corpus, rng_ngrama):
    """El umbral de «copiado» de la ventana de 45 palabras, recalculado exactamente como en
    la tabla principal: mismo orden de generación (contar_3, contar_2 y por último contar_1)
    con la misma `rng_ngrama`, para que este número sea el MISMO que ya imprime la tabla y no
    uno recalculado con el generador de azar parado en otro punto de su secuencia. contar_3 y
    contar_2 se generan y se tiran: aquí solo hacen falta para consumir su mismo tramo de azar
    antes de generar contar_1."""
    largo = LARGOS_VENTANA[0]
    puntos_l = puntos[:VENTANAS]
    _muestras_contar(3, puntos_l, palabras, largo, rng_ngrama)
    _muestras_contar(2, puntos_l, palabras, largo, rng_ngrama)
    contar_1 = _muestras_contar(1, puntos_l, palabras, largo, rng_ngrama)
    racha_c1, _ = medir(contar_1, corpus, umbral=largo + 1)
    return max(racha_c1) + MARGEN_UMBRAL


def _reportar_muestras(ruta_fichero, puntos, palabras, corpus, rng_ngrama):
    """--muestras: para cada muestra de 45 palabras de `ruta_fichero`, su racha máxima y su
    copiado con el umbral de la tabla de 45 palabras (el mismo número, no uno propio)."""
    largo = LARGOS_VENTANA[0]
    umbral = _umbral_45(puntos, palabras, corpus, rng_ngrama)
    _, cuerpo = _leer_fichero_muestras(ruta_fichero)
    muestras_45 = [(arranque, muestra) for (l, arranque, muestra) in cuerpo if l == largo]
    assert muestras_45, f"«{ruta_fichero}» no declara ninguna muestra de {largo} palabras"

    # L24 (9 de octubre): tabla editorial (regla 6 ter). Las cuentas no cambian.
    filas = []
    for i, (arranque, muestra) in enumerate(muestras_45, start=1):
        valores = rachas(muestra, corpus)
        racha_max = max(valores) if valores else 0
        cubiertas = _cobertura(valores, umbral)
        copiado = 100 * sum(cubiertas) / len(cubiertas) if cubiertas else 0.0
        filas.append([str(i), arranque, str(racha_max), pct(copiado, 1, de_uno=False)])
    print("\n".join(tabla_editorial(
        "Racha y copiado de cada muestra", ["muestra", "arranque", "racha", "copiado"], filas, "didd",
        [# Orden 97: nota para el lector; el porqué del umbral está en _umbral_45.
         f"La racha es la más larga de cada muestra, en palabras; el copiado, la parte de sus "
         f"{largo} palabras que cae dentro de rachas de {EN_LETRA[int(umbral)]} o más."])))


def _medir_largo(largo, n_ventanas, puntos, palabras, primera, corpus, ajenas, puntos_ajenos,
                  primera_ajena, muestras_red, rng_ngrama, rng_baraja):
    puntos_l = puntos[:n_ventanas]

    techo = [_ventana_literal(primera[t], palabras, largo) for t in puntos_l]
    suelo = [_ventana_literal(primera_ajena[t], ajenas, largo)
             for t in puntos_ajenos[:n_ventanas]]
    contar_3 = _muestras_contar(3, puntos_l, palabras, largo, rng_ngrama)
    contar_2 = _muestras_contar(2, puntos_l, palabras, largo, rng_ngrama)
    contar_1 = _muestras_contar(1, puntos_l, palabras, largo, rng_ngrama)

    generadores = {"techo": techo, "contar_3": contar_3, "contar_2": contar_2,
                   "contar_1": contar_1, "suelo": suelo}
    for nombre, muestras_por_tirada in muestras_red.items():
        muestras_l = [m for (l, _, m) in muestras_por_tirada if l == largo][:n_ventanas]
        assert len(muestras_l) == n_ventanas, (
            f"«{nombre}» declara {len(muestras_l)} muestras de {largo} palabras; "
            f"se esperaban {n_ventanas}"
        )
        generadores[nombre] = muestras_l

    # UMBRAL DE «COPIADO»: sale de la propia medición, por cada largo de ventana.
    racha_c1, _ = medir(contar_1, corpus, umbral=largo + 1)
    umbral = max(racha_c1) + MARGEN_UMBRAL

    filas = {}
    for nombre, muestras in generadores.items():
        racha_max, copiados = medir(muestras, corpus, umbral)
        filas[nombre] = _fila(nombre, racha_max, copiados)

    # BARAJADA: control, no una fila con sus propias VENTANAS muestras. Se construye
    # barajando (una sola vez, con random.Random(SEMILLA)) las muestras de contar_1, de
    # suelo, y de la tirada 1 de cada semilla, y se agrupan todas en una fila.
    fuentes_barajada = list(contar_1) + list(suelo)
    for nombre, muestras_por_tirada in muestras_red.items():
        if nombre.endswith("_t1"):
            fuentes_barajada += [m for (l, _, m) in muestras_por_tirada if l == largo][:n_ventanas]
    barajadas = [_barajar(m, rng_baraja) for m in fuentes_barajada]
    racha_b, copiados_b = medir(barajadas, corpus, umbral)
    filas["barajada"] = _fila("barajada", racha_b, copiados_b)

    orden = sorted(filas.values(), key=lambda f: (f["racha_med"], f["copiado_med"]),
                   reverse=True)
    return orden, umbral


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--muestras", metavar="FICHERO",
                     help="para cada muestra de 45 palabras de FICHERO, su racha y su "
                          "copiado con el umbral de la tabla")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())

    if args.muestras:
        texto = mr.cargar_texto()
        palabras = texto.split()
        vocab, _, _, _ = mr.vocabulario_y_datos(texto)
        corpus = _con_bordes(texto)
        puntos = arranques(palabras, set(vocab), n=VENTANAS)
        rng_ngrama = np.random.default_rng(SEMILLA)
        _reportar_muestras(args.muestras, puntos, palabras, corpus, rng_ngrama)
        return

    print(f"Medido el {date.today().isoformat()}.")
    print("No entrena nada: busca texto en texto. No depende de la máquina.\n")

    texto = mr.cargar_texto()
    palabras = texto.split()
    assert 600_000 <= len(palabras) <= 750_000, (
        f"se esperaban entre 600.000 y 750.000 palabras de corpus; hay {miles(len(palabras))}"
    )
    vocab, _, _, _ = mr.vocabulario_y_datos(texto)
    vocab = set(vocab)
    corpus = _con_bordes(texto)
    primera = _primera_aparicion(palabras)
    puntos = arranques(palabras, vocab, n=VENTANAS)

    ajenas = _palabras_libros_ajenos()
    vocab_ajenas = set(w for w, _ in Counter(ajenas).most_common(20_000))
    primera_ajena = _primera_aparicion(ajenas)
    puntos_ajenos = arranques(ajenas, vocab_ajenas, n=VENTANAS)

    ficheros = _ficheros_de_muestras()
    assert ficheros, f"no hay ningún fichero de muestras en «{DIR_MUESTRAS}»"
    assert len(ficheros) != 0, "0 ficheros de muestras no es un aprobado: es el fallo 4.33"

    muestras_red = {}
    arranques_declarados = None
    for ruta in ficheros:
        nombre = os.path.splitext(os.path.basename(ruta))[0]  # red_s<semilla>_t<tirada>
        cabecera, cuerpo = _leer_fichero_muestras(ruta)
        assert cuerpo, f"«{ruta}» no declara ninguna muestra"
        declarados = tuple(c[1] for c in cuerpo if c[0] == LARGOS_VENTANA[0])
        if arranques_declarados is None:
            arranques_declarados = declarados
        else:
            assert declarados == arranques_declarados, (
                f"«{ruta}» declara arranques distintos (o en otro orden) que el primer "
                f"fichero de muestras leído"
            )
        muestras_red[nombre] = cuerpo

    rng_ngrama = np.random.default_rng(SEMILLA)
    rng_baraja = random.Random(SEMILLA)

    resultados = {}
    for largo, n_ventanas in zip(LARGOS_VENTANA, (VENTANAS, VENTANAS_LARGAS)):
        filas, umbral = _medir_largo(largo, n_ventanas, puntos, palabras, primera, corpus,
                                      ajenas, puntos_ajenos, primera_ajena, muestras_red,
                                      rng_ngrama, rng_baraja)
        resultados[largo] = (filas, umbral)
        ancho_nombre = max(len(f["nombre"]) for f in filas) + 1
        print(f"--- VENTANA DE {largo} PALABRAS ({n_ventanas} muestras por generador) ---\n")
        _tabla(filas, ancho_nombre)
        print()
        for l in _clave_generadores():
            print(l)
        print()
        print(_linea_umbral(umbral))
        print()

    print("--- LA MISMA MEDIDA EN SEIS ENTRENAMIENTOS ---")
    print("Tres semillas; cada una entrenada dos veces con TODO idéntico.\n")
    largo_tabla3 = LARGOS_VENTANA[0]
    _, umbral_45 = resultados[largo_tabla3]
    cab3 = [
        f"{'semilla':<10}{'tirada':>7}{'huella de pesos':>17}{'racha, mediana':>15}"
        f"{'copiado, mediana':>17}",
        f"{'-' * 10}{'-' * 7}{'-' * 17}{'-' * 15}{'-' * 17}",
    ]
    for l in comprobar_ancho(cab3):
        print(l)
    for nombre in sorted(muestras_red):
        cabecera, cuerpo = _leer_fichero_muestras(os.path.join(DIR_MUESTRAS, nombre + ".txt"))
        muestras_l = [m for (l, _, m) in cuerpo if l == largo_tabla3][:VENTANAS]
        racha_max, copiados = medir(muestras_l, corpus, umbral_45)
        med_r = float(np.median(racha_max))
        med_c = float(np.median(copiados))
        fila = (f"{cabecera['semilla']:<10}{cabecera['tirada']:>7}"
                f"{cabecera['huella de pesos']:>17}{coma(med_r, 1):>15}"
                f"{pct(med_c, 1, de_uno=False):>17}")
        print(*comprobar_ancho([fila]), sep="")


if __name__ == "__main__":
    main()
