#!/usr/bin/env python3
"""Las cuentas del capítulo 1, medidas en vez de supuestas.

El capítulo 1 hace seis afirmaciones con número y ninguna tenía un programa detrás
(regla 1 bis). `ngrama.py` produce las muestras que el capítulo enseña, pero no cuenta
nada de esto. Aquí se cuenta, sobre el mismo Quijote y con la misma limpieza que usa
`ngrama.py`, para que los dos hablen del mismo texto:

  (a) «unos dos millones de letras, trescientas ochenta y cinco mil palabras»
  (b) «después de "qu" vino una "e" tantas miles de veces, una "i" tantas, y nada más,
      porque en castellano detrás de "qu" no va otra cosa»
  (c) «En el Quijote hay treinta y seis mil palabras distintas»
  (d) «las hojas posibles son treinta y seis mil por treinta y seis mil: mil
      trescientos millones. ¿Cuántas de ellas aparecen realmente en el Quijote?
      Ciento setenta y ocho mil. El cero coma cero uno por ciento.»
  (e) «de esas ciento setenta y ocho mil combinaciones que sí aparecen, ciento cuarenta
      mil aparecen una sola vez en todo el libro. Casi cuatro de cada cinco.»
  (f) «harían falta unos cuarenta y siete billones de hojas»

Y una séptima, que es la única del capítulo que se podía comprobar sobre la propia
muestra impresa:

  (g) «Han aparecido palabras reales. "Ya", "y", "de", "cue".»

      La muestra no se vuelve a generar aquí: se lee de `datos/salidas/ngrama.txt`, que
      es la que está impresa en el libro. Medir sobre otra muestra sería medir otra cosa.

CÓMO SE CUENTAN LAS PALABRAS (T24, decidido por Carlos el 25 de septiembre de 2026). Con el
mismo corte que `ngrama.trocear`: cada palabra es un trozo y cada signo de puntuación, otro.
Antes se cortaba por los espacios, y «figura», «figura,» y «figura.» eran tres palabras. Las
hojas se cuentan sobre los trozos, porque es lo que cuenta la máquina; las palabras que se le
dicen al lector se cuentan sin los signos.

Y dos cosas más, que pidió el capítulo 1 al reescribirse (T3 y T5):

  (h) las frases del ejemplo del hidalgo y del caballero de la triste figura, de los cuatro
      verbos de decir y de los nombres de Rocinante, contadas sobre los mismos trozos;
  (i) cuántos libros hay en el mundo según Google (2010) frente a los Quijotes que harían
      falta para la libreta de tres palabras. Esa cifra no se mide aquí: es de fuera, y va como
      constante con su fuente.

POR QUÉ ESTA MEDICIÓN NO DEPENDE DE LA MÁQUINA. Se cuentan letras y palabras de un
fichero de texto. El resultado es el mismo en cualquier ordenador y en cualquier día;
lo único que lo cambiaría es cambiar el corpus.

Uso:
    python casillas_vacias.py
    python casillas_vacias.py --selftest
"""

# ======================= CONSTANTES =======================

CORPUS = "../datos/quijote.txt"
MUESTRA = "../datos/salidas/ngrama.txt"
ROTULO_MUESTRA = "--- LETRAS, 1 letra de contexto ---"

PAR_VIGILADO = "qu"          # el par de letras que el capítulo dice que solo admite dos

# (h) Las frases que se cuentan, en el orden en que salen en el capítulo. Todas en minúsculas y
#     sin tildes de mayúscula, porque así queda el texto limpio.
CONTEXTO_CORTO = "la triste"                       # lo que miral selector en dos
FRASES_HIDALGO = ["hidalgo de la triste",          # lo que miral selector en cuatro
                  "hidalgo",
                  "caballero de la triste figura"]
FRASES_DECIR = ["dijo don quijote", "respondió don quijote",
                "replicó don quijote", "preguntó don quijote"]
NOMBRES_ROCINANTE = ["rocinante", "rocín", "caballo", "corcel"]

# (i) Recuento de libros publicados de Google Books. Leonid Taycher, «Books of the world, stand up
#     and be counted! All 129,864,880 of you», Inside Google Books, 5 de agosto de 2010,
#     http://booksearch.blogspot.com/2010/08/books-of-world-stand-up-and-be-counted.html
#     Cuenta EDICIONES, no obras («we count hardcover and paperback books produced from the same
#     text twice»). Comprobado contra la entrada original el 25 de septiembre de 2026.
LIBROS_GOOGLE_2010 = 129_864_880

# ==========================================================

import re
import sys
from collections import Counter
import platform
from datetime import datetime, timezone
from pathlib import Path

from formato import ANCHO_CAJA, coma, comprobar_ancho, miles, pct, tabla_editorial
from ngrama import (ALFABETO, SIGNOS, cargar_corpus, normalizar, palabra_de_ejemplo,
                    solo_palabras, trocear)

AQUI = Path(__file__).resolve().parent


# ---- cada cuenta en una función que se puede probar sola ---------------------------

def tras(secuencia, par):
    """Qué letra viene detrás de cada aparición de `par`, y cuántas veces cada una."""
    ancho = len(par)
    cuenta = Counter()
    for i in range(len(secuencia) - ancho):
        if secuencia[i:i + ancho] == par:
            cuenta[secuencia[i + ancho]] += 1
    return cuenta


def parejas(palabras):
    """Cada pareja de palabras seguidas, y cuántas veces aparece."""
    return Counter(zip(palabras, palabras[1:]))


def casillas(vocabulario, palabras_de_contexto):
    """Hojas que tendría la libreta si se miran esas palabras hacia atrás.

    Una hoja por cada combinación posible de palabras de contexto; dentro de cada
    hoja van los palitos de lo que vino después. Es el vocabulario elevado al
    número de palabras que se miran, que es como las cuenta el capítulo: mirando una
    palabra hay tantas hojas como palabras distintas, y mirando dos hay ese número
    al cuadrado."""
    return vocabulario ** palabras_de_contexto


def veces(trozos, frase):
    """Cuántas veces aparece la frase, trozo a trozo, en la lista de trozos."""
    f = frase.split()
    n = len(f)
    return sum(1 for i in range(len(trozos) - n + 1) if trozos[i:i + n] == f)


def lo_que_sigue(trozos, frase):
    """Qué palabra viene detrás de cada aparición de la frase, y cuántas veces."""
    f = frase.split()
    n = len(f)
    return Counter(trozos[i + n] for i in range(len(trozos) - n) if trozos[i:i + n] == f)


def de_una_sola_vez(cuenta):
    """Cuántas de las cosas contadas aparecieron exactamente una vez."""
    return sum(1 for n in cuenta.values() if n == 1)


def leer_muestra(ruta, rotulo):
    """La muestra impresa en el libro, tal como la dejó `ngrama.py`."""
    lineas = Path(ruta).read_text(encoding="utf-8").split("\n")
    i = lineas.index(rotulo)
    return lineas[i + 1].strip()


def palabras_reales(muestra, vocabulario):
    """Los trozos entre espacios de la muestra que son palabras del Quijote."""
    trozos = [t.strip(SIGNOS) for t in muestra.split() if t.strip(SIGNOS)]
    return trozos, [t for t in trozos if t in vocabulario]


# ---- selftest de tres partes -------------------------------------------------------

def selftest(secuencia=None, palabras=None):
    print("--- selftest ---")

    # [1] TEST NULO — el mismo texto con las letras barajadas y con las palabras
    #     barajadas. Ahí no queda ninguna regla del castellano. Si estas cuentas no
    #     distinguen el Quijote de su propia baraja, no están midiendo nada.
    import random
    baraja_letras = list(secuencia[:400_000])
    random.Random(1).shuffle(baraja_letras)
    tras_barajado = tras("".join(baraja_letras), PAR_VIGILADO)
    tras_real = tras(secuencia[:400_000], PAR_VIGILADO)

    baraja_palabras = list(palabras[:200_000])
    random.Random(1).shuffle(baraja_palabras)
    hapax_barajado = de_una_sola_vez(parejas(baraja_palabras)) / len(parejas(baraja_palabras))
    hapax_real = de_una_sola_vez(parejas(palabras[:200_000])) / len(parejas(palabras[:200_000]))

    print(f"[1] test nulo         letras distintas tras «{PAR_VIGILADO}»: "
          f"barajado={len(tras_barajado)}  real={len(tras_real)}; "
          f"parejas con un solo palito: barajado={pct(hapax_barajado, 0)} "
          f"real={pct(hapax_real, 0)}")
    assert len(tras_barajado) > 2 * len(tras_real), (len(tras_barajado), len(tras_real))
    assert hapax_barajado > hapax_real, (hapax_barajado, hapax_real)
    dijo_barajado = veces(baraja_palabras, FRASES_DECIR[0])
    dijo_real = veces(palabras[:200_000], FRASES_DECIR[0])
    print(f"                      «{FRASES_DECIR[0]}» en 200.000 trozos: "
          f"barajado={dijo_barajado}  real={dijo_real}")
    assert dijo_real > 10 * max(1, dijo_barajado), (dijo_real, dijo_barajado)

    sin_grupo = palabra_de_ejemplo(secuencia, "qz", "e")           # «qz» no sale en el Quijote
    print(f"                      grupo «qz» (no aparece): palabra de ejemplo = {sin_grupo}"
          " (sin palabra, como tiene que ser)")
    assert tras(secuencia, "qz") == Counter() and sin_grupo is None, sin_grupo

    # [2] SEÑAL IMPLANTADA — un texto diminuto cuyas respuestas se cuentan con el dedo.
    #     «a b a b a c»: parejas ab, ba, ab, ba, ac -> 3 distintas, de ellas 1 una sola
    #     vez; vocabulario de 3 palabras -> 9 hojas posibles. Y «quxquxqux»: detrás
    #     de «qu» solo hay una letra, tres veces.
    mini = "a b a b a c".split()
    p = parejas(mini)
    assert len(p) == 3, p
    assert de_una_sola_vez(p) == 1, p
    assert casillas(len(set(mini)), 1) == 3, casillas(len(set(mini)), 1)
    assert casillas(len(set(mini)), 2) == 9, casillas(len(set(mini)), 2)
    assert casillas(len(set(mini)), 3) == 27, casillas(len(set(mini)), 3)
    t = tras("quxquxqux", PAR_VIGILADO)
    assert t == Counter({"x": 3}), t
    assert veces(mini, "a b") == 2 and veces(mini, "a b a") == 2 and veces(mini, "c a") == 0
    assert lo_que_sigue(mini, "b") == Counter({"a": 2}), lo_que_sigue(mini, "b")
    trozos, reales = palabras_reales("hola, xqz hola", {"hola"})
    assert (len(trozos), len(reales)) == (3, 2), (trozos, reales)
    # La palabra de ejemplo, con un texto fabricado: gana la más frecuente; con empate, la que
    # sale antes; un signo detrás del grupo no está dentro de ninguna palabra.
    fabricado = "extra extra texto texto sexto extrano texte exterior ext otro"
    assert palabra_de_ejemplo(fabricado, "xt", "r") == "extra"          # extra 2, extrano 1
    assert palabra_de_ejemplo(fabricado, "xt", "o") == "texto"          # texto 2, sexto 1
    assert palabra_de_ejemplo(fabricado, "xt", "e") == "texte"          # empate: texte antes
    assert palabra_de_ejemplo("exterior texte", "xt", "e") == "exterior"  # empate al revés
    assert palabra_de_ejemplo(fabricado, "xt", " ") is None
    assert palabra_de_ejemplo(fabricado, "xt", "z") is None
    print("[2] señal implantada  texto de seis palabras: 3 parejas, 1 con un solo palito, "
          "9 hojas mirando dos palabras; «quxquxqux»: una letra detrás, 3 veces;\n"
          "                      palabra de ejemplo en un texto fabricado: la más frecuente, y "
          "con empate la primera")

    # [3] INVARIANTE DEL DOMINIO — cosas que tienen que ser verdad pase lo que pase,
    #     comprobadas sobre el Quijote entero.
    p = parejas(palabras)
    vocabulario = len(set(palabras))
    t = tras(secuencia, PAR_VIGILADO)
    assert sum(p.values()) == len(palabras) - 1, (sum(p.values()), len(palabras))
    assert len(p) <= casillas(vocabulario, 2), (len(p), vocabulario)
    assert de_una_sola_vez(p) <= len(p)
    assert set(t) <= set(ALFABETO), sorted(set(t) - set(ALFABETO))
    assert casillas(vocabulario, 3) == casillas(vocabulario, 2) * vocabulario
    for frase in FRASES_DECIR:                          # una frase no sale más que su final
        assert veces(palabras, frase) <= veces(palabras, frase.split(" ", 1)[1]), frase
    assert sum(lo_que_sigue(palabras, CONTEXTO_CORTO).values()) <= veces(palabras, CONTEXTO_CORTO)
    en_el_texto = set(re.findall(r"[^\W\d_]+", secuencia))
    for letra in t:
        ejemplo = palabra_de_ejemplo(secuencia, PAR_VIGILADO, letra)
        assert ejemplo and PAR_VIGILADO + letra in ejemplo and ejemplo in en_el_texto, (
            letra, ejemplo)
    print(f"[3] invariante        parejas contadas = trozos menos uno ({miles(sum(p.values()))}); "
          f"parejas vistas <= hojas; todo lo que va tras «{PAR_VIGILADO}» está en el alfabeto;\n"
          f"                      las {len(t)} palabras de ejemplo contienen «{PAR_VIGILADO}» "
          "y su letra, y salen en el texto")

    print("\nSELFTEST: las tres pruebas pasan.\n")


# ---- lo que se imprime --------------------------------------------------------------

def main():
    crudo = cargar_corpus(str(AQUI / CORPUS))
    secuencia = normalizar(crudo, ALFABETO)
    trozos_libro = trocear(secuencia)          # lo que cuenta la máquina: palabras y signos
    palabras = trozos_libro                     # las hojas se cuentan sobre esto
    solo = solo_palabras(trozos_libro)          # lo que se le dice al lector: palabras
    vocabulario = sorted(set(palabras))

    selftest(secuencia, palabras)
    if "--selftest" in sys.argv:
        return

    t = tras(secuencia, PAR_VIGILADO)
    total_par = sum(t.values())
    p = parejas(palabras)
    posibles = casillas(len(vocabulario), 2)
    una_vez = de_una_sola_vez(p)
    # La línea «hojas con más de un palito» (Orden 4.2) es hojas con algo apuntado menos las de
    # un solo palito; la lee figura_lo_poco_que_hay.py, que antes la calculaba ella.
    muestra = leer_muestra(AQUI / MUESTRA, ROTULO_MUESTRA)
    trozos, reales = palabras_reales(muestra, set(solo))
    tres = casillas(len(vocabulario), 3)
    quijotes_tres = round(tres / len(palabras))
    sigue_corto = lo_que_sigue(palabras, CONTEXTO_CORTO)

    lineas = [
        f"########## capítulo 1: lo que no cabe en la libreta ##########",
        f"fecha: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC",
        f"máquina: {platform.system()} {platform.machine()}. La medición no",
        "depende de la máquina: cuenta letras y palabras de un fichero",
        "corpus: datos/quijote.txt, limpiado y troceado como en ngrama.py:",
        "cada palabra es un trozo, y cada signo de puntuación, otro",
        "",
        "EL TEXTO QUE SE CUENTA",
        "",
        f"  letras (incluidos los espacios)        {miles(len(secuencia)):>14}",
        f"  palabras                               {miles(len(solo)):>14}",
        f"  signos de puntuación                   {miles(len(palabras) - len(solo)):>14}",
        f"  trozos que cuenta la máquina           {miles(len(palabras)):>14}",
        "",
        f"  palabras distintas                     {miles(len(set(solo))):>14}",
        f"  trozos distintos (palabras y {len(SIGNOS)} signos) {miles(len(vocabulario)):>13}",
        "",
    ]
    # L24 (9 de octubre de 2026): la hoja de «qu», en tabla editorial; antes, renglones
    # sangrados que el capítulo 1 copiaba. Ninguna figura lee esta parte.
    filas_qu = []
    for letra, n in t.most_common():
        parte = pct(n / total_par, 2)
        if parte.startswith("0,00"):
            parte = "menos de 0,01 %"      # un recuento pequeño no es un cero
        filas_qu.append([letra, miles(n), parte,
                         palabra_de_ejemplo(secuencia, PAR_VIGILADO, letra)])
    lineas += tabla_editorial(
        f"Qué letra viene detrás de «{PAR_VIGILADO}»",
        ["letra que va detrás", "veces", "de cada cien veces", "por ejemplo"], filas_qu, "cddi",
        [f"Las {miles(total_par)} veces que aparece «{PAR_VIGILADO}» en el Quijote.",
         "La última columna es una palabra del Quijote donde pasó: la que más veces sale.",
         f"Símbolos distintos detrás de «{PAR_VIGILADO}»: {len(t)}, de los {len(ALFABETO)} que "
         "admite el programa (letras, espacio y signos de puntuación)."])
    lineas += [
        "",
        "LA LIBRETA DE PAREJAS DE TROZOS",
        "una hoja por cada pareja de trozos que podría existir",
        "",
        f"  hojas posibles (trozos distintos al cuadrado)     "
        f"{miles(posibles):>16}",
        f"  hojas con algo apuntado (parejas vistas)         "
        f"{miles(len(p)):>16}",
        f"  hojas con algo apuntado, por cada cien posibles"
        f"{pct(len(p) / posibles, 4):>16}",
        "",
        f"  una hoja con algo apuntado de cada             "
        f"{miles(round(posibles / len(p))):>18}",
        "",
        f"  parejas que hay dentro del libro (trozos menos uno)  "
        f"{miles(len(palabras) - 1):>12}",
        f"  de cada cien de esas parejas, cuántas son distintas"
        f"{pct(len(p) / (len(palabras) - 1), 1):>14}",
        "",
        f"  parejas con un solo palito en todo el libro      "
        f"{miles(una_vez):>16}",
        f"  hojas con más de un palito                       "
        f"{miles(len(p) - una_vez):>16}",
        f"  de cada cien parejas vistas, las de un solo palito"
        f"{pct(una_vez / len(p), 1):>15}",
        "",
        "  Quijotes de texto que harían falta para poner una sola",
        "  palabra en cada hoja                            "
        f"{miles(round(posibles / len(palabras))):>16}",
        "",
        "SI SE MIRARAN TRES PALABRAS HACIA ATRÁS",
        "",
        f"  hojas posibles                          "
        f"{miles(tres):>16}",
        "  Quijotes de texto que harían falta para poner una sola",
        "  palabra en cada hoja                    "
        f"{miles(quijotes_tres):>16}",
        "",
        "  libros publicados en el mundo según Google (2010,",
        f"  ediciones, no obras; constante, no medido aquí)  {miles(LIBROS_GOOGLE_2010):>14}",
        f"  libros por cada Quijote que haría falta         "
        f"{coma(LIBROS_GOOGLE_2010 / quijotes_tres, 2):>15}",
        "",
        "LO QUE LA MÁQUINA NO PUEDE JUNTAR",
        "cuántas veces aparece cada frase en el Quijote, trozo a trozo",
        "",
        f"  «{CONTEXTO_CORTO}»{miles(veces(palabras, CONTEXTO_CORTO)):>{58 - len(CONTEXTO_CORTO)}}",
        "  y lo que viene detrás:",
    ]
    for trozo, n in sigue_corto.most_common():
        lineas.append(f"      «{trozo}»{miles(n):>{54 - len(trozo)}}")
    lineas.append("")
    for frase in FRASES_HIDALGO + [""] + FRASES_DECIR + [""] + NOMBRES_ROCINANTE:
        if not frase:
            lineas.append("")
            continue
        lineas.append(f"  «{frase}»{miles(veces(palabras, frase)):>{58 - len(frase)}}")
    lineas += [
        "",
        "PALABRAS REALES EN LA MUESTRA DE UNA LETRA DE CONTEXTO",
        "la muestra impresa en el libro, leída de datos/salidas/ngrama.txt",
        "",
        f"  trozos entre espacios                           {miles(len(trozos)):>16}",
        f"  de ellos, palabras que están en el Quijote      "
        f"{miles(len(reales)):>16}",
        f"  de cada cien trozos, palabras del Quijote       "
        f"{pct(len(reales) / len(trozos), 1):>16}",
        "",
        "  las palabras del Quijote que salieron, en orden:",
    ]
    fila = "   "
    for palabra in reales:
        if len(fila) + len(palabra) + 3 > ANCHO_CAJA:
            lineas.append(fila)
            fila = "   "
        fila += f" «{palabra}»"
    lineas.append(fila)

    # El ancho se comprueba fuera de las tablas editoriales: ésas las compone el libro.
    dentro = False
    for l in lineas:
        if l in ("::: tabla", "::: muestra"):
            dentro = True
        elif l == ":::":
            dentro = False
        elif not dentro:
            comprobar_ancho([l], ANCHO_CAJA)
        print(l)


if __name__ == "__main__":
    main()
