#!/usr/bin/env python3
"""
Capítulo 7 — el mismo texto en nueve idiomas, contado en trozos.

Pedido por Carlos el 10 de octubre de 2026 («me gustaría meter estas explicaciones en el libro»),
con el chino y el japonés fuera por decisión suya: un carácter chino lleva más significado que una
letra, y la comparación de trozos entre escrituras tan distintas no se lee limpia.

Usa solo el troceador del modelo del capítulo (el mismo de maquina_entera_detalle.py); no carga
el modelo. Mide tres cosas:

  1. CUÁNTOS TROZOS CUESTA LO MISMO. La Declaración Universal de Derechos Humanos entera, en
     nueve idiomas (datos/declaracion_universal/, fuente en datos/README.md), y cuántos trozos
     necesita cada uno, por cada 100 del castellano. Los espacios se normalizan (cada tramo de
     espacios y saltos de línea pasa a ser un espacio), porque las copias traen sangrías
     distintas y eso no es el idioma.
  2. EL COMIENZO DEL ARTÍCULO 1, en castellano y en inglés, trozo a trozo.
  3. QUÉ HAY EN LA CAJA. Las 151.665 entradas, según las letras de cada una: la escritura de la
     primera letra que lleva. Las que no llevan ninguna (cifras, signos, espacios) van aparte, y
     también las que son un pedazo de letra (bytes sueltos que solos no forman ninguna) y las
     marcas especiales del formato de conversación.

Uso:
    python idiomas_en_trozos.py > ../datos/salidas/idiomas_en_trozos.txt
    python idiomas_en_trozos.py --selftest
"""

# ======================= CONSTANTES =======================

MODELO = "Qwen/Qwen2.5-0.5B"      # el del capítulo 7; solo se usa su troceador
CARPETA = "../datos/declaracion_universal"
SEMILLA = 20261010                # para barajar las letras del test nulo

# (nombre en el libro, fichero, letras con que se escribe, escritura que se espera en el texto)
IDIOMAS = [
    ("castellano", "spa", "latinas", "LATIN"),
    ("inglés", "eng", "latinas", "LATIN"),
    ("francés", "fra", "latinas", "LATIN"),
    ("catalán", "cat", "latinas", "LATIN"),
    ("alemán", "deu", "latinas", "LATIN"),
    ("euskera", "eus", "latinas", "LATIN"),
    ("ruso", "rus", "cirílicas", "CYRILLIC"),
    ("griego", "ell_monotonic", "griegas", "GREEK"),
    ("hindi", "hin", "del hindi", "DEVANAGARI"),
]
REFERENCIA = "castellano"
PALABRAS_MINIMAS = 1300           # una Declaración entera tiene más; menos es un fichero roto
LETRAS_PROPIAS_MINIMAS = 0.80     # fracción de letras de la escritura esperada

# El comienzo del artículo 1, de la primera palabra a la que cierra lo mismo en los dos idiomas.
# No hasta el primer punto: en castellano la frase sigue («…y derechos y, dotados como están…»)
# y en inglés se corta («…and rights. They are endowed…»), y se compararían tramos distintos.
ARTICULO_1 = [("castellano", "Todos los seres humanos", "derechos"),
              ("inglés", "All human beings", "rights")]

# Escrituras de la caja, con su nombre en el libro. Lo que no está aquí va a «otras».
ESCRITURAS = [("LATIN", "latinas"), ("CJK", "chinas"), ("CYRILLIC", "cirílicas"),
              ("ARABIC", "árabes"), ("HANGUL", "coreanas"), ("HEBREW", "hebreas"),
              ("THAI", "tailandesas"), ("HIRAGANA", "japonesas"), ("KATAKANA", "japonesas"),
              ("GREEK", "griegas"), ("DEVANAGARI", "del hindi")]
ENTRADAS_EN_EL_CAPITULO = 151665  # lo que imprime maquina_entera_detalle.txt
LETRAS_EN_BYTES = ["a", "á", "ñ", "ж", "λ", "न"]   # castellano, ruso, griego, hindi

# selftest
REPETICIONES = 200                # « de» repetido: tienen que salir exactamente tantos trozos
CARACTERES_DESCONOCIDOS = 200     # del silabario lineal B, que la caja no tiene como piezas
PRIMERO_LINEAL_B, ULTIMO_LINEAL_B = 0x10000, 0x1005D
MINIMO_POR_DESCONOCIDO = 2        # trozos por carácter, como poco, si va byte a byte
RAZON_BARAJADO_MINIMA = 1.5       # barajar las letras tiene que encarecer el texto al menos así

# ==========================================================

from formato import coma, miles, tabla_editorial, unidos

import argparse
import datetime
import os
import platform
import random
import sys
import unicodedata

import transformers
from transformers import AutoTokenizer

AQUI = os.path.dirname(os.path.abspath(__file__))


def cargar():
    return AutoTokenizer.from_pretrained(MODELO)


def ids_de(tok, texto):
    return tok(texto, add_special_tokens=False)["input_ids"]


def escritura(ch):
    """La escritura de una letra, por el nombre que le da Unicode: LATIN, CYRILLIC…"""
    nombre = unicodedata.name(ch, "")
    for clave in [e for e, _ in ESCRITURAS] + ["DEVANAGARI"]:
        if nombre.startswith(clave):
            return clave
    return "OTRA"


def leer(fichero):
    ruta = os.path.join(AQUI, CARPETA, fichero + ".txt")
    assert os.path.exists(ruta), f"Se esperaba el texto en {ruta}; no existe"
    with open(ruta, encoding="utf-8") as f:
        crudo = f.read()
    # La caja normaliza a NFC antes de trocear (su normalizador es NFC()). El texto en hindi
    # trae letras con el punto ya pegado (ज़, U+095B) que NFC separa en letra y punto: no cambia
    # ni un trozo, pero sin normalizar aquí el texto rehecho no sería idéntico al leído. Visto
    # el 10 oct por la prueba 3 del selftest.
    return unicodedata.normalize("NFC", " ".join(crudo.split()))


def validar_entrada(tok):
    """Todo lo que el resto da por hecho, comprobado antes de contar nada."""
    assert len(tok) == ENTRADAS_EN_EL_CAPITULO, (
        f"Se esperaban {miles(ENTRADAS_EN_EL_CAPITULO)} entradas en la caja (las que cita el "
        f"capítulo 7); hay {miles(len(tok))}")
    textos = {}
    for nombre, fichero, _, esperada in IDIOMAS:
        t = leer(fichero)
        palabras = len(t.split())
        assert palabras >= PALABRAS_MINIMAS, (
            f"Se esperaba la Declaración entera en {nombre} ({PALABRAS_MINIMAS} palabras o más); "
            f"el fichero {fichero}.txt tiene {palabras}")
        letras = [c for c in t if c.isalpha()]
        propias = sum(escritura(c) == esperada for c in letras) / len(letras)
        assert propias >= LETRAS_PROPIAS_MINIMAS, (
            f"Se esperaba el {nombre} escrito con letras {esperada} (al menos "
            f"{coma(100 * LETRAS_PROPIAS_MINIMAS, 0)} %); son {coma(100 * propias, 1)} %")
        textos[nombre] = t
    assert REFERENCIA in textos, f"Se esperaba {REFERENCIA!r} entre los idiomas"
    return textos


def articulo_1(textos):
    """El comienzo del artículo 1, de `comienzo` a `final` (incluido)."""
    frases = []
    for nombre, comienzo, final in ARTICULO_1:
        t = textos[nombre]
        assert t.count(comienzo) == 1, (
            f"Se esperaba «{comienzo}» una sola vez en el texto en {nombre}; "
            f"sale {t.count(comienzo)}")
        i = t.find(comienzo)
        j = t.find(final, i)
        assert j > i, f"Se esperaba «{final}» detrás de «{comienzo}» en {nombre}"
        frases.append((nombre, t[i:j + len(final)]))
    return frases


def clase_de_entrada(tok, i, especiales):
    """A qué grupo va una entrada de la caja."""
    if i in especiales:
        return "marcas especiales"
    texto = tok.decode([i])
    for c in texto:
        if c.isalpha():
            e = escritura(c)
            return dict(ESCRITURAS).get(e, "otras")
    if "�" in texto:
        return "pedazos de letra"
    return "sin letras"


def censo(tok):
    especiales = set(tok.added_tokens_decoder)
    cuenta = {}
    for i in range(len(tok)):
        k = clase_de_entrada(tok, i, especiales)
        cuenta[k] = cuenta.get(k, 0) + 1
    return sorted(cuenta.items(), key=lambda kv: -kv[1])


# ------------------------------------------------------------------------ las tres mediciones

def bloque_declaracion(tok, textos):
    cuentas = {n: len(ids_de(tok, textos[n])) for n, *_ in IDIOMAS}
    base = cuentas[REFERENCIA]
    letras = {n: l for n, _, l, _ in IDIOMAS}
    orden = sorted(cuentas, key=lambda n: cuentas[n])
    filas = [[n, letras[n], miles(cuentas[n]), str(round(100 * cuentas[n] / base))]
             for n in orden]
    lineas = tabla_editorial(
        "La Declaración Universal de Derechos Humanos, entera, en trozos",
        ["idioma", "letras", "trozos", f"por cada 100 del {REFERENCIA}"], filas, "iidd",
        ["El mismo texto en cada idioma, contado con la caja de la máquina del capítulo."])
    return lineas, cuentas


def bloque_articulo_1(tok, textos):
    filas, cuentas = [], {}
    for nombre, frase in articulo_1(textos):
        piezas = [tok.decode([i]) for i in ids_de(tok, frase)]
        assert "".join(piezas) == frase, f"Los trozos de la frase en {nombre} no la rehacen"
        cuentas[nombre] = (frase, piezas)
        filas.append([nombre, str(len(piezas)),
                      unidos([p.replace(" ", "_") for p in piezas])])
    lineas = tabla_editorial(
        "El comienzo del artículo 1, trozo a trozo", ["idioma", "trozos", "cómo se parte"],
        filas, "idi",
        ["“|” separa dos trozos; el guion bajo es el espacio de delante, que la palabra lleva "
         "dentro de una frase."])
    return lineas, cuentas


def bloque_censo(tok):
    filas = censo(tok)
    total = sum(v for _, v in filas)
    assert total == len(tok), f"El censo suma {total}; la caja tiene {len(tok)}"
    lineas = tabla_editorial(
        "Las piezas de la caja, según sus letras", ["letras", "piezas", "de cada cien"],
        [[k, miles(v), coma(100 * v / total, 1)] for k, v in filas] +
        [["**en total**", f"**{miles(total)}**", "**100**"]], "idd",
        ["Cada pieza cuenta por la primera letra que lleva. «Pedazos de letra»: bytes que solos "
         "no forman ninguna."])
    return lineas, dict(filas)


# ------------------------------------------------------------------------------ selftest

def selftest(tok, textos):
    fallos = []
    rng = random.Random(SEMILLA)

    # 1. TEST NULO — las MISMAS letras del castellano, barajadas: misma longitud, mismas letras,
    #    ningún idioma. Si la cuenta de trozos midiera solo la longitud, saldría parecida. Si mide
    #    lo familiar que le resulta el texto a la caja, tiene que subir mucho.
    t = textos[REFERENCIA]
    letras = list(t)
    rng.shuffle(letras)
    barajado = "".join(letras)
    n_orig, n_bar = len(ids_de(tok, t)), len(ids_de(tok, barajado))
    razon = n_bar / n_orig
    print(f"[1] test nulo         {REFERENCIA}: {miles(n_orig)} trozos; con sus letras "
          f"barajadas: {miles(n_bar)} ({coma(razon, 2)} veces)")
    if razon < RAZON_BARAJADO_MINIMA:
        fallos.append(f"test nulo: barajar las letras debía encarecer el texto al menos "
                      f"{coma(RAZON_BARAJADO_MINIMA, 1)} veces; salió {coma(razon, 2)}")

    # 2. SEÑAL IMPLANTADA — dos respuestas conocidas de antemano. « de» repetido REPETICIONES
    #    veces: cada uno es una pieza de la caja, así que salen exactamente REPETICIONES trozos.
    #    Y CARACTERES_DESCONOCIDOS caracteres del silabario lineal B, que la caja no tiene: cada
    #    uno son cuatro bytes y tiene que costar al menos MINIMO_POR_DESCONOCIDO trozos.
    n_de = len(ids_de(tok, " de" * REPETICIONES))
    raros = "".join(chr(rng.randint(PRIMERO_LINEAL_B, ULTIMO_LINEAL_B))
                    for _ in range(CARACTERES_DESCONOCIDOS))
    n_raros = len(ids_de(tok, raros))
    print(f"[2] señal implantada  « de» × {REPETICIONES}: {n_de} trozos; "
          f"{CARACTERES_DESCONOCIDOS} caracteres que la caja no tiene: {miles(n_raros)} trozos "
          f"({coma(n_raros / CARACTERES_DESCONOCIDOS, 2)} por carácter)")
    if n_de != REPETICIONES:
        fallos.append(f"señal implantada: se esperaban {REPETICIONES} trozos; salieron {n_de}")
    if n_raros < MINIMO_POR_DESCONOCIDO * CARACTERES_DESCONOCIDOS:
        fallos.append(f"señal implantada: se esperaban al menos {MINIMO_POR_DESCONOCIDO} trozos "
                      f"por carácter desconocido; salieron {coma(n_raros / CARACTERES_DESCONOCIDOS, 2)}")

    # 3. INVARIANTE DEL DOMINIO — trocear no pierde nada: en los nueve idiomas, juntar los trozos
    #    devuelve el texto exacto. Y el censo reparte todas las entradas de la caja, ni una más.
    perdidos = [n for n, *_ in IDIOMAS
                if tok.decode(ids_de(tok, textos[n])) != textos[n]]
    _, cuenta = bloque_censo(tok)
    suma = sum(cuenta.values())
    print(f"[3] invariante        textos que no se rehacen al juntar sus trozos: {len(perdidos)} "
          f"de {len(IDIOMAS)}; el censo reparte {miles(suma)} de {miles(len(tok))} entradas")
    if perdidos:
        fallos.append(f"invariante: no se rehacen al juntar los trozos: {', '.join(perdidos)}")
    if suma != len(tok):
        fallos.append(f"invariante: el censo suma {suma}; la caja tiene {len(tok)}")

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

    tok = cargar()
    textos = validar_entrada(tok)
    if args.selftest:
        sys.exit(selftest(tok, textos))

    print(f"máquina: {platform.machine()}, {platform.system()} {platform.release()}")
    print(f"troceador: {MODELO}   transformers {transformers.__version__}")
    print(f"fecha: {datetime.date.today().isoformat()}")
    print()
    print("--- 1. CUÁNTOS TROZOS CUESTA LO MISMO ---\n")
    lineas, cuentas = bloque_declaracion(tok, textos)
    print("\n".join(lineas))
    for n, *_ in IDIOMAS:
        print(f"{n}: {miles(len(textos[n].split()))} palabras, {miles(len(textos[n]))} "
              f"caracteres, {miles(cuentas[n])} trozos")
    print("bytes que ocupa una letra: " + ", ".join(
        f"«{c}» {len(c.encode('utf-8'))}" for c in LETRAS_EN_BYTES))
    print("\n--- 2. EL COMIENZO DEL ARTÍCULO 1 ---\n")
    lineas, _ = bloque_articulo_1(tok, textos)
    print("\n".join(lineas))
    print("\n--- 3. QUÉ HAY EN LA CAJA ---\n")
    lineas, _ = bloque_censo(tok)
    print("\n".join(lineas))


if __name__ == "__main__":
    main()
