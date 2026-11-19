#!/usr/bin/env python3
"""
Capítulo 12 — ¿detrás de «Explica… a un niño de seis años» viene otra petición? (Orden 178, 19 de noviembre de 2026).

El capítulo 12 dice que, en los textos con que se entrenan estas máquinas, detrás de «Explica… a un niño de seis
años» suele venir otra línea igual, y que por eso GPT-3 continúa la lista en vez de contestar. Esa frase era una
deducción a partir de lo que escribió GPT-3, no un dato. Este programa la mide, y la medición puede llevar la contraria.

Qué hace. Lee en flujo, sin bajarla entera, una colección pública de páginas web de las que se usan para entrenar
estas máquinas: FineWeb (muestra `sample-10BT`, en inglés) y FineWeb-2 (`spa_Latn`, en castellano). Recorre los
documentos EN ORDEN (los de cada fichero, uno tras otro, desde el primero) hasta reunir CASOS_MINIMOS casos o hasta el
tope (TOPE_DOCUMENTOS documentos o TOPE_SEGUNDOS segundos), y cuenta cuántos documentos y cuánto tiempo hizo falta.

Qué busca (las expresiones regulares están abajo, escritas, y se imprimen en la salida). Un CASO es una línea de
como mucho MAX_LINEA caracteres que pide explicar o describir algo a un niño de seis o cinco años:
  - en inglés, «explain» o «describe» y, detrás, «6 year old», «six-year-old», «five year old», «like I'm five»
    y sus variantes;
  - en castellano, «Explica», «Explícale» o «Explícame» y, detrás, «a un niño de seis años», «de cinco años» o
    «pequeño».
El verbo puede ir en cualquier punto de la línea, antes de lo que dice del niño. Pero «explain it to your six year old
daughter» no es una petición, es una frase que habla de explicar. Por eso cada caso se cuenta DOS veces: entre TODOS los
casos, y entre los que son una PETICIÓN, que son los que EMPIEZAN con el verbo (quitadas las viñetas y la numeración, y
un «Please», un «Por favor» o una etiqueta corta como «Q:» o «Ejercicio 3.»). Los ejemplos salen de las peticiones.
Medido el 19 de noviembre de 2026 en una prueba: en el inglés de FineWeb, una línea así es rarísima (del orden de una
por cada cien mil documentos), y casi todas las que casan en sentido amplio no son peticiones.

Qué cuenta, en cada caso, mirando la LÍNEA SIGUIENTE NO VACÍA:
  (a) otra petición: empieza por un verbo de petición en imperativo (la lista de `por_que_ya_contesta.py` en inglés y
      su equivalente en castellano, la de `la_misma_peticion.py`) o acaba en «?»; antes se quitan viñetas y
      numeración («1.», «-», «•»);
  (b) otra cosa: una respuesta, un párrafo, un título…
Y además: cuántos casos forman parte de una LISTA (tres o más líneas seguidas, el propio caso incluido, que son
peticiones, sean o no casos); y el reparto de (a) frente a (b), con su intervalo (de Wilson, al 95 %), en cada idioma.
Una advertencia que la salida repite: los casos de un mismo documento no son independientes (una lista de cuatro
ejercicios da cuatro casos); por eso se cuenta también por DOCUMENTO (la primera línea de cada uno).

Ejemplos. De cada idioma, tres casos reales en los que detrás viene otra petición (en lo posible, dentro de una
lista) y dos en los que viene la respuesta, elegidos con la semilla (no a mano), recortados a unas cinco líneas, con
la dirección de la página. Se saltan los que lleven alguna de las palabras de PALABRAS_A_SALTAR (sexo, juego, farmacia…),
y la salida dice cuántos se saltaron. Es texto de terceros, sacado de la web: se imprimen solo los fragmentos.

Uso (desde codigo/; hace falta `pyarrow` y `huggingface_hub`, y red):
    python la_lista_de_ejercicios.py --selftest
    python la_lista_de_ejercicios.py > ../datos/salidas/la_lista_de_ejercicios.txt
    python la_lista_de_ejercicios.py --idioma inglés          # solo uno
El selftest no usa red: trabaja con documentos inventados.

POR QUÉ ESTA MEDICIÓN NO DEPENDE DE LA MÁQUINA. No hay modelo ni tiempos que cuenten: lee texto y cuenta. El tiempo que
se imprime es el de esta ejecución (la lectura por la red) y no es una medición del libro.
"""

# ======================= CONSTANTES =======================

CORPUS = {   # idioma -> (repositorio de Hugging Face, carpeta de ficheros parquet, cómo se llama)
    "inglés": ("datasets/HuggingFaceFW/fineweb", "sample/10BT", "FineWeb, muestra sample-10BT"),
    "castellano": ("datasets/HuggingFaceFW/fineweb-2", "data/spa_Latn/train", "FineWeb-2, spa_Latn"),
}
SEMILLA = 20261119
CASOS_MINIMOS = 300             # se para al reunir al menos esos casos en un idioma...
TOPE_DOCUMENTOS = 16_000_000    # ...o al recorrer tantos documentos (la muestra sample-10BT tiene unos quince millones)...
TOPE_SEGUNDOS = 10800           # ...o tras tantos segundos (por idioma)
LOTE = 2000                     # filas que se leen de golpe
MAX_LINEA = 240                 # una línea más larga no es una petición, es un párrafo
EJEMPLOS_PETICION = 3           # por idioma: la siguiente línea es otra petición
EJEMPLOS_RESPUESTA = 2          # por idioma: la siguiente línea es otra cosa
LINEAS_DE_EJEMPLO = 5           # el caso y las siguientes no vacías, hasta cinco
ANCHO_DE_EJEMPLO = 200          # caracteres por línea de un ejemplo, como mucho
MUESTRA_QUE_CASA = 6            # cuántas líneas que casan se enseñan
MUESTRA_QUE_NO_CASA = 6         # y cuántas que dicen lo del niño pero no casan
Z95 = 1.96
INFORME_CADA = 250_000          # documentos entre dos avisos de avance (por la salida de errores)

# Las expresiones regulares (sin distinguir mayúsculas de minúsculas), una por idioma. EL_NINO es lo que se dice del niño.
EN_NINO = (r"(?:\b(?:6|six|5|five)[\s-]*(?:years?|yrs?)[\s-]*olds?\b"
           r"|\blike\s+i(?:'|’)?\s*(?:m|am)\s+(?:5|five)\b)")
EN_PIDE = r"\b(?:explain|describe)\b"
ES_NINO = (r"\ba(?:\s+un|\s+una)?\s+niñ[oa]s?\b"
           r"(?:\s+de\s+(?:seis|6|cinco|5)\s+años|\s+pequeñ[oa]s?)")
ES_PIDE = r"\b(?:explica|explícale|explicale|explícame|explicame)\b"
EXPRESIONES = {"inglés": EN_PIDE + r"[^\n]{0,160}?" + EN_NINO,
               "castellano": ES_PIDE + r"[^\n]{0,160}?" + ES_NINO}
SOLO_EL_NINO = {"inglés": EN_NINO, "castellano": ES_NINO}      # para enseñar lo que dice del niño y no casa

# Al principio de la línea (quitadas viñetas y numeración): un «Please», un «Por favor» o una etiqueta corta, y el verbo.
ETIQUETA = r"(?:(?:q|question|task|exercise|ejercicio|pregunta|tarea|actividad)\s*\d*\s*[:.)-]\s*)?"
# Verbos de petición en imperativo (la lista de por_que_ya_contesta.py, VERBOS_EN, y la de la_misma_peticion.py,
# VERBOS_DE_PETICION, copiadas: este programa no carga ningún modelo).
VERBOS = {
    "inglés": ("explain", "write", "give", "translate", "describe", "tell", "list", "summarize", "summarise", "provide",
               "make", "create", "show", "say", "name", "define", "compare", "answer", "complete", "calculate", "find",
               "help", "suggest", "generate", "come", "draft", "outline", "state", "please", "convert", "rewrite"),
    "castellano": ("explica", "escribe", "dame", "traduce", "describe", "cuenta", "cuéntame", "dime", "redacta",
                   "resume", "enumera", "menciona", "haz", "hazme", "da", "pon", "indica", "responde", "completa",
                   "calcula", "busca", "ayúdame", "ayuda", "inventa", "compara", "ponme", "muéstrame", "enséñame",
                   "traduceme", "tradúceme", "escríbeme", "explícame", "sugiere", "propón", "lista", "explícale",
                   "explicale", "explicame", "define", "nombra", "enumera", "elabora", "diseña", "analiza", "justifica"),
}
# Lo que se quita del principio de una línea antes de mirar su primera palabra: viñetas, numeración, comillas, signos.
PRINCIPIO_DE_LINEA = r"^[\s\-\*•·–—\"“”'‘’«»(\[¿¡>#]*(?:\d+[.)]\s*|[a-zA-Z][.)]\s+)?[\s\-\*•·–—\"“”'‘’«»(\[¿¡]*"
# Palabras por las que se salta un ejemplo (no se imprime nada que dé vergüenza imprimir; la salida dice cuántos se saltaron).
PALABRAS_A_SALTAR = ("sex", "porn", "nude", "naked", "xxx", "erotic", "fuck", "shit", "bitch", "cock", "pussy", "dick",
                     "nsfw", "escort", "viagra", "casino", "betting", "pharmacy", "drug", "kill", "suicid", "violen", "gun",
                     "sexo", "porno", "desnud", "erót", "puta", "mierda", "polla", "coño", "follar", "prostit", "farmacia",
                     "apuestas", "matar", "suicid", "droga", "armas")

# ==========================================================

import argparse
import datetime
import platform
import random
import re
import sys
import time

from formato import coma, miles

CASO = {i: re.compile(e, re.I) for i, e in EXPRESIONES.items()}
EMPIEZA = {"inglés": re.compile(PRINCIPIO_DE_LINEA + ETIQUETA + r"(?:please[,\s]+)?" + EN_PIDE, re.I),
           "castellano": re.compile(PRINCIPIO_DE_LINEA + ETIQUETA + r"(?:por\s+favor[,\s]+)?" + ES_PIDE, re.I)}
NINO = {i: re.compile(e, re.I) for i, e in SOLO_EL_NINO.items()}
PRINCIPIO = re.compile(PRINCIPIO_DE_LINEA)
PRIMERA_PALABRA = re.compile(r"[A-Za-zÁ-ÿ]+")

assert CASOS_MINIMOS > 0 and TOPE_DOCUMENTOS > 0 and TOPE_SEGUNDOS > 0, "los topes tienen que ser positivos"
assert EJEMPLOS_PETICION >= 0 and EJEMPLOS_RESPUESTA >= 0 and LINEAS_DE_EJEMPLO >= 2, "mal número de ejemplos"
assert set(CORPUS) == set(EXPRESIONES) == set(VERBOS), "cada idioma lleva su corpus, su expresión y sus verbos"


# ---------------------------------------------------------------------------------------------
# Lo que se cuenta
# ---------------------------------------------------------------------------------------------

def es_peticion(linea, idioma):
    """(a): empieza por un verbo de petición en imperativo (quitadas viñetas y numeración) o acaba en «?»."""
    t = linea.strip()
    if not t:
        return False
    if t.endswith("?"):
        return True
    m = PRIMERA_PALABRA.match(PRINCIPIO.sub("", t))
    return bool(m) and m.group().lower() in VERBOS[idioma]


def siguiente_no_vacia(lineas, i):
    """El índice de la línea siguiente no vacía a la i, o None si no hay."""
    for j in range(i + 1, len(lineas)):
        if lineas[j].strip():
            return j
    return None


def es_lista(lineas, i, idioma):
    """¿La línea i forma parte de una racha de tres o más líneas no vacías seguidas que son peticiones? Las líneas
    en blanco no cortan la racha (una lista de ejercicios suele ir con renglones en blanco entre medias)."""
    def pide(j):
        return es_peticion(lineas[j], idioma) or bool(CASO[idioma].search(lineas[j]))
    n = 1
    j = i - 1
    while j >= 0:
        if lineas[j].strip():
            if not pide(j):
                break
            n += 1
        j -= 1
    j = siguiente_no_vacia(lineas, i)
    while j is not None and pide(j):
        n += 1
        j = siguiente_no_vacia(lineas, j)
    return n >= 3


def casos_del_documento(texto, idioma):
    """Los casos de un documento: [{linea, peticion, siguiente, es_a, en_lista, contexto}] (`peticion`: la línea EMPIEZA con el verbo). `contexto` son el caso y las
    siguientes líneas no vacías (hasta LINEAS_DE_EJEMPLO), ya recortadas."""
    if not CASO[idioma].search(texto):
        return []
    lineas = texto.split("\n")
    casos = []
    for i, l in enumerate(lineas):
        if len(l) <= MAX_LINEA and CASO[idioma].search(l):
            j = siguiente_no_vacia(lineas, i)
            ctx, k = [l.strip()[:ANCHO_DE_EJEMPLO]], i
            while len(ctx) < LINEAS_DE_EJEMPLO:
                k = siguiente_no_vacia(lineas, k)
                if k is None:
                    break
                ctx.append(lineas[k].strip()[:ANCHO_DE_EJEMPLO])
            casos.append({"linea": l.strip(), "peticion": bool(EMPIEZA[idioma].match(l)),
                          "siguiente": lineas[j].strip() if j is not None else None,
                          "es_a": j is not None and es_peticion(lineas[j], idioma),
                          "en_lista": es_lista(lineas, i, idioma), "contexto": ctx})
    return casos


def wilson(k, n):
    """El intervalo de Wilson al 95 % de una proporción k de n: (bajo, alto)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + Z95 ** 2 / n
    c = (p + Z95 ** 2 / (2 * n)) / d
    h = Z95 * ((p * (1 - p) / n + Z95 ** 2 / (4 * n ** 2)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def cuenta(casos):
    """Los números de una lista de casos: casos, a, b, en lista, y lo mismo por documento (su primer caso)."""
    a = sum(c["es_a"] for c in casos)
    return {"casos": len(casos), "a": a, "b": len(casos) - a, "en_lista": sum(c["en_lista"] for c in casos)}


def barajar_en_cada_documento(documentos, semilla):
    """La prueba nula: las líneas de cada documento, barajadas con una semilla fija."""
    rng = random.Random(semilla)
    out = []
    for t in documentos:
        ls = t.split("\n")
        rng.shuffle(ls)
        out.append("\n".join(ls))
    return out


# ---------------------------------------------------------------------------------------------
# Leer el corpus, en flujo
# ---------------------------------------------------------------------------------------------

def documentos_de(idioma):
    """Los documentos (texto, dirección) del corpus del idioma, en orden, leídos en flujo (nada se baja entero)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    repo, carpeta, _ = CORPUS[idioma]
    fs = HfFileSystem()
    ficheros = sorted(f for f in fs.ls(f"{repo}/{carpeta}/", detail=False) if f.endswith(".parquet"))
    assert ficheros, f"no encuentro los ficheros de {repo}/{carpeta}"
    for f in ficheros:
        pf = pq.ParquetFile(fs.open(f))
        for lote in pf.iter_batches(batch_size=LOTE, columns=["text", "url"]):
            yield from zip(lote.column("text").to_pylist(), lote.column("url").to_pylist())


def recorrer(idioma, fuente=None, quiet=False):
    """Recorre documentos hasta CASOS_MINIMOS casos o el tope. Devuelve un diccionario con los casos y lo que hizo falta."""
    t0, n_docs, casos, docs_con_caso, motivo = time.time(), 0, [], 0, "tope de documentos"
    flujo = fuente if fuente is not None else documentos_de(idioma)
    for texto, url in flujo:
        n_docs += 1
        c = casos_del_documento(texto, idioma)
        if c:
            docs_con_caso += 1
            for x in c:
                x["url"] = url
            casos += c
        if n_docs % INFORME_CADA == 0 and not quiet:
            print(f"  [{idioma}] {n_docs:,} documentos, {len(casos)} casos, {time.time() - t0:,.0f} s", file=sys.stderr, flush=True)
        if len(casos) >= CASOS_MINIMOS:
            motivo = "casos reunidos"
            break
        if n_docs >= TOPE_DOCUMENTOS:
            break
        if time.time() - t0 >= TOPE_SEGUNDOS:
            motivo = "tope de tiempo"
            break
    else:
        motivo = "se acabó el corpus"
    return {"casos": casos, "documentos": n_docs, "con_caso": docs_con_caso, "segundos": time.time() - t0, "motivo": motivo}


# ---------------------------------------------------------------------------------------------
# La salida
# ---------------------------------------------------------------------------------------------

def pct(x, n=1):
    return f"{100 * x:.{n}f}".replace(".", ",") + " %"


def apto(c):
    """¿Se puede imprimir? No lleva ninguna de PALABRAS_A_SALTAR ni en las líneas ni en la dirección."""
    t = (" ".join(c["contexto"]) + " " + c.get("url", "")).lower()
    return not any(p in t for p in PALABRAS_A_SALTAR)


def elegir_ejemplos(casos, n_peticion, n_respuesta, semilla):
    """(ejemplos con otra petición detrás, ejemplos con una respuesta detrás, cuántos se saltaron): con la semilla,
    en lo posible de listas, sin tocar los que lleven PALABRAS_A_SALTAR."""
    rng = random.Random(semilla)
    rechazados = set()                   # los casos distintos que se saltan (uno puede mirarse dos veces)
    def sacar(grupo, n):
        grupo = list(grupo)
        rng.shuffle(grupo)
        out = []
        for c in grupo:
            if len(out) == n:
                break
            if apto(c):
                out.append(c)
            else:
                rechazados.add(id(c))
        return out
    peticion = [c for c in casos if c["es_a"]]
    en_lista = [c for c in peticion if c["en_lista"]]
    respuesta = [c for c in casos if not c["es_a"] and c["siguiente"] is not None]
    # primero de las listas; si faltan, de las demás
    elegidos = sacar(en_lista, n_peticion)
    if len(elegidos) < n_peticion:
        resto = [c for c in peticion if c not in elegidos]
        elegidos += sacar(resto, n_peticion - len(elegidos))
    respondidos = sacar(respuesta, n_respuesta)
    return elegidos, respondidos, len(rechazados)


def bloque_de_numeros(casos):
    """Las líneas con los números de un conjunto de casos: (a), (b), listas, y lo mismo por documento."""
    c = cuenta(casos)
    ba, aa = wilson(c["a"], c["casos"])
    primeros = {}
    for x in casos:
        primeros.setdefault(x.get("url", id(x)), x)
    d = cuenta(list(primeros.values()))
    bd, ad = wilson(d["a"], d["casos"])
    n = max(1, c["casos"])
    return [f"  Casos: {c['casos']}.",
            f"    (a) detrás viene otra petición: {c['a']} ({pct(c['a'] / n)}; intervalo del 95 %: {pct(ba)} a {pct(aa)}).",
            f"    (b) detrás viene otra cosa: {c['b']} ({pct(c['b'] / n)}).",
            f"    Forman parte de una lista (tres o más peticiones seguidas): {c['en_lista']} ({pct(c['en_lista'] / n)}).",
            f"    Por documento (el primer caso de cada uno): {d['casos']} documentos; (a) {d['a']} "
            f"({pct(d['a'] / max(1, d['casos']))}; intervalo del 95 %: {pct(bd)} a {pct(ad)}); (b) {d['b']}; en lista {d['en_lista']}."]


def lineas_de_resultado(idioma, r, lineas_que_casan, lineas_que_no, semilla):
    """Las líneas de la salida de un idioma."""
    repo, carpeta, nombre = CORPUS[idioma]
    casos = r["casos"]
    peticiones = [x for x in casos if x["peticion"]]
    peticion, respuesta, saltados = elegir_ejemplos(peticiones, EJEMPLOS_PETICION, EJEMPLOS_RESPUESTA, semilla)
    s = ["", "=" * 74, f"{idioma.upper()}: {nombre}", "=" * 74,
         f"Recorridos {miles(r['documentos'])} documentos en {coma(r['segundos'] / 60)} minutos ({r['motivo']}; se pedían al menos "
         f"{CASOS_MINIMOS} casos, con un tope de {miles(TOPE_DOCUMENTOS)} documentos o {TOPE_SEGUNDOS // 60} minutos).",
         f"La expresión: {EXPRESIONES[idioma]}",
         f"Documentos con algún caso: {r['con_caso']:,}.".replace(",", "."), "",
         "TODOS LOS CASOS (la línea dice lo del niño detrás de «explain»/«explica», aunque sea una frase que habla de explicar):"]
    s += bloque_de_numeros(casos)
    s += ["", "SOLO LAS PETICIONES (la línea EMPIEZA con el verbo, quitadas viñetas, numeración, «please» o una etiqueta corta):"]
    s += bloque_de_numeros(peticiones)
    s += ["", "Una advertencia: los casos de un mismo documento no son independientes (una lista de ejercicios da varios casos); "
              "el intervalo por casos supone que lo son, y el de «por documento» se acerca más a lo que importa.", ""]
    s += ["Una muestra de lo que CASA (las primeras líneas que casaron):"]
    s += [f"  + {x[:ANCHO_DE_EJEMPLO]}" for x in lineas_que_casan[:MUESTRA_QUE_CASA]]
    s += ["", "Una muestra de lo que NO casa (líneas que dicen lo del niño y no casan: sin el verbo antes, o demasiado largas):"]
    s += [f"  - {x[:ANCHO_DE_EJEMPLO]}" for x in lineas_que_no[:MUESTRA_QUE_NO_CASA]]
    s += ["", f"Ejemplos de las PETICIONES, elegidos con la semilla {semilla} (no a mano); se saltaron {saltados} por llevar alguna "
              "de las palabras de PALABRAS_A_SALTAR. Es texto de terceros: solo los fragmentos, con su dirección."]
    for titulo, grupo in (("detrás viene OTRA PETICIÓN", peticion), ("detrás viene la RESPUESTA u otra cosa", respuesta)):
        s += ["", f"--- {titulo} ({len(grupo)}) ---"]
        if not grupo:
            s.append("(no hay ninguno)")
        for k, x in enumerate(grupo, 1):
            s += [f"Ejemplo {k}: {x.get('url', '')}" + ("  [forma parte de una lista]" if x["en_lista"] else "")]
            s += [f"    {l}" for l in x["contexto"]]
    return s


def lo_que_dice_del_nino_y_no_casa(texto, idioma):
    """Las líneas de un documento que dicen lo del niño pero que no casan con la expresión del caso."""
    out = []
    for l in texto.split("\n"):
        if NINO[idioma].search(l) and not (len(l) <= MAX_LINEA and CASO[idioma].search(l)):
            out.append(l.strip())
    return out


def medir_idioma(idioma):
    """Recorre el corpus del idioma; guarda además unas muestras de lo que casa y de lo que no."""
    que_casan, que_no = [], []
    def flujo():
        for texto, url in documentos_de(idioma):
            if len(que_no) < MUESTRA_QUE_NO_CASA and NINO[idioma].search(texto):
                que_no.extend(lo_que_dice_del_nino_y_no_casa(texto, idioma)[:2])
            yield texto, url
    r = recorrer(idioma, flujo())
    que_casan.extend(x["linea"] for x in r["casos"][:MUESTRA_QUE_CASA])
    return r, que_casan, que_no


# ---------------------------------------------------------------------------------------------
# Selftest (sin red)
# ---------------------------------------------------------------------------------------------

def _doc_lista_en(k):
    return "\n".join(["Fichas de clase", ""] + [f"{i + 1}. Explain {t} to a 6 year old." for i, t in
                      enumerate(["gravity", "the moon", "rain", "the sun"])] + ["", f"Texto de relleno número {k}."])


def selftest():
    fallos = []
    en, es = "inglés", "castellano"

    # 1. NULO — las líneas barajadas dentro de cada documento: el porcentaje de (a) no se parece al real. Se construye un
    #    corpus inventado de 60 documentos largos, la mitad con «Explain… 6 year old» seguido de otra petición y la
    #    mitad seguido de una respuesta, y diez líneas de relleno que no son peticiones.
    relleno = [f"La casa tiene {i} ventanas y un jardín." for i in range(10)]
    docs = []
    for k in range(30):
        docs.append("\n".join(["Explain how rain forms to a 6 year old.", "Explain how snow forms to a 6 year old.",
                               "Why is the sea salty?"] + relleno[:7]))
        docs.append("\n".join(["Explain how rain forms to a 6 year old.",
                               "Rain forms when water vapour in the air cools and turns into drops."] + relleno[:8]))
    real = cuenta([x for t in docs for x in casos_del_documento(t, en)])
    barajado = cuenta([x for t in barajar_en_cada_documento(docs, SEMILLA) for x in casos_del_documento(t, en)])
    p_real, p_barajado = real["a"] / real["casos"], barajado["a"] / barajado["casos"]
    ok1 = abs(p_real - p_barajado) > 0.25
    print(f"[1] test nulo         con las líneas barajadas, (a) pasa del {pct(p_real)} real al {pct(p_barajado)}: "
          f"{'no se parece, bien' if ok1 else 'SE PARECE: MAL'}")
    if not ok1:
        fallos.append("test nulo: barajadas las líneas, el porcentaje de (a) se parece al real")

    # 2. SEÑAL — un documento inventado con una lista de cuatro peticiones y otro con petición más respuesta, en inglés y
    #    en castellano: los cuenta bien.
    lista = casos_del_documento(_doc_lista_en(0), en)
    sola = casos_del_documento("Explain gravity to a 6 year old.\nGravity is the force that pulls things down.\n", en)
    lista_es = casos_del_documento("Ejercicios\n\n1. Explica la lluvia a un niño de seis años.\n2. Explícale la luna a un niño de cinco años.\n"
                                   "3. Explica el mar a un niño pequeño.\n4. Cuenta un cuento corto.\n\nGracias por leer.\n", es)
    sola_es = casos_del_documento("Explícale la gravedad a un niño de seis años.\n\nLa gravedad es la fuerza que tira de las cosas.\n", es)
    ruido = casos_del_documento("I will explain it to my six year old son.\nHe is happy.\n", en)   # un caso suelto, no una petición
    ok2 = (len(lista) == 4 and all(x["en_lista"] for x in lista) and [x["es_a"] for x in lista] == [True, True, True, False]
           and len(sola) == 1 and not sola[0]["es_a"] and not sola[0]["en_lista"]
           and len(lista_es) == 3 and all(x["en_lista"] for x in lista_es) and [x["es_a"] for x in lista_es] == [True, True, True]
           and len(sola_es) == 1 and not sola_es[0]["es_a"]
           and len(ruido) == 1 and not ruido[0]["peticion"] and all(x["peticion"] for x in lista + sola + lista_es + sola_es))
    print(f"[2] señal implantada  una lista de cuatro peticiones: {len(lista)} casos, en lista {sum(x['en_lista'] for x in lista)}, "
          f"(a) {sum(x['es_a'] for x in lista)}; petición más respuesta: {len(sola)} caso, (b) {sum(not x['es_a'] for x in sola)}; "
          f"lo mismo en castellano ({len(lista_es)} y {len(sola_es)}); una frase que habla de explicar: {len(ruido)} caso suelto y {sum(x['peticion'] for x in ruido)} petición: "
          f"{'bien' if ok2 else 'MAL'}")
    if not ok2:
        fallos.append("señal implantada: no cuenta bien la lista, la petición con respuesta o el castellano")

    # 3. INVARIANTE — (a) + (b) = casos, en el corpus inventado y con la lista; y las funciones de apoyo (el intervalo, la
    #    elección de ejemplos con la semilla, las palabras que se saltan).
    todos = [x for t in docs + [_doc_lista_en(1)] for x in casos_del_documento(t, en)]
    c = cuenta(todos)
    lo, hi = wilson(30, 60)
    e1 = elegir_ejemplos(todos, 3, 2, SEMILLA)
    e2 = elegir_ejemplos(todos, 3, 2, SEMILLA)
    feo = {"linea": "x", "siguiente": "y", "es_a": True, "en_lista": True, "contexto": ["Explain sex to a 6 year old."], "url": "a"}
    ok3 = (c["a"] + c["b"] == c["casos"] and lo < 0.5 < hi and wilson(0, 0) == (0.0, 0.0)
           and [x["linea"] for x in e1[0]] == [x["linea"] for x in e2[0]] and len(e1[0]) == 3 and len(e1[1]) == 2
           and not apto(feo) and elegir_ejemplos([feo], 1, 0, SEMILLA)[2] == 1)
    print(f"[3] invariante        (a) + (b) = casos: {c['a']} + {c['b']} = {c['casos']}; intervalo de 30 de 60: "
          f"{pct(lo)} a {pct(hi)}; los ejemplos, con la semilla, los mismos dos veces, y un ejemplo con una palabra de "
          f"PALABRAS_A_SALTAR se salta: {'bien' if ok3 else 'MAL'}")
    if not ok3:
        fallos.append("invariante: (a) + (b) no son los casos, o el intervalo, la semilla o las palabras a saltar fallan")
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
    ap.add_argument("--idioma", choices=sorted(CORPUS), help="mide solo ese idioma (por omisión, los dos)")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    import pyarrow
    print(f"Medido el {datetime.date.today()} en {platform.platform()}; pyarrow {pyarrow.__version__}; "
          f"{platform.python_version()}. Sin modelo: se lee texto y se cuenta.")
    print("Pregunta: en los textos con que se entrenan estas máquinas, ¿detrás de «Explica… a un niño de seis años» "
          "viene otra petición?")
    print(f"Semilla {SEMILLA}. Una línea es un CASO si pide explicar o describir algo a un niño de cinco o seis años "
          f"(como mucho {MAX_LINEA} caracteres). (a): la línea siguiente no vacía es otra petición (empieza por un verbo "
          "de petición en imperativo o acaba en «?»); (b): es otra cosa.")
    for idioma in ([args.idioma] if args.idioma else sorted(CORPUS)):
        print(f"Midiendo {idioma}…", file=sys.stderr, flush=True)
        r, que_casan, que_no = medir_idioma(idioma)
        print("\n".join(lineas_de_resultado(idioma, r, que_casan, que_no, SEMILLA)))


if __name__ == "__main__":
    main()
