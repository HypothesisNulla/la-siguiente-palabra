#!/usr/bin/env python3
"""Verifica que cada número de la PROSA de un capítulo salga de una salida guardada (regla 11.4, 11.11 y 11.19;
Orden 137.2 con el cambio de la 138, 8 nov 2026).

`verificar_cifras.py` mira el FORMATO de las salidas guardadas, y `verificar_muestras.py` mira que lo que el libro
copia de una salida (bloques y filas de tabla) sea literal. Nadie miraba los números que el autor escribe en sus
frases: «de cada 3.609», «cuarenta y una de cuarenta y tres». Este programa los saca de la prosa y busca cada uno
en las salidas guardadas de los programas de ese capítulo.

Uso:
    python verificar_numeros_prosa.py RAIZ SALIDAS                 los capítulos reformados (estricto)
    python verificar_numeros_prosa.py RAIZ SALIDAS --capitulos 1   esos capítulos (estricto; sale 1 si algún número no tiene salida)
    python verificar_numeros_prosa.py RAIZ SALIDAS --recuento      cuántos sin salida en cada capítulo (no suspende)
    python verificar_numeros_prosa.py --selftest

RAIZ es `libro-ia-libro/`; SALIDAS, `libro-ia-publico/datos/salidas/`. La prosa se saca con
`RAIZ/herramientas/texto_libro.py` (párrafos, listas y destacados; no tablas, muestras, citas en bloque ni pies de
figura; no bibliografías). Se leen tres ficheros de `RAIZ/notas/reforma-nov/`:

  capitulos-reformados.md   una línea «- capNN» por capítulo ya reescrito con la regla 11. Sin él, sin capítulos: el
                            verificador estricto no mira los capítulos viejos (se escribieron con la regla vieja y no
                            cumplen «cada número de la prosa sale de una salida»); entran uno a uno al reformarlos.
  salidas-por-capitulo.md   «- capNN: salida.txt, otra.txt» las salidas guardadas de los programas de ese capítulo.
                            Si un capítulo no está, se DERIVAN: las salidas que contienen alguna línea de los bloques,
                            tablas y citas del propio capítulo (orientativo, solo para el recuento).
  cifras-exentas.md         las exenciones, UNA POR NÚMERO (ver abajo).

QUÉ ES UN NÚMERO
  - Cifras: «3.609», «0,30», «12», «90,4 %». El punto es de millar (grupos de tres) y la coma, decimal.
    Se saltan los códigos pegados a letras (GPT-4, A1) y los ordinales (3.º no cuenta como 3.º, sí como 3).
  - Con letra, de diez para arriba, también compuestos y con escala: «diez», «cuarenta y una», «veintitrés mil»,
    «mil trescientos millones». «Un millón» cuenta; «un» y «una» sueltos no (son artículos).
  - Del «dos» al «nueve» con letra NO se comprueban: no se distinguen de una cuenta ordinaria («las dos letras de
    antes»). Se cuentan y se dicen, no se miran. «Cien» tras «cada» o «por» («de cada cien») es la unidad con que el
    libro explica una probabilidad, no una cifra: también se cuenta aparte.
  - Una cifra seguida de «mil», «millón», «millones», «billón» o «billones» es una sola cantidad («130 millones»).

QUÉ SIGNIFICA «TIENE SALIDA»
  Se saca de las salidas del capítulo cada número escrito con el formato del libro (los .txt; los .csv no). Un número
  de la prosa tiene salida si:
  1. ES IGUAL, como valor: «12» y «12,0» son el mismo; «0,30» y «30 %» no (ver el punto 4).
  2. CON DECIMALES: «7,7» tiene salida si la salida dice 7,66 o 7,7 (se redondea a los decimales de la prosa).
  3. UN PORCENTAJE entero («75 %») tiene salida si la salida dice 75,00 o 74,6.
  4. UNA PROPORCIÓN de salida («0,94») tiene también su porcentaje («94») y al revés.
  5. UNA APROXIMACIÓN marcada («unas treinta», «unos cinco», «cerca de», «alrededor de», «casi», «más de»,
     «menos de», «algo más de», «poco más de») tiene salida si hay un valor cercano, con el tamaño del redondeo del
     propio número: «treinta» redondea a decenas, «quinientos millones» a cientos de millones, «cinco» a unidades.
     «Unas/unos/cerca de/alrededor de/aproximadamente»: a menos de medio redondeo. «Más de/algo más de/poco más
     de»: el valor, por encima, a menos de un redondeo. «Menos de/casi/apenas»: el valor, por debajo.
  No son cifras (Orden 143): las décadas («los años sesenta») ni las palabras de tamaño sueltas, sin número delante
(«millones», «cientos de miles de millones»). Se salta, y no se cuenta.

Un número sin marcar y entero se compara exacto: «treinta» NO tiene salida si la salida dice 29,8.

EXENCIONES (Orden 137.2: «nada de exenciones genéricas»). Solo de la lista `cifras-exentas.md`, una línea por número:

    - capNN | número tal como está escrito | un trozo de su frase | motivo

  El trozo de frase tiene que tener al menos 12 caracteres y estar en la frase donde sale el número: así la exención
  vale para ESE número y no para el mismo número en otro sitio. El motivo es uno de estos tres, con su prueba:
  - «año» o «fecha», con «fuente: notas/fuentes/…» y ese fichero tiene que existir (los años y fechas se comprueban
    contra las fuentes);
  - «referencia» (número de capítulo, figura, tabla, página, apartado);
  - «inventado»: el ejemplo lo declara inventado; el párrafo tiene que decirlo con alguna de las palabras
    «inventad», «imagina», «supón», «ficticio».
  - «fuente»: una cifra que no es un año ni una fecha y viene de una fuente de fuera (la de un artículo): con
    «fuente: notas/fuentes/…», y el fichero tiene que existir.
  - «redondeo: de 79,40 %»: el número de la prosa es el redondeo de un valor exacto que SÍ está en una salida
    guardada. El valor exacto tiene que estar en las salidas del capítulo y, redondeado a los decimales del
    número de la prosa, tiene que dar ese número (Orden 140.1).
  - «redondeo: de 15.048.831» con un número escrito con letra («más de quince millones»): vale si el exacto, redondeado a la
    escala de la última palabra (millones: 10^6), da ese número (Orden 148), y además el número lleva su marcador delante
    («unos», «más de», «casi»…) y el marcador dice la verdad: «más de» pide un exacto mayor, «casi» uno menor (Orden 149).
  - «cálculo: (8.638 / 20.000) * 20.000; fuente: notas/fuentes/x.pdf»: el número es el resultado de una cuenta (Orden 141).
    La cuenta se escribe con + - * / ^ y paréntesis, y tiene que dar el número (con las mismas reglas de redondeo y de
    aproximación de arriba: «casi setenta» vale 2017 - 1948 = 69). Cada cifra de la cuenta tiene que estar en las
    salidas del capítulo o en el texto de alguna de las fuentes que se citan.
  - Con `año`, `fecha` o `fuente` el número tiene además que SALIR EN EL TEXTO de la fuente citada (los .pdf se leen con
    `pdftotext`; un .pdf sin texto, un escaneo sin capa de texto —menos de cien caracteres por página—, no se puede
    comprobar y pasa): «1.104» se busca como
    «1,104» y como «1104», que es como lo escriben los artículos en inglés.
  - «cálculo: …; supuestos: 100.000.000.000 = «cien mil millones»; 2 = «dos veces»»: una cuenta cuyas cifras son los SUPUESTOS
    del propio texto (Orden 143). Cada supuesto es «valor = «frase»», y la frase tiene que estar copiada del anexo del
    capítulo (`anexo-aNN-…`); las cifras de la cuenta tienen que ser esos valores (o salir de las salidas o de la fuente).
  - «ejemplo: «¿Y si lo probamos al revés?»»: el número es de un ejemplo inventado que el texto plantea como hipótesis. La
    exención copia la frase que plantea la hipótesis, y esa frase tiene que estar en el MISMO APARTADO (Orden 143).
  - «fuente-no-libre: fichas/x.md»: la fuente no es libre y no está en el repositorio, solo su ficha en `notas/fuentes/fichas/`.
    La ficha lleva tres líneas, «Cita: …» (autor, título, editorial, año), «Página: N» y «Frase: «…»» (de donde sale el dato),
    y el número tiene que salir en esa frase. Se cuentan APARTE en la salida: son datos que un programa no puede comprobar.
  - Con `año`, `fecha`, `fuente` y `cálculo` se puede añadir «frase: «…»»: si el número no sale tal cual en la fuente (una
    conversión, o escrito de otra manera), lo que se exige es que esa frase salga en su texto. Las fuentes `.html` se leen
    sin las etiquetas.
  - «suma a la vista: 5 + 1 + 7»: el número es una cuenta hecha delante del lector con números que están en el
    mismo párrafo. Los sumandos, escritos en la exención, tienen que salir en el párrafo (con cifra o con
    letra, del uno al nueve) y sumar el número exento (Orden 140.2).
  Una exención que no casa con ningún número (porque el texto cambió) es un error: la lista no puede quedarse vieja.

Los capítulos que no están reformados no se corrigen: `--recuento` dice cuántos números sin salida tienen, separados en
años, referencias y cifras, y cuántos sin salida en NINGUNA salida del repositorio (cota más baja).
"""

# ======================= CONSTANTES =======================

CARPETA_NOTAS = "notas/reforma-nov"
FICHERO_REFORMADOS = "capitulos-reformados.md"
FICHERO_SALIDAS = "salidas-por-capitulo.md"
FICHERO_EXENTAS = "cifras-exentas.md"

MIN_FRAGMENTO = 12                      # caracteres del trozo de frase de una exención
MOTIVOS = ("año", "fecha", "referencia", "inventado", "fuente", "redondeo", "suma a la vista", "cálculo", "ejemplo",
           "fuente-no-libre")
DECLARA_INVENTADO = ("inventad", "imagina", "supón", "ficticio")
REFERENCIAS = {"capítulo", "capítulos", "cap.", "caps.", "figura", "figuras", "tabla", "tablas", "página", "páginas",
               "pág.", "págs.", "apartado", "regla", "anexo", "nota", "paso", "punto", "línea"}
MESES = {"enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
         "noviembre", "diciembre"}
MARCAS_APROX = ("unas", "unos", "cerca de", "alrededor de", "aproximadamente")
MARCAS_MAS = ("algo más de", "poco más de", "más de")
MARCAS_MENOS = ("menos de", "casi", "apenas")
LARGO_FRASE_MIN = 3                     # el selftest: una frase de menos palabras no se mira
# ==========================================================

import argparse
import os
import re
import sys
import tempfile
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

AQUI = Path(__file__).resolve().parent

UNIDADES = {"dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9}
ARTICULOS = {"un": 1, "una": 1, "uno": 1}
DIEZ = {"diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15, "dieciséis": 16,
        "diecisiete": 17, "dieciocho": 18, "diecinueve": 19, "veinte": 20, "veintiuno": 21, "veintiún": 21,
        "veintiuna": 21, "veintidós": 22, "veintitrés": 23, "veinticuatro": 24, "veinticinco": 25,
        "veintiséis": 26, "veintisiete": 27, "veintiocho": 28, "veintinueve": 29}
DECENAS = {"treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70, "ochenta": 80, "noventa": 90}
CENTENAS = {"cien": 100, "ciento": 100, "doscientos": 200, "doscientas": 200, "trescientos": 300, "trescientas": 300,
            "cuatrocientos": 400, "cuatrocientas": 400, "quinientos": 500, "quinientas": 500, "seiscientos": 600,
            "seiscientas": 600, "setecientos": 700, "setecientas": 700, "ochocientos": 800, "ochocientas": 800,
            "novecientos": 900, "novecientas": 900}
ESCALAS = {"mil": 1000, "millón": 10 ** 6, "millones": 10 ** 6, "billón": 10 ** 12, "billones": 10 ** 12}

CIFRA = re.compile(r"(?<![A-Za-zÁ-ÿ_.,])(?:\d{1,3}(?:\.\d{3})+|\d+)(?:,\d+)?(?![A-Za-zÁ-ÿ_]|,\d)")
PALABRA = re.compile(r"[A-Za-zÁ-ÿ]+(?:\.)?")


# ---- el número, con su valor ------------------------------------------------------------

def valor_de(texto):
    """«3.609» -> 3609, «0,30» -> 0.30; y cuántos decimales lleva."""
    limpio = texto.replace(".", "").replace(",", ".")
    return Decimal(limpio), (len(texto.split(",")[1]) if "," in texto else 0)


def cuantos_ceros(n):
    """Cuántos ceros finales tiene un entero: el tamaño del redondeo con que se escribió."""
    n, ceros = int(n), 0
    while n and n % 10 == 0:
        n //= 10
        ceros += 1
    return ceros


def numeros_con_letra(texto):
    """Los números escritos con letra de la frase: [(texto, valor, empieza, acaba, pequeño)]. «Pequeño» es del dos
    al nueve, que se cuenta y no se comprueba."""
    palabras = [(m.group().rstrip(".").lower(), m.start(), m.end()) for m in re.finditer(r"[A-Za-zÁ-ÿ]+", texto)]
    n, salida, i = len(palabras), [], 0
    vocab = {**UNIDADES, **DIEZ, **DECENAS, **CENTENAS, **ESCALAS}
    while i < n:
        w = palabras[i][0]
        siguiente = palabras[i + 1][0] if i + 1 < n else ""
        if w in vocab or (w in ARTICULOS and siguiente in ESCALAS):
            j, run = i, []
            while j < n:
                w = palabras[j][0]
                prox = palabras[j + 1][0] if j + 1 < n else ""
                if j > i and re.search(r"\S", texto[palabras[j - 1][2]:palabras[j][1]]):
                    break                               # una coma, un punto o un guion corta el número
                if w in vocab or (w in ARTICULOS and prox in ESCALAS):
                    run.append(w)
                elif w == "y" and run and run[-1] in DECENAS and prox in {**UNIDADES, **ARTICULOS}:
                    run.append(w)
                else:
                    break
                j += 1
            if run and run[-1] == "y":
                run, j = run[:-1], j - 1
            anterior = palabras[i - 1][0] if i > 0 else ""
            if run and all(w in ("millón", "millones", "billón", "billones") for w in run):
                i = j                                   # «miles de millones»: una palabra de tamaño, no una cifra
                continue
            if len(run) == 1 and run[0] in DECENAS and anterior in ("años", "década", "décadas"):
                i = j                                   # «los años sesenta»: una década, no una cifra
                continue
            total, cur = 0, 0
            for w in run:
                if w == "y":
                    continue
                if w == "mil":
                    cur = (cur or 1) * 1000
                elif w in ("millón", "millones"):
                    total, cur = total + (cur or 1) * 10 ** 6, 0
                elif w in ("billón", "billones"):
                    total, cur = total + (cur or 1) * 10 ** 12, 0
                else:
                    cur += vocab.get(w, ARTICULOS.get(w, 0))
            valor = total + cur
            ini, fin = palabras[i][1], palabras[j - 1][2]
            salida.append((texto[ini:fin], Decimal(valor), ini, fin, run == [run[0]] and run[0] in UNIDADES))
            i = j
        else:
            i += 1
    return salida


def numeros_de(texto):
    """Todos los números de una frase, con cómo están escritos y su clase. Una lista de dicts."""
    hallados = []
    ocupado = []
    for m in CIFRA.finditer(texto):
        if re.search(r"[A-Za-z]-$", texto[:m.start()]):
            continue                                                 # GPT-4, A-1
        v, dec = valor_de(m.group())
        fin = m.end()
        despues = texto[fin:fin + 24]
        escala = re.match(r"\s+(mil|millón|millones|billón|billones)\b", despues)
        if escala:
            v, dec = v * ESCALAS[escala.group(1)], 0
            fin += escala.end()
        pct = bool(re.match(r"\s*%", texto[fin:fin + 3]))
        hallados.append(dict(texto=texto[m.start():fin], valor=v, dec=dec, ini=m.start(), fin=fin, forma="cifra",
                             pct=pct, escala=bool(escala)))
        ocupado.append((m.start(), fin))
    for t, v, ini, fin, pequeno in numeros_con_letra(texto):
        if any(a <= ini < b for a, b in ocupado):
            continue
        antes = texto[:ini].split()
        if v == 100 and antes and antes[-1].lower() in ("cada", "por"):
            forma = "unidad"
        else:
            forma = "letra_pequena" if pequeno else "letra"
        hallados.append(dict(texto=t, valor=v, dec=0, ini=ini, fin=fin, forma=forma, pct=False, escala=False))
    for h in hallados:
        h["marca"] = marca_de(texto[:h["ini"]])
        h["clase"] = clase_de(texto, h)
    return sorted(hallados, key=lambda h: h["ini"])


def marca_de(antes):
    """Si la aproximación está marcada justo antes del número: 'aprox', 'mas', 'menos' o None."""
    a = antes.rstrip().lower()
    for lista, nombre in ((MARCAS_MAS, "mas"), (MARCAS_MENOS, "menos"), (MARCAS_APROX, "aprox")):
        if any(re.search(r"(^|\W)" + re.escape(m) + r"$", a) for m in lista):
            return nombre
    return None


def clase_de(texto, h):
    """Año, fecha, referencia o cifra: para el recuento y para saber qué exención cabe."""
    if h["forma"] not in ("cifra", "letra", "letra_pequena"):
        return "unidad"
    antes = re.findall(r"[A-Za-zÁ-ÿ.]+", texto[:h["ini"]])
    despues = re.findall(r"[A-Za-zÁ-ÿ]+", texto[h["fin"]:h["fin"] + 40])
    if antes and antes[-1].lower() in REFERENCIAS:
        return "referencia"
    if h["forma"] == "cifra" and len(despues) >= 2 and despues[0].lower() == "de" and despues[1].lower() in MESES:
        return "fecha"
    if (h["forma"] == "cifra" and "." not in h["texto"] and "," not in h["texto"] and not h["pct"]
            and h["marca"] is None and h["texto"].isdigit() and 1000 <= int(h["texto"]) <= 2100):
        return "año"
    return "cifra"


# ---- lo que hay en las salidas guardadas ----------------------------------------------------

def valores_de_salidas(textos):
    """Los números de las salidas guardadas, como valores, con la variante «proporción ↔ porcentaje»."""
    vistos = set()
    for t in textos:
        for m in re.finditer(r"(?<![\w.,])(?:\d{1,3}(?:\.\d{3})+|\d+)(?:,\d+)?(?![\w])", t):
            v, _ = valor_de(m.group())
            vistos.add(v)
            if 0 < v < 1:
                vistos.add(v * 100)
            elif v > 1 and v <= 100:
                vistos.add(v / 100)
    return vistos


def redondeo(n):
    """El tamaño del redondeo con que se escribió un número entero: 30 -> 10, 500.000.000 -> 100.000.000."""
    return Decimal(10) ** cuantos_ceros(n) if n == n.to_integral_value() and n != 0 else Decimal(1)


def escala_de_la_palabra(texto):
    """La escala de un número escrito con letra, por su última palabra: «quince millones» → 10^6; None si no acaba en
    una palabra de escala (millones, billones)."""
    ultima = (re.findall(r"[a-záéíóúñ]+", texto.lower()) or [""])[-1]
    return {"millón": Decimal(10) ** 6, "millones": Decimal(10) ** 6, "billón": Decimal(10) ** 12,
            "billones": Decimal(10) ** 12}.get(ultima)


def redondeo_a_la_escala(texto, valor, marca, exacto):
    """Orden 149: un número escrito con letra que acaba en una palabra de escala («quince millones») es un redondeo, y
    como todo redondeo lleva su marcador delante («unos», «más de», «casi»…), y el marcador tiene que decir la
    verdad: con «más de», el exacto es mayor que el número (y menos de una escala más); con «casi» o «menos de», menor
    (y menos de una escala menos); con «unos», a menos de media escala. '' si vale; si no, por qué no."""
    escala = escala_de_la_palabra(texto)
    if not escala:
        return f"{exacto} redondeado no da {valor}"
    if (exacto / escala).quantize(Decimal(1), rounding=ROUND_HALF_UP) * escala != valor:
        return f"{exacto} redondeado a {escala:,.0f} no da {valor}"
    if not marca:
        return (f"«{texto}» es un redondeo de {exacto} a la escala de su palabra y tiene que llevar «unos», «más de» "
                "o «casi» delante")
    bien = {"mas": valor <= exacto < valor + escala, "menos": valor - escala < exacto <= valor,
            "aprox": abs(exacto - valor) <= escala / 2}[marca]
    if not bien:
        return f"el marcador de «{texto}» no dice la verdad: el exacto es {exacto}"
    return ""


def tiene_salida(h, valores):
    """Si el número de la prosa tiene un valor en las salidas, con las reglas de la cabecera."""
    v = h["valor"]
    if v in valores:
        return True
    if h["dec"] > 0:
        q = Decimal(1).scaleb(-h["dec"])
        if any(x.quantize(q, rounding=ROUND_HALF_UP) == v for x in valores):
            return True
    if h["pct"] and v == v.to_integral_value():
        if any(x.quantize(Decimal(1), rounding=ROUND_HALF_UP) == v for x in valores):
            return True
    marca = h["marca"]
    if marca:
        g = redondeo(v)
        if marca == "aprox":
            return any(abs(x - v) <= g / 2 for x in valores)
        if marca == "mas":
            return any(v <= x < v + g for x in valores)
        if marca == "menos":
            return any(v - g < x <= v for x in valores)
    return False


# ---- las listas de notas/reforma-nov ---------------------------------------------------------

def leer_lista(ruta):
    """Las líneas «- …» de un fichero de notas; vacío si no existe."""
    if not Path(ruta).exists():
        return []
    return [l[2:].strip() for l in Path(ruta).read_text(encoding="utf-8").splitlines() if l.startswith("- ")]


def capitulos_reformados(raiz):
    return {int(m.group(1)) for l in leer_lista(Path(raiz) / CARPETA_NOTAS / FICHERO_REFORMADOS)
            for m in [re.match(r"cap(\d+)\b", l)] if m}


def salidas_declaradas(raiz):
    """{capítulo: [ficheros]} de salidas-por-capitulo.md."""
    d = {}
    for l in leer_lista(Path(raiz) / CARPETA_NOTAS / FICHERO_SALIDAS):
        m = re.match(r"cap(\d+)\s*:\s*(.+)$", l)
        if m:
            d[int(m.group(1))] = [x.strip() for x in m.group(2).split(",") if x.strip()]
    return d


def leer_exenciones(raiz):
    """[(cap, número, fragmento, motivo, línea)] y los errores de formato de la lista."""
    exenciones, errores = [], []
    ruta = Path(raiz) / CARPETA_NOTAS / FICHERO_EXENTAS
    for l in leer_lista(ruta):
        if not re.match(r"cap\d+\b", l):
            continue                     # una viñeta de explicación, no una exención
        partes = [p.strip() for p in l.split(" | ")]
        if len(partes) != 4 or not re.match(r"cap\d+$", partes[0]):
            errores.append(f"exención mal formada (se esperaba «capNN | número | trozo de la frase | motivo»): {l}")
            continue
        cap, numero, fragmento, motivo = int(partes[0][3:]), partes[1], partes[2], partes[3]
        if len(fragmento) < MIN_FRAGMENTO:
            errores.append(f"exención genérica: el trozo de la frase «{fragmento}» tiene menos de {MIN_FRAGMENTO} "
                           f"caracteres, así que valdría para cualquier frase parecida: {l}")
            continue
        base = motivo.split(";")[0].split(":")[0].strip().lower()
        if base not in MOTIVOS:
            errores.append(f"exención con un motivo que no es uno de {MOTIVOS}: «{motivo}» en {l}")
            continue
        if base in ("año", "fecha", "fuente") or (base == "cálculo" and "fuente:" in motivo):
            rutas = fuentes_de(motivo)
            if not rutas or not all((Path(raiz) / r).exists() for r in rutas):
                errores.append(f"exención de {base} sin «fuente: notas/fuentes/…» que exista en {raiz}: {l}")
                continue
        if base == "ejemplo" and not re.search(r"ejemplo:\s*«[^»]{12,}»", motivo):
            errores.append(f"exención de ejemplo sin «ejemplo: «la frase que plantea la hipótesis»»: {l}")
            continue
        if base == "fuente-no-libre":
            fi = re.search(r"fuente-no-libre:\s*(fichas/[^\s;]+)", motivo)
            if not fi or not (Path(raiz) / "notas" / "fuentes" / fi.group(1)).exists():
                errores.append(f"exención de fuente-no-libre sin «fichas/…» que exista en notas/fuentes/: {l}")
                continue
        if base == "cálculo" and not re.search(r"cálculo:\s*[^;]*\d", motivo):
            errores.append(f"exención de cálculo sin «cálculo: la cuenta»: {l}")
            continue
        if base == "redondeo" and not re.search(r"redondeo:\s*de\s+\d[\d.]*(?:,\d+)?", motivo):
            errores.append(f"exención de redondeo sin «redondeo: de 79,40 %» (el valor exacto): {l}")
            continue
        if base == "suma a la vista" and not re.search(r"suma a la vista:\s*\d+(?:\s*\+\s*\d+)+", motivo):
            errores.append(f"exención de suma sin «suma a la vista: 5 + 1 + 7» (los sumandos): {l}")
            continue
        exenciones.append((cap, numero, fragmento, motivo, base, l))
    return exenciones, errores


def fuentes_de(motivo):
    """Las rutas «notas/fuentes/…» que cita un motivo, sin la puntuación de detrás."""
    return [r.rstrip(".,;") for r in re.findall(r"notas/fuentes/[^\s,;]+", motivo)]


_TEXTOS_DE_FUENTE = {}


def texto_de_fuente(raiz, ruta):
    """El texto de una fuente: un .pdf, con `pdftotext`; lo demás, tal cual. Vacío si no se pudo leer."""
    clave = (str(raiz), ruta)
    if clave not in _TEXTOS_DE_FUENTE:
        f = Path(raiz) / ruta
        try:
            if f.suffix.lower() == ".pdf":
                import subprocess
                t = subprocess.run(["pdftotext", "-layout", str(f), "-"], capture_output=True, text=True).stdout
                paginas = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(f)], capture_output=True,
                                                                        text=True).stdout).group(1))
                if len(t.strip()) < 100 * paginas:
                    t = ""      # un escaneo: algo de texto suelto (la cubierta de Google) pero no el texto del libro
            else:
                t = f.read_text(encoding="utf-8", errors="ignore")
                if f.suffix.lower() in (".html", ".htm"):
                    import html as _html
                    t = re.sub(r"<(script|style).*?</\1>", " ", t, flags=re.S)
                    t = _html.unescape(re.sub(r"<[^>]+>", " ", t))
        except OSError:
            t = ""
        _TEXTOS_DE_FUENTE[clave] = re.sub(r"\s+", " ", t)
    return _TEXTOS_DE_FUENTE[clave]


def formas_en_ingles(escrito):
    """«1.104» -> «1,104», «1104»; «0,43» -> «0.43»: como lo escribiría un artículo en inglés."""
    v, dec = valor_de(escrito)
    entero = int(v)
    formas = {str(entero)} if dec == 0 else {str(v)}
    if dec == 0 and entero >= 1000:
        formas.add(f"{entero:,}")
    return formas


def formas_de_valor(v):
    """Las formas en que un valor entero sale en un texto en inglés: 43, 1104, 1,104."""
    if v != v.to_integral_value():
        return {str(v.normalize())}
    n = int(v)
    return {str(n), f"{n:,}"} if n >= 1000 else {str(n)}


def numero_en_fuente(raiz, rutas, escrito, valor=None, frase=None):
    """True si el número (o, si se da, la frase) sale en el texto de alguna de las fuentes; None si ninguna tiene texto."""
    textos = [texto_de_fuente(raiz, r) for r in rutas]
    textos = [t for t in textos if t.strip()]
    if not textos:
        return None
    if frase:
        buscada = re.sub(r"\s+", " ", frase).strip().lower()
        return any(buscada in t.lower() for t in textos)
    formas = formas_de_valor(valor) if (valor is not None and not re.fullmatch(r"[\d.,]+", escrito)) \
        else formas_en_ingles(escrito)
    return any(re.search(r"(?<![\d.,])" + re.escape(f) + r"(?![\d]|[.,]\d)", t) for f in formas for t in textos)


def evaluar_cuenta(cuenta):
    """Evalúa una cuenta con + - * / ^ y paréntesis, con los números escritos como en el libro. Devuelve
    (resultado, [cifras de la cuenta]). Solo aritmética: no ejecuta nada más."""
    import ast
    cifras = re.findall(r"\d[\d.]*(?:,\d+)?", cuenta)
    expr = re.sub(r"\d[\d.]*(?:,\d+)?", lambda m: f"({valor_de(m.group())[0]})", cuenta).replace("^", "**")
    assert re.fullmatch(r"[\d\s+\-*/().]+", expr), f"Se esperaba una cuenta solo con cifras y + - * / ^ ( ); es «{cuenta}»"

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return Decimal(str(n.value))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Add):
                return a + b
            if isinstance(n.op, ast.Sub):
                return a - b
            if isinstance(n.op, ast.Mult):
                return a * b
            if isinstance(n.op, ast.Div):
                return a / b
            if isinstance(n.op, ast.Pow):
                return a ** int(b)
        raise AssertionError(f"Se esperaba solo aritmética en «{cuenta}»")
    return ev(ast.parse(expr, mode="eval")), cifras


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


# ---- el capítulo ---------------------------------------------------------------------------------

def prosa_del_capitulo(tl, fichero):
    """[(línea, frase, párrafo)] de la prosa del capítulo, sin títulos."""
    salida = []
    for linea, es_titulo, texto in tl.parrafos(Path(fichero).read_text(encoding="utf-8")):
        if es_titulo:
            continue
        for fr in tl.frases(texto):
            salida.append((linea, fr, texto))
    return salida


def lineas_de_bloques(texto):
    """Las líneas de bloques, tablas, citas y sangrados del capítulo (para derivar sus salidas)."""
    salida, dentro = [], False
    for l in texto.splitlines():
        if re.match(r"^:::\s*\{?\.?(tabla|muestra)", l):
            dentro = True
            continue
        if l.strip() == ":::":
            dentro = False
            continue
        if dentro or l.startswith("    ") or l.startswith(">") or l.startswith("|"):
            t = norm(l.strip(" >|"))
            if len(t) >= 14:
                salida.append(t)
    return salida


def cargar_salidas(carpeta):
    return {p.name: norm(p.read_text(encoding="utf-8", errors="ignore")) for p in sorted(Path(carpeta).glob("*.txt"))}


def salidas_del_capitulo(cap, texto_cap, declaradas, todas):
    """Las salidas del capítulo: las declaradas o, si no hay, las derivadas de sus bloques. Devuelve (nombres, derivadas)."""
    if cap in declaradas:
        faltan = [n for n in declaradas[cap] if n not in todas]
        assert not faltan, (f"Se esperaba que las salidas declaradas para el capítulo {cap} existieran; no están: "
                            f"{faltan}")
        return declaradas[cap], False
    lineas = lineas_de_bloques(texto_cap)
    return [n for n, t in todas.items() if any(l in t for l in lineas)], True


PEQUENOS = {1: ("uno", "un", "una"), 2: ("dos",), 3: ("tres",), 4: ("cuatro",), 5: ("cinco",), 6: ("seis",),
            7: ("siete",), 8: ("ocho",), 9: ("nueve",)}


def aparece_valor(parrafo, v):
    """Si el valor entero v sale en el párrafo, escrito con cifra o, del uno al nueve, con letra."""
    if any(h["valor"] == v for h in numeros_de(parrafo)):
        return True
    palabras = re.findall(r"[a-záéíóúü]+", parrafo.lower())
    return int(v) in PEQUENOS and any(w in palabras for w in PEQUENOS[int(v)])


def apartado_de(texto_md, linea):
    """El texto del apartado (hasta el título de nivel 1 o 2 siguiente) en el que cae la línea (empieza en 1)."""
    ls = texto_md.split("\n")
    i = min(max(linea - 1, 0), len(ls) - 1)
    ini = next((k for k in range(i, -1, -1) if re.match(r"#{1,2}\s", ls[k])), 0)
    fin = next((k for k in range(i + 1, len(ls)) if re.match(r"#{1,2}\s", ls[k])), len(ls))
    return re.sub(r"[*`]", "", " ".join(ls[ini:fin]))


def frase_de(motivo):
    m = re.search(r"frase:\s*«([^»]+)»", motivo)
    return m.group(1) if m else None


def exencion_no_vale(e, h, parrafo, valores, raiz=".", cap=None, apartado=""):
    """Por qué una exención concreta no vale para este número ('' si vale). Cada motivo tiene su prueba."""
    base, motivo = e[4], e[3]
    if base in ("año", "fecha", "fuente"):
        # el número tiene que salir en el texto de su fuente (Orden 141) o, si no sale tal cual, la frase (Orden 143);
        # un escaneo sin texto no se puede mirar
        frase = frase_de(motivo)
        esta = numero_en_fuente(raiz, fuentes_de(motivo), h["texto"], h["valor"], frase)
        if esta is False:
            return (f"la frase «{frase}» no sale en el texto de {', '.join(fuentes_de(motivo))}" if frase else
                    f"«{h['texto']}» no sale en el texto de {', '.join(fuentes_de(motivo))}")
    if base == "ejemplo":
        hip = re.search(r"ejemplo:\s*«([^»]+)»", motivo).group(1)
        if norm(hip) not in norm(apartado):
            return f"la frase de la hipótesis «{hip[:50]}…» no está en el mismo apartado que el número"
    if base == "fuente-no-libre":
        ficha = Path(raiz) / "notas" / "fuentes" / re.search(r"fuente-no-libre:\s*(fichas/[^\s;]+)", motivo).group(1)
        t = ficha.read_text(encoding="utf-8")
        cita, pagina, fr = (re.search(rf"^{k}:\s*(.+)$", t, re.M) for k in ("Cita", "Página", "Frase"))
        if not (cita and pagina and fr) or not re.search(r"\d", pagina.group(1)):
            return "la ficha tiene que llevar «Cita: …», «Página: N» (con número) y «Frase: «…»»"
        frase = frase_de(motivo)
        if frase:
            if re.sub(r"\s+", " ", frase).lower() not in re.sub(r"\s+", " ", fr.group(1)).lower():
                return f"la frase «{frase}» no está en la línea «Frase:» de la ficha"
        elif not any(re.search(r"(?<![\d.,])" + re.escape(f) + r"(?![\d])", fr.group(1))
                     for f in (formas_de_valor(h["valor"]) | {h["texto"]})):
            return f"«{h['texto']}» no sale en la línea «Frase:» de la ficha"
        h["no_libre"] = True
    if base == "cálculo":
        cuenta = re.search(r"cálculo:\s*([^;]+)", motivo).group(1).strip()
        try:
            resultado, cifras = evaluar_cuenta(cuenta)
        except (AssertionError, SyntaxError, ZeroDivisionError) as err:
            return f"la cuenta «{cuenta}» no se puede hacer: {err}"
        if not tiene_salida(h, {resultado}):
            return f"la cuenta «{cuenta}» da {resultado.quantize(Decimal('0.01'))} y no es «{h['texto']}»"
        rutas = fuentes_de(motivo)
        supuestos = {}
        if "supuestos:" in motivo:
            anexo = sorted((Path(raiz) / "manuscript").glob(f"anexo-a{int(cap or 0):02d}-*.md")) if cap else []
            texto_anexo = norm(re.sub(r"[*`]", "", anexo[0].read_text(encoding="utf-8"))).lower() if anexo else ""
            for par in motivo.split("supuestos:", 1)[1].split(";"):
                m = re.match(r"\s*(\d[\d.,]*)\s*=\s*«([^»]+)»", par)
                if not m:
                    return f"un supuesto mal escrito (se esperaba «valor = «frase»»): {par.strip()[:60]}"
                if norm(m.group(2)).lower() not in texto_anexo:
                    return f"el supuesto «{m.group(2)}» no está en el anexo del capítulo {cap}"
                supuestos[valor_de(m.group(1))[0]] = m.group(2)
        for c in cifras:
            v, _ = valor_de(c)
            if v in valores or v in supuestos:
                continue
            esta = numero_en_fuente(raiz, rutas, c) if rutas else False
            if not esta:
                return f"la cifra «{c}» de la cuenta no está en las salidas del capítulo ni en el texto de la fuente"
    if base == "inventado" and not any(p in parrafo.lower() for p in DECLARA_INVENTADO):
        return f"el párrafo no declara el ejemplo inventado ({'/'.join(DECLARA_INVENTADO)})"
    if base == "redondeo":
        exacto, dec = valor_de(re.search(r"redondeo:\s*de\s+(\d[\d.]*(?:,\d+)?)", motivo).group(1))
        if exacto not in valores:
            return f"el valor exacto {exacto} no está en las salidas guardadas del capítulo"
        if exacto.quantize(Decimal(1).scaleb(-h["dec"]), rounding=ROUND_HALF_UP) != h["valor"]:
            # Orden 148: un número escrito con la palabra de su escala («quince millones») es el redondeo del exacto a
            # esa escala (15.048.831 → 15.000.000), no a los decimales
            porque = redondeo_a_la_escala(h["texto"], h["valor"], h["marca"], exacto)
            if porque:
                return porque or f"{exacto} redondeado a {h['dec']} decimales no da {h['valor']}"
    if base == "suma a la vista":
        sumandos = [int(x) for x in re.findall(r"\d+", re.search(r"suma a la vista:([^;]*)", motivo).group(1))]
        if sum(sumandos) != h["valor"]:
            return f"{' + '.join(map(str, sumandos))} da {sum(sumandos)}, no {h['valor']}"
        faltan = [x for x in sumandos if not aparece_valor(parrafo, Decimal(x))]
        if faltan:
            return f"los sumandos {faltan} no salen en el párrafo"
    return ""


def revisar_capitulo(tl, raiz, cap, fichero, declaradas, todas, exenciones):
    """Devuelve (números, errores): cada número con su clase y si tiene salida, exención o ninguna."""
    texto = Path(fichero).read_text(encoding="utf-8")
    nombres, derivadas = salidas_del_capitulo(cap, texto, declaradas, todas)
    valores = valores_de_salidas([todas[n] for n in nombres])
    valores_todos = valores_de_salidas(todas.values())
    resultado, usadas = [], set()
    for linea, frase, parrafo in prosa_del_capitulo(tl, fichero):
        for h in numeros_de(frase):
            h.update(linea=linea, frase=frase)
            if h["forma"] in ("unidad", "letra_pequena"):
                h["estado"] = h["forma"]
            else:
                # Una exención escrita a mano manda sobre una coincidencia casual con una salida: «casi setenta
                # años» de 1948 a 2017 no sale de ninguna salida aunque haya un 69 en alguna (Orden 140).
                exenta = [e for e in exenciones if e[0] == cap and e[1] == h["texto"] and norm(e[2]) in norm(frase)]
                if exenta:
                    e = exenta[0]
                    usadas.add(e[5])
                    porque = exencion_no_vale(e, h, parrafo, valores, raiz, cap, apartado_de(texto, linea))
                    h["estado"] = "exento" if not porque else "exención no vale"
                    h["porque"] = porque
                elif tiene_salida(h, valores):
                    h["estado"] = "con salida"
                else:
                    h["estado"] = "sin salida"
                    h["en_ninguna"] = not tiene_salida(h, valores_todos)
            resultado.append(h)
    errores = []
    for e in exenciones:
        if e[0] == cap and e[5] not in usadas:
            errores.append(f"exención que no casa con ningún número del capítulo {cap} (¿cambió el texto?): {e[5]}")
    return resultado, errores, nombres, derivadas


# ---- el programa -----------------------------------------------------------------------------------

def cargar_texto_libro(raiz):
    herramientas = Path(raiz) / "herramientas"
    assert (herramientas / "texto_libro.py").exists(), (
        f"Se esperaba encontrar {herramientas / 'texto_libro.py'} (de ahí sale la prosa); no existe")
    sys.path.insert(0, str(herramientas))
    import texto_libro
    return texto_libro


def ficheros_por_capitulo(tl, raiz):
    """{número de capítulo: fichero} de Book.txt (solo los «capNN-»)."""
    d = {}
    for f in tl.ficheros_del_libro(raiz):
        m = re.match(r"cap(\d+)-", Path(f).name)
        if m:
            d[int(m.group(1))] = f
    return d


def correr(raiz, carpeta_salidas, capitulos, estricto=True, recuento=False):
    tl = cargar_texto_libro(raiz)
    por_cap = ficheros_por_capitulo(tl, raiz)
    todas = cargar_salidas(carpeta_salidas)
    declaradas = salidas_declaradas(raiz)
    exenciones, errores = leer_exenciones(raiz)
    for c in capitulos:
        assert c in por_cap, f"Se esperaba el capítulo {c} en Book.txt; hay {sorted(por_cap)}"
    fallos = list(errores)
    filas = []
    for cap in capitulos:
        numeros, errs, nombres, derivadas = revisar_capitulo(tl, raiz, cap, por_cap[cap], declaradas, todas, exenciones)
        fallos += errs
        cuenta = {}
        for h in numeros:
            cuenta[h["estado"]] = cuenta.get(h["estado"], 0) + 1
        sin = [h for h in numeros if h["estado"] == "sin salida"]
        mal = [h for h in numeros if h["estado"] == "exención no vale"]
        por_clase = {c: sum(1 for h in sin if h["clase"] == c) for c in ("año", "fecha", "referencia", "cifra")}
        cuenta["no_libre"] = sum(1 for h in numeros if h.get("no_libre"))
        filas.append((cap, len(numeros), cuenta, por_clase, sum(1 for h in sin if h.get("en_ninguna")), nombres, derivadas))
        if estricto:
            for h in sin:
                fallos.append(f"capítulo {cap}, línea {h['linea']}: «{h['texto']}» ({h['clase']}) no está en las salidas "
                              f"de su capítulo; frase: «{h['frase'][:110]}»")
            for h in mal:
                fallos.append(f"capítulo {cap}, línea {h['linea']}: «{h['texto']}» está exento y la exención no vale: "
                              f"{h['porque']}")
    return filas, fallos


def imprimir_recuento(filas):
    print(f"{'cap':>3} {'números':>8} {'con salida':>11} {'sin salida':>11} {'años':>5} {'fechas':>7} {'refs':>5} "
          f"{'cifras':>7} {'en ninguna':>11} {'pequeños':>9} {'unidad':>7}  salidas")
    tot = [0] * 9
    for cap, n, cuenta, clase, ninguna, nombres, derivadas in filas:
        sin = cuenta.get("sin salida", 0)
        fila = [n, cuenta.get("con salida", 0) + cuenta.get("exento", 0), sin, clase["año"], clase["fecha"], clase["referencia"],
                clase["cifra"], ninguna, cuenta.get("letra_pequena", 0)]
        tot = [a + b for a, b in zip(tot, fila)]
        print(f"{cap:>3} {fila[0]:>8} {fila[1]:>11} {fila[2]:>11} {fila[3]:>5} {fila[4]:>7} {fila[5]:>5} {fila[6]:>7} "
              f"{fila[7]:>11} {fila[8]:>9} {cuenta.get('unidad', 0):>7}  "
              f"{len(nombres)} {'derivadas' if derivadas else 'declaradas'}")
    print(f"{'tot':>3} " + " ".join(f"{x:>{w}}" for x, w in zip(tot, (8, 11, 11, 5, 7, 5, 7, 11, 9))))


# ---- selftest ------------------------------------------------------------------------------------------

def _libro_de_mentira(d, capitulo, salida, extra=None):
    """Un libro mínimo en `d`: manuscript/Book.txt, un capítulo, una salida y un texto_libro.py copiado de verdad."""
    raiz = Path(d) / "libro"
    (raiz / "manuscript").mkdir(parents=True)
    (raiz / "manuscript" / "Book.txt").write_text("cap01-x.md\n", encoding="utf-8")
    (raiz / "manuscript" / "cap01-x.md").write_text(capitulo, encoding="utf-8")
    (raiz / "notas" / "reforma-nov").mkdir(parents=True)
    (raiz / "notas" / "fuentes").mkdir(parents=True)
    (raiz / "notas" / "fuentes" / "shannon.md").write_text(
        "fuente de prueba: en 1948 y en 1913 y en 2017; 20,000 letras, 8,638 vocales, 1,104 parejas\n", encoding="utf-8")
    (raiz / "herramientas").mkdir()
    origen = Path(d).parent
    real = None
    for cand in (AQUI.parent.parent / "libro-ia-libro" / "herramientas" / "texto_libro.py",):
        if cand.exists():
            real = cand
    assert real, "Se esperaba encontrar libro-ia-libro/herramientas/texto_libro.py junto al repositorio público"
    (raiz / "herramientas" / "texto_libro.py").write_text(real.read_text(encoding="utf-8"), encoding="utf-8")
    sal = Path(d) / "salidas"
    sal.mkdir()
    (sal / "programa.txt").write_text(salida, encoding="utf-8")
    (raiz / "notas" / "reforma-nov" / FICHERO_SALIDAS).write_text("# salidas\n\n- cap01: programa.txt\n", encoding="utf-8")
    for nombre, contenido in (extra or {}).items():
        (raiz / "notas" / "reforma-nov" / nombre).write_text(contenido, encoding="utf-8")
    _ = origen
    return raiz, sal


def _selftest():
    fallos = []

    def prueba(texto, salida):
        """Los números de una frase contra una salida: devuelve {texto: tiene salida}."""
        v = valores_de_salidas([salida])
        return {h["texto"]: tiene_salida(h, v) for h in numeros_de(texto) if h["forma"] in ("cifra", "letra")}

    # 1. TEST NULO: una prosa sin números no tiene ninguno, y una con números que todos están en la salida no falla.
    sin = numeros_de("Una frase sin ninguna cuenta, solo con palabras y una pregunta. ¿Quién dijo eso?")
    hay = prueba("Salen 3.609 hojas, un 12,5 % y unas treinta veces.", "hojas 3.609\nporcentaje 12,50 %\nveces 29,8")
    print(f"[1] test nulo         frase sin números: {len(sin)}; frase con tres números que están en la salida: {hay}")
    if sin or not all(hay.values()) or len(hay) != 3:
        fallos.append(f"test nulo: se esperaban 0 números y tres con salida; salió {len(sin)} y {hay}")

    # 2. SEÑAL IMPLANTADA: un número inventado tiene que hacer saltar la alarma, y las formas del mismo número no.
    salida = "veces: 47\nhojas: 22.952\ngrande2: 526.000.000\ndiv: 7,66\nproporcion: 0,94\npct: 75,00 %\nentero: 12,0\ngrande: 526.794.304\nmedia: 29,8"
    casos = [
        ("Hay 9.999 hojas.", {"9.999": False}),                                   # el inventado
        ("Hay cuarenta y siete trozos.", {"cuarenta y siete": True}),             # con letra
        ("Hay cuarenta y ocho trozos.", {"cuarenta y ocho": False}),              # con letra, otro valor
        ("Son 7,7 en media.", {"7,7": True}),                                     # redondeo a los decimales de la prosa
        ("Son 7,8 en media.", {"7,8": False}),
        ("Salen unas treinta veces.", {"treinta": True}),                    # aproximación marcada
        ("Salen unas cuarenta veces.", {"cuarenta": False}),
        ("Salen treinta veces.", {"treinta": False}),                             # sin marca: exacto
        ("Son más de quinientos millones.", {"quinientos millones": True}),               # «más de» cubre los 526 millones
        ("Es un 75 % de casos y 12 de ellos.", {"75": True, "12": True}),       # porcentaje entero, 12 = 12,0
        ("Salió el 94 por cien.", {"94": True}),                                  # la proporción 0,94 es el 94
        ("Una palabra, 22.952 hojas; dos, quinientos veintiséis millones.", {"quinientos veintiséis millones": True}),  # la coma corta
    ]
    malos = []
    for frase, esperado in casos:
        obtenido = prueba(frase, salida)
        for k, v in esperado.items():
            claves = [x for x in obtenido if x == k or k.endswith(x) or x.endswith(k)]
            if not claves or obtenido[claves[0]] != v:
                malos.append((frase, k, v, obtenido))
    print(f"[2] señal implantada  {len(casos)} casos (inventado, con letra, redondeos, aproximaciones, porcentaje): "
          f"mal resueltos {len(malos)}")
    if malos:
        fallos.append(f"señal implantada: {malos}")
    # Orden 149: «quince millones» es un redondeo a millones de 15.048.831 y lleva marcador, y el marcador dice la verdad.
    exacto = Decimal(15048831)
    quince = Decimal(15000000)
    casos_e = [("quince millones", None, False),               # a secas: falla (sin marcador)
               ("quince millones", "mas", True),               # «más de quince millones»: bien
               ("quince millones", "menos", False),            # «casi quince millones»: el exacto es mayor: miente
               ("quince millones", "aprox", True),             # «unos quince millones»: a menos de medio millón
               ("dieciséis millones", "mas", False)]           # «más de dieciséis millones»: no es su redondeo
    r_e = [not redondeo_a_la_escala(t, Decimal(16000000) if "dieciséis" in t else quince, m, exacto) for t, m, _ in casos_e]
    print(f"[2] señal implantada  «quince millones» (15.048.831): a secas {'falla' if not r_e[0] else 'PASA'}; con «más de» "
          f"{'pasa' if r_e[1] else 'FALLA'}; con «casi» {'falla' if not r_e[2] else 'PASA'}; con «unos» "
          f"{'pasa' if r_e[3] else 'FALLA'}; «más de dieciséis millones» {'falla' if not r_e[4] else 'PASA'}; «cuarenta» "
          f"sin escala: {'sí' if escala_de_la_palabra('cuarenta') is None else 'NO'}")
    if r_e != [c[2] for c in casos_e] or escala_de_la_palabra("cuarenta") is not None:
        fallos.append(f"señal implantada: el redondeo a la escala de una palabra mal juzgado: {r_e}")

    # 3. INVARIANTES: de extremo a extremo, con un libro de mentira; exenciones concretas, genéricas y viejas.
    capitulo = ("# Un capítulo\n\n"
                "Tiene 3.609 hojas y 9.999 papeletas. En 1948 pasó algo. Es el capítulo 7 del libro. "
                "Imagina que hay 777 piezas, un ejemplo inventado. Salen unas treinta veces.\n")
    with tempfile.TemporaryDirectory() as d:
        raiz, sal = _libro_de_mentira(d, capitulo, "hojas 3.609\nveces 29,8\n")
        r0, e0 = correr(raiz, sal, [1], estricto=True)
        sin_salida = sorted({h["texto"] for h in revisar_capitulo(cargar_texto_libro(raiz), raiz, 1, raiz / "manuscript" / "cap01-x.md",
                                                              salidas_declaradas(raiz), cargar_salidas(sal), [])[0]
                                 if h["estado"] == "sin salida"})
        print(f"[3] invariante        libro de mentira: sin salida {sin_salida}; fallos sin exenciones: {len(e0)}")
        if sin_salida != ["1948", "7", "777", "9.999"] or len(e0) != 4:
            fallos.append(f"invariante: se esperaban sin salida 1948, 7, 777 y 9.999; salió {sin_salida} con {len(e0)} fallos")
        exentas = ("# exenciones\n\n"
                   "- cap01 | 1948 | En 1948 pasó algo | año; fuente: notas/fuentes/shannon.md\n"
                   "- cap01 | 7 | Es el capítulo 7 del libro | referencia\n"
                   "- cap01 | 777 | Imagina que hay 777 piezas | inventado\n")
        (raiz / "notas" / "reforma-nov" / FICHERO_EXENTAS).write_text(exentas, encoding="utf-8")
        _, e1 = correr(raiz, sal, [1], estricto=True)
        print(f"                      con tres exenciones concretas: fallos {len(e1)} (queda 9.999)")
        if len(e1) != 1 or "9.999" not in e1[0]:
            fallos.append(f"invariante: con las exenciones debía quedar solo 9.999; salió {e1}")
        # una genérica (trozo corto), una de año sin fuente, una vieja que ya no casa y una de inventado no declarado
        malas = ("- cap01 | 9.999 | papeletas | referencia\n"
                 "- cap01 | 1948 | En 1948 pasó algo | año; fuente: notas/fuentes/no_existe.md\n"
                 "- cap01 | 55 | Una frase que ya no está en el capítulo | referencia\n")
        (raiz / "notas" / "reforma-nov" / FICHERO_EXENTAS).write_text("# x\n\n" + malas, encoding="utf-8")
        _, e2 = correr(raiz, sal, [1], estricto=True)
        genericas = [e for e in e2 if "genérica" in e]
        sin_fuente = [e for e in e2 if "sin «fuente" in e]
        print(f"                      exenciones malas: genérica {len(genericas)}, año sin fuente {len(sin_fuente)} "
              f"(esperado 1 y 1); el 9.999 sigue sin salida: {any('9.999' in e for e in e2)}")
        if len(genericas) != 1 or len(sin_fuente) != 1 or not any("9.999" in e and "no está en las salidas" in e for e in e2):
            fallos.append(f"invariante: las exenciones malas debían rechazarse; salió {e2}")
        (raiz / "notas" / "reforma-nov" / FICHERO_EXENTAS).write_text(
            "# x\n\n- cap01 | 55 | Una frase que ya no está en el capítulo | referencia\n", encoding="utf-8")
        _, e3 = correr(raiz, sal, [1], estricto=True)
        vieja = [e for e in e3 if "no casa con ningún número" in e]
        if len(vieja) != 1:
            fallos.append(f"invariante: una exención vieja debía dar error; salió {e3}")
    # Los motivos de la Orden 140: redondeo (79 de 79,40 %), suma a la vista (5 + 1 + 7) y fuente.
    capitulo2 = ("# Otro\n\nLleva 79 papeletas la hoja. Cinco, uno y siete suman 13. "
                 "Y 1913 fue el año del artículo.\n")
    buenas = ("- cap01 | 79 | Lleva 79 papeletas la hoja | redondeo: de 79,40 %\n"
              "- cap01 | 13 | Cinco, uno y siete suman 13 | suma a la vista: 5 + 1 + 7\n"
              "- cap01 | 1913 | fue el año del artículo | año; fuente: notas/fuentes/shannon.md\n")
    malas = {
        "redondeo con un valor que no está en la salida": ("- cap01 | 79 | Lleva 79 papeletas la hoja | redondeo: de 78,40 %\n", "no está en las salidas"),
        "redondeo que no redondea al número": ("- cap01 | 79 | Lleva 79 papeletas la hoja | redondeo: de 79,60 %\n", "no está en las salidas"),
        "suma que no da": ("- cap01 | 13 | Cinco, uno y siete suman 13 | suma a la vista: 5 + 1 + 6\n", "da 12"),
        "suma con un sumando que no sale": ("- cap01 | 13 | Cinco, uno y siete suman 13 | suma a la vista: 5 + 2 + 6\n", "no salen en el párrafo"),
        "fuente que no existe": ("- cap01 | 1913 | fue el año del artículo | fuente: notas/fuentes/no_existe.pdf\n", "sin «fuente"),
    }
    # El motivo «cálculo» (Orden 141) y el texto de la fuente.
    capitulo3 = "# Otro más\n\nEsperaba unas 3.731 parejas. Pasaron casi setenta años. Y salieron 9.999 cosas.\n"
    buenas3 = ("- cap01 | 3.731 | Esperaba unas 3.731 | cálculo: (8.638 / 20.000) * (8.638 / 20.000) * 20.000; fuente: notas/fuentes/shannon.md\n"
               "- cap01 | setenta | casi setenta años | cálculo: 2017 - 1948; fuente: notas/fuentes/shannon.md\n"
               "- cap01 | 9.999 | salieron 9.999 cosas | fuente: notas/fuentes/shannon.md\n")
    malas3 = {
        "cálculo que no da el número": ("- cap01 | setenta | casi setenta años | cálculo: 2017 - 1900; fuente: notas/fuentes/shannon.md\n", "da 117"),
        "cifra de la cuenta que no está en la fuente": ("- cap01 | setenta | casi setenta años | cálculo: 2017 - 1947; fuente: notas/fuentes/shannon.md\n", "«1947» de la cuenta"),
        "cifra exenta que no sale en su fuente": ("- cap01 | 9.999 | salieron 9.999 cosas | fuente: notas/fuentes/shannon.md\n", "no sale en el texto"),
    }
    with tempfile.TemporaryDirectory() as d3:
        raiz3, sal3 = _libro_de_mentira(d3, capitulo3, "nada 1\n")
        ruta3 = raiz3 / "notas" / "reforma-nov" / FICHERO_EXENTAS
        ruta3.write_text("# x\n\n" + buenas3.replace("- cap01 | 9.999 | salieron 9.999 cosas | fuente: notas/fuentes/shannon.md\n", ""),
                         encoding="utf-8")
        _, ok3 = correr(raiz3, sal3, [1], estricto=True)
        quedan = [x for x in ok3 if "9.999" not in x]
        print(f"                      cálculo (3.731 y casi setenta) bien escrito: fallos {len(quedan)} (esperado 0; el 9.999 queda sin exención)")
        if quedan:
            fallos.append(f"invariante: las exenciones de cálculo buenas debían valer; salió {quedan}")
        rotas3 = []
        for nombre, (linea, quiero) in malas3.items():
            ruta3.write_text("# x\n\n" + linea, encoding="utf-8")
            _, e = correr(raiz3, sal3, [1], estricto=True)
            if not any(quiero in x for x in e):
                rotas3.append((nombre, e))
        print(f"                      tres exenciones malas (cuenta, cifra, fuente): sin rechazar {len(rotas3)} (esperado 0)")
        if rotas3:
            fallos.append(f"invariante: estas exenciones de cálculo malas debían rechazarse: {rotas3}")
    # Orden 143: décadas y escalas sueltas, ejemplo, supuestos, fuente no libre y frase.
    sueltas = numeros_de("Fue en los años sesenta; hay cientos de miles de millones de pesos.")
    print(f"                      décadas y palabras de tamaño sueltas no son cifras: {len(sueltas)} números (esperado 0)")
    if sueltas:
        fallos.append(f"invariante: las décadas y las escalas sueltas no debían contar; salió {[h['texto'] for h in sueltas]}")
    cap143 = ("# Cap\n\n## El colegio\n\nImagina que entra un abuelo. Y tu abuelo, con ochenta, entra el primero.\n\n"
              "## Otro apartado\n\nCon tantos pesos pasarían más de seis mil años. El mástil medía metro ochenta.\n"
              "Lo dijo en 1988 y sigue.\n")
    anexo143 = ("# A1. Cuentas\n\n## Otro apartado\n\nCon cien mil millones de pesos, dos pruebas por peso, a un segundo cada una, "
                "y un año de unos treinta y un millones y medio de segundos.\n")
    buenas143 = {
        "ejemplo": "- cap01 | ochenta | tu abuelo, con ochenta | ejemplo: «Imagina que entra un abuelo»\n",
        "supuestos": ("- cap01 | seis mil | pasarían más de seis mil años | cálculo: 100.000.000.000 * 2 / 31.500.000; supuestos: "
                      "100.000.000.000 = «cien mil millones de pesos»; 2 = «dos pruebas por peso»; 31.500.000 = "
                      "«treinta y un millones y medio de segundos»\n"),
        "no libre": "- cap01 | 1988 | Lo dijo en 1988 | fuente-no-libre: fichas/f.md\n",
        "frase": ("- cap01 | ochenta | medía metro ochenta | fuente: notas/fuentes/s.html; frase: «six feet high»\n"),
    }
    malas143 = {
        "ejemplo con la hipótesis en otro apartado": ("- cap01 | ochenta | tu abuelo, con ochenta | ejemplo: «El mástil medía metro ochenta»\n", "no está en el mismo apartado"),
        "supuesto que no está en el anexo": ("- cap01 | seis mil | pasarían más de seis mil años | cálculo: 100.000.000.000 * 2 / 31.500.000; supuestos: "
                                             "100.000.000.000 = «cien mil millones de pesos»; 2 = «tres pruebas por peso»; 31.500.000 = "
                                             "«treinta y un millones y medio de segundos»\n", "no está en el anexo"),
        "cuenta con una cifra que no es supuesto": ("- cap01 | seis mil | pasarían más de seis mil años | cálculo: 100.000.000.000 * 3 / 31.500.000; supuestos: "
                                                    "100.000.000.000 = «cien mil millones de pesos»; 2 = «dos pruebas por peso»; 31.500.000 = "
                                                    "«treinta y un millones y medio de segundos»\n", "no está en las salidas"),
        "frase que no sale en la fuente": ("- cap01 | ochenta | medía metro ochenta | fuente: notas/fuentes/s.html; frase: «seven feet high»\n", "no sale en el texto"),
    }
    with tempfile.TemporaryDirectory() as d4:
        raiz4, sal4 = _libro_de_mentira(d4, cap143, "nada 1\n")
        (raiz4 / "manuscript" / "anexo-a01-x.md").write_text(anexo143, encoding="utf-8")
        (raiz4 / "notas" / "fuentes" / "s.html").write_text("<html><body><p>racks, <b>six feet</b> high</p></body></html>", encoding="utf-8")
        (raiz4 / "notas" / "fuentes" / "fichas").mkdir()
        (raiz4 / "notas" / "fuentes" / "fichas" / "f.md").write_text(
            "Cita: A. Autor, Libro, Editorial, 1988.\nPágina: 12\nFrase: «Lo dijo en 1988 y sigue.»\n", encoding="utf-8")
        ruta4 = raiz4 / "notas" / "reforma-nov" / FICHERO_EXENTAS
        ruta4.write_text("# x\n\n" + "".join(buenas143.values()), encoding="utf-8")
        filas4, e4 = correr(raiz4, sal4, [1], estricto=True)
        print(f"                      ejemplo, supuestos, fuente no libre y frase, bien escritos: fallos {len(e4)} (esperado 0); "
              f"fuentes no libres contadas aparte: {filas4[0][2].get('no_libre')} (esperado 1)")
        if e4 or filas4[0][2].get("no_libre") != 1:
            fallos.append(f"invariante: las exenciones de la Orden 143 debían valer y contar una no libre; salió {e4}")
        rotas4 = []
        for nombre, (linea, quiero) in malas143.items():
            ruta4.write_text("# x\n\n" + linea, encoding="utf-8")
            _, e = correr(raiz4, sal4, [1], estricto=True)
            if not any(quiero in x for x in e):
                rotas4.append((nombre, e))
        # una ficha sin página y otra con la frase sin el número
        for nombre, ficha in {"ficha sin página": "Cita: A.\nFrase: «Lo dijo en 1988 y sigue.»\n",
                              "ficha sin el número en la frase": "Cita: A.\nPágina: 3\nFrase: «Lo dijo hace tiempo.»\n"}.items():
            (raiz4 / "notas" / "fuentes" / "fichas" / "f.md").write_text(ficha, encoding="utf-8")
            ruta4.write_text("# x\n\n" + buenas143["no libre"], encoding="utf-8")
            _, e = correr(raiz4, sal4, [1], estricto=True)
            if not any("ficha" in x or "Frase" in x for x in e):
                rotas4.append((nombre, e))
        print(f"                      seis exenciones malas de la Orden 143: sin rechazar {len(rotas4)} (esperado 0)")
        if rotas4:
            fallos.append(f"invariante: estas exenciones malas de la 143 debían rechazarse: {rotas4}")
    with tempfile.TemporaryDirectory() as d2:
        raiz2, sal2 = _libro_de_mentira(d2, capitulo2, "hoja 79,40 %\n")
        ruta = raiz2 / "notas" / "reforma-nov" / FICHERO_EXENTAS
        ruta.write_text("# x\n\n" + buenas, encoding="utf-8")
        _, ok = correr(raiz2, sal2, [1], estricto=True)
        print(f"                      redondeo, suma a la vista y fuente, bien escritos: fallos {len(ok)} (esperado 0)")
        if ok:
            fallos.append(f"invariante: las exenciones buenas de redondeo, suma y fuente debían valer; salió {ok}")
        rotas = []
        for nombre, (linea, quiero) in malas.items():
            ruta.write_text("# x\n\n" + linea, encoding="utf-8")
            _, e = correr(raiz2, sal2, [1], estricto=True)
            if not any(quiero in x for x in e):
                rotas.append((nombre, e))
        print(f"                      cinco exenciones malas de los motivos nuevos: sin rechazar {len(rotas)} (esperado 0)")
        if rotas:
            fallos.append(f"invariante: estas exenciones malas debían rechazarse: {rotas}")
    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Que cada número de la prosa salga de una salida guardada.")
    ap.add_argument("raiz", nargs="?", help="libro-ia-libro/")
    ap.add_argument("salidas", nargs="?", help="libro-ia-publico/datos/salidas/")
    ap.add_argument("--capitulos", help="capítulos a verificar, separados por comas (por defecto, los reformados)")
    ap.add_argument("--recuento", action="store_true", help="contar los sin salida de todos los capítulos, sin suspender")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    assert a.raiz and a.salidas, "Se esperaban la raíz del libro y la carpeta de salidas: RAIZ SALIDAS"
    if a.recuento:
        tl = cargar_texto_libro(a.raiz)
        filas, fallos = correr(a.raiz, a.salidas, sorted(ficheros_por_capitulo(tl, a.raiz)), estricto=False)
        imprimir_recuento(filas)
        sys.exit(0)
    capitulos = ([int(x) for x in a.capitulos.split(",")] if a.capitulos else sorted(capitulos_reformados(a.raiz)))
    if not capitulos:
        print("PASA: ningún capítulo reformado todavía (notas/reforma-nov/capitulos-reformados.md vacío o ausente).")
        sys.exit(0)
    filas, fallos = correr(a.raiz, a.salidas, capitulos, estricto=True)
    for cap, n, cuenta, clase, ninguna, nombres, derivadas in filas:
        print(f"capítulo {cap}: {n} números; con salida {cuenta.get('con salida', 0)}, exentos {cuenta.get('exento', 0)} (con fuente no libre, que un programa no puede comprobar: {cuenta.get('no_libre', 0)}), "
              f"del dos al nueve con letra (no se comprueban) {cuenta.get('letra_pequena', 0)}, «de cada cien» "
              f"{cuenta.get('unidad', 0)}, sin salida {cuenta.get('sin salida', 0)}; salidas: {', '.join(nombres)}")
    if fallos:
        for f in fallos:
            print("  -", f)
        print(f"\nFALLA: {len(fallos)} números o exenciones sin resolver.")
        sys.exit(1)
    print("PASA: cada número de la prosa sale de una salida guardada, o está exento con su motivo.")


if __name__ == "__main__":
    main()
