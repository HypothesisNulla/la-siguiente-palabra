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
MOTIVOS = ("año", "fecha", "referencia", "inventado")
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
        if base in ("año", "fecha"):
            f = re.search(r"fuente:\s*(notas/fuentes/\S+)", motivo)
            if not f or not (Path(raiz) / f.group(1).rstrip(".,;")).exists():
                errores.append(f"exención de {base} sin «fuente: notas/fuentes/…» que exista en {raiz}: {l}")
                continue
        exenciones.append((cap, numero, fragmento, motivo, base, l))
    return exenciones, errores


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
            elif tiene_salida(h, valores):
                h["estado"] = "con salida"
            else:
                exenta = [e for e in exenciones if e[0] == cap and e[1] == h["texto"] and norm(e[2]) in norm(frase)]
                if exenta:
                    e = exenta[0]
                    usadas.add(e[5])
                    ok = e[4] != "inventado" or any(p in parrafo.lower() for p in DECLARA_INVENTADO)
                    h["estado"] = "exento" if ok else "exención sin declarar inventado"
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
        mal = [h for h in numeros if h["estado"] == "exención sin declarar inventado"]
        por_clase = {c: sum(1 for h in sin if h["clase"] == c) for c in ("año", "fecha", "referencia", "cifra")}
        filas.append((cap, len(numeros), cuenta, por_clase, sum(1 for h in sin if h.get("en_ninguna")), nombres, derivadas))
        if estricto:
            for h in sin:
                fallos.append(f"capítulo {cap}, línea {h['linea']}: «{h['texto']}» ({h['clase']}) no está en las salidas "
                              f"de su capítulo; frase: «{h['frase'][:110]}»")
            for h in mal:
                fallos.append(f"capítulo {cap}, línea {h['linea']}: «{h['texto']}» está exento como inventado y su párrafo "
                              f"no lo declara ({'/'.join(DECLARA_INVENTADO)})")
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
    (raiz / "notas" / "fuentes" / "shannon.md").write_text("fuente de prueba\n", encoding="utf-8")
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
        print(f"capítulo {cap}: {n} números; con salida {cuenta.get('con salida', 0)}, exentos {cuenta.get('exento', 0)}, "
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
