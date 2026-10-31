#!/usr/bin/env python3
"""
Capítulo 14 — muchas vueltas seguidas, como trabaja un asistente de verdad: un cuaderno de programación
y una carpeta de ficheros.

`muchas_vueltas.py` dio a la máquina un intérprete que solo devolvía lo que el programa imprimía. Los
asistentes de hoy no funcionan así: su herramienta de código es un cuaderno. Guarda las variables de
una vuelta a la siguiente y, si la última línea del programa es una expresión, devuelve también su
valor aunque no lleve `print`. La máquina de aquel experimento escribió así (`len(coincidencias)` al
final) y no le volvió nada: el 4 de 8 lo decidió el montaje, no ella. Aquí se rehace con un cuaderno de
verdad y con un trabajo de los que se hacen con ellos: mirar una carpeta de ficheros de medidas,
averiguar qué hay dentro y sacar una cuenta.

Mismo modelo (el de 32.000 millones, adiestrado y comprimido), mismo formato de conversación con
herramientas, siempre el trozo más probable, hasta 12 vueltas y 800 trozos por respuesta. La
herramienta es una sola, `ejecutar_python`. Un único proceso de Python (`python3 -I`) por tarea,
que guarda las variables; lo que vuelve es lo que se imprime y, después, el valor de la última línea
(su `repr`) si es una expresión y no es `None`; los errores, con su `Traceback` (y la sesión sigue); 30
segundos por vuelta y 2.000 caracteres de salida como mucho, con aviso. Si una vuelta pasa de 30 s o
el proceso muere, se arranca otro limpio y las variables anteriores se pierden (y se le dice). Cada
tarea trabaja en una carpeta temporal nueva con solo los ocho ficheros de CSV, copiados con sus
nombres. El `python3` de la herramienta es el de Homebrew, sin pandas ni numpy: solo la biblioteca
estándar (`csv`, `statistics`…), y la máquina no puede instalar nada.

Cinco tareas (TAREAS), cada una una conversación aparte. Las respuestas buenas las calcula este
programa con dos implementaciones independientes (módulo `csv` y lectura a mano, ambas con
`decimal`), antes de cargar el modelo.

Cómo se decide si acierta (fijado antes de ejecutar):
  - Tareas 1, 2 y 4: el último número de la respuesta final coincide con el bueno con la precisión
    que da la máquina. Con menos decimales que el bueno, vale si redondeando el bueno (medio hacia
    arriba) a esos decimales sale lo mismo; con cero decimales, si coincide la parte entera. Un
    número como «3.369» puede ser decimal o de miles: se prueban las dos lecturas y vale si alguna
    acierta.
  - Tarea 3: el texto de la tarea buena está en la respuesta final (sin distinguir mayúsculas) y no
    está el de ninguna de las otras cinco.
  - Tarea 5: el último número es exacto (con puntos o espacios de miles).
  - «Las seis tareas» de la 3 son las seis de crecer.csv; «ninguna otra» son las otras cinco.
  - «Mediciones del procesador» (tarea 2) son los ficheros en_serie_barridos_cpu; los de la tarjeta
    gráfica (en_serie_barridos) están en la carpeta para que la máquina tenga que distinguirlos.

Una «vuelta» es un programa que se ejecuta. Un programa «da error» si el cuaderno recoge una
excepción, se pasa de tiempo o muere el proceso. «Se corrige» es: dio error al menos una vez y, aun
así, la respuesta final acierta. Una orden mal escrita no se ejecuta, no cuenta como respuesta y se
anota. Si una tarea resulta ambigua, se queda con su resultado.

Lo que hizo la máquina en cada vuelta (última columna de la tabla) lo escribe este programa a partir
del texto del programa de la vuelta, con una regla simple y en este orden (que_hace): si el programa
pide los ficheros de la carpeta (`listdir`, `glob`, `scandir`, `iterdir`, `walk`), «lista la
carpeta»; si usa `max`, `min`, `median`, `mean`, `sum`, `sorted`, `count` o `groupby`, «calcula»; si
solo abre o lee (`open`, `read_csv`, `csv.`, `read`), «lee los ficheros»; si no encaja con nada, el
programa recortado a una línea. Es una etiqueta de lectura rápida, no un análisis del programa.

La muestra, con la regla fijada antes de lanzar: la primera tarea, en el orden de la lista, que use
al menos tres vueltas; si ninguna llega a tres, la que más use, y la primera si empatan. Se imprime
entera, vuelta a vuelta.

Se ejecuta una sola vez. No se escriben aquí tiempos de las tareas.

Uso (desde codigo/, con HF_HUB_OFFLINE=1 para no descargar nada):
    python muchas_vueltas_carpeta.py --selftest
    python muchas_vueltas_carpeta.py > ../datos/salidas/muchas_vueltas_carpeta.txt
Las cinco conversaciones completas quedan en ../datos/salidas/muchas_vueltas_carpeta/tarea_N.txt.
"""

# ======================= CONSTANTES =======================

CARPETA_DATOS = "../datos/salidas"
FICHEROS = ["cuantos_hilos-A.csv", "cuantos_hilos-B.csv", "en_serie_barridos-A.csv",
            "en_serie_barridos-B.csv", "en_serie_barridos_cpu-A.csv", "en_serie_barridos_cpu-B.csv",
            "crecer.csv", "editar_contexto.csv"]
CARPETA_CONVERSACIONES = "../datos/salidas/muchas_vueltas_carpeta"

HERRAMIENTA = {
    "type": "function",
    "function": {
        "name": "ejecutar_python",
        "description": ("Runs Python code in a persistent session, like a notebook, and returns "
                        "what it prints and the value of the last expression. The working "
                        "directory contains CSV files with measurements."),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "The Python 3 code to run."},
            },
            "required": ["code"],
        },
    },
}

FINAL = " Los ficheros están en la carpeta de trabajo. Contesta con el resultado."

# (clave, rótulo de la tabla, enunciado sin el final común, cómo se comprueba)
TAREAS = [
    (1, "mayor ventaja, según los hilos",
     "En las mediciones hechas con distinto número de hilos, ¿cuál es la mayor ventaja que "
     "aparece, contando los dos ficheros (A y B)?", "decimal"),
    (2, "mediana de la ventaja en el procesador",
     "En las mediciones del procesador (no las de la tarjeta gráfica), ¿cuál es la mediana de la "
     "ventaja con el modelo de ancho 256 y textos de longitud 1024, juntando los ficheros A y B? "
     "Si hay un número par de valores, la mediana es la media de los dos del centro.", "decimal"),
    (3, "tarea que ningún modelo acierta",
     "En los resultados por tamaño de modelo, ¿qué tarea no acierta ningún modelo ni una sola "
     "vez?", "texto"),
    (4, "tiempo medio de «solo añadir»",
     "En las mediciones de editar el contexto, ¿cuál es el tiempo medio, en milisegundos, de la "
     "condición «solo añadir» con el modelo de 7.000 millones sin comprimir?", "decimal"),
    (5, "filas de todos los CSV",
     "¿Cuántas filas tienen en total todos los CSV de la carpeta, sin contar las cabeceras?",
     "entero"),
]

VUELTAS_MUESTRA = 3                 # la muestra: la primera tarea con al menos tantas vueltas
DECIMALES_TABLA = 4                 # decimales con que se enseña una respuesta buena decimal

# ==========================================================

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import re
import select
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from decimal import ROUND_FLOOR, ROUND_HALF_UP, Decimal

from formato import coma, miles, muestra_editorial, tabla_editorial
from muchas_vueltas import (ENTORNO, MAX_NUEVOS, MAX_SALIDA, MAX_VUELTAS, PYTHON, TIEMPO_PROGRAMA,
                            generador, leer_orden, lineas_de, orden, plural, resolver, si_no)
from ordenes_escritas import MODELO_GRANDE, cargar, ultimo_numero

MARCA = b"\x00RESP\x00"

# --------------------------- el cuaderno ---------------------------

# El proceso que guarda las variables. Lee una petición por línea (JSON con «code»), ejecuta el
# programa en un espacio de nombres que dura toda la tarea y contesta con una línea que empieza por
# la marca: {"salida": lo que imprimió y el valor de la última línea, "error": si hubo excepción}.
DRIVER = r'''
import ast, io, json, linecache, sys, traceback
MARCA = chr(0) + "RESP" + chr(0)
canal, entrada = sys.stdout, sys.stdin
espacio = {"__name__": "__main__"}
n = 0
while True:
    linea = entrada.readline()
    if not linea:
        break
    codigo = json.loads(linea)["code"]
    n += 1
    nombre = "<cell %d>" % n
    linecache.cache[nombre] = (len(codigo), None, codigo.splitlines(True), nombre)
    buf = io.StringIO()
    error = False
    guardado = sys.stdout, sys.stderr, sys.stdin
    sys.stdout = sys.stderr = buf
    sys.stdin = io.StringIO("")
    try:
        arbol = ast.parse(codigo, nombre, "exec")
        ultima = None
        if arbol.body and isinstance(arbol.body[-1], ast.Expr):
            ultima = ast.Expression(arbol.body.pop().value)
            ast.fix_missing_locations(ultima)
        exec(compile(arbol, nombre, "exec"), espacio)
        if ultima is not None:
            valor = eval(compile(ultima, nombre, "eval"), espacio)
            if valor is not None:
                print(repr(valor))
    except BaseException as e:
        error = True
        buf.write("Traceback (most recent call last):\n")
        buf.write("".join(traceback.format_tb(e.__traceback__.tb_next)))
        buf.write("".join(traceback.format_exception_only(type(e), e)))
    finally:
        sys.stdout, sys.stderr, sys.stdin = guardado
    canal.write(MARCA + json.dumps({"salida": buf.getvalue(), "error": error}) + "\n")
    canal.flush()
'''


class Cuaderno:
    """Una sesión de Python que guarda las variables, en una carpeta temporal con los ocho CSV."""

    def __init__(self):
        self.carpeta = tempfile.mkdtemp(prefix="cuaderno_")
        for f in FICHEROS:
            shutil.copyfile(os.path.join(CARPETA_DATOS, f), os.path.join(self.carpeta, f))
        self.proceso = None
        self._arrancar()

    def _arrancar(self):
        self.proceso = subprocess.Popen([PYTHON, "-I", "-c", DRIVER], cwd=self.carpeta, env=ENTORNO,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT)
        self.pendiente = b""

    def _matar(self):
        self.proceso.kill()
        self.proceso.wait()
        self.proceso.stdin.close()
        self.proceso.stdout.close()

    def _reiniciar(self, motivo):
        self._matar()
        self._arrancar()
        return (f"Error: {motivo}; la sesión se ha reiniciado y las variables anteriores se han "
                f"perdido."), True

    def ejecutar(self, codigo, tiempo=TIEMPO_PROGRAMA):
        """Ejecuta una vuelta. Devuelve (lo que vuelve a la máquina, si hubo error)."""
        limite = time.monotonic() + tiempo
        try:
            self.proceso.stdin.write((json.dumps({"code": codigo}) + "\n").encode("utf-8"))
            self.proceso.stdin.flush()
        except BrokenPipeError:
            salida, error = self._reiniciar("el proceso de Python había terminado")
            return self._recortar(salida), error
        fd = self.proceso.stdout.fileno()
        while True:
            i = self.pendiente.find(MARCA)
            j = self.pendiente.find(b"\n", i) if i >= 0 else -1
            if j >= 0:
                d = json.loads(self.pendiente[i + len(MARCA):j].decode("utf-8"))
                self.pendiente = self.pendiente[j + 1:]
                return self._recortar(d["salida"]), d["error"]
            resto = limite - time.monotonic()
            if resto <= 0 or not select.select([fd], [], [], resto)[0]:
                salida, error = self._reiniciar(f"el programa tardó más de {tiempo} s y se detuvo")
                return self._recortar(salida), error
            trozo = os.read(fd, 65536)
            if not trozo:
                salida, error = self._reiniciar("el proceso de Python terminó")
                return self._recortar(salida), error
            self.pendiente += trozo

    @staticmethod
    def _recortar(salida):
        if len(salida) > MAX_SALIDA:
            n = len(salida)
            salida = (salida[:MAX_SALIDA] + f"\n[Salida recortada: tenía {miles(n)} caracteres; "
                      f"se devuelven los primeros {miles(MAX_SALIDA)}.]")
        return salida

    def cerrar(self):
        self._matar()
        shutil.rmtree(self.carpeta)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cerrar()


# --------------------------- las respuestas buenas ---------------------------

def leer_a(nombre):
    """Un CSV con el módulo csv (A): una lista de diccionarios."""
    with open(os.path.join(CARPETA_DATOS, nombre), encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def leer_b(nombre):
    """El mismo CSV a mano (B): por líneas, cortando por las comas. Revienta si hay comillas."""
    with open(os.path.join(CARPETA_DATOS, nombre), encoding="utf-8", newline="") as fh:
        texto = fh.read()
    assert '"' not in texto, f"Se esperaba un CSV sin comillas en {nombre}"
    lineas = [l for l in texto.replace("\r\n", "\n").split("\n") if l != ""]
    cabecera = lineas[0].split(",")
    filas = []
    for l in lineas[1:]:
        celdas = l.split(",")
        assert len(celdas) == len(cabecera), f"Se esperaba una celda por columna en {nombre}: {l!r}"
        filas.append(dict(zip(cabecera, celdas)))
    return filas


def respuestas_buenas():
    """{tarea: respuesta buena} (Decimal, texto o entero), cada una de dos maneras que coinciden."""
    buenas = {}

    # 1. mayor ventaja en los dos ficheros de hilos
    a = max(Decimal(f["ventaja"]) for n in FICHEROS[0:2] for f in leer_a(n))
    b = None
    for n in FICHEROS[0:2]:
        for f in leer_b(n):
            v = Decimal(f["ventaja"])
            b = v if b is None or v > b else b
    assert a == b, f"Tarea 1: las dos cuentas no coinciden: {a} y {b}"
    buenas[1] = a

    # 2. mediana de la ventaja, procesador, ancho 256, longitud 1024, A y B juntos
    va = [Decimal(f["ventaja"]) for n in FICHEROS[4:6] for f in leer_a(n)
          if f["ancho"] == "256" and f["longitud"] == "1024"]
    vb = sorted(Decimal(f["ventaja"]) for n in FICHEROS[4:6] for f in leer_b(n)
                if int(f["ancho"]) == 256 and int(f["longitud"]) == 1024)
    assert len(va) == len(vb) and len(va) % 2 == 0, "Se esperaba un número par de valores"
    a = statistics.median(va)
    b = (vb[len(vb) // 2 - 1] + vb[len(vb) // 2]) / 2
    assert a == b, f"Tarea 2: las dos cuentas no coinciden: {a} y {b}"
    buenas[2] = a

    # 3. la tarea de crecer.csv que ningún modelo acierta
    ceros = {}
    for f in leer_a("crecer.csv"):
        ceros.setdefault(f["tarea"], []).append(Decimal(f["acierto"]))
    a = [t for t, v in ceros.items() if max(v) == 0]
    maximos = {}
    for f in leer_b("crecer.csv"):
        v = Decimal(f["acierto"])
        maximos[f["tarea"]] = max(v, maximos.get(f["tarea"], v))
    b = [t for t, v in maximos.items() if v == 0]
    assert a == b and len(a) == 1, f"Tarea 3: se esperaba una sola tarea sin aciertos: {a} y {b}"
    assert len(ceros) == 6, "Se esperaban seis tareas en crecer.csv"
    buenas[3] = a[0]

    # 4. tiempo medio de «solo añadir», 7B sin comprimir
    ta = [Decimal(f["tiempo_ms"]) for f in leer_a("editar_contexto.csv")
          if f["condicion"] == "solo añadir" and f["modelo"] == "7B sin comprimir"]
    tb, suma = 0, Decimal(0)
    for f in leer_b("editar_contexto.csv"):
        if f["condicion"] == "solo añadir" and f["modelo"] == "7B sin comprimir":
            suma += Decimal(f["tiempo_ms"])
            tb += 1
    a, b = statistics.mean(ta), suma / tb
    assert len(ta) == tb and abs(a - b) < Decimal("1e-20"), (
        f"Tarea 4: las dos cuentas no coinciden: {a} y {b}")
    buenas[4] = a

    # 5. filas de todos los CSV, sin cabeceras
    a = sum(len(leer_a(n)) for n in FICHEROS)
    b = 0
    for n in FICHEROS:
        with open(os.path.join(CARPETA_DATOS, n), "rb") as fh:
            b += len([l for l in fh.read().split(b"\n") if l.strip()]) - 1
    assert a == b, f"Tarea 5: las dos cuentas no coinciden: {a} y {b}"
    buenas[5] = a

    assert set(buenas) == {t[0] for t in TAREAS}, "Se esperaba una respuesta buena por tarea"
    return buenas


# --------------------------- cuándo acierta ---------------------------

NUMERO = re.compile(r"\d{1,3}(?:[   .,]\d{3})+(?:[.,]\d+)?(?!\d)|\d+(?:[.,]\d+)?")


def ultimo_token(texto):
    """El último número del texto, tal como está escrito (None si no hay)."""
    ms = NUMERO.findall(texto)
    return ms[-1] if ms else None


def lecturas(token):
    """Las maneras de leer un número escrito: [(valor, decimales escritos)]. Los espacios son
    miles; con punto y coma, el último es el decimal; un solo separador seguido de tres cifras
    (3.369) es ambiguo y da las dos lecturas."""
    t = re.sub(r"[   ]", "", token)
    puntos, comas = t.count("."), t.count(",")
    if puntos and comas:
        decimal = "." if t.rfind(".") > t.rfind(",") else ","
        miles_ = "," if decimal == "." else "."
        ent, dec = t.replace(miles_, "").split(decimal)
        return [(Decimal(ent + "." + dec), len(dec))]
    if puntos + comas == 0:
        return [(Decimal(t), 0)]
    sep = "." if puntos else ","
    if puntos + comas > 1:
        return [(Decimal(t.replace(sep, "")), 0)]
    ent, dec = t.split(sep)
    como_decimal = (Decimal(ent + "." + dec), len(dec))
    if len(dec) == 3 and 1 <= len(ent) <= 3 and not ent.startswith("0"):
        return [como_decimal, (Decimal(ent + dec), 0)]
    return [como_decimal]


def coincide(valor, decimales, bueno):
    """Si un número escrito con `decimales` decimales vale por el bueno: redondeado a esos decimales
    (medio hacia arriba) o, sin decimales, por la parte entera."""
    if decimales == 0:
        return valor == bueno.to_integral_value(ROUND_FLOOR)
    return valor == bueno.quantize(Decimal(1).scaleb(-decimales), rounding=ROUND_HALF_UP)


def acierta(tipo, buena, final, nombres=()):
    """Si la respuesta final acierta. `nombres` son los nombres de las seis tareas (para la 3)."""
    if final is None:
        return False
    if tipo == "decimal":
        token = ultimo_token(final)
        return token is not None and any(coincide(v, d, buena) for v, d in lecturas(token))
    if tipo == "entero":
        return ultimo_numero(final) == buena
    f = final.casefold()
    return buena.casefold() in f and not any(o.casefold() in f for o in nombres if o != buena)


def nombres_de_tareas():
    return sorted({f["tarea"] for f in leer_a("crecer.csv")})


def evaluar(r, tipo, buena):
    r["acierta"] = acierta(tipo, buena, r["final"], nombres_de_tareas())
    r["se_corrige"] = r["errores"] > 0 and r["acierta"]
    return r


# --------------------------- qué hace cada vuelta ---------------------------

def que_hace(codigo):
    """Una etiqueta de lectura rápida para un programa, con una regla simple y en este orden."""
    c = codigo.lower()
    if any(p in c for p in ("listdir", "glob", "scandir", "iterdir", "walk")):
        return "lista la carpeta"
    if any(p in c for p in ("max(", "min(", "median", "mean", "sum(", "sorted(", "count",
                            "groupby")):
        return "calcula"
    if any(p in c for p in ("open(", "read_csv", "csv.", "read")):
        return "lee los ficheros"
    una = " ".join(codigo.split())
    return una if len(una) <= 40 else una[:37] + "..."


def vueltas_dichas(r):
    """«lista la carpeta; calcula (error); calcula» a partir del rastro."""
    out, errores = [], [it[2] for it in r["rastro"] if it[0] == "vuelve"]
    programas = [it[1] for it in r["rastro"] if it[0] == "programa"]
    for codigo, error in zip(programas, errores):
        out.append(que_hace(codigo) + (" (error)" if error else ""))
    return "; ".join(out) if out else "ninguna"


# --------------------------- selftest ---------------------------

def selftest():
    ok = True
    buenas = respuestas_buenas()
    tipos = {t[0]: t[3] for t in TAREAS}
    enunciados = {t[0]: t[2] + FINAL for t in TAREAS}

    # [1] test nulo: una máquina de mentira que nunca usa la herramienta y contesta mal da 0
    #     aciertos (y 0 vueltas, y el cuaderno no llega a abrirse).
    llamadas = []

    def espia(codigo):
        llamadas.append(codigo)
        return "", False

    def nunca(mensajes):
        return "No lo sé: creo que es 7."

    rs = [evaluar(resolver(nunca, espia, enunciados[c]), tipos[c], buenas[c]) for c in buenas]
    aciertos, vueltas = sum(r["acierta"] for r in rs), sum(r["vueltas"] for r in rs)
    p1 = aciertos == 0 and vueltas == 0 and not llamadas
    ok &= p1
    print(f"[1] test nulo         máquina que nunca usa la herramienta y contesta mal, en las "
          f"{len(rs)} tareas: {aciertos} aciertos, {vueltas} vueltas, {len(llamadas)} programas "
          f"ejecutados: {'bien' if p1 else 'MAL'}")

    # [2] señal implantada: una máquina de mentira que en la vuelta 1 lista la carpeta, en la 2
    #     escribe un programa con error, en la 3 lo corrige (usando una variable de la vuelta 1 y
    #     acabando en una expresión sin print) y luego contesta lo que le vuelve. Tiene que dar:
    #     acierta, 3 vueltas, 1 error, «se corrige».
    v1 = "import os\narchivos = sorted(os.listdir('.'))\narchivos"
    v2 = "len(archivos) + variable_que_no_existe"
    v3 = ("import csv\nsum(len(list(csv.reader(open(a, encoding='utf-8')))) - 1 "
          "for a in archivos)")

    def corrige(mensajes):
        n = sum(m["role"] == "tool" for m in mensajes)
        if n < 3:
            return orden((v1, v2, v3)[n])
        return f"Entre todos los CSV hay {mensajes[-1]['content'].strip()} filas."

    with Cuaderno() as cuaderno:
        r = evaluar(resolver(corrige, cuaderno.ejecutar, enunciados[5]), tipos[5], buenas[5])
    p2 = (r["acierta"] and r["vueltas"] == 3 and r["errores"] == 1 and r["se_corrige"]
          and r["primer_error"] == 2 and r["siguio"])
    # El cuaderno guarda una variable de la vuelta 1 a la 2 y devuelve la última línea sin print.
    with Cuaderno() as cuaderno:
        cuaderno.ejecutar("x = 41")
        guarda, e_guarda = cuaderno.ejecutar("x + 1")
        mezcla, _ = cuaderno.ejecutar("print('a')\n3")
        nada, _ = cuaderno.ejecutar("None")
        texto_v, _ = cuaderno.ejecutar("'abc'")
    cuaderno_bien = (guarda == "42\n" and not e_guarda and mezcla == "a\n3\n" and nada == ""
                     and texto_v == "'abc'\n")
    p2 = p2 and cuaderno_bien
    ok &= p2
    print(f"[2] señal implantada  lista la carpeta, error, corrección y respuesta: "
          f"acierta={r['acierta']}, vueltas={r['vueltas']}, errores={r['errores']}, "
          f"se corrige={r['se_corrige']}; el cuaderno guarda variables y devuelve la última "
          f"línea sin print: {cuaderno_bien}: {'bien' if p2 else 'MAL'}")

    # [3] invariante: las dos implementaciones de las respuestas buenas coinciden en las cinco
    #     tareas (respuestas_buenas() revienta si no); el cuaderno cumple lo que dice (solo los
    #     ocho ficheros, el error sale con su Traceback y la sesión sigue, el tiempo agotado
    #     reinicia, la salida se recorta con aviso, la plantilla lleva la herramienta); y las
    #     reglas de «acierta» y de «qué hace cada vuelta» hacen lo que dicen.
    with Cuaderno() as cuaderno:
        lista, _ = cuaderno.ejecutar("import os\nsorted(os.listdir('.'))")
        cuaderno.ejecutar("y = 5")
        traza, e_traza = cuaderno.ejecutar("y + sin_definir")
        sigue, _ = cuaderno.ejecutar("y")
        sintaxis, e_sintaxis = cuaderno.ejecutar("def f(:\n  pass")
        tarde, e_tarde = cuaderno.ejecutar("while True: pass", tiempo=2)
        perdida, _ = cuaderno.ejecutar("y")
        largo, _ = cuaderno.ejecutar("print('x' * 5000)")
    solo_los_ocho = lista.strip() == repr(sorted(FICHEROS))
    errores_bien = (e_traza and "Traceback" in traza and "NameError" in traza and sigue == "5\n"
                    and e_sintaxis and "SyntaxError" in sintaxis
                    and e_tarde and "tardó más de 2 s" in tarde and "NameError" in perdida)
    recorta_bien = (len(largo) < 5000 and "Salida recortada" in largo
                    and largo.startswith("x" * MAX_SALIDA))
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(MODELO_GRANDE[1], local_files_only=True)
    plantilla = tk.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False,
                                       add_generation_prompt=True, tools=[HERRAMIENTA])
    plantilla_bien = ('"ejecutar_python"' in plantilla and "persistent session" in plantilla
                      and len(tk.encode("<tool_call>", add_special_tokens=False)) == 1)
    b = Decimal("6.6989")
    casos = [("6.6989", b, True), ("6,7", b, True), ("6.70", b, True), ("6.69", b, False),
             ("6", b, True), ("7", b, False), ("unos 6,6989 s", b, True),
             ("3.369,21", Decimal("3369.2105"), True), ("3,369.2", Decimal("3369.2105"), True),
             ("3 369,2", Decimal("3369.2105"), True), ("3.369", Decimal("3369.2105"), True),
             ("3.5", Decimal("3369.2105"), False), ("sin número", b, False)]
    reglas_bien = all(acierta("decimal", bueno, texto) == esperado for texto, bueno, esperado in casos)
    n6 = ["datos del mundo", "plurales", "razonar sobre una frase", "seguir un patrón inventado",
          "sumar dos cifras", "traducir del inglés"]
    reglas_bien &= (acierta("texto", n6[3], "Es «Seguir un patrón inventado».", n6)
                    and not acierta("texto", n6[3], "Son seguir un patrón inventado y plurales.", n6)
                    and not acierta("texto", n6[3], "Ninguna.", n6)
                    and acierta("entero", 658, "Hay 658 filas.") and acierta("entero", 658, "658")
                    and not acierta("entero", 658, "Hay 657 filas."))
    etiquetas_bien = [que_hace("import os\nprint(os.listdir('.'))"), que_hace("x = max(v)"),
                      que_hace("open('a.csv').readline()"), que_hace("x = 1")] == [
        "lista la carpeta", "calcula", "lee los ficheros", "x = 1"]
    p3 = (solo_los_ocho and errores_bien and recorta_bien and plantilla_bien and reglas_bien
          and etiquetas_bien)
    ok &= p3
    print(f"[3] invariante        las dos implementaciones coinciden en las {len(buenas)} tareas; "
          f"cuaderno con solo los ocho ficheros: {solo_los_ocho}; error con Traceback y sesión que "
          f"sigue, tiempo agotado que reinicia: {errores_bien}; salida recortada con aviso: "
          f"{recorta_bien}; plantilla con la herramienta: {plantilla_bien}; reglas de «acierta»: "
          f"{reglas_bien}; etiquetas de las vueltas: {etiquetas_bien}: {'bien' if p3 else 'MAL'}")

    print()
    print("SELFTEST: las tres pruebas pasan." if ok else "SELFTEST: FALLA.")
    return ok


# --------------------------- la salida ---------------------------

def bueno_dicho(tipo, buena):
    if tipo == "decimal":
        return coma(float(buena), DECIMALES_TABLA)
    return miles(buena) if tipo == "entero" else buena


def contesto_dicho(r, tipo):
    """Lo que se lee en la respuesta final, tal como lo escribió la máquina."""
    if r["final"] is None:
        return "sin respuesta"
    if tipo == "texto":
        nombrados = [t for t in nombres_de_tareas() if t.casefold() in r["final"].casefold()]
        return " y ".join(nombrados) if nombrados else "ninguna de las seis"
    return ultimo_token(r["final"]) or "ningún número"


def escribir_conversacion(r, clave, rotulo, buena, tipo):
    """La conversación entera de una tarea, tal cual, a su fichero."""
    os.makedirs(CARPETA_CONVERSACIONES, exist_ok=True)
    enunciado = next(t[2] for t in TAREAS if t[0] == clave) + FINAL
    out = [f"Tarea {clave}: {rotulo}", f"Respuesta buena: {buena}", "", "Enunciado:", enunciado, ""]
    vuelta = 0
    for item in r["rastro"]:
        if item[0] == "escribe":
            out += ["----- La máquina escribe -----", item[1], ""]
        elif item[0] == "programa":
            vuelta += 1
            out += [f"----- Programa que se ejecuta (vuelta {vuelta}) -----", item[1], ""]
        else:
            out += [f"----- Lo que vuelve a la máquina "
                    f"({'con error' if item[2] else 'sin error'}) -----", item[1], ""]
    out += ["----- Resultado -----",
            f"acierta: {si_no(r['acierta'])}; vueltas: {r['vueltas']}; programas con error: "
            f"{r['errores']}; se corrige: {si_no(r['se_corrige'])}; órdenes mal escritas: "
            f"{r['ordenes_mal']}; respuesta final: "
            f"{'ninguna' if r['final'] is None else 'la última escritura'}"]
    with open(os.path.join(CARPETA_CONVERSACIONES, f"tarea_{clave}.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)

    t0 = time.time()
    print(f"máquina: {platform.platform()}; Python {platform.python_version()}")
    print(f"mlx {importlib.metadata.version('mlx')}; mlx-lm {importlib.metadata.version('mlx-lm')}")
    buenas = respuestas_buenas()                       # antes de cargar el modelo
    nombre, repo = MODELO_GRANDE
    modelo, tok, _ = cargar(repo)
    import mlx.core as mx
    print(f"modelo: {nombre} ({repo})")
    print(f"dispositivo de cálculo: {mx.default_device()}")
    version = subprocess.run([PYTHON, "-I", "--version"], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, env=ENTORNO, check=True)
    paquetes = subprocess.run([PYTHON, "-I", "-c",
                               "import importlib.util as u; print(', '.join(p + ': ' + "
                               "('sí' if u.find_spec(p) else 'no') for p in ('pandas', 'numpy')))"],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=ENTORNO,
                              check=True)
    print(f"herramienta: {shutil.which(PYTHON, path=ENTORNO['PATH'])} "
          f"({version.stdout.decode().strip()}), con -I, {TIEMPO_PROGRAMA} s por vuelta; "
          f"{paquetes.stdout.decode().strip()}")
    print()
    gen = generador(modelo, tok, HERRAMIENTA)

    resultados = []
    for clave, rotulo, enunciado, tipo in TAREAS:
        t1 = time.time()
        with Cuaderno() as cuaderno:
            r = resolver(gen, cuaderno.ejecutar, enunciado + FINAL)
        evaluar(r, tipo, buenas[clave])
        r.update(clave=clave, rotulo=rotulo, tipo=tipo, buena=buenas[clave])
        resultados.append(r)
        escribir_conversacion(r, clave, rotulo, buenas[clave], tipo)
        print(f"tarea {clave}: acierta={si_no(r['acierta'])} vueltas={r['vueltas']} "
              f"errores={r['errores']} ({time.time() - t1:.0f} s)", file=sys.stderr, flush=True)
    print(f"tiempo total: {time.time() - t0:.0f} s", file=sys.stderr, flush=True)

    out = informe(resultados)
    print("\n".join(out))


def informe(resultados):
    """La tabla, las respuestas buenas y la muestra, como lista de renglones."""
    out = []
    filas = []
    for r in resultados:
        corrige = si_no(r["se_corrige"]) if r["errores"] else "—"
        filas.append([f"{r['clave']}. {r['rotulo']}", si_no(r["acierta"]), str(r["vueltas"]),
                      str(r["errores"]), corrige, vueltas_dichas(r)])
    aciertos = sum(r["acierta"] for r in resultados)
    notas = ["La máquina trabaja como en un cuaderno de programación: lo que define en una vuelta "
             "sigue ahí en la siguiente, y le vuelve lo que el programa imprime y el valor de su "
             "última línea. Solo tiene ocho ficheros de medidas.",
             "Acierta: su respuesta final trae el resultado bueno. Vueltas: los programas que "
             "escribió y se ejecutaron antes de contestar. Con error: cuántos acabaron en error. "
             "Se corrige: si, habiendo tenido algún error, la respuesta final acierta.",
             "La última columna dice, vuelta por vuelta, qué hacía el programa: «lista la carpeta» "
             "si pide los ficheros, «calcula» si usa máximos, medias, sumas, cuentas u "
             "ordenaciones, «lee los ficheros» si solo los abre; si no encaja, el programa recortado."]
    if any(r["ordenes_mal"] for r in resultados):
        notas.append("Alguna orden salió mal escrita y no se ejecutó: "
                     + ", ".join(f"tarea {r['clave']}" for r in resultados if r["ordenes_mal"])
                     + ".")
    out += tabla_editorial(
        f"Cinco tareas con un cuaderno y una carpeta de ficheros: aciertan {aciertos}",
        ["la tarea", "acierta", "vueltas", "con error", "se corrige", "qué hizo en cada vuelta"],
        filas, "icddci", notas)
    out.append("")

    filas = [[f"{r['clave']}. {r['rotulo']}", bueno_dicho(r["tipo"], r["buena"]),
              contesto_dicho(r, r["tipo"])] for r in resultados]
    out += tabla_editorial(
        "Lo que había que contestar y lo que se lee en la respuesta final",
        ["la tarea", "respuesta buena", "lo que contestó"], filas, "idd",
        [f"La respuesta buena la calcula un programa antes de preguntar, de dos maneras que tienen "
         f"que dar lo mismo; las decimales se enseñan con {DECIMALES_TABLA} decimales. La última "
         f"columna es el último número de la respuesta final de la máquina, tal como lo escribió "
         f"(en la tarea 3, las tareas que nombra)."])
    out.append("")

    muestra = next((r for r in resultados if r["vueltas"] >= VUELTAS_MUESTRA), None)
    if muestra is None:
        mas = max(r["vueltas"] for r in resultados)
        muestra = next(r for r in resultados if r["vueltas"] == mas)
        motivo = (f"ninguna llega a {VUELTAS_MUESTRA} vueltas: la que más usa, con "
                  f"{plural(mas, 'vuelta', 'vueltas')} (la primera, si empatan)")
    else:
        motivo = (f"la primera, en el orden de la lista, con al menos {VUELTAS_MUESTRA} vueltas "
                  f"(usa {muestra['vueltas']})")
    out += [f"muestra elegida: tarea {muestra['clave']}, {motivo}", ""]
    enunciado = next(t[2] for t in TAREAS if t[0] == muestra["clave"]) + FINAL
    lineas = lineas_de(f"Tarea {muestra['clave']}: {enunciado}") + [""]
    vuelta = 0
    for item in muestra["rastro"]:
        if item[0] == "escribe":
            lineas += ["Escribe la máquina:"] + lineas_de(item[1]) + [""]
        elif item[0] == "programa":
            vuelta += 1
            lineas += [f"Se ejecuta este programa (vuelta {vuelta}):"] + lineas_de(item[1]) + [""]
        else:
            # (el salto de línea con que acaba una salida no se enseña; a la máquina sí le llegó)
            lineas += [f"Vuelve a la máquina ({'con error' if item[2] else 'sin error'}):"] \
                + lineas_de(item[1].rstrip("\n") or "(no vuelve nada)") + [""]
    lineas.pop()
    if muestra["errores"]:
        hecho = (f"El programa de la vuelta {muestra['primer_error']} dio error; "
                 f"{'al final acierta' if muestra['acierta'] else 'al final falla'}.")
    else:
        hecho = f"Ningún programa dio error; {'al final acierta' if muestra['acierta'] else 'al final falla'}."
    out += muestra_editorial(
        "Una tarea entera, vuelta a vuelta: lo que escribe la máquina, el programa y lo que vuelve",
        lineas,
        [f"Tarea {muestra['clave']} ({muestra['rotulo']}): "
         f"{plural(muestra['vueltas'], 'vuelta', 'vueltas')} en total, "
         f"{plural(muestra['errores'], 'programa con error', 'programas con error')}. " + hecho])
    return out


if __name__ == "__main__":
    main()
