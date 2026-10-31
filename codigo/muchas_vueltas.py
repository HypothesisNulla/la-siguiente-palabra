#!/usr/bin/env python3
"""
Capítulo 14 — muchas vueltas seguidas: la misma máquina, con una herramienta, en bucle.

En `ordenes_escritas` la máquina escribía una orden, el programa de alrededor la ejecutaba, y con
el resultado delante la máquina contestaba: una vuelta. Aquí la herramienta es un intérprete de
Python y el bucle puede dar hasta doce vueltas: la máquina escribe un programa, el programa de
alrededor lo ejecuta y le devuelve lo que imprime (o el error), y la máquina sigue escribiendo
hasta que contesta sin orden. Todo sigue siendo la siguiente palabra; lo que cambia es cuántas
veces se repite el ciclo «escribir, ejecutar, volver a escribir» antes de la respuesta final.

El modelo es el de 32.000 millones, adiestrado y comprimido (el de `ordenes_escritas`), con el
mismo formato de conversación con herramientas y cogiendo siempre el trozo más probable: no hay
sorteo. La herramienta es una sola, `ejecutar_python`. Cada programa se ejecuta en una carpeta
temporal nueva que contiene solo una copia de `datos/quijote.txt`, con `python3 -I` y 30 segundos
como máximo; lo que vuelve es la salida y los errores juntos, recortados a 2.000 caracteres (y se
dice si se recortó).

Ocho tareas sobre ese texto (TAREAS, abajo). Las respuestas buenas las calcula este programa con
dos implementaciones independientes que tienen que coincidir, ANTES de cargar el modelo. Una
tarea acierta si el último número de la respuesta final es el bueno (en la 4, si la palabra buena
está en la respuesta final); si se acaban las vueltas sin respuesta, es un fallo.

Cómo se leen los enunciados al calcular las respuestas buenas (fijado antes de ejecutar):
  - «Sancho como palabra entera, con esa mayúscula» (tarea 1): ni antes ni después hay una letra.
  - En las tareas 6 y 8, «aparece «X»» es el número de veces que el texto «X» sale, con esa
    mayúscula, aunque vaya dentro de otra palabra (como `str.count`); el enunciado no pide palabra
    entera más que en la tarea 1.
  - Las líneas se cuentan como las cuenta Python al leer el fichero (CRLF o LF, una línea por
    salto); el fichero acaba con un renglón en blanco, que es la última línea.
  - Cada tarea es una conversación aparte, así que la 5 repite la definición de «palabra» de la 4.

Una «vuelta» es un programa que se ejecuta. Un programa «da error» si su salida lleva «Traceback»
o su código de salida no es 0 (también si se agota el tiempo). «Se corrige» es: dio error al
menos una vez y, aun así, la respuesta final acierta. Una orden mal escrita (no es JSON, no
nombra la herramienta, no trae el programa) no se ejecuta, no cuenta como respuesta y se anota.

La muestra para el libro, con la regla fijada antes de lanzar: la primera tarea, en el orden de
la lista, en la que algún programa da error y la máquina sigue escribiendo; si ninguna da error,
la tarea 1. Se imprime entera, vuelta a vuelta.

Se ejecuta una sola vez y no se cambia una tarea después de ver el resultado. No se escriben aquí
tiempos de pared de las tareas: el que se dice es el de la ejecución entera, en la respuesta.

Corrección del 31 de octubre de 2026 (Orden 115), hecha DESPUÉS de ver el resultado y dicha aquí
para que se sepa: la primera lectura de la tarea 4 («la palabra buena está en la respuesta final»)
dio por buena la respuesta «muy», porque «Quijote» salía en la frase («el texto de "Don Quijote"»).
Ahora la respuesta de la tarea 4 es la última palabra entre comillas de la respuesta final y, si no
hay comillas, la última palabra. Con esa lectura el recuento pasa de 5 a 4 aciertos; el programa
no se volvió a ejecutar: `--regenerar` rehace `muchas_vueltas.txt` a partir de las ocho
conversaciones guardadas, que no cambian (solo la última línea de `tarea_4.txt`, el resumen, que
repite si acierta). Y este experimento se sustituye por `muchas_vueltas_carpeta.py`, porque su
ejecutor devolvía solo lo que el programa imprime, y los asistentes de hoy funcionan como un
cuaderno de programación: devuelven también el valor de la última línea aunque no lleve `print`.
Los cuatro fallos (tareas 1, 4, 5 y 7) fueron exactamente eso: un programa que acababa en una
expresión sin `print`, a la máquina le volvió un texto vacío y contestó un número o una palabra
que no salían de ningún cálculo.

Uso (desde codigo/, con HF_HUB_OFFLINE=1 para no descargar nada):
    python muchas_vueltas.py --selftest
    python muchas_vueltas.py > ../datos/salidas/muchas_vueltas.txt
    python muchas_vueltas.py --regenerar        # sin cargar el modelo: rehace el .txt (Orden 115)
Las ocho conversaciones completas quedan en ../datos/salidas/muchas_vueltas/tarea_N.txt.
"""

# ======================= CONSTANTES =======================

QUIJOTE = "../datos/quijote.txt"
FICHERO = "quijote.txt"             # cómo se llama la copia dentro de la carpeta temporal
CARPETA_CONVERSACIONES = "../datos/salidas/muchas_vueltas"
SALIDA_TXT = "../datos/salidas/muchas_vueltas.txt"

PYTHON = "python3"                  # con -I: sin variables PYTHON*, sin paquetes del usuario
TIEMPO_PROGRAMA = 30                # segundos que puede tardar un programa, como mucho
MAX_SALIDA = 2000                   # caracteres de salida que vuelven a la máquina, como mucho
MAX_VUELTAS = 12                    # programas que se ejecutan por tarea, como mucho
MAX_NUEVOS = 800                    # trozos por respuesta, como mucho

LETRAS = "abcdefghijklmnopqrstuvwxyzáéíóúüñ"        # las letras de una «palabra» (tareas 4 y 5)
PALABRA_LARGA = 6                   # «de más de seis letras»

HERRAMIENTA = {
    "type": "function",
    "function": {
        "name": "ejecutar_python",
        "description": ("Runs a Python program and returns what it prints (stdout and stderr). "
                        "The text of Don Quixote is in the file quijote.txt in the working "
                        "directory."),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "The Python 3 program to run."},
            },
            "required": ["code"],
        },
    },
}

FINAL = (" El texto está en el fichero quijote.txt de la carpeta de trabajo. Úsalo entero, tal "
         "cual. Contesta con el resultado.")
DEFINICION_PALABRA = ("Una palabra es una secuencia seguida de letras (de la a a la z, con á, é, í, "
                      "ó, ú, ü y ñ), sin distinguir mayúsculas.")

# (clave, rótulo de la tabla, enunciado sin el final común, cómo se comprueba)
TAREAS = [
    (1, "«Sancho» como palabra entera",
     "¿Cuántas veces aparece «Sancho» como palabra entera, con esa mayúscula?", "numero"),
    (2, "líneas que empiezan por «Capítulo»",
     "¿Cuántas líneas empiezan exactamente por «Capítulo» (con mayúscula, en el primer carácter "
     "de la línea)?", "numero"),
    (3, "primera línea con «Dulcinea»",
     "¿En qué número de línea aparece por primera vez «Dulcinea»? (La primera línea del fichero "
     "es la 1.)", "numero"),
    (4, "palabra larga más repetida",
     DEFINICION_PALABRA + " ¿Cuál es la palabra de más de seis letras que más veces aparece?",
     "palabra"),
    (5, "palabras distintas",
     DEFINICION_PALABRA + " ¿Cuántas palabras distintas tiene el texto?", "numero"),
    (6, "«Rocinante» en el primer capítulo",
     "¿Cuántas veces aparece «Rocinante» entre la primera línea que empieza por «Capítulo "
     "primero» y la primera que empieza por «Capítulo II», sin contar esas dos líneas?", "numero"),
    (7, "capítulo con más líneas",
     "Los capítulos empiezan en las líneas que empiezan por «Capítulo». Cada uno llega hasta la "
     "línea anterior al siguiente; el último, hasta el final del fichero. ¿En qué número de "
     "línea empieza el capítulo con más líneas?", "numero"),
    (8, "«Dulcinea» en la segunda mitad",
     "Si el fichero tiene N líneas, la segunda mitad son las líneas de N//2 + 1 a N. ¿Cuántas "
     "veces aparece «Dulcinea» en la segunda mitad?", "numero"),
]

# ==========================================================

import argparse
import collections
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time

from formato import miles, muestra_editorial, tabla_editorial
from ordenes_escritas import MODELO_GRANDE, cargar, escribir, renglones, ultimo_numero

ENTORNO = {"PATH": os.environ["PATH"], "LC_ALL": "en_US.UTF-8"}
ORDEN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
RE_LETRAS = re.compile("[" + LETRAS + "]+")


# --------------------------- las respuestas buenas ---------------------------

def leer_texto():
    """El fichero de dos maneras: leído por Python en modo texto (A) y leído en bytes y troceado a
    mano por los saltos de línea (B). Las dos tienen que dar las mismas líneas."""
    with open(QUIJOTE, encoding="utf-8") as fh:
        lineas_a = fh.read().splitlines()
    with open(QUIJOTE, "rb") as fh:
        crudo = fh.read().decode("utf-8")
    lineas_b = crudo.replace("\r\n", "\n").split("\n")
    if lineas_b[-1] == "":
        lineas_b.pop()
    assert "\r" not in "".join(lineas_b), "Se esperaban saltos de línea CRLF o LF; hay otro"
    assert not any(c in crudo for c in "\x0b\x0c\x1c\x1d\x1e\x85  "), (
        "Se esperaba que el fichero no tuviera separadores de línea raros")
    assert lineas_a == lineas_b, "Se esperaba que las dos lecturas del fichero dieran las mismas líneas"
    return lineas_a, lineas_b


def palabras_b(texto):
    """Las palabras de `texto` a mano, letra a letra (B)."""
    out, actual = [], []
    for c in texto:
        for d in c.lower():
            if d in LETRAS:
                actual.append(d)
            elif actual:
                out.append("".join(actual))
                actual = []
    if actual:
        out.append("".join(actual))
    return out


def contar_palabra_entera_b(texto, palabra):
    """Cuántas veces sale `palabra` con ninguna letra pegada delante ni detrás (B)."""
    n, i = 0, 0
    while True:
        i = texto.find(palabra, i)
        if i < 0:
            return n
        antes = texto[i - 1] if i > 0 else " "
        j = i + len(palabra)
        despues = texto[j] if j < len(texto) else " "
        if not antes.isalpha() and not despues.isalpha():
            n += 1
        i += 1


def unico_maximo(cuentas):
    """La clave con la cuenta más alta; revienta si hay empate arriba."""
    orden = sorted(cuentas.items(), key=lambda kv: -kv[1])
    assert orden[0][1] > orden[1][1], (
        f"Se esperaba un máximo sin empate; hay {orden[0]} y {orden[1]}")
    return orden[0][0]


def respuestas_buenas():
    """{tarea: respuesta buena}, cada una calculada de dos maneras que tienen que coincidir."""
    la, lb = leer_texto()
    ta, tb = "\n".join(la), "\n".join(lb)
    n = len(la)
    assert n == len(lb)
    buenas = {}

    # 1. «Sancho» como palabra entera
    a = len(re.findall(r"(?<![^\W\d_])Sancho(?![^\W\d_])", ta))
    b = contar_palabra_entera_b(tb, "Sancho")
    assert a == b, f"Tarea 1: las dos cuentas no coinciden: {a} y {b}"
    buenas[1] = a

    # 2. líneas que empiezan por «Capítulo»
    a = sum(1 for l in la if l.startswith("Capítulo"))
    b = len(re.findall(r"^Capítulo", tb, re.M))
    assert a == b, f"Tarea 2: las dos cuentas no coinciden: {a} y {b}"
    buenas[2] = a

    # 3. primera línea con «Dulcinea»
    a = next(i for i, l in enumerate(la, 1) if "Dulcinea" in l)
    m = re.search("Dulcinea", tb)
    b = tb.count("\n", 0, m.start()) + 1
    assert a == b, f"Tarea 3: las dos cuentas no coinciden: {a} y {b}"
    buenas[3] = a

    # 4 y 5. palabras (letras de LETRAS, sin distinguir mayúsculas)
    pa = collections.Counter(RE_LETRAS.findall(ta.lower()))
    pb = collections.Counter(palabras_b(tb))
    assert pa == pb, "Tarea 4: las dos maneras de partir el texto en palabras no coinciden"
    a = unico_maximo({w: c for w, c in pa.items() if len(w) > PALABRA_LARGA})
    b = unico_maximo({w: c for w, c in pb.items() if len(w) > PALABRA_LARGA})
    assert a == b, f"Tarea 4: las dos cuentas no coinciden: {a} y {b}"
    buenas[4] = a
    a, b = len(set(RE_LETRAS.findall(ta.lower()))), len(set(palabras_b(tb)))
    assert a == b, f"Tarea 5: las dos cuentas no coinciden: {a} y {b}"
    buenas[5] = a

    # 6. «Rocinante» entre «Capítulo primero» y «Capítulo II»
    i1 = next(i for i, l in enumerate(la) if l.startswith("Capítulo primero"))
    i2 = next(i for i, l in enumerate(la) if l.startswith("Capítulo II"))
    assert i1 < i2, "Se esperaba «Capítulo primero» antes que «Capítulo II»"
    a = sum(l.count("Rocinante") for l in la[i1 + 1:i2])
    dentro, b = False, 0
    for l in lb:
        if not dentro:
            dentro = l.startswith("Capítulo primero")
            continue
        if l.startswith("Capítulo II"):
            break
        b += l.count("Rocinante")
    assert a == b, f"Tarea 6: las dos cuentas no coinciden: {a} y {b}"
    buenas[6] = a

    # 7. línea donde empieza el capítulo con más líneas
    inicios = [i for i, l in enumerate(la, 1) if l.startswith("Capítulo")]
    largos = {ini: fin - ini for ini, fin in zip(inicios, inicios[1:] + [n + 1])}
    a = unico_maximo(largos)
    mejor, mejor_ini, ini_actual = 0, None, None
    for i, l in enumerate(lb, 1):
        if l.startswith("Capítulo"):
            if ini_actual is not None and i - ini_actual > mejor:
                mejor, mejor_ini = i - ini_actual, ini_actual
            ini_actual = i
    if n + 1 - ini_actual > mejor:
        mejor, mejor_ini = n + 1 - ini_actual, ini_actual
    assert a == mejor_ini, f"Tarea 7: las dos cuentas no coinciden: {a} y {mejor_ini}"
    buenas[7] = a

    # 8. «Dulcinea» en las líneas N//2 + 1 a N
    a = sum(l.count("Dulcinea") for l in la[n // 2:])
    desplazamiento = sum(len(l) + 1 for l in lb[:n // 2])
    b = tb.count("Dulcinea", desplazamiento)
    assert a == b, f"Tarea 8: las dos cuentas no coinciden: {a} y {b}"
    buenas[8] = a

    assert set(buenas) == {t[0] for t in TAREAS}, "Se esperaba una respuesta buena por tarea"
    return buenas


COMILLAS = re.compile(r'«([^»]*)»|"([^"]*)"|“([^”]*)”')


def palabra_respondida(final):
    """La palabra que da una respuesta final como respuesta (tarea 4): la última palabra entre
    comillas («…», "…" o “…”); si no hay comillas, la última palabra. None si no hay ninguna.
    (Orden 115: antes bastaba con que la palabra buena estuviera en cualquier sitio de la
    respuesta; ver la nota del principio.)"""
    for m in reversed(list(COMILLAS.finditer(final))):
        palabras = RE_LETRAS.findall(next(g for g in m.groups() if g is not None).lower())
        if palabras:
            return palabras[-1]
    palabras = RE_LETRAS.findall(final.lower())
    return palabras[-1] if palabras else None


def acierta(tipo, buena, final):
    """Si la respuesta final acierta. Sin respuesta final (None), no."""
    if final is None:
        return False
    if tipo == "palabra":
        return palabra_respondida(final) == buena
    return ultimo_numero(final) == buena


# --------------------------- la herramienta ---------------------------

def ejecutar_python(codigo, tiempo=TIEMPO_PROGRAMA):
    """Ejecuta `codigo` en una carpeta temporal nueva, con solo una copia del Quijote, con
    `python3 -I` y un tiempo máximo. Devuelve (lo que vuelve a la máquina, si hubo error)."""
    with tempfile.TemporaryDirectory() as carpeta:
        shutil.copyfile(QUIJOTE, os.path.join(carpeta, FICHERO))
        try:
            r = subprocess.run([PYTHON, "-I", "-c", codigo], cwd=carpeta, env=ENTORNO,
                               stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, timeout=tiempo)
            salida = r.stdout.decode("utf-8", errors="replace")
            error = r.returncode != 0 or "Traceback" in salida
        except subprocess.TimeoutExpired as e:
            salida = (e.stdout or b"").decode("utf-8", errors="replace")
            salida += f"\nError: el programa tardó más de {tiempo} s y se detuvo."
            error = True
    if len(salida) > MAX_SALIDA:
        n = len(salida)
        salida = (salida[:MAX_SALIDA] + f"\n[Salida recortada: tenía {miles(n)} caracteres; se "
                  f"devuelven los primeros {miles(MAX_SALIDA)}.]")
    return salida, error


def leer_orden(texto):
    """El programa de una orden bien escrita: (código, None). Si no, (None, motivo)."""
    m = ORDEN.search(texto)
    if not m:
        return None, "no hay orden"
    try:
        d = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None, "la orden no es JSON"
    if not isinstance(d, dict) or d.get("name") != HERRAMIENTA["function"]["name"]:
        return None, "la orden no nombra la herramienta"
    args = d.get("arguments")
    if not isinstance(args, dict) or not isinstance(args.get("code"), str):
        return None, "la orden no trae el programa"
    return args["code"], None


def orden(codigo):
    """El texto de una orden bien escrita con este programa (para las máquinas de mentira)."""
    cuerpo = {"name": HERRAMIENTA["function"]["name"], "arguments": {"code": codigo}}
    return "<tool_call>\n" + json.dumps(cuerpo, ensure_ascii=False) + "\n</tool_call>"


# --------------------------- el bucle ---------------------------

def resolver(generar, ejecutar, enunciado):
    """Una tarea entera. `generar(mensajes)` devuelve lo que escribe la máquina; `ejecutar(código)`,
    (salida, hubo_error). Hasta MAX_VUELTAS programas ejecutados; la escritura siguiente al último
    ya no se ejecuta: si es otra orden, no hay respuesta."""
    mensajes = [{"role": "user", "content": enunciado}]
    rastro, vueltas, errores, ordenes_mal = [], 0, 0, 0
    primer_error, final, siguio = None, None, False
    for escritura in range(MAX_VUELTAS + 1):
        texto = generar(mensajes)
        rastro.append(("escribe", texto))
        if primer_error is not None:
            siguio = True                         # escribe algo después de su primer error
        codigo, motivo = leer_orden(texto)
        if codigo is None:
            if "<tool_call>" in texto:
                ordenes_mal += 1                  # quiso abrir una orden y no está bien escrita
            else:
                final = texto
            break
        if escritura == MAX_VUELTAS:
            break
        vueltas += 1
        salida, error = ejecutar(codigo)
        rastro += [("programa", codigo), ("vuelve", salida, error)]
        if error:
            errores += 1
            if primer_error is None:
                primer_error = vueltas
        mensajes += [{"role": "assistant", "content": texto}, {"role": "tool", "content": salida}]
    return {"final": final, "vueltas": vueltas, "errores": errores, "primer_error": primer_error,
            "siguio": siguio, "ordenes_mal": ordenes_mal, "rastro": rastro, "mensajes": mensajes}


def generador(modelo, tok, herramienta=HERRAMIENTA):
    def generar(mensajes):
        texto = tok.apply_chat_template(mensajes, tokenize=False, add_generation_prompt=True,
                                        tools=[herramienta])
        assert f'"{herramienta["function"]["name"]}"' in texto and "<tool_call>" in texto, (
            "Se esperaba la descripción de la herramienta en el enunciado; no está:\n" + texto)
        return escribir(modelo, tok, texto, MAX_NUEVOS)[0]
    return generar


def enunciado_de(clave):
    return next(t[2] for t in TAREAS if t[0] == clave) + FINAL


def evaluar(r, tipo, buena):
    """Añade al resultado de resolver si acierta y si se corrige."""
    r["acierta"] = acierta(tipo, buena, r["final"])
    r["se_corrige"] = r["errores"] > 0 and r["acierta"]
    return r


# --------------------------- selftest ---------------------------

def selftest():
    ok = True
    buenas = respuestas_buenas()
    tipos = {t[0]: t[3] for t in TAREAS}

    # [1] test nulo: una máquina de mentira que nunca usa la herramienta y contesta mal tiene que
    #     dar 0 aciertos y 0 vueltas (y el ejecutor no llega a llamarse).
    llamadas = []

    def espia(codigo):
        llamadas.append(codigo)
        return "", False

    def nunca(mensajes):
        return "Creo que la respuesta es 7, y la palabra, «zzz»."

    rs = [evaluar(resolver(nunca, espia, enunciado_de(c)), tipos[c], buenas[c]) for c in buenas]
    aciertos, vueltas = sum(r["acierta"] for r in rs), sum(r["vueltas"] for r in rs)
    p1 = aciertos == 0 and vueltas == 0 and not llamadas
    ok &= p1
    print(f"[1] test nulo         máquina que nunca usa la herramienta y contesta mal, en las "
          f"{len(rs)} tareas: {aciertos} aciertos, {vueltas} vueltas, {len(llamadas)} programas "
          f"ejecutados: {'bien' if p1 else 'MAL'}")

    # [2] señal implantada: una máquina de mentira que escribe un programa con un error, lo
    #     corrige y contesta con lo que le vuelve. Se ejecuta de verdad (python3 -I, carpeta
    #     temporal) y tiene que dar: acierta, 2 vueltas, 1 error, «se corrige».
    roto = "print(open('quijote.txt', encoding='utf-8').read().conut('Sancho'))"
    bueno = ("import re\nt = open('quijote.txt', encoding='utf-8').read()\n"
             "print(len(re.findall(r'(?<![^\\W\\d_])Sancho(?![^\\W\\d_])', t)))")

    def corrige(mensajes):
        n_vuelven = sum(m["role"] == "tool" for m in mensajes)
        if n_vuelven == 0:
            return orden(roto)
        if n_vuelven == 1:
            return orden(bueno)
        return f"«Sancho» aparece {mensajes[-1]['content'].strip()} veces."

    r = evaluar(resolver(corrige, ejecutar_python, enunciado_de(1)), tipos[1], buenas[1])
    p2 = (r["acierta"] and r["vueltas"] == 2 and r["errores"] == 1 and r["se_corrige"]
          and r["primer_error"] == 1 and r["siguio"])
    ok &= p2
    print(f"[2] señal implantada  programa con error, corregido: acierta={r['acierta']}, "
          f"vueltas={r['vueltas']}, errores={r['errores']}, se corrige={r['se_corrige']}: "
          f"{'bien' if p2 else 'MAL'}")

    # [3] invariante: las dos implementaciones de las respuestas buenas coinciden en las ocho
    #     tareas (respuestas_buenas() revienta si no); y el ejecutor cumple lo que dice: carpeta
    #     nueva con solo una copia idéntica del Quijote, python aislado (-I), error por código de
    #     salida, por «Traceback» y por tiempo, y salida recortada con aviso.
    def prueba(codigo, **kw):
        return ejecutar_python(codigo, **kw)

    carpeta, _ = prueba("import os; print(sorted(os.listdir('.')))")
    copia, _ = prueba("import hashlib; print(hashlib.sha256(open('quijote.txt','rb').read())"
                      ".hexdigest())")
    import hashlib
    with open(QUIJOTE, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    aislado, _ = prueba("import sys; print(sys.flags.isolated)")
    _, e_ok = prueba("print('hola')")
    _, e_salida = prueba("import sys; sys.exit(3)")
    _, e_traza = prueba("print('Traceback (de mentira)')")
    texto_t, e_tiempo = prueba("while True: pass", tiempo=2)
    largo, _ = prueba("print('x' * 5000)")
    recortado = (len(largo) < 5000 and "Salida recortada" in largo
                 and largo.startswith("x" * MAX_SALIDA))
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(MODELO_GRANDE[1], local_files_only=True)
    plantilla = tk.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False,
                                       add_generation_prompt=True, tools=[HERRAMIENTA])
    solo_el_quijote = carpeta.strip() == "['" + FICHERO + "']" and copia.strip() == sha
    errores_bien = e_salida and e_traza and e_tiempo and "tardó más de 2 s" in texto_t
    plantilla_bien = ('"ejecutar_python"' in plantilla
                      and len(tk.encode("<tool_call>", add_special_tokens=False)) == 1)
    lecturas = [palabra_respondida('El texto de "Don Quijote" es "muy".'),
                palabra_respondida("La palabra es quijote."),
                palabra_respondida("«señor» y también «Don Quijote»"),
                palabra_respondida("1234"), palabra_respondida('Es "muy" y "".')]
    lectura_bien = lecturas == ["muy", "quijote", "quijote", None, "muy"]
    p3 = (solo_el_quijote and aislado.strip() == "1" and not e_ok and errores_bien and recortado
          and plantilla_bien and lectura_bien)
    ok &= p3
    print(f"[3] invariante        las dos implementaciones coinciden en las {len(buenas)} tareas; "
          f"carpeta solo con una copia idéntica del Quijote: {solo_el_quijote}; python aislado: "
          f"{aislado.strip() == '1'}; error por código de salida, «Traceback» y tiempo: "
          f"{errores_bien}; sin error si todo va bien: {not e_ok}; salida recortada con aviso: "
          f"{recortado}; plantilla con la herramienta: {plantilla_bien}; respuesta de la tarea 4: "
          f"{lectura_bien}: "
          f"{'bien' if p3 else 'MAL'}")

    print()
    print("SELFTEST: las tres pruebas pasan." if ok else "SELFTEST: FALLA.")
    return ok


# --------------------------- la salida ---------------------------

def plural(n, uno, varios):
    return f"{n} {uno if n == 1 else varios}"


def si_no(x):
    return "sí" if x else "no"


def lo_que_contesto(r, tipo, buena):
    """Lo que se lee en la respuesta final de la máquina, para la segunda tabla."""
    if r["final"] is None:
        return "sin respuesta"
    if tipo == "palabra":
        return palabra_respondida(r["final"]) or "ninguna palabra"
    n = ultimo_numero(r["final"])
    return "ningún número" if n is None else miles(n)


def escribir_conversacion(r, clave, rotulo, buena):
    """La conversación entera de una tarea, tal cual, a su fichero."""
    os.makedirs(CARPETA_CONVERSACIONES, exist_ok=True)
    out = [f"Tarea {clave}: {rotulo}", f"Respuesta buena: {buena}", "",
           "Enunciado:", enunciado_de(clave), ""]
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


def lineas_de(texto):
    """Un texto de máquina o de programa en renglones que caben (sin tabuladores)."""
    return renglones(texto.expandtabs(4))


def informe(resultados):
    """Las dos tablas y la muestra, como lista de renglones."""
    out = []
    # ---- la tabla
    filas = []
    for r in resultados:
        corrige = si_no(r["se_corrige"]) if r["errores"] else "—"
        filas.append([f"{r['clave']}. {r['rotulo']}", si_no(r["acierta"]), str(r["vueltas"]),
                      str(r["errores"]), corrige])
    aciertos = sum(r["acierta"] for r in resultados)
    notas = ["El texto de Don Quijote no está en lo que recibe la máquina: solo puede verlo "
             "escribiendo programas de Python que lo lean y leyendo lo que esos programas imprimen.",
             "Acierta: su respuesta final trae el resultado bueno. Vueltas: los programas que "
             "escribió y se ejecutaron antes de contestar. Con error: cuántos de ellos acabaron en "
             "error. Se corrige: si, habiendo tenido algún error, la respuesta final acierta."]
    if any(r["ordenes_mal"] for r in resultados):
        notas.append("Alguna orden salió mal escrita y no se ejecutó: "
                     + ", ".join(f"tarea {r['clave']}" for r in resultados if r["ordenes_mal"])
                     + ".")
    out += (tabla_editorial(
        f"Las {len(resultados)} tareas sobre el Quijote con un programa en bucle: aciertan "
        f"{aciertos}",
        ["la tarea", "acierta", "vueltas", "con error", "se corrige"], filas, "icddc", notas))
    out.append("")

    # ---- lo que había que contestar y lo que contestó
    filas = [[f"{r['clave']}. {r['rotulo']}", miles(r["buena"]) if isinstance(r["buena"], int)
              else r["buena"], lo_que_contesto(r, r["tipo"], r["buena"])] for r in resultados]
    out += (tabla_editorial(
        "Lo que había que contestar y lo que se lee en la respuesta final",
        ["la tarea", "respuesta buena", "lo que contestó"], filas, "idd",
        ["La respuesta buena la calcula un programa antes de preguntar, de dos maneras que tienen "
         "que dar lo mismo. La última columna es el último número de la respuesta final de la "
         "máquina (en la tarea 4, la última palabra entre comillas, o la última palabra)."]))
    out.append("")

    # ---- la muestra
    muestra = next((r for r in resultados if r["errores"] > 0 and r["siguio"]), None)
    if muestra is None:
        muestra, motivo = resultados[0], "ningún programa dio error: se toma la tarea 1"
    else:
        motivo = (f"la primera, en el orden de la lista, en la que un programa dio error y la "
                  f"máquina siguió (el de la vuelta {muestra['primer_error']})")
    out += [f"muestra elegida: tarea {muestra['clave']}, {motivo}", ""]
    lineas = lineas_de(f"Tarea {muestra['clave']}: {enunciado_de(muestra['clave'])}") + [""]
    vuelta = 0
    for item in muestra["rastro"]:
        if item[0] == "escribe":
            lineas += ["Escribe la máquina:"] + lineas_de(item[1]) + [""]
        elif item[0] == "programa":
            vuelta += 1
            lineas += [f"Se ejecuta este programa (vuelta {vuelta}):"] + lineas_de(item[1]) + [""]
        else:
            # (el salto de línea con que acaba un print no se enseña; a la máquina sí le llegó)
            lineas += [f"Vuelve a la máquina ({'con error' if item[2] else 'sin error'}):"] \
                + lineas_de(item[1].rstrip("\n") or "(no imprimió nada)") + [""]
    lineas.pop()
    if muestra["errores"]:
        hecho = (f"El programa de la vuelta {muestra['primer_error']} dio error y la máquina "
                 f"siguió escribiendo; al final {'acierta' if muestra['acierta'] else 'falla'}.")
    else:
        hecho = (f"Ningún programa dio error; al final "
                 f"{'acierta' if muestra['acierta'] else 'falla'}.")
    out += (muestra_editorial(
        "Una tarea entera, vuelta a vuelta: lo que escribe la máquina, el programa y lo que vuelve",
        lineas,
        [f"Tarea {muestra['clave']} ({muestra['rotulo']}): "
         f"{plural(muestra['vueltas'], 'vuelta', 'vueltas')} en total, "
         f"{plural(muestra['errores'], 'programa con error', 'programas con error')}. " + hecho]))
    return out


SECCION = re.compile(r"^----- (.*?) -----$", re.M)


def reconstruir(clave):
    """Lo que dejó resolver() para una tarea, leído de su fichero de conversación (Orden 115). El
    texto de cada sección es el que se escribió: entre el título y el siguiente van un salto
    delante y dos detrás."""
    with open(os.path.join(CARPETA_CONVERSACIONES, f"tarea_{clave}.txt"), encoding="utf-8") as fh:
        texto = fh.read()
    partes = SECCION.split(texto)
    rastro, vueltas, errores, ordenes_mal = [], 0, 0, 0
    primer_error, siguio = None, False
    for titulo, cuerpo in zip(partes[1::2], partes[2::2]):
        cuerpo = cuerpo[1:-2]
        if titulo == "La máquina escribe":
            rastro.append(("escribe", cuerpo))
            siguio = siguio or primer_error is not None
            if leer_orden(cuerpo)[0] is None and "<tool_call>" in cuerpo:
                ordenes_mal += 1
        elif titulo.startswith("Programa que se ejecuta"):
            vueltas += 1
            rastro.append(("programa", cuerpo))
        elif titulo.startswith("Lo que vuelve a la máquina"):
            error = titulo.endswith("(con error)")
            rastro.append(("vuelve", cuerpo, error))
            if error:
                errores += 1
                primer_error = primer_error or vueltas
    ultima = rastro[-1]
    final = ultima[1] if (ultima[0] == "escribe" and leer_orden(ultima[1])[0] is None
                          and "<tool_call>" not in ultima[1]) else None
    return {"final": final, "vueltas": vueltas, "errores": errores, "primer_error": primer_error,
            "siguio": siguio, "ordenes_mal": ordenes_mal, "rastro": rastro, "texto": texto}


NOTA_ORDEN_115 = [
    "Nota del 31 de octubre de 2026 (Orden 115). La lectura de la tarea 4 se corrigió después de ver "
    "el resultado: la primera daba por buena la respuesta «muy» porque «Quijote» salía en la frase; "
    "ahora la respuesta es la última palabra entre comillas de la respuesta final, o la última "
    "palabra si no hay comillas. Con la lectura corregida acierta 4 de 8 y no 5.",
    "Esta salida se ha regenerado a partir de las ocho conversaciones guardadas, que no han cambiado, "
    "sin volver a ejecutar el modelo. Y el experimento se sustituye por muchas_vueltas_carpeta.py "
    "porque el ejecutor de este no devolvía el valor de la última línea, como hacen los asistentes de "
    "hoy: los cuatro fallos son un programa que acaba en una expresión sin print.",
]


def regenerar():
    """Rehace SALIDA_TXT y los ficheros de conversación con la lectura corregida, sin modelo."""
    with open(SALIDA_TXT, encoding="utf-8") as fh:
        cabecera = fh.read().split("\n\n")[0].split("\n")
    buenas = respuestas_buenas()
    resultados = []
    for clave, rotulo, _, tipo in TAREAS:
        r = evaluar(reconstruir(clave), tipo, buenas[clave])
        r.update(clave=clave, rotulo=rotulo, tipo=tipo, buena=buenas[clave])
        antes = r.pop("texto")
        assert antes.startswith(f"Tarea {clave}: {rotulo}\nRespuesta buena: {buenas[clave]}\n\n"
                                f"Enunciado:\n{enunciado_de(clave)}\n\n"), (
            f"Se esperaba el enunciado guardado de la tarea {clave}; es otro")
        escribir_conversacion(r, clave, rotulo, buenas[clave])
        with open(os.path.join(CARPETA_CONVERSACIONES, f"tarea_{clave}.txt"), encoding="utf-8") as fh:
            despues = fh.read()
        marca = "----- Resultado -----"
        assert despues.split(marca)[0] == antes.split(marca)[0], (
            f"Se esperaba la misma conversación en la tarea {clave}; ha cambiado")
        resultados.append(r)
    salida = cabecera + [""] + NOTA_ORDEN_115 + [""] + informe(resultados)
    with open(SALIDA_TXT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(salida) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--regenerar", action="store_true",
                    help="rehace la salida desde las conversaciones guardadas, sin cargar el modelo")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if args.regenerar:
        regenerar()
        return

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
    print(f"herramienta: {shutil.which(PYTHON, path=ENTORNO['PATH'])} "
          f"({version.stdout.decode().strip()}), con -I, {TIEMPO_PROGRAMA} s como máximo")
    print()
    gen = generador(modelo, tok)

    resultados = []
    for clave, rotulo, _, tipo in TAREAS:
        t1 = time.time()
        r = evaluar(resolver(gen, ejecutar_python, enunciado_de(clave)), tipo, buenas[clave])
        r.update(clave=clave, rotulo=rotulo, tipo=tipo, buena=buenas[clave])
        resultados.append(r)
        escribir_conversacion(r, clave, rotulo, buenas[clave])
        print(f"tarea {clave}: acierta={si_no(r['acierta'])} vueltas={r['vueltas']} "
              f"errores={r['errores']} ({time.time() - t1:.0f} s)", file=sys.stderr, flush=True)
    print(f"tiempo total: {time.time() - t0:.0f} s", file=sys.stderr, flush=True)

    print("\n".join(informe(resultados)))


if __name__ == "__main__":
    main()
