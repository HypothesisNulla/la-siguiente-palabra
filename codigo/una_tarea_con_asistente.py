#!/usr/bin/env python3
"""
Capítulo 14 — una tarea entera, hecha por un asistente de hoy.

Los dos experimentos anteriores (`muchas_vueltas.py` y `muchas_vueltas_carpeta.py`) montaban el bucle a
mano con el modelo del portátil: inventaba ficheros o se paraba. Lo que el lector tiene que ver es cómo
trabaja un asistente de hoy, vuelta a vuelta, en una tarea real. Aquí la hace Claude Code, el
asistente de programación de Anthropic, en modo no interactivo (`claude -p`), con una sola instrucción
y sin más ayuda, y se guarda la conversación entera.

La tarea es la 1 de `muchas_vueltas_carpeta.py`: «En las mediciones hechas con distinto número de
hilos, ¿cuál es la mayor ventaja que aparece, contando los dos ficheros (A y B)? Los ficheros están en
la carpeta de trabajo. Contesta con el resultado.» La carpeta es una carpeta temporal nueva, fuera de
cualquier repositorio y sin ningún CLAUDE.md, con solo copias de los ocho CSV de aquel experimento. La
respuesta buena (6,6989) la calcula `muchas_vueltas_carpeta.respuestas_buenas()` con sus dos
implementaciones antes de llamar al asistente. Acierta con la misma regla que aquel experimento: el
número que la respuesta final da como resultado coincide con la mayor ventaja con la precisión que dé el
asistente (ver «El acierto», abajo).

Cómo se lanza (ARGUMENTOS en el programa):
  - `--safe-mode`: sin CLAUDE.md, ni memoria, ni skills, plugins, hooks, servidores MCP, agentes ni
    comandos del usuario; quedan las herramientas, los plugins y las skills que trae Claude Code.
  - `--restricted --tools Bash,Read,Glob,Grep`: solo se puede leer y ejecutar (no escribir con una
    herramienta de edición, ni internet con WebFetch/WebSearch); las herramientas de ficheros quedan
    confinadas a la carpeta de trabajo; no lee los ajustes del usuario.
  - `--settings` con el sandbox de Claude Code para la terminal (AJUSTES): sin red (ningún dominio
    permitido), sin leer /Users y sin escribir fuera de la carpeta de trabajo y los temporales del
    propio sandbox. Se comprobó con un diagnóstico aparte, antes (ver la Respuesta 116).
  - `--permission-prompts none`: nada que pida permiso se concede; `--allowedTools` deja pasar solo
    esas cuatro herramientas. `--no-session-persistence`: no deja sesión guardada.
  - Sin `--model` ni `--effort`: el modelo es el que usa Claude Code por omisión y se apunta de la
    salida. Tope de gasto (PRESUPUESTO_USD) y de tiempo (TIEMPO_MAXIMO) por si se desbocara.
Lo que carga igualmente, y se dice: dos datos que Claude Code añade por su cuenta (el correo de la
cuenta y unas líneas de atribución para los commits); ni CLAUDE.md ni memoria.

El programa convierte la salida `stream-json` en una muestra para el libro, vuelta a vuelta: lo que
escribe el asistente, la orden que da (herramienta y argumentos; el programa entero si es un programa)
y lo que le vuelve (recortado si es largo, y avisando). La salida en bruto se guarda sin tocar. Una
«vuelta» es una orden que da el asistente. Una orden «da error» si lo que le vuelve viene marcado como
error o lleva «Traceback». No se repite ni se cambia la instrucción.

El acierto, corregido el 31 de octubre de 2026 (Orden 117), DESPUÉS de ver el resultado y dicho aquí y
en la salida para que se sepa: la primera regla, la de `muchas_vueltas_carpeta.py`, miraba el último número
de la respuesta final, y esta respuesta acaba hablando de otros barridos («…se queda en torno a 5,6×»)
después de dar la buena («La mayor ventaja es **6,70× (6,6989)**»). Ahora cuenta como respuesta el número
que la respuesta final da como resultado (resultado_dicho): el primero que va detrás de «la mayor ventaja
es»; si esa frase no está, el primero en negrita que lleve un número; y si tampoco, el último número del
texto (la regla de antes). Con esa regla acierta.

La muestra para el libro (muestra_de): lo que el asistente escribe, las órdenes y lo que le vuelve, enteros,
salvo el primer retorno, el de `ls -la && head -50 *`: del listado se quitan las columnas del propietario
y del grupo (quitar columnas, no cambiar letras: regla 6; el nombre de usuario del Mac no se publica) y de
los ficheros se enseñan solo la cabecera y las tres primeras filas de `cuantos_hilos-A.csv`; lo demás se
recorta, diciendo cuánto falta. La salida en bruto (stream.jsonl) lleva ese nombre de usuario y no se
publica: está en el repositorio privado del libro, en notas/fuentes/una_tarea_con_asistente/, junto a
meta.json (lo que no va en el stream: segundos, código de salida y ficheros de la carpeta al terminar). No
lleva el correo de la cuenta.

Uso (desde codigo/):
    python una_tarea_con_asistente.py --selftest
    python una_tarea_con_asistente.py > ../datos/salidas/una_tarea_con_asistente.txt   # llama al asistente
    python una_tarea_con_asistente.py --desde RUTA/stream.jsonl > ../datos/salidas/una_tarea_con_asistente.txt
La segunda forma no llama a nadie: rehace la salida a partir de un stream.jsonl guardado y de su meta.json.
La salida en bruto de una ejecución queda en ../datos/salidas/una_tarea_con_asistente/ (ignorada por git).
"""

# ======================= CONSTANTES =======================

DONDE = "/private/tmp"              # dónde se crea la carpeta de trabajo (fuera de todo repositorio)
CARPETA_SALIDA = "../datos/salidas/una_tarea_con_asistente"
TAREA = 1                           # la tarea de muchas_vueltas_carpeta.py que se le da
HERRAMIENTAS = ["Bash", "Read", "Glob", "Grep"]
AJUSTES = {"sandbox": {"enabled": True, "autoAllowBashIfSandboxed": True,
                       "allowUnsandboxedCommands": False,
                       "network": {"allowedDomains": []},
                       "filesystem": {"denyRead": ["/Users"]}}}
PRESUPUESTO_USD = 5                 # tope de gasto de la ejecución
TIEMPO_MAXIMO = 900                 # segundos, como mucho, de la ejecución entera
LIMITE_TEXTO = 3000                 # caracteres de lo que escribe el asistente que se enseñan
LIMITE_ORDEN = 4000                 # ... de una orden (el programa entero, si cabe)
LIMITE_VUELVE = 1500                # ... de lo que vuelve de una orden
MAX_LINEAS_BLOQUE = 40              # un bloque de la muestra tiene que ser de menos (verificar_anchos.py)
MARCA_AJENA_INICIO = ("[salida ajena: empieza — lo que escribe Claude Code y lo que le devuelven sus "
                      "herramientas; no se revisa el formato de sus números (regla 6)]")
MARCA_AJENA_FIN = "[salida ajena: termina]"

# ==========================================================

import argparse
import getpass
import glob
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time

from formato import coma, miles, muestra_editorial, tabla_editorial
from muchas_vueltas import lineas_de, plural, si_no
from muchas_vueltas_carpeta import (CARPETA_DATOS, FICHEROS, FINAL, NUMERO, TAREAS, acierta,
                                    coincide, lecturas,
                                    respuestas_buenas, ultimo_token)


# --------------------------- lanzar al asistente ---------------------------

def buscar_claude():
    """El ejecutable de Claude Code: el de la variable CLAUDE_BIN, el del PATH o el que trae la
    aplicación de escritorio (el más reciente)."""
    if os.environ.get("CLAUDE_BIN"):
        return os.environ["CLAUDE_BIN"]
    en_path = shutil.which("claude")
    if en_path:
        return en_path
    ruta = os.path.expanduser("~/Library/Application Support/Claude/claude-code/*/*/claude.app/"
                              "Contents/MacOS/claude")
    candidatos = sorted(glob.glob(ruta))
    assert candidatos, "Se esperaba encontrar el ejecutable de Claude Code; no está"
    return candidatos[-1]


def instruccion():
    """La instrucción, literal: la tarea 1 de muchas_vueltas_carpeta.py con su final común."""
    return next(t[2] for t in TAREAS if t[0] == TAREA) + FINAL


def argumentos(claude, texto):
    """La línea de órdenes con la que se lanza al asistente."""
    return [claude, "-p", texto, "--output-format", "stream-json", "--verbose", "--safe-mode",
            "--restricted", "--tools", ",".join(HERRAMIENTAS),
            "--allowedTools", ",".join(HERRAMIENTAS), "--permission-prompts", "none",
            "--no-session-persistence", "--max-budget-usd", str(PRESUPUESTO_USD),
            "--settings", json.dumps(AJUSTES)]


def preparar_carpeta():
    """Una carpeta temporal nueva, fuera de cualquier repositorio, sin CLAUDE.md en ella ni en
    ninguna carpeta de arriba, con solo copias de los ocho CSV."""
    carpeta = tempfile.mkdtemp(prefix="asistente_", dir=DONDE)
    for f in FICHEROS:
        shutil.copyfile(os.path.join(CARPETA_DATOS, f), os.path.join(carpeta, f))
    comprobar_carpeta(carpeta)
    return carpeta


def comprobar_carpeta(carpeta):
    real = os.path.realpath(carpeta)
    sitio = real
    while True:
        assert not os.path.exists(os.path.join(sitio, ".git")), (
            f"Se esperaba una carpeta fuera de todo repositorio; hay un .git en {sitio}")
        for nombre in ("CLAUDE.md", ".claude"):
            assert not os.path.exists(os.path.join(sitio, nombre)), (
                f"Se esperaba una carpeta sin {nombre} en ella ni en las de arriba; está en {sitio}")
        if sitio == os.path.dirname(sitio):
            break
        sitio = os.path.dirname(sitio)
    assert sorted(os.listdir(real)) == sorted(FICHEROS), (
        f"Se esperaban solo los ocho CSV en la carpeta; hay {sorted(os.listdir(real))}")


# --------------------------- leer la salida ---------------------------

def texto_de(contenido):
    """El texto de un tool_result: una cadena o una lista de bloques de texto."""
    if isinstance(contenido, str):
        return contenido
    return "\n".join(b.get("text", "") for b in contenido if isinstance(b, dict))


def leer_stream(lineas):
    """De las líneas de stream-json a: la cabecera (evento init), los pasos en su orden y el
    resultado. Un paso es ("texto", texto), ("orden", id, herramienta, argumentos) o ("vuelve", id,
    texto, error); un «Traceback» en lo que vuelve también cuenta como error."""
    eventos = [json.loads(l) for l in lineas if l.strip()]
    init = next((e for e in eventos if e.get("type") == "system" and e.get("subtype") == "init"), None)
    resultado = next((e for e in reversed(eventos) if e.get("type") == "result"), None)
    pasos, pensamientos = [], 0
    for e in eventos:
        if e.get("type") == "assistant":
            for b in e["message"]["content"]:
                if b["type"] == "text" and b["text"].strip():
                    pasos.append(("texto", b["text"]))
                elif b["type"] == "tool_use":
                    pasos.append(("orden", b["id"], b["name"], b["input"]))
                elif b["type"] == "thinking":
                    pensamientos += 1
        elif e.get("type") == "user" and isinstance(e["message"]["content"], list):
            for b in e["message"]["content"]:
                if b.get("type") == "tool_result":
                    t = texto_de(b.get("content", ""))
                    pasos.append(("vuelve", b["tool_use_id"], t,
                                  bool(b.get("is_error")) or "Traceback (most recent call last)" in t))
    ordenes = [p for p in pasos if p[0] == "orden"]
    vuelven = [p for p in pasos if p[0] == "vuelve"]
    return {"init": init, "pasos": pasos, "resultado": resultado, "pensamientos": pensamientos,
            "vueltas": len(ordenes), "errores": sum(v[3] for v in vuelven),
            "herramientas": [o[2] for o in ordenes]}


MARCA_RESULTADO = re.compile(r"la mayor ventaja es[\s:*]*", re.I)
NEGRITA = re.compile(r"\*\*(.+?)\*\*", re.S)


def resultado_dicho(texto):
    """El número (tal como está escrito) que la respuesta final da como resultado: el primero que va
    detrás de «la mayor ventaja es»; si esa frase no está, el primero en negrita que lleve un número; y
    si tampoco, el último número del texto (la regla de muchas_vueltas_carpeta.py). None si no hay."""
    m = MARCA_RESULTADO.search(texto)
    if m:
        n = NUMERO.search(texto[m.end():])
        if n:
            return n.group(0)
    for b in NEGRITA.finditer(texto):
        n = NUMERO.search(b.group(1))
        if n:
            return n.group(0)
    todos = NUMERO.findall(texto)
    return todos[-1] if todos else None


def acierta_asistente(buena, final):
    """Si el número que la respuesta final da como resultado vale por la respuesta buena, con la
    precisión que dé (se prueban las dos lecturas de «3.369», como en muchas_vueltas_carpeta.py)."""
    if final is None:
        return False
    token = resultado_dicho(final)
    return token is not None and any(coincide(v, d, buena) for v, d in lecturas(token))


LS_LINEA = re.compile(r"^(\S+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\d+)\s+(\w{3}\s+\d+\s+[\d:]{4,5})\s+(.+)$")


def sin_propietario(linea):
    """Una línea de `ls -la` sin las columnas del propietario y del grupo (y solo esas dos)."""
    m = LS_LINEA.match(linea)
    if not m:
        return linea
    perm, enlaces, _, _, tam, fecha, nombre = m.groups()
    return f"{perm} {enlaces} {tam} {fecha} {nombre}"


def retorno_libro(texto, fichero="cuantos_hilos-A.csv", filas=3):
    """Del retorno de `ls -la && head -50 *` para el libro: el listado sin propietario ni grupo, y de
    los ficheros solo la cabecera y las tres primeras filas de `fichero`. Devuelve (renglones, cuántos
    caracteres del retorno original faltan)."""
    lineas = texto.split("\n")
    primero = next(i for i, l in enumerate(lineas) if l.startswith("==> "))
    k = lineas.index(f"==> {fichero} <==")
    fin = next((j for j in range(k + 1, len(lineas)) if lineas[j].startswith("==> ")), len(lineas))
    seccion = lineas[k:min(k + 2 + filas, fin)]          # el rótulo, la cabecera del CSV y las filas
    enseñado = lineas[:primero] + seccion
    faltan = len(texto) - len("\n".join(enseñado))
    return [sin_propietario(l) for l in lineas[:primero]] + seccion, faltan


def respuesta_final(leido):
    r = leido["resultado"]
    if r and isinstance(r.get("result"), str):
        return r["result"]
    textos = [p[1] for p in leido["pasos"] if p[0] == "texto"]
    return textos[-1] if textos else None


# --------------------------- selftest ---------------------------

def evento(tipo, **kw):
    return json.dumps({"type": tipo, **kw})


def stream_de_mentira(pasos, final):
    """Un stream-json grabado de mentira: `pasos` es una lista de ("texto", t), ("orden", nombre,
    argumentos) o ("vuelve", texto, error)."""
    out = [evento("system", subtype="init", claude_code_version="0.0.0", model="de-mentira",
                  tools=HERRAMIENTAS, cwd="/x", mcp_servers=[], skills=[], plugins=[], agents=[],
                  apiKeySource="de-mentira")]
    n = 0
    for p in pasos:
        if p[0] == "texto":
            out.append(evento("assistant", message={"content": [{"type": "text", "text": p[1]}]}))
        elif p[0] == "orden":
            n += 1
            out.append(evento("assistant", message={"content": [
                {"type": "tool_use", "id": f"t{n}", "name": p[1], "input": p[2]}]}))
        else:
            out.append(evento("user", message={"content": [
                {"type": "tool_result", "tool_use_id": f"t{n}", "content": p[1],
                 "is_error": p[2]}]}))
    out.append(evento("result", subtype="success", result=final, duration_ms=1000))
    return out


def selftest():
    ok = True
    buenas = respuestas_buenas()
    buena = buenas[TAREA]

    # [1] test nulo: un stream de mentira sin ninguna herramienta da 0 vueltas (y no acierta).
    l = leer_stream(stream_de_mentira([("texto", "No lo sé.")], "No lo sé: creo que es 7."))
    p1 = (l["vueltas"] == 0 and l["errores"] == 0 and l["herramientas"] == []
          and not acierta("decimal", buena, respuesta_final(l)))
    ok &= p1
    print(f"[1] test nulo         stream sin ninguna herramienta: {l['vueltas']} vueltas, "
          f"{l['errores']} errores, acierta={acierta('decimal', buena, respuesta_final(l))}: "
          f"{'bien' if p1 else 'MAL'}")

    # [2] señal implantada: un stream de mentira con tres herramientas, una con error, da 3
    #     vueltas y 1 error, en su orden, y acierta con la respuesta buena.
    pasos = [("texto", "Miro la carpeta."), ("orden", "Bash", {"command": "ls"}),
             ("vuelve", "a.csv\nb.csv", False),
             ("orden", "Read", {"file_path": "/nada.csv"}), ("vuelve", "File does not exist", True),
             ("orden", "Bash", {"command": "python3 -c 'print(6.6989)'"}), ("vuelve", "6.6989", False)]
    esta = ("La mayor ventaja es **6,70× (6,6989)**. Sale del fichero **A**, en el barrido 1. En los "
            "demás barridos de ese caso la ventaja se queda en torno a 5,6×.")
    l = leer_stream(stream_de_mentira(pasos, esta))
    orden_bien = [p[0] for p in l["pasos"]] == ["texto", "orden", "vuelve", "orden", "vuelve",
                                                 "orden", "vuelve"]
    p2 = (l["vueltas"] == 3 and l["errores"] == 1 and l["herramientas"] == ["Bash", "Read", "Bash"]
          and [p[3] for p in l["pasos"] if p[0] == "vuelve"] == [False, True, False] and orden_bien
          and acierta_asistente(buena, respuesta_final(l)))
    # La regla del acierto con esta respuesta: acierta (6,70 → 6,6989 con dos decimales), aunque el
    # último número sea 5,6; y no acierta si lo que va detrás de «la mayor ventaja es» es otro.
    regla_bien = (resultado_dicho(esta) == "6,70" and acierta_asistente(buena, esta)
                  and not acierta("decimal", buena, esta)          # la regla de antes sí fallaba
                  and not acierta_asistente(buena, "La mayor ventaja es 5,6, y la de A, 6,6989.")
                  and acierta_asistente(buena, "Resultado: **6,6989** (en el fichero A).")
                  and acierta_asistente(buena, "6,6989") and not acierta_asistente(buena, None))
    p2 = p2 and regla_bien
    ok &= p2
    print(f"[2] señal implantada  tres herramientas, una con error: {l['vueltas']} vueltas, "
          f"{l['errores']} errores, herramientas {l['herramientas']}, en su orden: {orden_bien}; "
          f"regla del acierto con la respuesta de verdad (6,70× (6,6989), último número 5,6): "
          f"{regla_bien}: {'bien' if p2 else 'MAL'}")

    # [3] invariante: las dos implementaciones de la respuesta buena coinciden (respuestas_buenas()
    #     revienta si no); la línea de órdenes lleva el confinamiento; la carpeta que se prepara
    #     está fuera de todo repositorio y sin CLAUDE.md, con solo los ocho CSV; y el recorte dice
    #     lo que recorta.
    linea = argumentos("claude", "x")
    confina = (all(f in linea for f in ("--safe-mode", "--restricted", "--no-session-persistence"))
               and linea[linea.index("--tools") + 1] == "Bash,Read,Glob,Grep"
               and linea[linea.index("--permission-prompts") + 1] == "none"
               and not any(p in " ".join(linea) for p in ("WebFetch", "WebSearch", "Edit", "Write")))
    aj = json.loads(linea[linea.index("--settings") + 1])["sandbox"]
    confina &= (aj["enabled"] and aj["network"]["allowedDomains"] == []
                and aj["allowUnsandboxedCommands"] is False and aj["filesystem"]["denyRead"] == ["/Users"])
    carpeta = preparar_carpeta()
    try:
        carpeta_bien = sorted(os.listdir(carpeta)) == sorted(FICHEROS)
    finally:
        shutil.rmtree(carpeta)
    largo = "y" * (LIMITE_VUELVE + 500)
    recorte = recortar(largo, LIMITE_VUELVE)
    recorte_bien = (recorte.startswith("y" * LIMITE_VUELVE) and "recortado" in recorte
                    and recortar("corto", LIMITE_VUELVE) == "corto"
                    and instruccion().startswith("En las mediciones hechas con distinto número de hilos"))
    # La muestra del libro: sin el nombre de usuario (el del Mac que ejecuta esto, sin escribirlo
    # aquí) ni el grupo, con solo esas dos columnas quitadas del listado, y con el recorte dicho.
    usuario, grupo = getpass.getuser(), "grupoprueba"
    ls = [f"drwx------@  11 {usuario}  {grupo}   352 Oct 31 09:28 .",
          f"-rw-r--r--@   1 {usuario}  {grupo}  4329 Oct 31 09:28 cuantos_hilos-A.csv",
          f"drwxrwxrwt  264 root  wheel  8448 Oct 31 09:28 .."]
    filas_csv = ["barrido,hilos,longitud,t_en_orden,t_a_la_vez,ventaja"] + [
        f"{i},1,64,0.01,0.007,1.4{i}" for i in range(1, 8)]
    retorno = "\n".join(["total 88"] + ls + ["==> crecer.csv <==", "modelo,numeros,tarea,acierto",
                          "a,1,b,0.4", "==> cuantos_hilos-A.csv <=="] + filas_csv)
    falso = leer_stream(stream_de_mentira([
        ("orden", "Bash", {"command": "ls -la && head -50 *"}), ("vuelve", retorno, False)], esta))
    texto_muestra = "\n".join(muestra_de(falso))
    renglones_ls, faltan = retorno_libro(retorno)
    antes = [l.split() for l in ls]
    despues = [sin_propietario(l).split() for l in ls]
    solo_dos = all(d == a[:2] + a[4:] and len(a) - len(d) == 2 for a, d in zip(antes, despues))
    sin_usuario = (usuario not in texto_muestra and grupo not in texto_muestra
                   and usuario in retorno and "cuantos_hilos-A.csv" in texto_muestra
                   and "3,1,64,0.01,0.007,1.43" in texto_muestra
                   and "4,1,64,0.01,0.007,1.44" not in texto_muestra
                   and "a,1,b,0.4" not in texto_muestra and "recortado: faltan" in texto_muestra
                   and faltan > 0 and "total 88" in texto_muestra)
    # La muestra en dos bloques (Orden 118): se parte justo antes de la segunda orden, cada bloque
    # cabe en una página (menos de MAX_LINEAS_BLOQUE líneas), y las dos marcas de salida ajena
    # rodean los bloques y nada más (la cabecera y la tabla quedan fuera, para que se revisen).
    tres = leer_stream(stream_de_mentira([
        ("orden", "Bash", {"command": "ls -la && head -50 *"}), ("vuelve", retorno, False),
        ("texto", "Ahora calculo."), ("orden", "Bash", {"command": "echo 6.6989"}),
        ("vuelve", "6.6989", False), ("orden", "Bash", {"command": "echo otra"}),
        ("vuelve", "otra", False)], esta))
    bloques = bloques_muestra(tres, esta, "3 vueltas, 0 órdenes con error", True)
    partido_bien = (len(bloques) == 2 and bloques[0][0] == "La primera vuelta: mirar la carpeta"
                    and bloques[1][0] == "La segunda vuelta y la respuesta"
                    and not any("echo 6.6989" in l for l in bloques[0][1])
                    and "Ahora calculo." in " ".join(bloques[1][1])
                    and bloques[1][1][0] == "Escribe el asistente:"
                    and "columnas del propietario" in " ".join(bloques[0][2])
                    and all(len(b[1]) < MAX_LINEAS_BLOQUE for b in bloques))
    todo = informe("\n".join(stream_de_mentira([
        ("orden", "Bash", {"command": "ls"}), ("vuelve", "a.csv", False),
        ("orden", "Bash", {"command": "echo 6.6989"}), ("vuelve", "6.6989", False)], esta)),
        buena, 14, 0, [], "0.0.0", "claude")
    ini = [i for i, l in enumerate(todo) if l.startswith("[salida ajena: empieza")]
    fin = [i for i, l in enumerate(todo) if l == MARCA_AJENA_FIN]
    marcas_bien = (len(ini) == 1 and len(fin) == 1 and ini[0] < fin[0]
                   and todo.index("::: tabla") < ini[0]
                   and [i for i, l in enumerate(todo) if l == "::: muestra"][0] > ini[0]
                   and [i for i, l in enumerate(todo) if l == ":::"][-1] < fin[0])
    p3 = (confina and carpeta_bien and recorte_bien and buena == buenas[1] and sin_usuario
          and solo_dos and partido_bien and marcas_bien)
    ok &= p3
    print(f"[3] invariante        las dos implementaciones coinciden (respuesta buena {buena}); "
          f"línea de órdenes con el confinamiento: {confina}; carpeta nueva, fuera de todo "
          f"repositorio, sin CLAUDE.md y con solo los ocho CSV: {carpeta_bien}; recorte que lo "
          f"dice: {recorte_bien}; en la muestra del libro no sale el nombre de usuario ni el grupo y "
          f"el recorte lo dice: {sin_usuario}; las columnas quitadas son solo esas dos: {solo_dos}; "
          f"la muestra en dos bloques, cada uno de menos de {MAX_LINEAS_BLOQUE} líneas: "
          f"{partido_bien}; marcas de salida ajena alrededor de los bloques y de nada más: "
          f"{marcas_bien}: {'bien' if p3 else 'MAL'}")

    print()
    print("SELFTEST: las tres pruebas pasan." if ok else "SELFTEST: FALLA.")
    return ok


# --------------------------- la salida ---------------------------

def recortar(texto, limite):
    """El texto, o sus primeros `limite` caracteres y un aviso de cuántos faltan."""
    if len(texto) <= limite:
        return texto
    return texto[:limite] + f"\n[... recortado: faltan {miles(len(texto) - limite)} caracteres]"


def orden_dicha(nombre, args):
    """Una orden: la herramienta y sus argumentos; si es una terminal, el programa entero."""
    if nombre == "Bash" and isinstance(args.get("command"), str):
        cabeza = [f"Da esta orden (herramienta {nombre}):"]
        if args.get("description"):
            cabeza.append(f"Descripción que le pone: {args['description']}")
        return cabeza + recortar(args["command"], LIMITE_ORDEN).split("\n")
    return [f"Da esta orden (herramienta {nombre}):"] + recortar(
        json.dumps(args, ensure_ascii=False), LIMITE_ORDEN).split("\n")


def muestra_de(leido, desde=0, hasta=None, con_instruccion=True):
    """Los renglones de la muestra, vuelta a vuelta: los pasos de `desde` a `hasta` (todos, por
    omisión), con la instrucción delante si `con_instruccion`."""
    lineas = lineas_de(f"Instrucción: {instruccion()}") + [""] if con_instruccion else []
    primero = True                      # el primer retorno es el que se recorta para el libro
    for p in leido["pasos"][desde:hasta]:
        if p[0] == "texto":
            lineas += ["Escribe el asistente:"] + lineas_de(recortar(p[1], LIMITE_TEXTO)) + [""]
        elif p[0] == "orden":
            lineas += [l for t in orden_dicha(p[2], p[3]) for l in lineas_de(t)] + [""]
        elif primero and "\n==> cuantos_hilos-A.csv <==" in p[2]:
            primero = False
            enseñado, faltan = retorno_libro(p[2])
            enseñado = enseñado + [f"[... recortado: faltan {miles(faltan)} caracteres]"]
            lineas += [f"Le vuelve ({'con error' if p[3] else 'sin error'}):"] \
                + [l for t in enseñado for l in lineas_de(t)] + [""]
        else:
            primero = False
            salida = recortar(p[2].rstrip("\n"), LIMITE_VUELVE) or "(no vuelve nada)"
            lineas += [f"Le vuelve ({'con error' if p[3] else 'sin error'}):"] + lineas_de(salida) + [""]
    if lineas and lineas[-1] == "":
        lineas.pop()
    return lineas


def bloques_muestra(leido, final, cuenta, gana):
    """La muestra del libro en DOS bloques (Orden 118; uno solo de 61 líneas no cabe en una página,
    verificar_anchos.py): la primera vuelta (la instrucción, la primera orden y lo que le vuelve) y
    la segunda vuelta con la respuesta. Se parte justo antes de la segunda orden. Devuelve
    [(título, renglones, notas)]."""
    ordenes = [i for i, p in enumerate(leido["pasos"]) if p[0] == "orden"]
    textos = [p[1] for p in leido["pasos"] if p[0] == "texto"]
    ultimo_texto = textos[-1] if textos else None   # (si es la respuesta final, no se repite)
    cola = ([] if ultimo_texto == final else
            ["", "Respuesta final:"] + lineas_de(recortar(final or "(ninguna)", LIMITE_TEXTO)))
    veredicto = f"En total, {cuenta}; {'acierta' if gana else 'no acierta'}."
    if len(ordenes) < 2:
        return [("La conversación entera: lo que escribe, la orden que da y lo que le vuelve",
                 muestra_de(leido) + cola, [veredicto])]
    corte = ordenes[1]
    # (el texto que el asistente escribe justo antes de su segunda orden va con la segunda vuelta)
    while corte > 0 and leido["pasos"][corte - 1][0] == "texto":
        corte -= 1
    primera = muestra_de(leido, 0, corte)
    segunda = muestra_de(leido, corte, None, con_instruccion=False) + cola
    return [("La primera vuelta: mirar la carpeta", primera,
             ["Cada paso es lo que escribe el asistente, la orden que da con una herramienta (la "
              "terminal) y lo que esa herramienta le devuelve.",
              "De lo que vuelve a la primera orden se enseñan solo el listado de ficheros, al que se "
              "le han quitado las columnas del propietario y del grupo, y el principio de uno de "
              "ellos; se dice cuánto falta."]),
            ("La segunda vuelta y la respuesta", segunda,
             ["Las órdenes, lo que vuelve y la respuesta están tal cual.", veredicto])]


def lanzar(carpeta):
    """Lanza al asistente en `carpeta`; devuelve (stdout, stderr, segundos, código de salida)."""
    t0 = time.time()
    try:
        r = subprocess.run(argumentos(buscar_claude(), instruccion()), cwd=carpeta,
                           stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=TIEMPO_MAXIMO)
        return r.stdout.decode("utf-8"), r.stderr.decode("utf-8"), time.time() - t0, r.returncode
    except subprocess.TimeoutExpired as e:
        return ((e.stdout or b"").decode("utf-8", errors="replace"),
                (e.stderr or b"").decode("utf-8", errors="replace") + "\n[tiempo máximo agotado]",
                time.time() - t0, None)


def informe(salida, buena, segundos, codigo, dejados, version, claude):
    """Las líneas de la salida: cabecera, tabla y muestra, a partir del stream-json."""
    leido = leer_stream(salida.split("\n"))
    init, res = leido["init"], leido["resultado"] or {}
    final = respuesta_final(leido)
    gana = acierta_asistente(buena, final)
    out = [f"máquina: {platform.platform()}; Python {platform.python_version()}",
           f"asistente: {version}, modelo {init['model']}; clave de acceso: "
           f"{init.get('apiKeySource')}; ejecutable {claude.replace(os.path.expanduser('~'), '~')}",
           f"herramientas que tuvo: {', '.join(init['tools'])}; servidores MCP: "
           f"{len(init['mcp_servers'])}; skills: {len(init.get('skills', []))} y plugins: "
           f"{len(init.get('plugins', []))}, todos de los que trae Claude Code; agentes: "
           f"{', '.join(init.get('agents', []))}",
           "carga: ni CLAUDE.md ni memoria ni instrucciones del usuario (--safe-mode); Claude Code "
           "añade por su cuenta el correo de la cuenta y unas líneas de atribución para commits",
           f"confinamiento: sin red, sin leer /Users, sin escribir fuera de la carpeta de trabajo; "
           f"en la carpeta de trabajo, al terminar, además de los ocho CSV: "
           f"{', '.join(dejados) if dejados else 'nada'}; "
           f"bloques de razonamiento interno (no se ven): {leido['pensamientos']}",
           f"fin: {res.get('subtype')}, código de salida {codigo}, denegaciones de permiso: "
           f"{len(res.get('permission_denials', []))}",
           "acierto: cuenta como respuesta el número que la respuesta final da como resultado (el que va "
           "detrás de «la mayor ventaja es»). Esta regla se corrigió el 31 de octubre de 2026 después de "
           "ver el resultado: la primera miraba el último número, y la respuesta acaba hablando de otros "
           "barridos",
           "salida en bruto: no se publica aquí porque lleva el nombre de usuario del Mac; está en el "
           "repositorio privado del libro, en notas/fuentes/una_tarea_con_asistente/. Lo de abajo es la "
           "muestra ya procesada", ""]

    filas = [["la respuesta buena", coma(float(buena), 4)],
             ["lo que da como resultado",
              (resultado_dicho(final) if final else None) or "ninguno"],
             ["¿acierta?", si_no(gana)],
             ["vueltas (órdenes que dio)", str(leido["vueltas"])],
             ["órdenes con error", str(leido["errores"])],
             ["duración", f"{segundos:.0f} s"]]
    out += tabla_editorial(
        "Una tarea entera, hecha por un asistente de hoy",
        ["lo que se mide", "resultado"], filas, "id",
        ["Es la tarea de los ficheros de medidas: la mayor ventaja en los dos ficheros de hilos, con "
         "la instrucción tal cual, sin más ayuda y sin acceso a nada fuera de su carpeta.",
         "Acierta si el número que da como resultado, el que va detrás de «la mayor ventaja es», es la "
         "mayor ventaja. Las vueltas son las órdenes que da; con error, las que le devuelven un fallo. "
         "La duración es la de esta ejecución, en este portátil y con conexión al servicio."])
    out.append("")
    cuenta = (f"{plural(leido['vueltas'], 'vuelta', 'vueltas')}, "
              f"{plural(leido['errores'], 'orden con error', 'órdenes con error')}")
    out.append(MARCA_AJENA_INICIO)
    for k, (titulo, renglones, notas) in enumerate(bloques_muestra(leido, final, cuenta, gana)):
        assert len(renglones) < MAX_LINEAS_BLOQUE, (
            f"Se esperaba un bloque de menos de {MAX_LINEAS_BLOQUE} líneas; «{titulo}» tiene "
            f"{len(renglones)}")
        out += ([""] if k else []) + muestra_editorial(titulo, renglones, notas)
    out.append(MARCA_AJENA_FIN)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--desde", metavar="STREAM",
                    help="no llama al asistente: rehace la salida desde un stream.jsonl guardado y "
                         "su meta.json (al lado)")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)

    buenas = respuestas_buenas()                       # antes de llamar al asistente
    buena = buenas[TAREA]
    if args.desde:
        with open(args.desde, encoding="utf-8") as fh:
            salida = fh.read()
        with open(os.path.join(os.path.dirname(args.desde), "meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
    else:
        claude = buscar_claude()
        version = subprocess.run([claude, "--version"], stdout=subprocess.PIPE, check=True)
        carpeta = preparar_carpeta()
        try:
            salida, errores, segundos, codigo = lanzar(carpeta)
            dejados = sorted(set(os.listdir(carpeta)) - set(FICHEROS))
        finally:
            shutil.rmtree(carpeta)
        meta = {"segundos": round(segundos), "codigo": codigo, "dejados": dejados,
                "version": version.stdout.decode().strip(), "claude": claude}
        os.makedirs(CARPETA_SALIDA, exist_ok=True)
        with open(os.path.join(CARPETA_SALIDA, "stream.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(salida)
        with open(os.path.join(CARPETA_SALIDA, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
        if errores.strip():
            with open(os.path.join(CARPETA_SALIDA, "stderr.txt"), "w", encoding="utf-8") as fh:
                fh.write(errores)

    print("\n".join(informe(salida, buena, meta["segundos"], meta["codigo"], meta["dejados"],
                            meta["version"], meta["claude"])))


if __name__ == "__main__":
    main()
