#!/usr/bin/env python3
"""
Vigila las negritas de la prosa (regla 11.29, Orden 236, 24 nov 2026).

Carlos, 24 nov 2026: «usamos las negritas en la prosa de forma algo aleatoria, sin ningún criterio concreto».
El criterio aprobado: en negrita solo el término nuevo la primera vez que se explica y, como mucho, una idea clave por
apartado; además, la pregunta final de cada capítulo. Todo lo demás, en letra normal (o, para un contraste de una palabra,
en cursiva).

Este verificador lee el manuscrito en el orden de Book.txt y suspende si hay una negrita en la prosa (fuera del código, de
los bloques «::: tabla» y «::: muestra» y de las filas de tabla con barras) que no esté en la lista de negritas permitidas,
`herramientas/negritas_permitidas.tsv` del libro (columnas: fichero, texto de la negrita). Esa lista es del libro (lleva su
texto) y por eso vive en el repositorio privado; aquí solo está el programa.

Para añadir una negrita: se decide con Carlos, se escribe en el manuscrito y se añade a la lista en el mismo commit.

Uso:
    python verificar_negritas.py ../../libro-ia-libro
    python verificar_negritas.py --selftest
"""

# ======================= CONSTANTES =======================

LISTA = "herramientas/negritas_permitidas.tsv"
ORDEN = "manuscript/Book.txt"
MANUSCRITO = "manuscript"
BLOQUES_QUE_SE_SALTAN = ("tabla", "muestra")      # los imprimen los programas: sus negritas no son de la prosa

# ==========================================================

import re
import sys
from pathlib import Path

CODIGO = re.compile(r"```.*?```|`[^`\n]*`", re.S)
NEGRITA = re.compile(r"\*\*(.+?)\*\*", re.S)
ABRE_BLOQUE = re.compile(r"^\s*:{3,}\s*\{?\.?([A-Za-záéíóúñ_-]+)")
CIERRA_BLOQUE = re.compile(r"^\s*:{3,}\s*$")


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def en_blanco(m):
    """El texto del código, sustituido por espacios (los saltos de línea se conservan)."""
    return re.sub(r"[^\n]", " ", m.group(0))


def prosa_sin_bloques(texto):
    """El texto con el código, los bloques de tabla y muestra y las filas de tabla en blanco (misma longitud)."""
    texto = CODIGO.sub(en_blanco, texto)
    salida, pila = [], []
    for linea in texto.split("\n"):
        if CIERRA_BLOQUE.match(linea):
            if pila:
                pila.pop()
            salida.append(" " * len(linea))
            continue
        m = ABRE_BLOQUE.match(linea)
        if m:
            pila.append(m.group(1))
            salida.append(" " * len(linea))
            continue
        if any(b in BLOQUES_QUE_SE_SALTAN for b in pila) or linea.lstrip().startswith("|"):
            salida.append(" " * len(linea))
        else:
            salida.append(linea)
    return "\n".join(salida)


def negritas_de_la_prosa(texto):
    """Lista de (línea, texto normalizado) de cada negrita de la prosa."""
    limpio = prosa_sin_bloques(texto)
    return [(limpio.count("\n", 0, m.start()) + 1, norm(m.group(1))) for m in NEGRITA.finditer(limpio)]


def lee_lista(ruta):
    permitidas = {}
    for n, linea in enumerate(Path(ruta).read_text(encoding="utf-8").split("\n"), 1):
        if not linea.strip() or linea.startswith("#"):
            continue
        partes = linea.split("\t")
        if n == 1 and partes[0] == "fichero":
            continue
        if len(partes) != 2:
            raise SystemExit(f"{ruta}, línea {n}: se esperan dos columnas (fichero, texto) separadas por tabulador")
        permitidas.setdefault(partes[0], set()).add(norm(partes[1]))
    return permitidas


def comprueba(raiz):
    raiz = Path(raiz)
    permitidas = lee_lista(raiz / LISTA)
    problemas, vistas = [], 0
    ficheros = [f.strip() for f in (raiz / ORDEN).read_text(encoding="utf-8").split("\n") if f.strip()]
    for f in ficheros:
        ruta = raiz / MANUSCRITO / f
        if not ruta.exists():
            continue
        for linea, texto in negritas_de_la_prosa(ruta.read_text(encoding="utf-8")):
            vistas += 1
            if texto not in permitidas.get(f, set()):
                problemas.append(f"{f}, línea {linea}: negrita que no está en la lista de permitidas: «{texto[:70]}»")
    return problemas, vistas, sum(len(v) for v in permitidas.values())


def selftest():
    # 1. señal: una negrita inventada en la prosa salta, una de la lista no
    texto = "Una frase con **un término nuevo** y otra con **negrita inventada**.\n"
    vistas = [t for _, t in negritas_de_la_prosa(texto)]
    permitidas = {"un término nuevo"}
    saltan = [t for t in vistas if t not in permitidas]
    ok1 = vistas == ["un término nuevo", "negrita inventada"] and saltan == ["negrita inventada"]
    # 2. nulo: el código, las tablas y las muestras no cuentan
    texto = ("Prosa sin negritas.\n\n```\n**en un bloque de código**\n```\n\nY `**en código en línea**`.\n\n"
             "::: tabla\n**en una tabla**\n:::\n\n::: muestra\n**en una muestra**\n:::\n\n| a | **en una fila** |\n")
    ok2 = negritas_de_la_prosa(texto) == []
    # 3. invariante: otros bloques (destacado) son prosa y vigilan; una negrita partida en dos líneas se une; un bloque
    #    anidado que se cierra devuelve la prosa a su sitio
    texto = ("::: destacado\n**dentro de un destacado**\n:::\n\n::: tabla\n**fuera**\n:::\n\n"
             "**partida en\ndos líneas** y luego **normal**\n")
    esperado = ["dentro de un destacado", "partida en dos líneas", "normal"]
    ok3 = [t for _, t in negritas_de_la_prosa(texto)] == esperado
    print(f"[1] señal        una negrita inventada salta y una de la lista no: {ok1}")
    print(f"[2] nulo         código, tablas, muestras y filas con barras no cuentan como prosa: {ok2}")
    print(f"[3] invariante   un destacado sí es prosa, una negrita partida se une y tras cerrar una tabla se vuelve a vigilar: {ok3}")
    if ok1 and ok2 and ok3:
        print("SELFTEST: las tres pruebas pasan.")
        return 0
    print("SELFTEST FALLA")
    return 1


def main():
    if sys.argv[1:] == ["--selftest"]:
        sys.exit(selftest())
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    problemas, vistas, listadas = comprueba(sys.argv[1])
    for p in problemas:
        print(p)
    if problemas:
        print(f"FALLA: {len(problemas)} negrita(s) de la prosa fuera de la lista ({vistas} vistas, {listadas} permitidas).")
        sys.exit(1)
    print(f"verificar_negritas: {vistas} negritas en la prosa, todas en la lista ({listadas} permitidas).")


if __name__ == "__main__":
    main()
