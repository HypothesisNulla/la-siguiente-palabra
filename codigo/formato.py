#!/usr/bin/env python3
"""Cómo se imprimen las tablas del libro: lo ancha que puede ser una línea y cómo se
escribe un número en castellano.

Vive en un solo sitio porque todos los programas del libro imprimen para la misma página.
Si cada uno lleva su propia copia, al mes hay tres anchos distintos y dos maneras de
escribir el mismo decimal, y el lector lo nota aunque no sepa decir qué le chirría.

EL ANCHO. Una tabla que no cabe sale partida o pisando el margen, y entonces no es una
tabla: es un borrón. El ancho está MEDIDO, no supuesto: se compuso una página de prueba
con el preámbulo real del libro (papel de 6 x 9 pulgadas, caja de 4,45 pulgadas, los
bloques de programa a \\footnotesize) y se fue alargando una línea de caracteres iguales
hasta que xelatex avisó de que se salía.

  - bloque normal:             con 68 no avisa; con 69 avisa (se sale 4,42 pt).
  - bloque dentro de una cita: con 62 no avisa; con 63 avisa (se sale 4,07 pt). Medido el 20 de
    septiembre de 2026, con la cita a 1,4 em por lado (fallo 4.44); antes de eso eran 64.

La cita es más estrecha porque su entorno mete margen por los dos lados. Si cambia el
papel, la caja o el cuerpo de letra, estos dos números se vuelven a MEDIR.

LA COMA. El libro está en castellano, y en castellano el decimal va con coma y el signo
de porcentaje con un espacio delante: 90,4 %. Que el programa imprima 90.4% y el libro
escriba 90,4 % es una diferencia invisible mientras se escribe y muy visible impresa.
"""

import re

ANCHO_CAJA = 68
ANCHO_CAJA_CITA = 62   # medido el 20 sep 2026 con el preámbulo actual (cita con margen de 1,4 em)


def comprobar_ancho(lineas, limite=ANCHO_CAJA):
    """Revienta si alguna línea no cabe. Se llama justo antes de imprimir una tabla: más
    vale que falle aquí, donde se lee el mensaje, que en la página impresa."""
    for l in lineas:
        assert len(l) <= limite, (
            f"Se esperaba una línea de {limite} caracteres como mucho; "
            f"se encontró una de {len(l)}: {l!r}"
        )
    return lineas


def coma(x, decimales=1):
    """Un número con coma decimal. 90.4 -> '90,4'."""
    return f"{x:.{decimales}f}".replace(".", ",")


def pct(x, decimales=1, de_uno=True):
    """Un porcentaje a la castellana: coma decimal y espacio antes del signo.

    de_uno=True espera la proporción (0,904 -> '90,4 %'); de_uno=False espera el número
    ya en porcentaje (90.4 -> '90,4 %'). Se dice cuál se le pasa porque confundirlos
    multiplica o divide por cien en silencio, y un dato cien veces mayor es un dato falso."""
    valor = 100 * x if de_uno else x
    return coma(valor, decimales) + " %"


# ---- La tabla de «qué trozo puede venir ahora» -----------------------------------
# Sale en el capítulo 7 y vuelve en el 12, y tiene que salir IGUAL las dos veces: el
# lector reconoce una tabla que ya ha visto mucho antes que un dato que ya ha leído.
ANCHO_TROZO = 12
ANCHO_PROB = 12
ESCALA_BARRA = 28    # almohadillas que valen el 100 %


def tabla_de_probabilidades(pares, con_barra=True, decimales=2):
    """La lista de continuaciones posibles, con rótulo de columna y barra.

    Los rótulos no son adorno. Sin ellos son dos columnas de cifras, y el lector que se
    encuentre la tabla al volver la página no tiene manera de saber qué es cada una.
    La barra no añade ningún dato: deja ver de un vistazo lo que el número dice de una
    en una, que para eso el ojo humano es mucho mejor que para comparar cifras."""
    lineas = [
        f"{'trozo':>{ANCHO_TROZO}}   {'probabilidad':>{ANCHO_PROB}}",
        f"{'-' * ANCHO_TROZO}   {'-' * ANCHO_PROB}",
    ]
    # El trozo llega ya escrito como se va a leer: el guion bajo que marca el espacio
    # lo pone quien conoce el troceador, no esta función, que si no acaba convirtiendo
    # en guion bajo el espacio de un rótulo como «(el resto)».
    for palabra, v in pares:
        fila = f"{palabra:>{ANCHO_TROZO}}   {pct(v, decimales):>{ANCHO_PROB}}"
        if con_barra:
            fila += "  " + "#" * max(1, int(round(v * ESCALA_BARRA)))
        lineas.append(fila.rstrip())
    return comprobar_ancho(lineas, ANCHO_CAJA_CITA)


def miles(n):
    """Un número entero con el separador de miles del castellano: 151936 -> '151.936'.

    Python escribe 151,936 con la coma inglesa, y en un libro en castellano eso se lee
    como un decimal. No es una coquetería: cambia el número que el lector entiende."""
    return f"{int(n):,}".replace(",", ".")


# ---- Tablas con estilo editorial (Carlos, 8 de octubre de 2026) ---------------------
# «Me gustaría que la salida de los scripts saquen tablas con un estilo editorial claro, con
# encabezados y con estilo. Ahora quedan muy pobres.» El programa sigue haciendo las cuentas y
# el libro sigue copiando su salida tal cual (regla 6). Lo que cambia es lo que imprime: en vez de
# columnas alineadas a espacios para un monoespaciado, un bloque con título, una tabla de barras
# en markdown y una nota. El filtro `libro-ia-libro/pdf/tablas.lua` lo convierte en una tabla
# de libro: «Tabla N» y el título en versalitas, filetes arriba, bajo los rótulos y al final,
# rótulos en cursiva, barras grises de verdad y la nota en gris pequeño.
#
#     ::: tabla
#     Lo que puede venir detrás de «La capital de Francia es»
#
#     | trozo siguiente | probabilidad | |
#     |:--|--:|:--|
#     | `_una` | 26,6 % | [barra:26.6] |
#
#     Modelo de 7.000 millones, en crudo. «_» marca el espacio pegado delante del trozo.
#     :::
#
# Cada nota va en un solo renglón: el verificador compara cada renglón del bloque con la salida.

ALINEA = {"i": ":--", "d": "--:", "c": ":-:"}


def celda(x):
    """Una celda: texto sin barras verticales sueltas, que partirían la fila."""
    return str(x).replace("|", "\\|")


def trozo(t):
    """Un trozo de texto tal como lo ve la máquina, en monoespaciado: `_una`."""
    assert "`" not in t, f"Se esperaba un trozo sin acentos graves; se encontró {t!r}"
    return f"`{t}`"


def unidos(piezas):
    """Los trozos de una frase, en UN solo monoespaciado y separados por una barra, sin
    espacios: `Todos|_los|_ser|es`. En papel no se ve el hueco entre dos códigos seguidos
    (Orden 71, 25 oct 2026). En una celda de tabla, tabla_editorial escapa cada barra (`\\|`)."""
    for t in piezas:
        assert "`" not in t and "|" not in t, f"Se esperaba un trozo sin ` ni |; se encontró {t!r}"
    return "`" + "|".join(piezas) + "`"


def barra(p):
    """Una barra gris del largo de la probabilidad p (de 0 a 1). La pinta el filtro."""
    assert 0 <= p <= 1 + 1e-9, f"Se esperaba una probabilidad entre 0 y 1; se encontró {p}"
    return f"[barra:{100 * p:.1f}]"


def tabla_editorial(titulo, rotulos, filas, alineacion, notas=()):
    """Las líneas de un bloque «::: tabla». `alineacion` es una letra por columna: i
    (izquierda), d (derecha) o c (centro). La negrita de una fila (un total) se pone con
    **…** en sus celdas."""
    assert len(alineacion) == len(rotulos), (
        f"Se esperaba una alineación por columna ({len(rotulos)}); hay {len(alineacion)}")
    for f in filas:
        assert len(f) == len(rotulos), (
            f"Se esperaban {len(rotulos)} celdas por fila; una tiene {len(f)}: {f}")
    for x in [titulo, *notas]:
        assert x and "\n" not in x, f"Se esperaba un título o nota de un solo renglón: {x!r}"
    out = ["::: tabla", titulo, "",
           "| " + " | ".join(celda(r) for r in rotulos) + " |",
           "|" + "|".join(ALINEA[a] for a in alineacion) + "|"]
    out += ["| " + " | ".join(celda(c) for c in f) + " |" for f in filas]
    if notas:
        out.append("")
        for n in notas:
            out += [n, ""]
        out.pop()
    out.append(":::")
    return out


def partir(texto, ancho=ANCHO_CAJA):
    """Un texto de máquina en renglones de `ancho` como mucho, cortando solo en un espacio, que
    es lo único que se quita: unidos con un espacio, los renglones vuelven a dar el texto. Ni
    los espacios dobles ni los signos se tocan. Para las muestras largas (L24, 9 oct 2026)."""
    renglones = []
    while len(texto) > ancho:
        corte = texto.rfind(" ", 0, ancho + 1)
        assert corte > 0, f"Se esperaba un espacio donde partir el texto: {texto[:ancho]!r}"
        renglones.append(texto[:corte])
        texto = texto[corte + 1:]
    renglones.append(texto)
    return renglones


def muestra_editorial(titulo, lineas, notas=()):
    """Un texto literal (lo que escribe o lo que recibe la máquina), en monoespaciado, con su
    título y su nota, como las tablas. No lleva número: no es una tabla."""
    for x in [titulo, *notas]:
        assert x and "\n" not in x, f"Se esperaba un título o nota de un solo renglón: {x!r}"
    cuerpo = comprobar_ancho(["    " + l for l in lineas], ANCHO_CAJA + 4)
    out = ["::: muestra", titulo, ""] + cuerpo
    if notas:
        out.append("")
        for n in notas:
            out += [n, ""]
        out.pop()
    out.append(":::")
    return out


def leer_tablas(texto):
    """Las tablas editoriales de una salida, para los programas que dibujan con sus números:
    {título: (rótulos, filas, notas)}. Las celdas, sin las ** de la negrita. Una muestra sale
    como (None, sus renglones sin la sangría, notas). Un título repetido revienta: sería ambiguo
    de cuál se habla."""
    L = texto.split("\n")
    out = {}
    i = 0
    while i < len(L):
        if L[i] == "::: muestra":
            titulo = L[i + 1]
            j = i + 2
            while L[j] != ":::":
                j += 1
            cuerpo = L[i + 2:j]
            assert titulo not in out, f"Se esperaba un título distinto por tabla; «{titulo}» se repite"
            out[titulo] = (None, [l[4:] for l in cuerpo if l.startswith("    ")],
                           [l for l in cuerpo if l.strip() and not l.startswith("    ")])
            i = j
        elif L[i] == "::: tabla":
            titulo = L[i + 1]
            j = i + 2
            while L[j] != ":::":
                j += 1
            cuerpo = [l for l in L[i + 2:j]]
            filas_md = [l for l in cuerpo if l.startswith("|")]
            notas = [l for l in cuerpo if l.strip() and not l.startswith("|")]
            celdas = lambda l: [c.strip().strip("*").replace("\\|", "|")
                                for c in re.split(r"(?<!\\)\|", l.strip()[1:-1])]
            assert len(filas_md) >= 2, f"Se esperaba cabecera y alineación en «{titulo}»"
            assert titulo not in out, f"Se esperaba un título distinto por tabla; «{titulo}» se repite"
            out[titulo] = (celdas(filas_md[0]), [celdas(l) for l in filas_md[2:]], notas)
            i = j
        i += 1
    return out
