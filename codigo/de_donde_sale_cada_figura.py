#!/usr/bin/env python3
"""
De dónde sale cada figura del libro: una tabla con todas, en orden.

Desde el 17 de octubre de 2026 (Orden 4) las figuras ya no llevan debajo, escrito por el
programa que las dibuja, el texto de dónde vienen sus datos. Eso no se pierde: se junta aquí,
en una sola tabla, calculada y no escrita a mano. Para cada figura del libro, por orden de
lectura: su número, el capítulo, el programa que la dibuja y las salidas guardadas que ese
programa lee (o «ninguna»: el programa no lee ningún fichero).

Cómo se sabe cada cosa, sin que nadie lo apunte:

  - el orden y el número, de `manuscript/Book.txt` y de las líneas «![pie](figuras/x.png)» de cada
    capítulo; por defecto con los borradores de `notas/borradores/en-curso/` encima del
    manuscrito (así sale el PDF de lectura), o con `--sin-borradores`, el manuscrito solo;
  - el programa, de qué programa de `codigo/` nombra el fichero «x.png» en una cadena de su
    código (no en su docstring);
  - las salidas, de qué ficheros de `datos/` nombra ese programa en sus cadenas.

Y revienta, sin escribir nada, si una figura no tiene programa, si tiene más de uno, o si la
numeración no coincide con los «Figura N:» del PDF (de lectura, o del manuscrito con
`--sin-borradores`), que se lee con `pdftotext`.

POR QUÉ ESTA MEDICIÓN NO DEPENDE DE LA MÁQUINA. Se leen ficheros de texto.

Uso:
    python de_donde_sale_cada_figura.py --selftest
    python de_donde_sale_cada_figura.py > ../datos/salidas/de_donde_sale_cada_figura.txt
    python de_donde_sale_cada_figura.py --sin-borradores
"""

# ======================= CONSTANTES =======================

LIBRO = "../../libro-ia-libro"
ORDEN = "manuscript/Book.txt"
MANUSCRITO = "manuscript"
BORRADORES = "notas/borradores/en-curso"
PDF_LECTURA = "../../PDF/La-siguiente-palabra-lectura.pdf"
PDF_MANUSCRITO = "../../PDF/La-siguiente-palabra.pdf"
CODIGO = "."
DATOS = "../datos"

FIGURA = r'^!\[(.*?)\]\(figuras/([^)]+)\.png\)'    # «![pie](figuras/x.png)»
ANCHO_MAXIMO = 56                                    # letras: programa más largo + salida más larga
FILAS_POR_TABLA = 20                                 # figuras por tabla, para que quepa en una página
LARGO_PIE = 25                                       # letras del pie que se cotejan con el PDF
DATO = r'datos/(?:salidas/)?([\w.\-]+\.(?:txt|csv|json))'

# Selftest
FIGURAS_MINI = "![Uno.](figuras/a.png)\n\n![Dos.](figuras/b.png)\n"

# ==========================================================

import argparse
import ast
import os
import platform
import re
import subprocess
import sys
import tempfile
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from formato import ANCHO_CAJA, comprobar_ancho, tabla_editorial

AQUI = Path(__file__).resolve().parent


# ---- el libro --------------------------------------------------------------------------

def capitulo(fichero):
    """«cap03-muchos.md» -> «3»; el prólogo, el epílogo y el apéndice, por su nombre."""
    if fichero.startswith("00-prologo"):
        return "prólogo"
    if fichero.startswith("98-epilogo"):
        return "epílogo"
    if fichero.startswith("99-apendice"):
        return "apéndice"
    m = re.match(r"cap(\d+)-", fichero)
    assert m, f"Se esperaba un nombre de capítulo «capNN-…» y se encontró «{fichero}»"
    return str(int(m.group(1)))


def figuras_del_libro(libro, con_borradores=True):
    """Cada figura en orden de lectura: [{numero, fichero, capitulo, nombre, pie}]."""
    libro = Path(libro)
    ficheros = [l.strip() for l in (libro / ORDEN).read_text(encoding="utf-8").split("\n")
                if l.strip()]
    figuras = []
    for f in ficheros:
        ruta = libro / BORRADORES / f
        if not (con_borradores and ruta.exists()):
            ruta = libro / MANUSCRITO / f
        for linea in ruta.read_text(encoding="utf-8").split("\n"):
            m = re.match(FIGURA, linea)
            if m:
                figuras.append({"numero": len(figuras) + 1, "fichero": f,
                                "capitulo": capitulo(f), "nombre": m.group(2), "pie": m.group(1)})
    assert figuras, f"No encontré ninguna figura en {libro}"
    return figuras


# ---- los programas ---------------------------------------------------------------------

def cadenas(ruta):
    """Las cadenas de un programa, sin su docstring de módulo ni los de sus funciones."""
    arbol = ast.parse(Path(ruta).read_text(encoding="utf-8"))
    docstrings = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            cuerpo = nodo.body
            if cuerpo and isinstance(cuerpo[0], ast.Expr) and isinstance(cuerpo[0].value, ast.Constant) \
                    and isinstance(cuerpo[0].value.value, str):
                docstrings.add(id(cuerpo[0].value))
    return [n.value for n in ast.walk(arbol)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings]


def programa_de(nombre, programas):
    """El único programa que nombra «nombre.png» en una cadena de su código. Revienta si no hay
    ninguno o si hay más de uno. `programas`: {fichero: [cadenas]}."""
    patron = re.compile(rf"(?<![\w]){re.escape(nombre)}\.png")
    candidatos = sorted(p for p, cs in programas.items() if any(patron.search(c) for c in cs))
    assert candidatos, f"la figura «{nombre}.png» no tiene programa: ninguno nombra ese fichero"
    assert len(candidatos) == 1, (
        f"la figura «{nombre}.png» tiene {len(candidatos)} programas: {', '.join(candidatos)}")
    return candidatos[0]


def datos_que_lee(cs):
    """Los ficheros de `datos/` que nombran las cadenas de un programa, sin repetir y en orden."""
    vistos = []
    for c in cs:
        for m in re.finditer(DATO, c):
            ruta = m.group(0)
            if ruta not in vistos:
                vistos.append(ruta)
    return vistos


def leer_programas(codigo):
    """Las cadenas de cada programa de `codigo/`, menos los verificadores y este mismo programa
    (cuyo selftest nombra figuras inventadas)."""
    return {p.name: cadenas(p) for p in sorted(Path(codigo).glob("*.py"))
            if not p.name.startswith("verificar_") and p.name != Path(__file__).name}


def nombre_corto(programa):
    """«figura_la_escalera.py» -> «la_escalera»: en la tabla, el programa sin el prefijo común, para
    que la columna no ensanche la tabla más allá de la caja (un programa que no empieza por
    «figura_», como figuras_el_comite_por_dentro.py, va entero, sin la extensión)."""
    return re.sub(r"^figura_", "", programa[:-3])


def nombre_de_salida(ruta):
    """Cómo se escribe una salida en la tabla, para que quepa en la caja: las de datos/salidas/ sin
    la ruta y sin «.txt» (la extensión de casi todas); las que no son .txt, con la suya; y las que
    están en datos/ y no en datos/salidas/, con un asterisco delante."""
    if ruta.startswith("datos/salidas/"):
        nombre = ruta[len("datos/salidas/"):]
        return nombre[:-4] if nombre.endswith(".txt") else nombre
    return "*" + ruta[len("datos/"):]


def filas_de(figuras, programas, datos):
    """Las filas de la tabla; revienta si una figura no tiene programa o tiene más de uno, o si
    un fichero que el programa dice leer no existe."""
    filas = []
    for f in figuras:
        prog = programa_de(f["nombre"], programas)
        leidos = datos_que_lee(programas[prog])
        for d in leidos:
            assert (Path(datos) / d[len("datos/"):]).exists(), (
                f"«{prog}» nombra «{d}» y ese fichero no existe")
        corto = [nombre_de_salida(d) for d in leidos]
        filas.append([str(f["numero"]), f["capitulo"], nombre_corto(prog),
                      ", ".join(corto) if corto else "ninguna"])
    return filas


# ---- el PDF ----------------------------------------------------------------------------

def solo_letras(t):
    t = unicodedata.normalize("NFKC", t).lower()
    return re.sub(r"[\W_]+", "", t)


def pies_del_pdf(pdf):
    """Los «Figura N: pie» del PDF, por orden de aparición: [(N, texto del pie)]."""
    texto = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True,
                           check=True).stdout
    # Solo al principio de línea, como en verificar_figuras.py: «la Figura 5: …» a media frase es
    # prosa que nombra una figura, no su pie.
    pies = []
    for m in re.finditer(r"^Figura (\d+): ", texto, re.M):
        pies.append((int(m.group(1)), re.sub(r"\s+", " ", texto[m.end():m.end() + 200])))
    return pies


def cotejar_con_pdf(figuras, pies):
    """Revienta si la numeración de las figuras del libro no es la de los «Figura N:» del PDF: los
    mismos números, sin huecos ni repeticiones, y el pie de cada una empieza como el del libro."""
    numeros = [n for n, _ in pies]
    assert numeros == list(range(1, len(figuras) + 1)), (
        f"el libro tiene {len(figuras)} figuras y el PDF «Figura N:» con N = {numeros[:5]}…"
        f"{numeros[-3:]} ({len(numeros)} pies): no coinciden")
    for f, (n, texto) in zip(figuras, pies):
        esperado = solo_letras(re.split(r"`\\ref", f["pie"])[0])[:LARGO_PIE]
        tiene = solo_letras(texto)[:len(esperado)]
        assert esperado == tiene, (
            f"la figura {n} («{f['nombre']}»): el pie del libro empieza «{esperado}» y el del PDF "
            f"«{tiene}»")


# ---- la salida -------------------------------------------------------------------------

def repartir(filas):
    """Las filas en trozos consecutivos, cada uno con a lo sumo FILAS_POR_TABLA filas y de modo que
    el programa más largo y la salida más larga de un mismo trozo sumen como mucho ANCHO_MAXIMO
    letras: la tabla de libro ajusta una sola columna al ancho de la caja, y si entre las dos
    se la comen, un nombre de fichero (que no se parte) se sale de ella. Medido con xelatex el 17
    de octubre de 2026: con 54 y 56 cabe; con 58 («lo_que_encuentra_en_un_paso» junto a
    «*en_que_orden_aprende_curva.csv»), se sale 29 pt."""
    trozos, actual, prog, salida = [], [], 0, 0
    for f in filas:
        p = len(f[2])
        s = max(len(x) for x in f[3].split(", ")) if f[3] != "ninguna" else 0
        if actual and (len(actual) == FILAS_POR_TABLA or max(prog, p) + max(salida, s) > ANCHO_MAXIMO):
            trozos.append(actual)
            actual, prog, salida = [], 0, 0
        actual.append(f)
        prog, salida = max(prog, p), max(salida, s)
    trozos.append(actual)
    assert [f for t in trozos for f in t] == filas, "el reparto perdió o repitió filas"
    return trozos


def tablas(filas):
    """La tabla en trozos de a lo sumo FILAS_POR_TABLA figuras: el libro no parte una tabla entre
    dos páginas, y una de 58 filas no cabe en una (probado con xelatex: caja desbordada)."""
    trozos = repartir(filas)
    out = []
    for k, trozo in enumerate(trozos):
        titulo = (f"De dónde sale cada figura, de la {trozo[0][0]} a la {trozo[-1][0]}"
                  if len(trozos) > 1 else "De dónde sale cada figura")
        out.append(tabla_editorial(
            titulo, ["fig.", "cap.", "programa", "salidas que lee"], trozo, "ddii",
            [# Orden 97: nota para el lector, y sin «< >» (el generador de Word los toma por una marca).
             "«Programa» es el nombre del programa que dibuja la figura: en codigo/, el fichero se "
             "llama así con «figura_» delante y «.py» detrás, salvo cuando el nombre ya empieza por "
             "«figuras_», como figuras_el_comite_por_dentro, que entonces es el nombre entero. "
             "«Salidas que lee» son ficheros de datos/salidas/, con el nombre sin el .txt (si acaban "
             "de otra manera, va escrito); con asterisco, un fichero de datos/ que no está en "
             "salidas/. «Ninguna» quiere decir que el programa no lee ningún fichero: dibuja un "
             "esquema, o calcula él mismo los números, o los trae escritos como ejemplo."]))
    return out


# ---- selftest --------------------------------------------------------------------------

def selftest():
    fallos = []

    # 1. TEST NULO — una figura que ningún programa dibuja, o que dibujan dos, revienta; y una
    #    numeración que no es la del PDF también.
    programas = {"figura_a.py": ['../figuras/a.png', "../datos/salidas/x.txt"],
                 "figura_b.py": ['../figuras/b.png'],
                 "figura_b2.py": ['../figuras/b.png']}
    malas = []
    for nombre, motivo in (("c", "sin programa"), ("b", "dos programas")):
        try:
            programa_de(nombre, programas); malas.append(motivo)
        except AssertionError:
            pass
    figs = [{"numero": 1, "capitulo": "1", "nombre": "a", "pie": "Uno."},
            {"numero": 2, "capitulo": "1", "nombre": "b", "pie": "Dos."}]
    try:
        cotejar_con_pdf(figs, [(1, "Uno."), (3, "Dos.")]); malas.append("numeración distinta")
    except AssertionError:
        pass
    print(f"[1] test nulo         una figura sin programa, con dos programas y con otra numeración: "
          f"{3 - len(malas)} de 3 revientan")
    if malas:
        fallos.append(f"test nulo: no revienta con {malas}")

    # 2. SEÑAL IMPLANTADA — un libro de dos figuras montado a propósito: la primera la dibuja un
    #    programa que lee una salida y la segunda otro que no lee nada; se leen bien el orden, el
    #    capítulo, el programa y las salidas, y «una_capa» no se confunde con «que_es_una_capa».
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        (t / "libro" / "manuscript").mkdir(parents=True)
        (t / "libro" / "notas" / "borradores" / "en-curso").mkdir(parents=True)
        (t / "libro" / "manuscript" / "Book.txt").write_text("cap01-uno.md\ncap02-dos.md\n")
        (t / "libro" / "manuscript" / "cap01-uno.md").write_text("![Uno.](figuras/a.png)\n")
        (t / "libro" / "manuscript" / "cap02-dos.md").write_text("![Viejo.](figuras/zzz.png)\n")
        (t / "libro" / "notas" / "borradores" / "en-curso" / "cap02-dos.md").write_text(
            "![Dos.](figuras/una_capa.png)\n")
        (t / "datos" / "salidas").mkdir(parents=True)
        (t / "datos" / "salidas" / "x.txt").write_text("1\n")
        (t / "codigo").mkdir()
        (t / "codigo" / "figura_a.py").write_text('"""Docstring: nombra otra.png."""\nS = "../datos/salidas/x.txt"\nD = "../figuras/a.png"\n')
        (t / "codigo" / "figura_una_capa.py").write_text('D = "../figuras/una_capa.png"\n')
        (t / "codigo" / "figura_que_es_una_capa.py").write_text('D = "../figuras/que_es_una_capa.png"\n')
        con = figuras_del_libro(t / "libro")
        sin = figuras_del_libro(t / "libro", con_borradores=False)
        filas = filas_de(con, leer_programas(t / "codigo"), t / "datos")
    ok = (filas == [["1", "1", "a", "x"], ["2", "2", "una_capa", "ninguna"]]
          and [f["nombre"] for f in sin] == ["a", "zzz"])
    print(f"[2] señal implantada  dos figuras: {filas[0][2]} lee {filas[0][3]}; {filas[1][2]}: "
          f"{filas[1][3]}; con y sin borradores: {'bien' if ok else 'MAL'}")
    if not ok:
        fallos.append(f"señal implantada: salió {filas} y {[f['nombre'] for f in sin]}")

    # 3. INVARIANTE — en el libro de verdad: todas las figuras tienen un programa y solo uno, los
    #    números van de 1 a N, la tabla cabe y cada fila tiene sus cuatro columnas.
    libro = AQUI / LIBRO
    figuras = figuras_del_libro(libro)
    programas = leer_programas(AQUI / CODIGO)
    filas = filas_de(figuras, programas, AQUI / DATOS)
    trozos = tablas(filas)
    print(f"[3] invariante        {len(figuras)} figuras, cada una con su programa; numeradas de 1 a "
          f"{figuras[-1]['numero']}; {sum(1 for f in filas if f[3] == 'ninguna')} "
          f"sin fichero; en {len(trozos)} tablas de {FILAS_POR_TABLA} filas como mucho")
    if sum(len(x) for x in repartir(filas)) != len(filas) or len(trozos) < -(-len(filas) // FILAS_POR_TABLA):
        fallos.append("invariante: las tablas no reparten todas las filas")
    if [f["numero"] for f in figuras] != list(range(1, len(figuras) + 1)) or len(filas) != len(figuras):
        fallos.append("invariante: la numeración no va de 1 a N")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


# ---- main ------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--sin-borradores", action="store_true",
                    help="solo el manuscrito, cotejado con el PDF del manuscrito")
    ap.add_argument("--pdf", help="otro PDF con el que cotejar la numeración")
    args = ap.parse_args()

    print("--- selftest ---")
    codigo = selftest()
    if codigo or args.selftest:
        sys.exit(codigo)

    con_borradores = not args.sin_borradores
    figuras = figuras_del_libro(AQUI / LIBRO, con_borradores)
    programas = leer_programas(AQUI / CODIGO)
    filas = filas_de(figuras, programas, AQUI / DATOS)
    pdf = Path(args.pdf) if args.pdf else AQUI / (PDF_LECTURA if con_borradores else PDF_MANUSCRITO)
    cotejar_con_pdf(figuras, pies_del_pdf(pdf))

    L = [
        "",
        "########## de dónde sale cada figura del libro ##########",
        f"fecha: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC",
        f"máquina: {platform.system()} {platform.machine()}. La medición no",
        "depende de la máquina: lee ficheros de texto",
        f"libro: {'con los borradores de en-curso/' if con_borradores else 'solo el manuscrito'};",
        f"numeración cotejada con {pdf.name}",
        "",
    ]
    for l in comprobar_ancho([l.rstrip() for l in L], ANCHO_CAJA):
        print(l)
    for trozo in tablas(filas):
        for l in trozo:
            print(l)
        print()


if __name__ == "__main__":
    main()
