#!/usr/bin/env python3
"""
Cuántos programas hay en `codigo/`, y de qué clase.

El apéndice del libro dice cuántos programas tiene el repositorio, y esa cifra se quedaba vieja
cada vez que se añadía uno. Aquí se cuenta, con una regla escrita, y la cifra del libro sale de
esta salida.

LA REGLA. Cada programa `codigo/*.py` cae en UNA clase, mirando en este orden:

  1. verificador: su nombre empieza por `verificar_`;
  2. programa de figura: su nombre empieza por `figura_` o `figuras_` (el segundo cubre
     `figuras_el_comite_por_dentro.py`, que dibuja cinco figuras de una vez);
  3. programa de medición: no es de las dos clases anteriores y escribe una salida en
     `datos/salidas/`, que se sabe de dos maneras: porque ahí hay un fichero `.txt` o `.csv` cuyo
     nombre es el del programa o empieza por el del programa y un guion bajo
     (`en_serie_o_a_la_vez_cpu.txt` es del programa `en_serie_o_a_la_vez.py`); o porque el
     programa dice en su código que escribe en `datos/salidas/<su nombre>.` (la segunda hace que
     este mismo programa sea de medición antes de que exista su salida, y no cambie de clase la
     primera vez que se guarda);
  4. programa que ayuda a los demás: todo lo demás (el formato de las tablas, la hoja de estilo
     de las figuras, un módulo que otros importan…).

No se cuenta ningún fichero que no sea `.py` (hay un `medir_en_mac.sh`) ni los de las carpetas
de dentro de `datos/salidas/`.

EL CORPUS (Orden 14, 17 oct 2026). El apéndice dice cuántos libros hay en `datos/entrada/corpus_es/` y
cuántos bytes ocupan. Se cuentan los ficheros de esa carpeta (solo los de dentro, no los de
subcarpetas) y se suman los bytes **de los ficheros** (su tamaño, el que dice `st_size`), no los
bloques que ocupan en el disco, que dependen del disco.

POR QUÉ ESTA MEDICIÓN NO DEPENDE DE LA MÁQUINA. Se cuentan ficheros.

Uso:
    python cuenta_el_repositorio.py --selftest
    python cuenta_el_repositorio.py > ../datos/salidas/cuenta_el_repositorio.txt
"""

# ======================= CONSTANTES =======================

CODIGO = "."
SALIDAS = "../datos/salidas"
CORPUS = "../datos/entrada/corpus_es"      # los libros de dominio público con los que se entrena

CLASES = ["verificadores", "programas de figuras", "programas de medición",
          "programas que ayudan a los demás"]

# ==========================================================

import argparse
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from formato import ANCHO_CAJA, coma, comprobar_ancho, miles, partir, tabla_editorial

AQUI = Path(__file__).resolve().parent


def clase_de(nombre, salidas, fuente=""):
    """La clase de un programa (el nombre del fichero, con .py) según la regla de arriba.
    `salidas`: los nombres de ficheros de datos/salidas/; `fuente`: el código del programa."""
    raiz = nombre[:-3]
    if nombre.startswith("verificar_"):
        return CLASES[0]
    if nombre.startswith(("figura_", "figuras_")):
        return CLASES[1]
    if f"datos/salidas/{raiz}." in fuente:
        return CLASES[2]
    for s in salidas:
        if s.endswith((".txt", ".csv")):
            parada = s.rsplit(".", 1)[0]
            if parada == raiz or parada.startswith(raiz + "_"):
                return CLASES[2]
    return CLASES[3]


def contar(codigo, salidas):
    """{clase: [nombres de programa sin .py]} de los .py de `codigo`, y cada uno en una sola clase."""
    nombres = sorted(p.name for p in Path(codigo).glob("*.py"))
    ficheros = [p.name for p in Path(salidas).iterdir() if p.is_file()] if Path(salidas).exists() else []
    clases = {c: [] for c in CLASES}
    for n in nombres:
        fuente = (Path(codigo) / n).read_text(encoding="utf-8")
        clases[clase_de(n, ficheros, fuente)].append(n[:-3])
    return clases


def contar_corpus(carpeta):
    """(cuántos ficheros, cuántos bytes suman) en la carpeta, sin entrar en subcarpetas. Los bytes
    son los de los ficheros (`st_size`), no los bloques del disco."""
    ficheros = [p for p in Path(carpeta).iterdir() if p.is_file()] if Path(carpeta).exists() else []
    return len(ficheros), sum(p.stat().st_size for p in ficheros)


# ---- selftest --------------------------------------------------------------------------

def selftest():
    fallos = []

    # 1. TEST NULO — una carpeta sin ningún programa tiene cero en las cuatro clases.
    with tempfile.TemporaryDirectory() as t:
        (Path(t) / "codigo").mkdir()
        (Path(t) / "salidas").mkdir()
        vacio = contar(Path(t) / "codigo", Path(t) / "salidas")
    print(f"[1] test nulo         una carpeta sin programas: {[len(v) for v in vacio.values()]}")
    if any(vacio.values()):
        fallos.append(f"test nulo: cuenta programas donde no hay ninguno: {vacio}")

    # 2. SEÑAL IMPLANTADA — una carpeta con un programa de cada clase, y uno que parece de
    #    medición por su nombre pero no tiene salida, con los que ayudan; un verificador con salida
    #    sigue siendo verificador; un programa cuya salida empieza por su nombre y un guion bajo,
    #    o que dice que escribe en datos/salidas/<su nombre>., es de medición.
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        (t / "codigo").mkdir()
        (t / "salidas").mkdir()
        for n in ("verificar_a.py", "figura_b.py", "medidor.py", "ayuda.py", "sin_salida.py",
                  "serie.py", "figuras_c.py"):
            (t / "codigo" / n).write_text("")
        (t / "codigo" / "propio.py").write_text('"""python propio.py > ../datos/salidas/propio.txt"""')
        (t / "codigo" / "notas.sh").write_text("")
        for s in ("medidor.txt", "verificar_a.txt", "serie_cpu.csv", "otra.txt"):
            (t / "salidas" / s).write_text("")
        (t / "salidas" / "carpeta").mkdir()
        c = contar(t / "codigo", t / "salidas")
    esperado = {CLASES[0]: ["verificar_a"], CLASES[1]: ["figura_b", "figuras_c"],
                CLASES[2]: ["medidor", "propio", "serie"], CLASES[3]: ["ayuda", "sin_salida"]}
    print(f"[2] señal implantada  8 programas, 1 .sh y 4 salidas: "
          f"{[len(v) for v in c.values()]}; como se espera: {'sí' if c == esperado else 'NO'}")
    if c != esperado:
        fallos.append(f"señal implantada: salió {c}")

    # 3. INVARIANTE — en el repositorio de verdad, cada programa está en una clase y solo en una, y
    #    las cuatro cuentas suman todos los .py de codigo/.
    clases = contar(AQUI / CODIGO, AQUI / SALIDAS)
    todos = sorted(p.name[:-3] for p in (AQUI / CODIGO).glob("*.py"))
    juntos = sorted(n for v in clases.values() for n in v)
    ok = juntos == todos and len(set(juntos)) == len(juntos)
    print(f"[3] invariante        {sum(len(v) for v in clases.values())} programas en cuatro clases "
          f"({', '.join(str(len(v)) for v in clases.values())}); de los {len(todos)} .py de codigo/, "
          f"cada uno en una sola: {'sí' if ok else 'NO'}")
    if not ok:
        fallos.append("invariante: algún programa falta, o está en dos clases")

    # y el corpus: una carpeta vacía suma cero; una de tres ficheros de 10, 20 y 30 bytes, con una
    # subcarpeta que no cuenta, suma 3 y 60; y en el corpus de verdad, los bytes de st_size son los que
    # se leen al abrir cada fichero.
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        (t / "vacio").mkdir()
        (t / "tres").mkdir()
        (t / "tres" / "sub").mkdir()
        for nombre, n in (("a", 10), ("b", 20), ("c", 30)):
            (t / "tres" / nombre).write_bytes(b"x" * n)
        (t / "tres" / "sub" / "d").write_bytes(b"x" * 1000)
        vacio, tres = contar_corpus(t / "vacio"), contar_corpus(t / "tres")
    n_c, b_c = contar_corpus(AQUI / CORPUS)
    leidos = sum(len(p.read_bytes()) for p in (AQUI / CORPUS).iterdir() if p.is_file())
    print(f"                      corpus: vacío {vacio}; tres ficheros de 10, 20 y 30 bytes {tres}; "
          f"el de verdad, {miles(n_c)} ficheros y {miles(b_c)} bytes, leídos {miles(leidos)}")
    if vacio != (0, 0) or tres != (3, 60) or leidos != b_c or n_c == 0:
        fallos.append(f"corpus: vacío {vacio}, tres {tres}, st_size {b_c} y leídos {leidos}")

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
    args = ap.parse_args()
    print("--- selftest ---")
    codigo = selftest()
    if codigo or args.selftest:
        sys.exit(codigo)

    clases = contar(AQUI / CODIGO, AQUI / SALIDAS)
    total = sum(len(v) for v in clases.values())
    L = [
        "",
        "########## cuántos programas hay en codigo/ ##########",
        f"fecha: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC",
        f"máquina: {platform.system()} {platform.machine()}. La medición no",
        "depende de la máquina: cuenta ficheros",
        "",
        "La regla, por orden: verificador (verificar_*), figura (figura_*",
        "o figuras_*), medición (tiene salida en datos/salidas/ con su",
        "nombre) y, si no, ayuda a los demás.",
        "",
    ]
    for l in comprobar_ancho([l.rstrip() for l in L], ANCHO_CAJA):
        print(l)
    for l in tabla_editorial(
            "Programas de codigo/, por clase", ["clase", "programas"],
            [[c, str(len(clases[c]))] for c in CLASES] + [["**en total**", f"**{total}**"]], "id",
            ["Cada programa está en una sola clase."]):
        print(l)
    n_corpus, b_corpus = contar_corpus(AQUI / CORPUS)
    print()
    print("EL CORPUS DE ENTRENAMIENTO")
    print()
    print(f"  ficheros de datos/entrada/corpus_es/                {miles(n_corpus):>14}")
    print(f"  bytes que suman (los de los ficheros)       {miles(b_corpus):>14}")
    print(f"  en millones de bytes                        {coma(b_corpus / 1e6, 1):>14}")
    for c in CLASES:
        cabecera = f"{c.upper()} ({len(clases[c])})"
        print()
        print(cabecera)
        print()
        for l in partir(", ".join(clases[c]) + ".", ANCHO_CAJA - 2):
            print("  " + l)


if __name__ == "__main__":
    main()
