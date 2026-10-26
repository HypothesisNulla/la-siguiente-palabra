#!/usr/bin/env python3
"""
Busca los sitios donde el libro usa un concepto de un capítulo anterior sin refrescarlo.

Sale de la lectura de Carlos (11 oct 2026): el lector es una persona que no recuerda con
precisión lo que quedó atrás, y se pierde si el texto da por sabido un concepto explicado
capítulos antes. Criterio suyo: se refresca CUALQUIER concepto de un capítulo anterior, en
su primer uso dentro de cada capítulo.

No decide nada: da candidatos para leer. Dos listas.

  1. Términos del vocabulario (notas/VOCABULARIO.md): para cada término y cada capítulo
     posterior al de su bautizo, el primer uso en el cuerpo de ese capítulo, con su párrafo.
  2. Referencias a un capítulo anterior («el capítulo 4», «capítulos 5 y 7»): cada una, con su
     frase. Un número de capítulo no le recuerda nada a quien lo ha olvidado.

No cuentan los títulos (#), las filas de tabla (|), los bloques de tabla (::: tabla … :::),
los bloques de código ni las líneas de figura (![…]). Sin azar: misma entrada, misma salida.

Uso:
    python buscar_recordatorios.py ../../libro-ia-libro
    python buscar_recordatorios.py ../../libro-ia-libro --selftest
"""

# ======================= CONSTANTES =======================

VOCABULARIO = "notas/VOCABULARIO.md"
ORDEN = "manuscript/Book.txt"
MANUSCRITO = "manuscript"
MARCADO = r'`\\[a-záéíóúñ]+\{(.*?)\}`\{=latex\}'   # envoltorio de imprenta: es formato, no texto
LETRA = "0-9A-Za-zÁÉÍÓÚÜÑáéíóúüñ"                  # lo que puede ir pegado a una palabra
MAX_PARRAFO = 420                                  # caracteres del párrafo que se enseñan
REF_CAPITULO = r'[Cc]ap[íi]tulos? (\d+)((?:(?:, | y | o )\d+)*)'

# ==========================================================

import argparse
import os
import re
import shutil
import sys
import tempfile

from verificar_vocabulario import leer_vocabulario, leer_orden


def numero_de(fichero):
    """El número de capítulo de «cap07-…md»; 0 para el prólogo, 98 para el epílogo, 99 para el apéndice."""
    m = re.match(r'cap(\d+)-', fichero)
    if m:
        return int(m.group(1))
    if fichero.startswith("00"):
        return 0
    if fichero.startswith("98"):
        return 98
    if fichero.startswith("99"):
        return 99
    raise AssertionError(f"Se esperaba un fichero «capNN-…», «00-…», «98-…» o «99-…»; se encontró «{fichero}»")


def cuerpo(ruta):
    """Las líneas del fichero con su número; las que no son cuerpo, vacías.

    Se dejan vacías (no se quitan) para que los números de línea sigan siendo los del
    fichero y los párrafos se corten donde se cortan en el texto."""
    assert os.path.exists(ruta), f"Se esperaba encontrar {ruta}; no existe"
    salida, en_tabla, en_codigo = [], False, False
    for linea in open(ruta, encoding="utf-8"):
        t = linea.rstrip("\n")
        s = t.strip()
        if s.startswith("```"):
            en_codigo = not en_codigo
            salida.append("")
            continue
        if s.startswith("::: tabla"):
            en_tabla = True
            salida.append("")
            continue
        if en_tabla and s.startswith(":::"):
            en_tabla = False
            salida.append("")
            continue
        if en_codigo or en_tabla or s.startswith(("#", "|", "![", ":::")):
            salida.append("")
            continue
        salida.append(re.sub(MARCADO, r'\1', t))
    assert not en_codigo, f"{ruta}: un bloque de código se abre y no se cierra"
    assert not en_tabla, f"{ruta}: un bloque «::: tabla» se abre y no se cierra"
    return salida


def parrafo(lineas, i):
    """El párrafo (líneas seguidas no vacías) que contiene la línea i, en una sola cadena."""
    a = i
    while a > 0 and lineas[a - 1].strip():
        a -= 1
    b = i
    while b + 1 < len(lineas) and lineas[b + 1].strip():
        b += 1
    texto = " ".join(x.strip() for x in lineas[a:b + 1])
    return texto if len(texto) <= MAX_PARRAFO else texto[:MAX_PARRAFO] + " […]"


def primera_linea(lineas, pat):
    """Índice de la línea donde empieza el primer acierto del patrón, buscando párrafo a
    párrafo para no perder un término partido por un salto de línea («el o / exclusivo»)."""
    i, n = 0, len(lineas)
    while i < n:
        if not lineas[i].strip():
            i += 1
            continue
        j = i
        while j + 1 < n and lineas[j + 1].strip():
            j += 1
        trozos, inicio = [], []
        for k in range(i, j + 1):
            inicio.append(sum(len(x) + 1 for x in trozos))
            trozos.append(lineas[k].strip())
        m = pat.search(" ".join(trozos))
        if m:
            k = max(x for x in range(len(inicio)) if inicio[x] <= m.start())
            return i + k
        i = j + 1
    return None


def patron(formas):
    alternativas = "|".join(re.escape(f) for f in sorted(formas, key=len, reverse=True))
    return re.compile(rf'(?<![{LETRA}])(?:{alternativas})(?![{LETRA}])', re.IGNORECASE)


def ficheros_a_leer(raiz, con_borradores):
    """(nombre, ruta) en el orden de Book.txt. Con borradores, lo que haya con el mismo nombre
    en notas/borradores/en-curso/ sustituye al capítulo, como hace construir_lectura.sh."""
    salida = []
    for f in leer_orden(raiz):
        ruta = os.path.join(raiz, MANUSCRITO, f)
        borrador = os.path.join(raiz, "notas", "borradores", "en-curso", f)
        if con_borradores and os.path.exists(borrador):
            ruta = borrador
        salida.append((f, ruta))
    return salida


def buscar(raiz, con_borradores=False):
    """Devuelve (usos, referencias). Cada uso: término, capítulo del bautizo, fichero, línea,
    párrafo. Cada referencia: fichero, línea, capítulos citados, párrafo."""
    terminos = leer_vocabulario(raiz)
    lista = ficheros_a_leer(raiz, con_borradores)
    nombres = [f for f, _ in lista]
    textos = {f: cuerpo(r) for f, r in lista}
    usos = []
    for t in terminos:
        assert t["fichero"] in nombres, \
            f"El término «{t['termino']}» se bautiza en {t['fichero']}, que no está en {ORDEN}"
        b = nombres.index(t["fichero"])
        pat = patron(t["formas"])
        for f in nombres[b + 1:]:
            lineas = textos[f]
            i = primera_linea(lineas, pat)
            if i is not None:
                usos.append({"termino": t["termino"], "bautizo": numero_de(t["fichero"]),
                             "fichero": f, "linea": i + 1, "parrafo": parrafo(lineas, i)})
    referencias = []
    for f in nombres:
        aqui = numero_de(f)
        lineas = textos[f]
        for i, linea in enumerate(lineas):
            for m in re.finditer(REF_CAPITULO, linea):
                citados = [int(x) for x in re.findall(r'\d+', m.group(0))]
                anteriores = [c for c in citados if c < aqui]
                if anteriores:
                    referencias.append({"fichero": f, "linea": i + 1, "citados": anteriores,
                                        "parrafo": parrafo(lineas, i)})
    return usos, referencias


def imprimir(usos, referencias):
    print("1. TÉRMINOS DEL VOCABULARIO: primer uso en cada capítulo posterior al bautizo")
    print(f"   {len(usos)} usos\n")
    for u in usos:
        print(f"[{u['termino']}] bautizo cap. {u['bautizo']} -> {u['fichero']}:{u['linea']}")
        print(f"    {u['parrafo']}\n")
    print("2. REFERENCIAS A UN CAPÍTULO ANTERIOR")
    print(f"   {len(referencias)} referencias\n")
    for r in referencias:
        cit = ", ".join(str(c) for c in r["citados"])
        print(f"[cap. {cit}] {r['fichero']}:{r['linea']}")
        print(f"    {r['parrafo']}\n")


def montar_libro(dir_, capitulos):
    """Un libro de prueba en dir_: {nombre: texto}, en ese orden, y un vocabulario."""
    os.makedirs(os.path.join(dir_, "manuscript"))
    os.makedirs(os.path.join(dir_, "notas"))
    with open(os.path.join(dir_, ORDEN), "w", encoding="utf-8") as fh:
        fh.write("\n".join(capitulos) + "\n")
    for nombre, texto in capitulos.items():
        with open(os.path.join(dir_, MANUSCRITO, nombre), "w", encoding="utf-8") as fh:
            fh.write(texto)
    with open(os.path.join(dir_, VOCABULARIO), "w", encoding="utf-8") as fh:
        fh.write("## Términos\n\ncodo | codo, codos | cap01-a.md | que aquí llamaré **el codo**\n"
                 "o exclusivo | o exclusivo | cap01-a.md | **el o exclusivo**\n")


def selftest(raiz):
    """Tres pruebas: nula, señal implantada e invariante sobre el libro de verdad."""
    base = ("# 1. A\n\nUna neurona que aquí llamaré **el codo** dice lo que sobra.\n\n"
            "Y **el o exclusivo**.\n")
    tmp = tempfile.mkdtemp()
    try:
        # 1. Nula: el término no vuelve a salir; las referencias son al mismo capítulo o a
        #    uno posterior. No debe salir ningún candidato.
        d = os.path.join(tmp, "nula")
        montar_libro(d, {"cap01-a.md": base,
                         "cap02-b.md": "# 2. B\n\nAquí no sale. Lo verás en el capítulo 3.\n",
                         "cap03-c.md": "# 3. El codo\n\n| codo | 1 |\n\nNada. Este capítulo 3.\n"})
        usos, refs = buscar(d)
        assert usos == [] and refs == [], \
            f"Prueba nula: se esperaban 0 usos y 0 referencias; salieron {len(usos)} y {len(refs)}"
        print("selftest 1 (nula): PASA, 0 candidatos donde no hay ninguno")

        # 2. Señal implantada: el codo sale en el título (no cuenta), en una tabla (no cuenta),
        #    y en el cuerpo en las líneas 7 y 9 del cap. 2 (cuenta solo la primera); en el cap. 3,
        #    en la línea 3. Una referencia al capítulo 1 en el cap. 3, línea 5.
        d = os.path.join(tmp, "senal")
        c2 = ("# 2. Los codos\n\n::: tabla\nel codo | 2\n:::\n\n"
              "Aquí vuelve el Codo, sin más.\n\nY otra vez los codos.\n\n"
              "Una frase larga que parte el o\nexclusivo en dos líneas.\n")
        c3 = "# 3. C\n\nCon codos.\n\nComo en el capítulo 1 y en el capítulo 3.\n"
        montar_libro(d, {"cap01-a.md": base, "cap02-b.md": c2, "cap03-c.md": c3})
        usos, refs = buscar(d)
        sitios = [(u["fichero"], u["linea"]) for u in usos]
        esperado = [("cap02-b.md", 7), ("cap03-c.md", 3), ("cap02-b.md", 11)]
        assert sitios == esperado, \
            f"Prueba implantada: se esperaba {esperado} (el último, partido en dos líneas); salió {sitios}"
        rs = [(r["fichero"], r["linea"], r["citados"]) for r in refs]
        assert rs == [("cap03-c.md", 5, [1])], \
            f"Prueba implantada: se esperaba la referencia cap03-c.md:5 al cap. 1; salió {rs}"
        print("selftest 2 (señal implantada): PASA, los dos usos y la referencia en su sitio")
    finally:
        shutil.rmtree(tmp)

    # 3. Invariante sobre el libro de verdad: cada uso está en un capítulo posterior al del
    #    bautizo, es el primero del término en ese capítulo, y no hay más de uno por par.
    usos, refs = buscar(raiz)
    terminos = {t["termino"]: t for t in leer_vocabulario(raiz)}
    nombres = leer_orden(raiz)
    vistos = set()
    for u in usos:
        t = terminos[u["termino"]]
        assert nombres.index(u["fichero"]) > nombres.index(t["fichero"]), \
            f"Invariante: «{u['termino']}» en {u['fichero']}, que no va después de {t['fichero']}"
        lineas = cuerpo(os.path.join(raiz, MANUSCRITO, u["fichero"]))
        antes = [l for l in lineas[:u["linea"] - 1] if patron(t["formas"]).search(l)]
        assert not antes, f"Invariante: «{u['termino']}» ya salía antes de {u['fichero']}:{u['linea']}"
        par = (u["termino"], u["fichero"])
        assert par not in vistos, f"Invariante: dos usos de «{par[0]}» en {par[1]}"
        vistos.add(par)
    print(f"selftest 3 (invariante): PASA, {len(usos)} usos, cada uno el primero de su capítulo "
          f"y posterior al bautizo; {len(refs)} referencias")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("raiz", help="carpeta libro-ia-libro")
    p.add_argument("--borradores", action="store_true",
                   help="leer los capítulos de notas/borradores/en-curso/ cuando los haya")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        return selftest(a.raiz)
    usos, referencias = buscar(a.raiz, a.borradores)
    imprimir(usos, referencias)
    return 0


if __name__ == "__main__":
    sys.exit(main())
