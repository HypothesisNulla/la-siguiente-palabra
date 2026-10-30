#!/usr/bin/env python3
"""
Comprueba que ninguna palabra que el libro se apropia se usa antes de bautizarla.

Sale de la ley primera: el lector no puede sostener una palabra que todavía no le han
dado. Los comentarios 1, 6, 9 y 13 de la lectura eran todos este mismo fallo, cazados de
uno en uno. Esto los caza todos a la vez, y los que vengan.

Recorre el manuscrito en el orden de Book.txt, busca dónde se bautiza cada término y dónde
se usa por primera vez, y suspende si el uso va por delante del bautizo.

Los títulos (líneas que empiezan por #) no cuentan como uso: ver notas/VOCABULARIO.md.

También vigila los puentes (regla 5 septies, 30 de octubre de 2026): las palabras de la prensa
que el lector va a encontrar mientras lee —parámetros, modelo de lenguaje / LLM, token, GPU,
ventana de contexto, prompt— se dicen una vez, donde el libro presenta la suya, y después el libro
sigue con la suya. Lee el apartado «Puentes» de notas/VOCABULARIO.md y suspende si:
  (a) «token» o «parámetro», que siguen prohibidos, aparecen en otro sitio que no sea la línea
      de su puente, un bloque de tabla o de muestra, el capítulo 16 o la bibliografía;
  (b) alguna palabra de un puente aparece, en el orden del libro, antes de su puente (quedan
      fuera las tablas, las muestras y los títulos de la bibliografía);
  (c) falta un puente: su frase ancla no está en el libro, o está en otro fichero.
Como todo este verificador, se pasa sobre el libro entero con los borradores encima
(manuscript/ de una copia consolidada, ver README §6); sobre el manuscript/ viejo, los puentes
que aún no existen suspenden.

Uso:
    python verificar_vocabulario.py ../../libro-ia-libro
    python verificar_vocabulario.py ../../libro-ia-libro --selftest
"""

# ======================= CONSTANTES =======================

VOCABULARIO = "notas/VOCABULARIO.md"
ORDEN = "manuscript/Book.txt"
MANUSCRITO = "manuscript"
MARCADO = r'`\\[a-záéíóúñ]+\{(.*?)\}`\{=latex\}'   # envoltorio de imprenta: es formato, no texto
CITA = r'`\\(?:ref|dondeesta|pageref|vpageref)\{[^}]*\}`\{=latex\}'   # cita a una figura o tabla: la etiqueta no es texto
SECCION_JUBILADAS = "## Palabras jubiladas"
SECCION_PUENTES = "## Puentes"
PROHIBIDAS_CON_PUENTE = ("token", "parametro")   # las de la regla 1 que un puente deja decir una vez
EL_MAPA = "mapa-de-los-sotanos"                  # el capítulo de los nombres técnicos: ahí valen todas

# ==========================================================

import argparse
import os
import re
import sys
import unicodedata


def norm(s):
    """Minúsculas y sin tildes, para buscar «parametro» y que valga «parámetro»."""
    return "".join(c for c in unicodedata.normalize("NFD", s.lower())
                   if unicodedata.category(c) != "Mn")


def leer_vocabulario(raiz):
    """Las líneas de VOCABULARIO.md con cuatro campos separados por |."""
    ruta = os.path.join(raiz, VOCABULARIO)
    assert os.path.exists(ruta), f"Se esperaba encontrar {ruta}; no existe"
    terminos = []
    for n, linea in enumerate(open(ruta, encoding="utf-8"), 1):
        if linea.count("|") != 3 or linea[:1] in ("#", " ", "\t"):
            continue
        termino, formas, fichero, ancla = [x.strip() for x in linea.split("|")]
        assert formas and fichero and ancla, \
            f"{VOCABULARIO} línea {n}: se esperaban cuatro campos con contenido; se encontró «{linea.strip()}»"
        terminos.append({"termino": termino,
                         "formas": [f.strip() for f in formas.split(",")],
                         "fichero": fichero,
                         "ancla": ancla})
    assert terminos, f"Se esperaba al menos un término declarado en {ruta}; no hay ninguno"
    return terminos


def leer_jubiladas(raiz):
    """Las palabras retiradas y las frases donde sí se les permite volver.

    Una explicación que se escribe a trozos acumula sinónimos: cada pasada deja una
    palabra nueva y no retira la vieja. Al final la misma cosa se llama de tres maneras en
    tres páginas seguidas y el lector cree que son tres cosas. Esto lo caza."""
    ruta = os.path.join(raiz, VOCABULARIO)
    dentro = False
    jubiladas, excepciones = [], []
    for n, linea in enumerate(open(ruta, encoding="utf-8"), 1):
        if linea.startswith("## "):
            dentro = linea.strip() == SECCION_JUBILADAS
            continue
        if not dentro:
            continue
        m = re.match(r"^-\s+`([^`]+)`", linea)
        if m:
            excepciones.append(m.group(1).strip())
            continue
        if linea.count("|") != 2 or linea[:1] in ("#", " ", "\t"):
            continue
        jubilada, buena, motivo = [x.strip() for x in linea.split("|")]
        assert jubilada and buena and motivo, \
            f"{VOCABULARIO} línea {n}: se esperaban tres campos con contenido; se encontró «{linea.strip()}»"
        jubiladas.append({"jubilada": jubilada, "buena": buena, "motivo": motivo})
    return jubiladas, excepciones


def buscar_jubiladas(lineas, jubiladas, excepciones):
    """Dónde ha vuelto una palabra retirada. Los títulos SÍ cuentan aquí: un apartado
    titulado con la palabra vieja es justo la manera en que estas cosas sobreviven."""
    avisos = []
    for j in jubiladas:
        patron = re.compile(r"\b" + re.escape(j["jubilada"]) + r"\b", re.IGNORECASE)
        for f, n, texto in lineas:
            if not patron.search(texto):
                continue
            if any(e in texto for e in excepciones):
                continue
            avisos.append((f, n, j, texto.strip()))
    return avisos


def leer_orden(raiz):
    ruta = os.path.join(raiz, ORDEN)
    assert os.path.exists(ruta), f"Se esperaba encontrar {ruta}; no existe"
    ficheros = [l.strip() for l in open(ruta, encoding="utf-8") if l.strip()]
    for f in ficheros:
        entero = os.path.join(raiz, MANUSCRITO, f)
        assert os.path.exists(entero), f"{ORDEN} nombra {f}, que no existe en {MANUSCRITO}/"
    return ficheros


def lineas_del_libro(raiz, orden):
    """Todo el manuscrito en orden de lectura, como (fichero, nº de línea, texto)."""
    for f in orden:
        with open(os.path.join(raiz, MANUSCRITO, f), encoding="utf-8") as fh:
            for n, linea in enumerate(fh, 1):
                yield f, n, re.sub(MARCADO, r"\1", re.sub(CITA, "", linea.rstrip("\n")))


def leer_puentes(raiz):
    """Las filas del apartado «Puentes»: palabra | formas | palabra del libro | fichero | ancla."""
    ruta = os.path.join(raiz, VOCABULARIO)
    dentro, puentes = False, []
    for n, linea in enumerate(open(ruta, encoding="utf-8"), 1):
        if linea.startswith("## "):
            dentro = linea.strip() == SECCION_PUENTES
            continue
        if not dentro or linea.count("|") != 4 or linea[:1] in ("#", " ", "\t"):
            continue
        palabra, formas, libro, fichero, ancla = [x.strip() for x in linea.split("|")]
        assert palabra and formas and libro and fichero and ancla, \
            f"{VOCABULARIO} línea {n}: se esperaban cinco campos con contenido; se encontró «{linea.strip()}»"
        puentes.append({"palabra": palabra,
                        "formas": [f.strip() for f in formas.split(",")],
                        "fichero": fichero,
                        "ancla": ancla})
    return puentes


def lineas_con_sitio(raiz, orden):
    """Como lineas_del_libro, y con una marca más: si la línea cae dentro de un bloque de
    tabla o de muestra, o de una sección titulada «Bibliografía». Ahí las palabras de los
    puentes no cuentan (son salida de máquina, texto que imprime un programa o títulos de
    otros): (fichero, nº de línea, texto, aparte)."""
    for f in orden:
        en_bloque = en_bibliografia = False
        with open(os.path.join(raiz, MANUSCRITO, f), encoding="utf-8") as fh:
            for n, linea in enumerate(fh, 1):
                crudo = linea.rstrip("\n")
                if crudo.strip() in ("::: tabla", "::: muestra"):
                    en_bloque = True
                elif en_bloque and crudo.strip() == ":::":
                    en_bloque = False
                titulo = re.match(r"#+\s+(.*)", crudo)
                if titulo:
                    en_bibliografia = norm(titulo.group(1)).startswith("bibliografia")
                yield (f, n, re.sub(MARCADO, r"\1", re.sub(CITA, "", crudo)),
                       en_bloque or en_bibliografia)


def revisar_puentes(raiz):
    """(filas, fallos) de los puentes: ver el docstring del módulo, (a), (b) y (c)."""
    puentes = leer_puentes(raiz)
    if not puentes:
        return [], []       # un libro sin el apartado «Puentes» no tiene puentes que vigilar
    lineas = list(lineas_con_sitio(raiz, leer_orden(raiz)))
    fallos, filas, su_linea = [], [], {}
    for p in puentes:
        pos = next((i for i, (f, n, t, _) in enumerate(lineas) if p["ancla"] in t), None)
        if pos is None:
            fallos.append(f"puente «{p['palabra']}»: falta el puente; la frase ancla «{p['ancla']}» "
                          f"no está en ninguna línea del libro")
            continue
        f, n = lineas[pos][0], lineas[pos][1]
        su_linea[norm(p["palabra"])] = (f, n)
        if f != p["fichero"]:
            fallos.append(f"puente «{p['palabra']}»: declarado en {p['fichero']} y encontrado en {f}")
        patron = re.compile(r"\b(" + "|".join(re.escape(x) for x in p["formas"]) + r")\b",
                            re.IGNORECASE)
        # (b) en el orden del libro, ninguna palabra del puente antes de él
        antes = next(((f2, n2, t2) for f2, n2, t2, aparte in lineas[:pos]
                      if not aparte and patron.search(t2)), None)
        if antes:
            fallos.append(f"puente «{p['palabra']}»: la palabra sale en {antes[0]} línea {antes[1]} "
                          f"y su puente no llega hasta {f} línea {n}\n      {antes[2].strip()[:90]}")
        filas.append((f"puente «{p['palabra']}»", f"{f}:{n}", f"{antes[0]}:{antes[1]}" if antes else f"{f}:{n}"))
    # (a) «token» y «parámetro», solo en su puente (o en tablas, muestras, bibliografía y el mapa)
    for f, n, t, aparte in lineas:
        if aparte or EL_MAPA in f:
            continue
        for prohibida in PROHIBIDAS_CON_PUENTE:
            if prohibida in norm(t) \
                    and not any(prohibida in k and v == (f, n) for k, v in su_linea.items()):
                fallos.append(f"«{prohibida}» fuera de su puente en {f} línea {n}\n      {t.strip()[:90]}")
    return filas, fallos


def sitio(lineas, condicion):
    """Primera posición (índice de lectura) donde se cumple la condición, o None."""
    for i, (f, n, texto) in enumerate(lineas):
        if condicion(texto):
            return i, f, n
    return None


def revisar(raiz):
    terminos = leer_vocabulario(raiz)
    jubiladas, excepciones = leer_jubiladas(raiz)
    lineas = list(lineas_del_libro(raiz, leer_orden(raiz)))
    cuerpo = [(f, n, t) for f, n, t in lineas if not t.lstrip().startswith("#")]
    fallos, filas = [], []

    for t in terminos:
        patron = re.compile(r"\b(" + "|".join(re.escape(x) for x in t["formas"]) + r")\b",
                            re.IGNORECASE)
        bautizo = sitio(lineas, lambda x, a=t["ancla"]: a in x)
        if bautizo is None:
            fallos.append(f"«{t['termino']}»: no encuentro el bautizo. El ancla declarada es "
                          f"«{t['ancla']}» y no aparece en ninguna línea del manuscrito")
            continue
        if bautizo[1] != t["fichero"]:
            fallos.append(f"«{t['termino']}»: el bautizo está declarado en {t['fichero']} "
                          f"y lo he encontrado en {bautizo[1]}")
        uso = sitio(cuerpo, lambda x, p=patron: bool(p.search(x)))
        if uso is None:
            fallos.append(f"«{t['termino']}»: está declarado y no se usa en ninguna parte")
            continue
        # El propio bautizo es un uso: vale que coincidan, no vale que el uso vaya antes.
        pos_bautizo = next(i for i, (f, n, _) in enumerate(cuerpo)
                           if (f, n) == (bautizo[1], bautizo[2]))
        filas.append((t["termino"], f"{bautizo[1]}:{bautizo[2]}", f"{uso[1]}:{uso[2]}"))
        if uso[0] < pos_bautizo:
            fallos.append(f"«{t['termino']}»: se usa en {uso[1]} línea {uso[2]} y no se bautiza "
                          f"hasta {bautizo[1]} línea {bautizo[2]}")

    for f, n, j, texto in buscar_jubiladas(lineas, jubiladas, excepciones):
        fallos.append(f"palabra jubilada «{j['jubilada']}» en {f} línea {n}: "
                      f"ahora se dice «{j['buena']}» ({j['motivo']})\n      {texto[:90]}")
    filas_puentes, fallos_puentes = revisar_puentes(raiz)
    return filas + filas_puentes, fallos + fallos_puentes, len(jubiladas)


def imprimir(filas, fallos, n_jubiladas):
    ancho = max([len(f[0]) for f in filas] + [8])
    print(f"{'término':<{ancho}}  {'se bautiza en':<34}  primer uso")
    for termino, bautizo, uso in filas:
        print(f"{termino:<{ancho}}  {bautizo:<34}  {uso}")
    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    n_puentes = sum(1 for f in filas if f[0].startswith("puente "))
    print(f"PASA: los {len(filas) - n_puentes} términos se presentan antes de usarse, los "
          f"{n_puentes} puentes van antes de su palabra, y ninguna de las "
          f"{n_jubiladas} palabras jubiladas ha vuelto.")
    return 0


# ============================ SELFTEST ============================

def selftest(raiz):
    """Tres pruebas sobre el propio verificador, que es lo que hay que poder creerse."""
    import tempfile, shutil
    fallos = []

    # [1] Test nulo: un término inventado que no se usa en ninguna parte no puede dar
    #     «bien»; tiene que quejarse de que no lo encuentra.
    with tempfile.TemporaryDirectory() as tmp:
        copia = os.path.join(tmp, "libro")
        shutil.copytree(raiz, copia, ignore=shutil.ignore_patterns(".git", "pdf"))
        with open(os.path.join(copia, VOCABULARIO), "a", encoding="utf-8") as fh:
            fh.write("\nzarandaja | zarandaja, zarandajas | cap02-perceptron.md | **zarandaja**\n")
        _, f2, _ = revisar(copia)
        ok = any("zarandaja" in x for x in f2)
        print(f"[1] test nulo         un término que no existe en el libro: "
              f"{'se queja, bien' if ok else 'NO se queja'}")
        if not ok:
            fallos.append("test nulo: un término inexistente pasó la revisión")

    # [2] Señal implantada: se mueve el bautizo de «pesos» al final del libro y el
    #     verificador tiene que cazar el uso adelantado.
    with tempfile.TemporaryDirectory() as tmp:
        copia = os.path.join(tmp, "libro")
        shutil.copytree(raiz, copia, ignore=shutil.ignore_patterns(".git", "pdf"))
        voc = os.path.join(copia, VOCABULARIO)
        texto = open(voc, encoding="utf-8").read().replace(
            "pesos | peso, pesos | cap02-perceptron.md | se les llama **los pesos**",
            "pesos | peso, pesos | cap13-mapa-de-los-sotanos.md | los nombres reales")
        open(voc, "w", encoding="utf-8").write(texto)
        _, f2, _ = revisar(copia)
        ok = any("«pesos»" in x and "no se bautiza hasta" in x for x in f2)
        print(f"[2] señal implantada  bautizo de «pesos» movido al final: "
              f"{'lo caza' if ok else 'NO lo caza'}")
        if not ok:
            fallos.append("señal implantada: moví el bautizo de «pesos» al capítulo 13 y no lo cazó")

    # [2 bis] La otra señal implantada: se mete una palabra jubilada en un TÍTULO, que es
    #         por donde vuelven de verdad, y tiene que cazarla.
    with tempfile.TemporaryDirectory() as tmp:
        copia = os.path.join(tmp, "libro")
        shutil.copytree(raiz, copia, ignore=shutil.ignore_patterns(".git", "pdf"))
        cap = os.path.join(copia, MANUSCRITO, "cap02-perceptron.md")
        # Se añade un título nuevo al final, en vez de cambiar uno que existe: así la
        # prueba no depende de cómo se llamen hoy los apartados del capítulo 2.
        texto = open(cap, encoding="utf-8").read() + "\n\n## Dieciséis lámparas posibles\n"
        open(cap, "w", encoding="utf-8").write(texto)
        _, f2b, _ = revisar(copia)
        ok = any("jubilada «lámparas»" in x for x in f2b)
        print(f"[2] señal implantada  «lámparas» devuelta a un título: "
              f"{'la caza' if ok else 'NO la caza'}")
        if not ok:
            fallos.append("señal implantada: devolví «lámparas» a un título y no la cazó")

    # [3] Invariante del dominio: los títulos no cuentan como uso. «neurona» está en el
    #     título del capítulo 2 y se bautiza dentro; tiene que pasar.
    filas, f3, _ = revisar(raiz)
    ok = not any("«neurona»" in x for x in f3)
    print(f"[3] invariante        «neurona» en el título del capítulo 2: "
          f"{'no cuenta como uso, bien' if ok else 'cuenta como uso, mal'}")
    if not ok:
        fallos.append("invariante: el título del capítulo 2 se está contando como uso")

    # [4] a [7] Los puentes (regla 5 septies), sobre un libro de juguete de tres capítulos: así
    #     la prueba no depende de si el libro real ya los tiene puestos. Puentes de juguete:
    #     «token» en el capítulo 1 y «GPU» en el 2; el 3 es el mapa de los sótanos.
    base = {
        "cap01-a.md": ("# Uno\n\nUn muestrario con una cosa de máquina.\n\n::: muestra\n"
                       "    la GPU dijo hola\n:::\n\nA estos trozos, en la prensa y en inglés, "
                       "se les llama *tokens*; aquí, trozos.\n\n::: tabla\n| trozo | token |\n:::\n"),
        "cap02-b.md": "# Dos\n\nLa tarjeta gráfica —en inglés, GPU— se inventó para los videojuegos.\n",
        "cap03-mapa-de-los-sotanos.md": "# Tres\n\nAquí valen token, parámetro y GPU.\n",
    }

    def juguete(cambios=(), ancla_token="se les llama *tokens*"):
        """Revisa el libro de juguete con `cambios` = [(fichero, viejo, nuevo)]; devuelve los fallos."""
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "notas"))
            os.makedirs(os.path.join(tmp, MANUSCRITO))
            with open(os.path.join(tmp, VOCABULARIO), "w", encoding="utf-8") as fh:
                fh.write("# Vocabulario\n\n## Puentes\n\n"
                         f"token | token, tokens | trozo | cap01-a.md | {ancla_token}\n"
                         "GPU | GPU | tarjeta gráfica | cap02-b.md | —en inglés, GPU—\n")
            with open(os.path.join(tmp, ORDEN), "w", encoding="utf-8") as fh:
                fh.write("\n".join(base) + "\n")
            textos = dict(base)
            for fichero, viejo, nuevo in cambios:
                assert viejo in textos[fichero], f"el libro de juguete no tiene «{viejo}»"
                textos[fichero] = textos[fichero].replace(viejo, nuevo)
            for fichero, texto in textos.items():
                with open(os.path.join(tmp, MANUSCRITO, fichero), "w", encoding="utf-8") as fh:
                    fh.write(texto)
            return revisar_puentes(tmp)[1]

    f4 = juguete()
    ok = not f4
    print(f"[4] puentes en su sitio  «token» y «GPU» en su puente, en una tabla o muestra antes "
          f"de él y en el mapa: {'pasa, bien' if ok else 'SUSPENDE, mal'}")
    if not ok:
        fallos.append(f"puentes en su sitio: el libro de juguete debía pasar y suspende: {f4[0]}")

    f5 = juguete([("cap02-b.md", "videojuegos.", "videojuegos. Cada token es un número.")])
    ok = any("«token» fuera de su puente" in x for x in f5)
    print(f"[5] señal implantada  «token» fuera de su puente: {'lo caza' if ok else 'NO lo caza'}")
    if not ok:
        fallos.append("señal implantada: «token» en un capítulo, fuera de su puente, no se cazó")

    f6 = juguete([("cap01-a.md", "se les llama *tokens*", "se llaman de otra manera")])
    ok = any("falta el puente" in x for x in f6)
    print(f"[6] señal implantada  un puente que falta: {'lo caza' if ok else 'NO lo caza'}")
    if not ok:
        fallos.append("señal implantada: un puente que falta no se cazó")

    f7 = juguete([("cap01-a.md", "Un muestrario", "Una GPU y un muestrario")])
    ok = any("«GPU»" in x and "su puente no llega" in x for x in f7)
    print(f"[7] señal implantada  «GPU» antes de su puente: {'lo caza' if ok else 'NO lo caza'}")
    if not ok:
        fallos.append("señal implantada: «GPU» antes de su puente no se cazó")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan, y los cuatro casos de los puentes.")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("raiz", help="raíz del repositorio del libro")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        return selftest(a.raiz)
    return imprimir(*revisar(a.raiz))


if __name__ == "__main__":
    sys.exit(main())
