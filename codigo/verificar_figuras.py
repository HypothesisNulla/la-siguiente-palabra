#!/usr/bin/env python3
"""
Comprueba que cada figura cae en la misma página que el texto que la explica.

Sale de un fallo que solo se ve leyendo el PDF y que no se ve nunca leyendo el
manuscrito: dos de las cuatro figuras del libro quedaban al pie de una página y su
explicación empezaba en la siguiente. El lector leía «el cuadro de la derecha» con la
figura ya pasada. En el manuscrito estaban pegadas; el salto lo ponía la imprenta.

Y como la paginación cambia cada vez que se toca un párrafo anterior, esto no se puede
vigilar a ojo: hay que comprobarlo en cada compilación.

Con una excepción, decidida el 21 de septiembre de 2026: una figura que ocupa la página
entera no deja sitio para ninguna línea detrás, así que su explicación empieza por fuerza en
la página siguiente. Eso no es el fallo que este verificador persigue —que la figura aparezca
DESPUÉS de su explicación—, de modo que a esas figuras se les exige la página siguiente en vez
de la misma. Que una figura ocupe la página entera se comprueba aquí, en el PDF, no se declara
a mano: se mira si en esa página queda algo más que la figura y su pie.

Y otra, decidida por Carlos el 9 de octubre de 2026: la figura que dejaba un hueco en blanco
delante flota ({#fig-x .flota} en el borrador), como las tablas flotantes: va donde se la
anuncia si cabe y, si no, arriba de la página siguiente. El texto que la presenta la nombra por
su número (\ref{fig-x}), así que a esa figura no se le mira el texto de detrás sino su cita: la
primera «figura N» del PDF tiene que estar en la página de la figura o en la anterior; o más
atrás, si lo que hay en medio son solo páginas de otra figura sin texto.

Y otra, decidida por Carlos el 23 de octubre de 2026: la cita tiene que llegar donde el texto
empieza a hablar de la figura, y hay figuras que reúnen cosas que el texto presenta en dos
momentos (la del comité del cap. 2: su recuadro 1 se explica veinte líneas antes que el 2). Para
esas no vale exigirlo a la PRIMERA cita. Ahora se exige: (1) que la última cita antes de la figura
esté en su página o en la anterior, y (2) que toda cita de más atrás diga en qué página está la
figura («Figura N, en la página P», lo que escribe la macro «dondeesta» cuando está lejos). Así el lector
nunca lee la explicación sin saber dónde está el dibujo.

Y dos cosas de la Orden 140 (8 nov 2026), para el capítulo 1 reformado:

  - 11.10, las remisiones. Un capítulo reformado no usa `\\dondeesta` («en esta misma página») sino
    `\\remision{etiqueta}`: «la Figura 3, de arriba,», «de abajo», «de la página siguiente», «de la página
    anterior». Para esos capítulos (los que llevan la macro) se comprueba que el PDF dice exactamente eso
    tantas veces como remisiones hay, y que ninguna dice «de la página N» (una figura lejana: no se remite a lo
    que el lector no tiene delante). Los demás capítulos siguen con la regla vieja.
  - 11.23, los anexos. Una figura de un anexo (`anexo-aNN-…`) no se nombra por su número: cada nota se explica sola
    y la figura va pegada a la suya. A esas figuras no se les pide cita ni texto que las explique; solo que
    estén en el PDF (cuentan en el recuento de pies de figura).

Y una de la Orden 143: una figura del cuerpo que el texto no nombra por su número, la foto del Mark I del cap. 2 (el
pie «El Mark I. Foto: Marina de los Estados Unidos, 1960.» basta), puede declararse en `notas/FIGURAS-SIN-CITA.md`, una
línea «figuras/x.png | motivo». Se trata como las de los anexos: solo tiene que estar en el PDF.

Y una de la Orden 148 (10 nov 2026), la regla 11.9 nueva: «Una figura del cuerpo no lleva números que el cuerpo no
explique». Lo que se puede comprobar: que ninguna figura del cuerpo de los capítulos reformados (los de
`notas/reforma-nov/capitulos-reformados.md`) lleve una cifra con decimales en sus rótulos. Se lee el texto de la imagen
con `tesseract` (si no está instalado, el verificador lo dice y no puede comprobarlo) y se busca una cifra con coma
decimal («0,25»; «3.609» es un millar, no un decimal). Las figuras de los capítulos 1 y 3 que ya la llevaban se declaran,
mientras no se rehagan, en `notas/FIGURAS-CON-DECIMALES.md` (una línea «figuras/x.png | motivo»); una declarada que ya
no lleva decimales sale como fallo, para que se quite de la lista.

Uso:
    python verificar_figuras.py ../../libro-ia-libro ../../PDF/La-siguiente-palabra.pdf
    python verificar_figuras.py ../../libro-ia-libro ../../PDF/La-siguiente-palabra.pdf --selftest
"""

# ======================= CONSTANTES =======================

ORDEN = "manuscript/Book.txt"
MANUSCRITO = "manuscript"
FIGURA = r'^!\[(.*?)\]\((figuras/[^)]+)\)'
FLOTA = r'\)\{#([\w-]+) \.flota\}\s*$'   # ![pie](figuras/x.png){#fig-x .flota}
LARGO_ANCLA = 45          # cuántos caracteres del texto explicativo se buscan en el PDF
SOBRA_PAGINA_ENTERA = 120 # caracteres que quedan en una página, quitados los del pie de la
                          # figura, cuando en esa página no hay nada más que la figura: la
                          # cabecera del capítulo, el «Figura N:» y el folio. Medido sobre el
                          # libro entero: las páginas que solo llevan figura dejan entre 41 y
                          # 51 caracteres, y la que menos deja de todas las demás, 288.

# ==========================================================

import argparse
import os
import re
import shutil
import subprocess
import sys

CON_DECIMALES = "notas/FIGURAS-CON-DECIMALES.md"
REFORMADOS = "notas/reforma-nov/capitulos-reformados.md"
DECIMAL = re.compile(r"(?<![\d.,])\d+,\d+(?!\d)")    # «0,25»; no «3.609» (un millar) ni «1.019»


def texto_por_pagina(pdf):
    n = int(subprocess.run(["pdfinfo", pdf], capture_output=True, text=True,
                           check=True).stdout.split("Pages:")[1].split()[0])
    paginas = {}
    for p in range(1, n + 1):
        t = subprocess.run(["pdftotext", "-f", str(p), "-l", str(p), pdf, "-"],
                           capture_output=True, text=True, check=True).stdout
        paginas[p] = re.sub(r"\s+", " ", t)
    return paginas


def paginas_con_figura(pdf):
    """La página de cada figura, por su número: la página donde está su pie, «Figura N:».
    Hasta el 9 de octubre de 2026 se emparejaba la figura k del manuscrito con la k-ésima página
    que tuviera imagen; con figuras flotantes, dos pueden caer en la misma página y una puede
    salir detrás de la siguiente, y eso descuadraba todo lo que venía después. El número del pie
    lo pone LaTeX en el orden del manuscrito, así que no se descuadra. Se exige además que en esa
    página haya una imagen de verdad."""
    con_imagen = set()
    for l in subprocess.run(["pdfimages", "-list", pdf], capture_output=True, text=True,
                            check=True).stdout.split("\n")[2:]:
        if l.strip():
            con_imagen.add(int(l.split()[0]))
    n = int(subprocess.run(["pdfinfo", pdf], capture_output=True, text=True,
                           check=True).stdout.split("Pages:")[1].split()[0])
    pie = {}
    for p in range(1, n + 1):
        t = subprocess.run(["pdftotext", "-f", str(p), "-l", str(p), pdf, "-"],
                           capture_output=True, text=True, check=True).stdout
        for num in re.findall(r"^Figura (\d+):", t, re.M):
            assert int(num) not in pie, f"el pie de la figura {num} sale dos veces"
            assert p in con_imagen, f"la página {p} tiene el pie de la figura {num} y ninguna imagen"
            pie[int(num)] = p
    assert pie, f"no encontré ningún «Figura N:» en {pdf}"
    assert sorted(pie) == list(range(1, len(pie) + 1)), \
        f"los pies de figura no van de 1 a {len(pie)} sin huecos: {sorted(pie)}"
    return [pie[k] for k in range(1, len(pie) + 1)]


def sin_cita_declaradas(raiz):
    """Las figuras que `notas/FIGURAS-SIN-CITA.md` declara sin cita por número: {«figuras/x.png»}."""
    ruta = os.path.join(raiz, "notas", "FIGURAS-SIN-CITA.md")
    if not os.path.exists(ruta):
        return set()
    return {l.split("|")[0].strip() for l in open(ruta, encoding="utf-8")
            if re.match(r"figuras/\S+\.png\s*\|", l)}


def figuras_del_manuscrito(raiz):
    """Cada figura del libro, en orden de lectura, con la frase que la explica:
    la primera línea de texto que viene después."""
    ficheros = [l.strip() for l in
                open(os.path.join(raiz, ORDEN), encoding="utf-8") if l.strip()]
    salida = []
    for f in ficheros:
        lineas = open(os.path.join(raiz, MANUSCRITO, f), encoding="utf-8").read().split("\n")
        for i, l in enumerate(lineas):
            m = re.match(FIGURA, l)
            if not m:
                continue
            # El texto que explica una figura es prosa, no un encabezado. Un encabezado no
            # explica nada, y además su texto sale también en el índice, así que buscarlo en
            # el PDF daría con la página del índice y no con la del capítulo.
            # Tampoco una tabla editorial («::: tabla» … «:::»): su título sale en versalitas y
            # sus filas, en columnas, así que no se encuentran tal cual en el PDF. Si la figura va
            # seguida de una tabla, el ancla es la prosa que viene detrás de la tabla.
            es_anexo = re.match(r"anexo-a\d+", f) is not None or m.group(2) in sin_cita_declaradas(raiz)
            siguiente, dentro = "", 0
            for x in lineas[i + 1:]:
                if x.startswith(":::") and x.strip(": ") != "":
                    dentro += 1
                elif x.strip() and set(x.strip()) == {":"}:
                    dentro = max(0, dentro - 1)
                elif x.strip() and not dentro and not x.lstrip().startswith("#"):
                    siguiente = x
                    break
            if es_anexo and not siguiente:
                siguiente = "(figura de anexo)"      # no hace falta texto detrás: no se busca
            assert siguiente, f"{f}: la figura {m.group(2)} no tiene texto detrás"
            limpia = lambda t: re.sub(r"\s+", " ", re.sub(r"\*\*|\*|`", "", t)).strip()
            # Una cita por número, `\ref{fig-x}`{=latex}, sale en el PDF como un número que aquí no
            # se conoce: el ancla se corta delante de ella, o se toma detrás si delante no queda
            # bastante texto.
            # Desde el 17 de octubre de 2026 detrás de cada cita va su macro \dondeesta{fig-x} («, en la
            # página siguiente,»): se corta también por ella, o su texto sin expandir se colaría en
            # el ancla y no se encontraría en el PDF.
            # (Orden 140: también las macros de remisión y de notas, `\remision`, `\marca` y `\entrada`.)
            trozos = re.split(r"`\\(?:ref|dondeesta|remision|marca|entrada)\{[^}]+\}(?:\{[^}]*\})?`\{=latex\}",
                              siguiente)
            siguiente = max(trozos, key=lambda x: len(limpia(x))) \
                if len(limpia(trozos[0])) < 15 else trozos[0]
            fl = re.search(FLOTA, l)
            if ".flota" in l:
                assert fl, f"{f}: figura {m.group(2)} marcada para flotar sin el formato " \
                           f"{{#fig-x .flota}}: {l[-60:]}"
            if fl and not es_anexo:
                citas = sum(x.count("\\ref{" + fl.group(1) + "}") for x in lineas)
                assert citas >= 1, f"{f}: la figura flotante {fl.group(1)} no se nombra " \
                                   f"por su número en el capítulo"
            salida.append({"fichero": f, "figura": m.group(2),
                           "pie": limpia(m.group(1)),
                           "ancla": limpia(siguiente)[:LARGO_ANCLA],
                           "flota": bool(fl), "anexo": es_anexo})
    assert salida, f"No encontré ninguna figura en el manuscrito de {raiz}"
    return salida


def pagina_solo_con_la_figura(texto, pie):
    """¿En esta página no hay nada más que la figura y su pie? Se le quita al texto de la
    página lo que ocupa el pie; lo que queda es la cabecera del capítulo y el folio."""
    pelado = lambda t: re.sub(r"\s", "", t)
    return len(pelado(texto)) - len(pelado(pie)) < SOBRA_PAGINA_ENTERA


def solo_figuras_entre(desde, hasta, imgs, figs, paginas):
    """¿Las páginas que hay entre la cita y la figura son todas páginas de una figura sola, sin
    texto? Pasa cuando otra flotante, de página entera, se coloca justo antes (cap. 3, 9 oct 2026:
    la figura 15 ocupa sola la página siguiente a la que cita la 16). El lector no se salta texto:
    pasa una página que es solo figura."""
    entre = range(desde + 1, hasta)
    if not entre:
        return False
    for p in entre:
        pies = [figs[j]["pie"] for j, q in enumerate(imgs) if q == p and j < len(figs)]
        if not pies or not pagina_solo_con_la_figura(paginas[p], " ".join(pies)):
            return False
    return True


def juzgar_flotante(fig, n, pag_img, imgs, figs, paginas):
    """La regla de las flotantes (9 y 23 oct 2026). Devuelve el fallo, o None si pasa."""
    cita = re.compile(rf"[Ff]igura {n}(?!\d)")
    con_pagina = re.compile(rf"[Ff]igura {n}, en la página \d+")
    pag_pie = imgs[n - 1] if n - 1 < len(imgs) else None
    citas = [p for p in sorted(paginas)
             if len(cita.findall(paginas[p])) > (1 if p == pag_pie else 0)]
    if not citas:
        return f"{fig}: flota, y no encuentro en el PDF ninguna cita «figura {n}»"
    previas = [p for p in citas if p <= pag_img]
    if not previas:
        return (f"{fig}: flota y está en la página {pag_img}; todas sus citas, «figura {n}», "
                f"vienen detrás, la primera en la {citas[0]}")
    cerca = lambda p: p in (pag_img, pag_img - 1) or solo_figuras_entre(p, pag_img, imgs, figs, paginas)
    if not cerca(max(previas)):
        return (f"{fig}: flota y está en la página {pag_img}; tenía que haber una cita, «figura "
                f"{n}», en esa página o en la anterior, y la más cercana está en la {max(previas)}")
    for p in previas:
        if not cerca(p) and len(cita.findall(paginas[p])) > len(con_pagina.findall(paginas[p])):
            return (f"{fig}: flota y está en la página {pag_img}; la cita de la página {p} está "
                    f"lejos y no dice en qué página está («figura {n}, en la página …»)")
    return None


def revisar(raiz, pdf, desplazar=0):
    figs = figuras_del_manuscrito(raiz)
    imgs = paginas_con_figura(pdf)
    paginas = texto_por_pagina(pdf)
    fallos, filas = [], []
    if len(imgs) != len(figs):
        fallos.append(f"el manuscrito tiene {len(figs)} figuras y el PDF {len(imgs)} pies de "
                      f"figura; o falta alguna o sobra")
    for k, f in enumerate(figs):
        ancla = figs[(k + desplazar) % len(figs)]["ancla"]
        pag_img = imgs[k] if k < len(imgs) else None
        if f.get("anexo") and desplazar == 0:
            # Figura de un anexo (11.23): no se nombra ni se busca su texto; solo tiene que estar en el PDF.
            filas.append((f["figura"], pag_img, None))
            if pag_img is None:
                fallos.append(f"{f['figura']}: figura de un anexo que no está en el PDF")
            continue
        pag_txt = next((p for p in paginas if ancla in paginas[p]), None)
        filas.append((f["figura"], pag_img, pag_txt))
        if pag_txt is None:
            fallos.append(f"{f['figura']}: no encuentro en el PDF el texto que la explica "
                          f"(«{ancla}…»)")
        elif pag_img is not None:
            entera = pagina_solo_con_la_figura(paginas[pag_img], f["pie"])
            esperada = pag_img + 1 if entera else pag_img
            if f["flota"]:
                # La flotante se busca por su cita: «Figura N» en el texto (juzgar_flotante).
                # (Con «desplazar», el selftest busca a propósito la cita de otra figura.)
                n = (k + desplazar) % len(figs) + 1
                fallo = juzgar_flotante(f["figura"], n, pag_img, imgs, figs, paginas)
                if fallo:
                    fallos.append(fallo)
            elif pag_txt != esperada and entera:
                fallos.append(f"{f['figura']}: ocupa la página {pag_img} entera, así que su "
                              f"explicación tenía que empezar en la {esperada}, y empieza "
                              f"en la {pag_txt}")
            elif pag_txt != esperada:
                fallos.append(f"{f['figura']}: la figura está en la página {pag_img} y el "
                              f"texto que la explica empieza en la {pag_txt}")
    return filas, fallos


REMISION = re.compile(r"(?:[Ff]igura|[Tt]abla) \d+, (de arriba|de abajo|de la página siguiente|de la página anterior)")
REMISION_LEJOS = re.compile(r"(?:[Ff]igura|[Tt]abla) \d+, de la página (\d+)")
DISTANCIA_MAXIMA = 2     # Orden 144.2: «de la página N» vale si la figura cae a dos páginas de su cita, no a más


def texto_de_imagen(ruta):
    """El texto que lleva una imagen, con tesseract; None si tesseract no está instalado."""
    if not shutil.which("tesseract"):
        return None
    return subprocess.run(["tesseract", str(ruta), "-", "-l", "eng", "--psm", "11"], capture_output=True,
                          text=True).stdout


def decimales_de(texto):
    """Las cifras con coma decimal que salen en un texto."""
    return DECIMAL.findall(texto or "")


def declaradas_con_decimales(raiz):
    """{figuras/x.png: motivo} de notas/FIGURAS-CON-DECIMALES.md."""
    ruta = os.path.join(raiz, CON_DECIMALES)
    if not os.path.exists(ruta):
        return {}
    return {a.strip(): b.strip() for a, _, b in (l.partition("|") for l in open(ruta, encoding="utf-8").read().split("\n")
            if l.startswith("figuras/"))}


def figuras_del_cuerpo_reformado(raiz):
    """Las figuras (figuras/x.png) que cita el cuerpo de los capítulos reformados, en orden y sin repetir."""
    ruta = os.path.join(raiz, REFORMADOS)
    caps = [int(m.group(1)) for l in open(ruta, encoding="utf-8") for m in [re.match(r"-\s*cap(\d+)\b", l)] if m] \
        if os.path.exists(ruta) else []
    figs = []
    for n in caps:
        for f in sorted(os.listdir(os.path.join(raiz, MANUSCRITO))):
            if re.match(rf"cap{n:02d}-.*\.md$", f):
                for l in open(os.path.join(raiz, MANUSCRITO, f), encoding="utf-8"):
                    m = re.match(FIGURA, l)
                    if m and m.group(2) not in figs:
                        figs.append(m.group(2))
    return figs


def juzgar_decimales(con_decimales, declaradas):
    """La 11.9 de la Orden 148. `con_decimales` = {figura: [cifras con decimales]} de las del cuerpo reformado.
    Falla una con decimales que no está declarada, y una declarada que ya no los lleva."""
    fallos = []
    for fig, cifras in con_decimales.items():
        if cifras and fig not in declaradas:
            fallos.append(f"11.9: {fig} lleva en sus rótulos cifras con decimales ({', '.join(cifras[:3])}), y una "
                          f"figura del cuerpo no lleva números que el cuerpo no explique")
        if not cifras and fig in declaradas:
            fallos.append(f"11.9: {fig} está en {CON_DECIMALES} y ya no lleva decimales: quítala de la lista")
    for fig in declaradas:
        if fig not in con_decimales:
            fallos.append(f"11.9: {fig} está en {CON_DECIMALES} y no es una figura del cuerpo de un capítulo reformado")
    return fallos


def sin_decimales(raiz):
    """(cuántas figuras se han leído, fallos, aviso): lee el texto de cada figura del cuerpo reformado."""
    figs = figuras_del_cuerpo_reformado(raiz)
    if not shutil.which("tesseract"):
        return 0, [], "AVISO: tesseract no está instalado: no se han podido comprobar los decimales de las figuras (11.9)"
    d = {f: decimales_de(texto_de_imagen(os.path.join(raiz, f))) for f in figs}
    return len(figs), juzgar_decimales(d, declaradas_con_decimales(raiz)), ""


def folio_de(texto_pagina):
    """El número de página impreso: el último número de la página (el pie). None si no acaba en un número."""
    m = re.search(r"(\d+)\s*$", texto_pagina)
    return int(m.group(1)) if m else None


def juzgar_remisiones(n_en_manuscrito, texto_pdf, paginas=None):
    """La 11.10 (Orden 140): el PDF dice «arriba», «abajo», «de la página siguiente» o «de la página anterior»
    tantas veces como remisiones `\\remision` hay en el manuscrito, y ninguna remite a una página lejana.
    Desde la Orden 144.2 también vale «de la página N» si N está a DOS páginas de la página donde se dice (el
    folio de la página de la cita, contra N): hace falta `paginas` ({página del PDF: texto}) para saber dónde se
    dice; sin ella, todo «de la página N» se da por lejano."""
    t = re.sub(r"(\w)- (\w)", r"\1\2", re.sub(r"\s+", " ", texto_pdf))   # palabras partidas a fin de renglón
    fallos = []
    dichas, lejos = len(REMISION.findall(t)), []
    if paginas is None:
        lejos = [m.group(0) for m in REMISION_LEJOS.finditer(t)]
    else:
        for tx in paginas.values():
            tp = re.sub(r"(\w)- (\w)", r"\1\2", re.sub(r"\s+", " ", tx))
            for m in REMISION_LEJOS.finditer(tp):
                f = folio_de(tp)
                if f is not None and abs(int(m.group(1)) - f) <= DISTANCIA_MAXIMA:
                    dichas += 1               # «de la página 32», dicho en la 30: la figura cae a dos páginas
                else:
                    lejos.append(m.group(0) + (f"  (dicho en la página {f})" if f else ""))
    if dichas != n_en_manuscrito:
        fallos.append(f"11.10: el manuscrito tiene {n_en_manuscrito} remisiones «\\remision» y el PDF dice «de "
                      f"arriba», «de abajo», «de la página siguiente» o «de la anterior» {dichas} veces")
    if lejos:
        fallos.append(f"11.10: {len(lejos)} remisiones a una página lejana, que el lector no tiene delante: {lejos[:3]}")
    return fallos


def remisiones(raiz, pdf):
    """Cuántas `\\remision{…}` hay en la prosa del manuscrito y las fallas de la 11.10 en el PDF."""
    ficheros = [l.strip() for l in open(os.path.join(raiz, ORDEN), encoding="utf-8") if l.strip()]
    n = 0
    for f in ficheros:
        for l in open(os.path.join(raiz, MANUSCRITO, f), encoding="utf-8").read().split("\n"):
            if not l.lstrip().startswith("!["):
                n += len(re.findall(r"`\\remision\{[\w-]+\}`\{=latex\}", l))
    if n == 0:
        return 0, []
    paginas = texto_por_pagina(pdf)
    texto = " ".join(paginas.values())
    return n, juzgar_remisiones(n, texto, paginas)


def imprimir(filas, fallos, n_rem=0, n_dec=0, aviso=""):
    print(f"{'figura':<30}{'imagen':>8}{'texto':>8}")
    for fig, a, b in filas:
        print(f"{fig:<30}{str(a):>8}{str(b):>8}")
    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print(f"PASA: las {len(filas)} figuras caen en la misma página que su explicación, las "
          f"de página entera, justo delante de ella, y las flotantes, con una cita en su página "
          f"o en la anterior y las citas de más atrás diciendo su página."
          + (f" {n_rem} remisiones de la 11.10 («de arriba», «de abajo»…) dichas en el PDF." if n_rem else "")
          + (f" {n_dec} figuras del cuerpo reformado, sin cifras con decimales (11.9)." if n_dec else ""))
    if aviso:
        print(aviso)
    return 0


# ============================ SELFTEST ============================

def selftest(raiz, pdf):
    fallos = []

    # [1] Test nulo: si se busca un texto que no está, tiene que decirlo, no callar.
    paginas = texto_por_pagina(pdf)
    hay = any("zarandaja pentagonal" in t for t in paginas.values())
    print(f"[1] test nulo         un texto que no está en el libro: "
          f"{'no aparece, bien' if not hay else 'APARECE, mal'}")
    if hay:
        fallos.append("test nulo: encontré en el PDF un texto que no debería estar")

    # [2] Señal implantada: se emparejan a propósito las figuras con la explicación de
    #     la siguiente. Casi todas tienen que saltar; si no salta ninguna, no comprueba.
    _, f2 = revisar(raiz, pdf, desplazar=1)
    print(f"[2] señal implantada  emparejadas a la figura equivocada: {len(f2)} quejas")
    if not f2:
        fallos.append("señal implantada: emparejé cada figura con la explicación de otra "
                      "y no se quejó de ninguna")

    # [2 bis] Señales implantadas en la regla del 23 oct, sobre la primera flotante del libro:
    #   (a) se le quitan las citas cercanas y se le pone una lejana que sí dice su página:
    #       tiene que saltar, porque falta la cita cerca;
    #   (b) se le deja todo y se añade una cita lejana que NO dice su página: tiene que saltar;
    #   (c) se le añade una cita lejana que sí la dice: tiene que pasar.
    figs = figuras_del_manuscrito(raiz)
    imgs = paginas_con_figura(pdf)
    k = next((j for j, g in enumerate(figs) if g["flota"] and imgs[j] > 3), None)
    if k is None:
        fallos.append("señal implantada 23 oct: no hay ninguna flotante donde probar")
    else:
        n, pi = k + 1, imgs[k]
        cita = re.compile(rf"[Ff]igura {n}(?!\d)(?!:)")
        lejos = pi - 3
        a = dict(paginas)
        for p in (pi - 1, pi):
            a[p] = cita.sub("Figura X", a[p])
        a[lejos] = a[lejos] + f" la Figura {n}, en la página 99, "
        b = dict(paginas); b[lejos] = b[lejos] + f" la Figura {n}. "
        c = dict(paginas); c[lejos] = c[lejos] + f" la Figura {n}, en la página 99, "
        ra, rb, rc = (juzgar_flotante("prueba", n, pi, imgs, figs, x) for x in (a, b, c))
        print(f"[2] señal implantada  flotante {n}: sin cita cerca {'salta' if ra else 'NO salta'}; "
              f"cita lejos sin página {'salta' if rb else 'NO salta'}; "
              f"cita lejos con página {'pasa' if not rc else 'NO pasa'}")
        if not ra:
            fallos.append("señal implantada: una flotante sin cita cerca pasó")
        if not rb:
            fallos.append("señal implantada: una cita lejana sin su página pasó")
        if rc:
            fallos.append(f"señal implantada: una cita lejana con su página no pasó: {rc}")

    # [2 ter] La 11.10: tres textos de juguete. Uno bien (dos remisiones dichas), uno con una que falta y uno con
    #   una que remite a una página lejana: el primero pasa y los otros dos saltan.
    bien = "ver la Figura 1, de arriba, y la Figura 2, de la página siguiente, aquí"
    falta = "ver la Figura 1, de arriba, y la Figura 2 sin más"
    lejos = "ver la Figura 1, de arriba, y la Figura 2, de la página 99, aquí"
    partida = "la Figura 4, de la pá- gina anterior, y la Figura 5, de abajo."
    r = [juzgar_remisiones(2, bien), juzgar_remisiones(2, falta), juzgar_remisiones(2, lejos), juzgar_remisiones(2, partida)]
    # Orden 144.2: «de la página N» vale a dos páginas de la cita (folio 30 → 32) y no a tres (30 → 33)
    dos = {1: "ver la Figura 1, de arriba, y la Figura 2, de la página 32, aquí 30"}
    tres = {1: "ver la Figura 1, de arriba, y la Figura 2, de la página 33, aquí 30"}
    sinfolio = {1: "ver la Figura 1, de arriba, y la Figura 2, de la página 32, aquí"}
    r2 = [juzgar_remisiones(2, " ".join(d.values()), d) for d in (dos, tres, sinfolio)]
    print(f"[2] señal implantada  «de la página N» (Orden 144.2): a dos páginas {'pasa' if not r2[0] else 'NO pasa'}; "
          f"a tres {'salta' if r2[1] else 'NO salta'}; sin folio donde se dice {'salta' if r2[2] else 'NO salta'}")
    if r2[0] or not r2[1] or not r2[2]:
        fallos.append(f"señal implantada: «de la página N» a dos páginas mal juzgado: {r2}")
    print(f"[2] señal implantada  remisiones de la 11.10: bien {'pasa' if not r[0] else 'NO pasa'}; una que falta "
          f"{'salta' if r[1] else 'NO salta'}; una lejana {'salta' if r[2] else 'NO salta'}; palabra partida "
          f"{'pasa' if not r[3] else 'NO pasa'}")
    if r[0] or not r[1] or not r[2] or r[3]:
        fallos.append(f"señal implantada: remisiones de la 11.10 mal juzgadas: {r}")

    # [2 quater] La 11.9 de la Orden 148, sobre dos imágenes de juguete (una con «0,25» y otra sin cifras con
    #   decimales) y sobre listas de juguete: una figura con decimales sin declarar salta; una declarada que ya
    #   no los lleva salta; una con decimales y declarada pasa; un millar («3.609») no es un decimal.
    d1 = {"figuras/a.png": ["0,25"], "figuras/b.png": []}
    r1 = [juzgar_decimales(d1, {}), juzgar_decimales(d1, {"figuras/a.png": "x"}),
          juzgar_decimales({"figuras/a.png": ["0,25"]}, {"figuras/a.png": "x", "figuras/b.png": "y"}),
          juzgar_decimales({"figuras/b.png": []}, {"figuras/b.png": "x"})]
    mil = decimales_de("526.794.304 hojas y 3.609 y 1.019 palabras") + decimales_de("de 0 a 1")
    con = decimales_de("el parecido es 0,25 y el otro, 0,70")
    print(f"[2] señal implantada  decimales de la 11.9: sin declarar {'salta' if r1[0] else 'NO salta'}; declarada pasa "
          f"{'sí' if not juzgar_decimales({'figuras/a.png': ['0,25']}, {'figuras/a.png': 'x'}) else 'NO'}; declarada sin "
          f"decimales {'salta' if r1[3] else 'NO salta'}; un millar cuenta como decimal: {'sí, mal' if mil else 'no, bien'}; "
          f"«0,25 y 0,70»: {len(con)} cifras")
    if not r1[0] or juzgar_decimales({"figuras/a.png": ["0,25"]}, {"figuras/a.png": "x"}) or not r1[3] or mil or len(con) != 2:
        fallos.append(f"señal implantada: los decimales de la 11.9 mal juzgados: {r1} {mil} {con}")
    if shutil.which("tesseract"):
        import tempfile
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        res = []
        for rotulo in ("el parecido: 0,25", "el parecido: casi nada"):
            fig_, ax_ = plt.subplots(figsize=(4, 1.2))
            ax_.axis("off")
            ax_.text(0.5, 0.5, rotulo, ha="center", va="center", fontsize=20)
            with tempfile.TemporaryDirectory() as t_:
                fig_.savefig(os.path.join(t_, "x.png"), dpi=150)
                res.append(decimales_de(texto_de_imagen(os.path.join(t_, "x.png"))))
            plt.close(fig_)
        print(f"[2] señal implantada  imagen con «0,25»: {res[0]}; imagen sin decimales: {res[1]}")
        if not res[0] or res[1]:
            fallos.append(f"señal implantada: tesseract y los decimales mal leídos en imágenes de juguete: {res}")

    # [3] Invariante del dominio: el libro tal como está, tiene que pasar.
    filas, f3 = revisar(raiz, pdf)
    print(f"[3] invariante        el libro tal como está: {len(f3)} fallos en "
          f"{len(filas)} figuras")
    if f3:
        fallos.append(f"invariante: {f3[0]}")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("raiz")
    p.add_argument("pdf")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    assert os.path.exists(a.pdf), f"Se esperaba el PDF compilado en {a.pdf}; no existe"
    if a.selftest:
        return selftest(a.raiz, a.pdf)
    filas, fallos = revisar(a.raiz, a.pdf)
    n_rem, f_rem = remisiones(a.raiz, a.pdf)
    n_dec, f_dec, aviso = sin_decimales(a.raiz)
    return imprimir(filas, fallos + f_rem + f_dec, n_rem, n_dec, aviso)


if __name__ == "__main__":
    sys.exit(main())
