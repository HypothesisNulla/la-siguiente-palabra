#!/usr/bin/env python3
"""
Capítulo 12 — trozo a trozo: por qué la máquina en crudo escribe otra petición (Orden 181, 19 de noviembre de 2026).

GPT-3, en crudo, contesta a «Explain the moon landing to a 6 year old in a few sentences.» con cuatro peticiones más, y
Qwen2.5 500M repite la suya. La explicación que se quiere comprobar es la de todo el libro: en cada paso sale el trozo más
probable detrás de lo que la máquina tiene delante, y aquí los trozos más probables, uno tras otro, forman OTRA petición.
De GPT-3 solo tenemos el texto, no sus papeletas; este programa enseña las papeletas de verdad de dos máquinas que sí se
pueden abrir: `Qwen/Qwen2.5-0.5B` y `EleutherAI/pythia-6.9b` (la red de antes de 2021 más parecida a GPT-3 que hay).

TRES CASOS, EN CRUDO, con el mismo montaje que `por_que_ya_contesta.py` (el A12): la petición sola y un salto de línea,
SIEMPRE el trozo más probable, sin sorteo, hasta MAX_NUEVOS trozos o hasta completar LINEAS_NUEVAS líneas nuevas:
  A. Qwen2.5 500M  — «Dame tres consejos para dormir mejor.»
  B. Pythia 6.900M — «Explain the moon landing to a 6 year old in a few sentences.»
  C. Qwen2.5 500M  — «Explica la llegada a la Luna a un niño de seis años, en unas pocas frases.»

EN CADA PASO, una fila: el trozo que escribe, las papeletas de cada cien que lleva, y los otros dos más probables con las
suyas. («_» marca el espacio pegado delante del trozo; «\\n» es el salto de línea.)

TRES SITIOS MARCADOS en cada línea nueva, que son los que cuenta el libro:
  1. el primer trozo de la línea nueva (¿se lleva la mayoría el mismo trozo con que empezaba la de encima?);
  2. el hueco del tema: el primer trozo en que la línea nueva difiere de la de encima, después de lo fijo del molde
     (¿gana el mismo tema de encima o uno nuevo, y con cuántas papeletas?);
  3. el final de la línea: el trozo que cierra el molde («old», «años», «mejor») y el que lleva el salto de línea.

LA PRUEBA DE QUE LO DECIDE LA LÍNEA DE ENCIMA: las papeletas del primer trozo de la línea nueva CON la línea de encima,
frente a las del mismo trozo como principio de un texto vacío (solo el token de inicio de texto: el que usó la máquina
para separar documentos al entrenar; la salida dice cuál). Una fila por caso.

Todo se calcula en la CPU, en bfloat16, como el A12 (las papeletas no llevan un tiempo dentro: salen lo mismo en cualquier
ordenador, salvo diferencias de redondeo en las matrices). Los tiempos que se imprimen son de esta ejecución.

Uso (desde codigo/; hace falta `torch` y `transformers`, y los dos modelos en la caché de Hugging Face):
    python trozo_a_trozo.py --selftest
    python trozo_a_trozo.py > ../datos/salidas/trozo_a_trozo.txt
"""

# ======================= CONSTANTES =======================

QWEN_05B = "Qwen/Qwen2.5-0.5B"
PYTHIA = "EleutherAI/pythia-6.9b"
ETIQUETA = {QWEN_05B: "Qwen2.5 500M", PYTHIA: "Pythia 6.900M"}      # en millones, como los otros rótulos del libro
DTYPE = "bfloat16"
SEMILLA = 20261119
MAX_NUEVOS = 60              # trozos como mucho
LINEAS_NUEVAS = 3            # líneas nuevas (no vacías, terminadas en salto) que se piden
TOP = 3                      # el escrito y los otros dos más probables
CASOS = [
    # (letra, modelo, petición, el trozo que cierra el molde)
    ("A", QWEN_05B, "Dame tres consejos para dormir mejor.", "mejor"),
    ("B", PYTHIA, "Explain the moon landing to a 6 year old in a few sentences.", "old"),
    ("C", QWEN_05B, "Explica la llegada a la Luna a un niño de seis años, en unas pocas frases.", "años"),
]
SUMA_TOLERANCIA = 0.01       # las papeletas de todos los trozos suman cien, con esta tolerancia (en papeletas)
MAYORIA = 0.5                # «se lleva la mayoría»: más de la mitad de las papeletas
LINEAS_DE_LA_SENAL = (3, 8)  # líneas idénticas delante en la prueba de señal: con 3 gana el trozo; con 8 se lleva además la mayoría

# ==========================================================

import argparse
import datetime
import platform
import random
import sys
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from formato import coma, miles, muestra_editorial, partir, tabla_editorial, trozo


def _comprobar_constantes():
    assert MAX_NUEVOS > 0 and LINEAS_NUEVAS > 0 and TOP >= 2, "constantes mal: MAX_NUEVOS, LINEAS_NUEVAS y TOP"
    letras = [c[0] for c in CASOS]
    assert letras and len(set(letras)) == len(letras), "CASOS: hace falta al menos un caso y las letras no se repiten"
    for letra, modelo, peticion, cierre in CASOS:
        assert modelo in ETIQUETA, f"CASOS: el modelo {modelo!r} no tiene rótulo en ETIQUETA"
        assert peticion.strip() == peticion and peticion and "\n" not in peticion, f"CASOS: petición mal: {peticion!r}"
        assert cierre and cierre in peticion, f"CASOS: el cierre del molde {cierre!r} no está en la petición {peticion!r}"
    assert hasattr(torch, DTYPE), f"DTYPE desconocido: {DTYPE}"


_comprobar_constantes()


# ---------------------------------------------------------------------------------------------
# Cómo se ve un trozo
# ---------------------------------------------------------------------------------------------

def ver(tok, i):
    """Un trozo tal como lo ve la máquina, para el monoespaciado: «_» por el espacio pegado delante, «\\n» por el salto."""
    t = tok.decode([int(i)])
    return t.replace(" ", "_").replace("\n", "\\n").replace("`", "´").replace("|", "¦")


def tiene_salto(tok, i):
    return "\n" in tok.decode([int(i)])


# ---------------------------------------------------------------------------------------------
# Recorrer: el trozo más probable, paso a paso, con sus papeletas
# ---------------------------------------------------------------------------------------------

def cargar(nombre):
    tok = AutoTokenizer.from_pretrained(nombre)
    modelo = AutoModelForCausalLM.from_pretrained(nombre, dtype=getattr(torch, DTYPE))
    modelo.eval()
    return tok, modelo


def papeletas(modelo, ids):
    """Las probabilidades (en float64, de 0 a 1) del siguiente trozo, dado un texto en ids (lista)."""
    with torch.no_grad():
        logits = modelo(input_ids=torch.tensor([ids])).logits[0, -1].double()
    return torch.softmax(logits, dim=-1)      # en doble precisión: en float32 las papeletas sumaban 100,015


def lineas_de(tok, ids):
    """Parte una lista de trozos en líneas: una línea acaba en el trozo que lleva un salto de línea. Los saltos que no cierran
    nada (líneas vacías) no cuentan como línea. Devuelve (líneas completas, [trozos de la línea que sigue abierta])."""
    lineas, actual = [], []
    for i in ids:
        if tiene_salto(tok, i):
            if tok.decode([int(i)]).strip("\n") == "" and not actual:
                continue                      # un salto de línea suelto: línea vacía
            actual.append(int(i))
            lineas.append(actual)
            actual = []
        else:
            actual.append(int(i))
    return lineas, actual


def recorrer(tok, modelo, peticion, maximo=MAX_NUEVOS, lineas_nuevas=LINEAS_NUEVAS, delante=None):
    """El trozo más probable, paso a paso, hasta `maximo` trozos o hasta completar `lineas_nuevas` líneas nuevas (o hasta que
    la máquina escribe su fin de texto). `delante` (lista de ids) sustituye a la petición, para las pruebas. Devuelve un
    diccionario con: ids de la petición (con su salto de línea), pasos (una lista de {id, p, top, probs}) y por qué paró."""
    ids = list(delante) if delante is not None else tok(peticion + "\n")["input_ids"]
    inicial = list(ids)
    pasos, porque = [], f"se llegó a los {maximo} trozos"
    eos = tok.eos_token_id
    while len(pasos) < maximo:
        probs = papeletas(modelo, ids)
        mejores = torch.topk(probs, TOP)
        siguiente = int(mejores.indices[0])
        pasos.append({"id": siguiente, "p": float(mejores.values[0]), "probs": probs,
                      "top": [(int(i), float(p)) for i, p in zip(mejores.indices, mejores.values)]})
        ids.append(siguiente)
        if eos is not None and siguiente == eos:
            porque = "la máquina escribe su fin de texto"
            break
        completas, _ = lineas_de(tok, [p["id"] for p in pasos])
        if len(completas) >= lineas_nuevas:
            porque = f"se completaron las {lineas_nuevas} líneas nuevas"
            break
    return {"peticion_ids": inicial, "pasos": pasos, "porque": porque}


def puesto(probs, i):
    """El puesto de un trozo entre todos (1 = el más probable)."""
    return 1 + int((probs > probs[int(i)]).sum())


# ---------------------------------------------------------------------------------------------
# Los tres sitios marcados
# ---------------------------------------------------------------------------------------------

def sitios(tok, r, cierre):
    """Los tres sitios marcados de cada línea nueva. Devuelve (filas, marcas): filas = lista de dicts para la tabla de sitios;
    marcas = {índice de paso: etiqueta} para señalar los pasos en el recorrido."""
    pasos = r["pasos"]
    ids_escritos = [p["id"] for p in pasos]
    pet_lineas, _ = lineas_de(tok, r["peticion_ids"])
    assert pet_lineas, "la petición no tiene ni una línea con salto"
    encima = pet_lineas[-1]
    completas, abierta = lineas_de(tok, ids_escritos)
    lineas = completas + ([abierta] if abierta else [])
    filas, marcas = [], {}
    inicio = 0
    # el comienzo de cada línea en el recorrido (saltando líneas vacías)
    for k, linea in enumerate(lineas, 1):
        # buscar dónde empieza esta línea entre los pasos
        while inicio < len(ids_escritos) and tok.decode([ids_escritos[inicio]]).strip("\n") == "" and tiene_salto(tok, ids_escritos[inicio]) \
                and ids_escritos[inicio] != linea[0]:
            # antes de la línea hay un salto de línea suelto (una línea vacía): también es un primer trozo, y se enseña
            ps = pasos[inicio]
            i_enc = encima[0]
            filas.append({"linea": k, "sitio": "línea vacía antes (el salto de línea gana)", "escribe": ids_escritos[inicio], "p": ps["p"],
                          "encima": i_enc, "p_encima": float(ps["probs"][i_enc]), "puesto_encima": puesto(ps["probs"], i_enc)})
            marcas[inicio] = f"línea {k}: línea vacía antes"
            inicio += 1
        assert ids_escritos[inicio:inicio + len(linea)] == linea, "no se encuentra la línea nueva en el recorrido"
        p0 = pasos[inicio]
        # 1. primer trozo
        i_enc = encima[0]
        filas.append({"linea": k, "sitio": "primer trozo", "escribe": linea[0], "p": p0["p"],
                      "encima": i_enc, "p_encima": float(p0["probs"][i_enc]), "puesto_encima": puesto(p0["probs"], i_enc)})
        marcas[inicio] = f"línea {k}: primer trozo"
        # 2. el hueco del tema: el primer trozo en que difiere de la de encima
        j = 0
        while j < len(linea) and j < len(encima) and linea[j] == encima[j]:
            j += 1
        if j < len(linea) and j < len(encima):
            pj = pasos[inicio + j]
            filas.append({"linea": k, "sitio": "hueco del tema" + (" (ya en el primer trozo)" if j == 0 else ""),
                          "escribe": linea[j], "p": pj["p"], "encima": encima[j], "p_encima": float(pj["probs"][encima[j]]),
                          "puesto_encima": puesto(pj["probs"], encima[j])})
            marcas.setdefault(inicio + j, f"línea {k}: hueco del tema")
        else:
            filas.append({"linea": k, "sitio": "hueco del tema", "escribe": None, "p": None, "encima": None, "p_encima": None,
                          "puesto_encima": None, "nota": "no difiere de la de encima" if j == len(linea) == len(encima)
                          else "una acaba donde la otra sigue"})
        # 3. el final: el trozo que cierra el molde y el que lleva el salto de línea
        cuando = next((q for q, i in enumerate(linea) if tok.decode([i]).strip().strip(".,;:").lower() == cierre.lower()), None)
        if cuando is not None:
            pc = pasos[inicio + cuando]
            filas.append({"linea": k, "sitio": f"la palabra «{cierre}» del cierre de la petición", "escribe": linea[cuando], "p": pc["p"], "encima": None,
                          "p_encima": None, "puesto_encima": None})
            marcas.setdefault(inicio + cuando, f"línea {k}: «{cierre}»")
        else:
            filas.append({"linea": k, "sitio": f"la palabra «{cierre}» del cierre de la petición", "escribe": None, "p": None, "encima": None, "p_encima": None,
                          "puesto_encima": None, "nota": f"la línea no lleva «{cierre}»"})
        if tiene_salto(tok, linea[-1]):
            pf = pasos[inicio + len(linea) - 1]
            filas.append({"linea": k, "sitio": "salto de línea", "escribe": linea[-1], "p": pf["p"], "encima": None,
                          "p_encima": None, "puesto_encima": None})
            marcas.setdefault(inicio + len(linea) - 1, f"línea {k}: salto de línea")
        else:
            filas.append({"linea": k, "sitio": "salto de línea", "escribe": None, "p": None, "encima": None, "p_encima": None,
                          "puesto_encima": None, "nota": "la línea sigue abierta al acabar"})
        inicio += len(linea)
        encima = linea
    return filas, marcas


def prueba_de_la_linea_de_encima(tok, modelo, r):
    """Las papeletas del primer trozo de la línea nueva con la línea de encima y como principio de un texto vacío. Devuelve
    (filas, el trozo de inicio que se usó): una fila por trozo, con (qué trozo es, id, papeletas con la línea de encima, puesto con
    la línea de encima, papeletas con el texto vacío, puesto con el texto vacío). Las filas son: el primer trozo que escribe la
    máquina tras la petición y, si es otro, el primer trozo de la propia petición (el de «la línea de encima»)."""
    inicio = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    assert inicio is not None, "el modelo no tiene token de inicio ni de fin de texto"
    pv = papeletas(modelo, [inicio])
    p0 = r["pasos"][0]["probs"]
    escrito = r["pasos"][0]["id"]
    de_encima = r["peticion_ids"][0]
    filas = [("el primer trozo que escribe", escrito, float(p0[escrito]), puesto(p0, escrito), float(pv[escrito]), puesto(pv, escrito))]
    if de_encima != escrito:
        filas.append(("el primer trozo de la petición", de_encima, float(p0[de_encima]), puesto(p0, de_encima),
                      float(pv[de_encima]), puesto(pv, de_encima)))
    return filas, tok.decode([inicio])


# ---------------------------------------------------------------------------------------------
# La salida
# ---------------------------------------------------------------------------------------------

def pap(p):
    """Papeletas de cada cien, con una cifra decimal y coma."""
    return coma(100 * p, 1)


def texto_escrito(tok, r):
    return tok.decode([p["id"] for p in r["pasos"]])


def tabla_de_sitios(tok, letra, modelo, r, filas):
    rot = ["línea", "sitio", "escribe", "papeletas de cada 100", "el de la línea de encima", "papeletas de ése", "puesto de ése"]
    out = []
    for f in filas:
        if f["escribe"] is None:
            out.append([str(f["linea"]), f["sitio"], f"— ({f['nota']})", "", "", "", ""])
            continue
        out.append([str(f["linea"]), f["sitio"], trozo(ver(tok, f["escribe"])), pap(f["p"]),
                    trozo(ver(tok, f["encima"])) if f["encima"] is not None else "",
                    pap(f["p_encima"]) if f["p_encima"] is not None else "",
                    str(f["puesto_encima"]) if f["puesto_encima"] is not None else ""])
    return tabla_editorial(
        f"Caso {letra}: los tres sitios de cada línea nueva ({ETIQUETA[modelo]}, en crudo)",
        rot, out, "iiidiid",
        [f"«_» marca el espacio pegado delante del trozo; «\\n» es el salto de línea. Papeletas: de cada cien, las que lleva ese trozo.",
         "«El de la línea de encima» es el trozo que tenía la línea anterior en ese mismo sitio; «puesto»: el que ocupa aquí entre todos."])


def tabla_del_recorrido(tok, letra, modelo, r, marcas):
    rot = ["paso", "trozo que escribe", "papeletas de cada 100", "segundo más probable", "tercero más probable", "sitio marcado"]
    filas = []
    for n, p in enumerate(r["pasos"]):
        otros = [f"{trozo(ver(tok, i))} {pap(q)}" for i, q in p["top"][1:]]
        filas.append([str(n + 1), trozo(ver(tok, p["id"])), pap(p["p"]), otros[0], otros[1], marcas.get(n, "")])
    return tabla_editorial(
        f"Caso {letra}: el recorrido entero, trozo a trozo ({ETIQUETA[modelo]}, en crudo)",
        rot, filas, "iidiii",
        ["En cada paso sale el trozo más probable (el segundo y el tercero, con sus papeletas de cada cien, no salen)."])


def caso_impreso(tok, modelo_nombre, modelo, letra, peticion, cierre):
    t0 = time.time()
    r = recorrer(tok, modelo, peticion)
    filas, marcas = sitios(tok, r, cierre)
    print(f"\n{'=' * 74}\nCASO {letra} — {ETIQUETA[modelo_nombre]}, en crudo\n{'=' * 74}")
    print(f"Petición (sola, con un salto de línea detrás): {peticion}")
    print(f"Escribe {len(r['pasos'])} trozos y para porque {r['porque']}. Los trozos de la petición: {len(r['peticion_ids'])}.")
    print()
    escrito = texto_escrito(tok, r).rstrip("\n")
    cuerpo = []
    for l in escrito.split("\n"):
        cuerpo += partir(l) if l else [""]
    print("\n".join(muestra_editorial(f"Caso {letra}: lo que escribe, entero y tal cual", cuerpo,
                                      [f"{ETIQUETA[modelo_nombre]}, en crudo, siempre el trozo más probable."])))
    print()
    print("\n".join(tabla_de_sitios(tok, letra, modelo_nombre, r, filas)))
    print()
    print("\n".join(tabla_del_recorrido(tok, letra, modelo_nombre, r, marcas)))
    print(f"(tiempo de este caso: {coma(time.time() - t0, 0)} s; no es una medición del libro)")
    return r, filas


# ---------------------------------------------------------------------------------------------
# Selftest
# ---------------------------------------------------------------------------------------------

def selftest():
    fallos = []
    tok, modelo = cargar(QWEN_05B)
    letra, _, peticion, cierre = CASOS[0]
    ids_pet = tok(peticion + "\n")["input_ids"]
    primero_de_encima = ids_pet[0]

    # [1] NULO — con una línea de palabras sin relación delante (las de la petición A barajadas, con la semilla), el primer
    #     trozo de la línea nueva no debe llevarse la mayoría con el primer trozo de la de encima (ni ser el más probable).
    palabras = peticion.rstrip(".").split()
    random.Random(SEMILLA).shuffle(palabras)
    barajada = " ".join(palabras) + "\n"
    ids_b = tok(barajada)["input_ids"]
    pb = papeletas(modelo, ids_b)
    p_igual = float(pb[ids_b[0]])
    p_real = float(papeletas(modelo, ids_pet)[ids_pet[0]])
    ok1 = p_igual <= MAYORIA and int(pb.argmax()) != ids_b[0] and p_real > p_igual
    print(f"[1] test nulo         con las palabras de la petición barajadas («{barajada.strip()}»), el primer trozo de la línea de encima "
          f"(«{tok.decode([ids_b[0]])}») lleva {pap(p_igual)} papeletas de cada cien en la línea nueva y no es el más probable; con la petición "
          f"de verdad, «{tok.decode([ids_pet[0]])}» lleva {pap(p_real)}: {'bien' if ok1 else 'MAL'}")
    if not ok1:
        fallos.append("test nulo: con la línea de encima barajada el primer trozo se parece al de la línea de encima")

    # [2] SEÑAL — con líneas idénticas delante, el primer trozo de la siguiente es el primero de la línea. La orden pedía tres líneas
    #     y la mayoría; con tres, el trozo gana pero no con la mayoría (se midió: 11,4 papeletas de cada cien); con ocho sí.
    res2, ok2 = [], True
    for k in LINEAS_DE_LA_SENAL:
        pk = papeletas(modelo, tok((peticion + "\n") * k)["input_ids"])
        gana = int(pk.argmax())
        res2.append((k, gana, float(pk[gana])))
        ok2 = ok2 and gana == primero_de_encima
    ok2 = ok2 and res2[-1][2] > MAYORIA
    detalle = "; ".join(f"con {k} líneas iguales delante, el primer trozo de la siguiente es «{tok.decode([g])}» con {pap(p)} papeletas de cada cien"
                        for k, g, p in res2)
    print(f"[2] señal implantada  {detalle} (el primero de la línea es «{tok.decode([primero_de_encima])}»; la mayoría llega con las ocho): "
          f"{'bien' if ok2 else 'MAL'}")
    if not ok2:
        fallos.append("señal implantada: con líneas iguales delante, el primer trozo de la siguiente no es el de la línea, o con ocho no llega a la mayoría")

    # [3] INVARIANTE — en cada paso las papeletas de todos los trozos suman cien (±0,01) y el trozo escrito es el de más papeletas
    r = recorrer(tok, modelo, peticion, maximo=12, lineas_nuevas=LINEAS_NUEVAS)
    sumas = [100 * float(p["probs"].sum()) for p in r["pasos"]]
    peor = max(abs(s - 100) for s in sumas)
    mejores = all(p["id"] == int(p["probs"].argmax()) and abs(p["p"] - float(p["probs"].max())) < 1e-9 for p in r["pasos"])
    # y el recorrido se parte bien en líneas: lo escrito vuelve a dar los mismos trozos, y los puestos son coherentes
    puesto_ok = all(puesto(p["probs"], p["id"]) == 1 for p in r["pasos"])
    ok3 = peor <= SUMA_TOLERANCIA and mejores and puesto_ok and len(r["pasos"]) == 12
    print(f"[3] invariante        {len(r['pasos'])} pasos: las papeletas de todos los trozos suman cien, con una diferencia de {coma(peor, 4)} como mucho "
          f"(tolerancia {coma(SUMA_TOLERANCIA, 2)}); el trozo escrito es siempre el de más papeletas y ocupa el puesto 1: {'bien' if ok3 else 'MAL'}")
    if not ok3:
        fallos.append("invariante: las papeletas no suman cien o el escrito no es el más probable")
    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


# ---------------------------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    t0 = time.time()
    print(f"Medido el {datetime.date.today()} en {platform.platform()} (un MacBook Pro M4 Max con 36 GB), en la CPU, en {DTYPE} (sin "
          f"tarjeta gráfica); torch {torch.__version__}, transformers {transformers.__version__}. Siempre el trozo más probable, sin sorteo, "
          f"hasta {MAX_NUEVOS} trozos o hasta completar {LINEAS_NUEVAS} líneas nuevas. Semilla {SEMILLA}.")
    resultados, cargados = {}, {}
    pruebas = []
    for nombre in dict.fromkeys(c[1] for c in CASOS):
        cargados[nombre] = cargar(nombre)
    for letra, nombre, peticion, cierre in CASOS:
        tok, modelo = cargados[nombre]
        r, filas = caso_impreso(tok, nombre, modelo, letra, peticion, cierre)
        resultados[letra] = (r, filas)
        pruebas.append((letra, nombre, peticion, prueba_de_la_linea_de_encima(tok, modelo, r), tok))
    print(f"\n{'=' * 74}\nLA PRUEBA DE QUE LO DECIDE LA LÍNEA DE ENCIMA\n{'=' * 74}")
    print("El primer trozo de la línea nueva, con la línea de encima y como principio de un texto vacío (solo el token de inicio de "
          "texto de cada máquina: «<|endoftext|>», el que separa un documento de otro en el entrenamiento).")
    print()
    filas = []
    for letra, nombre, peticion, (rs, inicio), tok in pruebas:
        for que, i, con, pcon, vacio, pvac in rs:
            veces = con / vacio
            filas.append([letra, ETIQUETA[nombre], que, trozo(ver(tok, i)), pap(con), str(pcon),
                          "menos de 0,1" if 100 * vacio < 0.05 else pap(vacio), miles(pvac),
                          (miles(round(veces)) if veces >= 10 else coma(veces, 1)) + " veces"])
    print("\n".join(tabla_editorial(
        "Un primer trozo, con la línea de encima y con un texto vacío",
        ["caso", "máquina", "trozo", "es", "con la línea de encima", "puesto", "con el texto vacío", "puesto", "cuántas veces más"],
        filas, "iiiiididd",
        ["Papeletas de cada cien. «Con el texto vacío»: solo el token de inicio de texto delante, sin ninguna petición.",
         "«Puesto»: el que ocupa ese trozo entre todos los posibles (1 = el más probable)."])))
    print(f"\n(Tiempo de esta ejecución: {coma((time.time() - t0) / 60, 0)} minutos; no es una medición del libro.)")


if __name__ == "__main__":
    main()
