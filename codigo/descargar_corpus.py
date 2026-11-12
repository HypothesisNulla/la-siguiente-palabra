#!/usr/bin/env python3
"""Descarga los libros del capítulo 5 desde Proyecto Gutenberg, a partir de la lista
`datos/entrada/candidatos_es.txt`. Uso: python descargar_corpus.py"""
import os, urllib.request
from concurrent.futures import ThreadPoolExecutor

LISTA = "../datos/entrada/candidatos_es.txt"
DESTINO = "../datos/entrada/corpus_es"
MIN_BYTES = 20_000

def bajar(numero):
    ruta = os.path.join(DESTINO, f"{numero}.txt")
    if os.path.exists(ruta) and os.path.getsize(ruta) > MIN_BYTES:
        return True
    for url in (f"https://www.gutenberg.org/cache/epub/{numero}/pg{numero}.txt",
                f"https://www.gutenberg.org/files/{numero}/{numero}-0.txt"):
        try:
            pet = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            texto = urllib.request.urlopen(pet, timeout=60).read().decode("utf-8", "replace")
            if len(texto) > MIN_BYTES:
                open(ruta, "w", encoding="utf-8").write(texto)
                return True
        except Exception:
            pass
    return False

if __name__ == "__main__":
    os.makedirs(DESTINO, exist_ok=True)
    numeros = [l.split("\t")[0] for l in open(LISTA, encoding="utf-8")]
    with ThreadPoolExecutor(max_workers=10) as ex:
        ok = sum(ex.map(bajar, numeros))
    print(f"{ok} de {len(numeros)} libros descargados en {DESTINO}")
