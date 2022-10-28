import numpy as np
from enum import Enum
import os
import tqdm
import tqdm.asyncio
from typing import Dict

from rdflib import Graph, URIRef
import wget

import gzip
import shutil

import asyncio
import aiohttp

from ontobert.data import load_tsv


DATA_PATH = "data"


def _data_path() -> str:
    """
    Returns the absolute path to the "data" folder in the root.
    """
    return os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            os.path.pardir,
            os.path.pardir,
            DATA_PATH,
        )
    )


class Mode(Enum):
    Test = 1
    Train = 2
    Valid = 3
    Full = 4


def _load_ontology_classes(dataset: str) -> np.ndarray:
    filepath = os.path.join(_data_path(), "kg", dataset.lower(), "mid2class.tsv")
    return load_tsv(filepath)


def _freebase_mids_to_wikidata(mids: np.array) -> Dict:
    """
    This function generates three files in the data/kg/fb15k dir:
        * mid2name.tsv: contains the textual label for each of the Freebase
        mids in mids;
        * mid2desc.tsv: contains the textual description for each of the Freebase
        mids in mids;
        * mid2class.tsv: contains the ontological class for each of the Freebase
        mids in mids;
    """

    freebase_wikidata_mappings = (
        "http://storage.googleapis.com/freebase-public/fb2w.nt.gz"
    )
    mappings_archive = os.path.join(_data_path(), "kg", "fb2w.nt.gz")
    mappings_file = os.path.join(_data_path(), "kg", "fb2w.nt")

    mid2name_file = os.path.join(_data_path(), "kg", "fb15k", "mid2name.tsv")
    mid2desc_file = os.path.join(_data_path(), "kg", "fb15k", "mid2desc.tsv")
    mid2class_file = os.path.join(_data_path(), "kg", "fb15k", "mid2class.tsv")

    if not os.path.exists(mappings_archive):
        wget.download(freebase_wikidata_mappings, mappings_archive)

    if not os.path.exists(mappings_file):
        with gzip.open(mappings_archive, "rb") as f_in:
            with open(mappings_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

    print("Reading RDF freebase MIDs to Wikidata mapping file..")

    g = Graph()
    g.parse(mappings_file)

    uris = dict()  # a dictionary of ("MID": "Wikidata URI")
    mid2name = dict()  # a dictionary of ("MID": "entity_label")
    mid2desc = dict()  # a dictionary of ("MID": "entity_description")

    mid2class = list()  # a list of tuples ("mid", "class")

    wid2label = dict()

    mids_not_mapped = []  # a list containing all the MIDs whose Wikidata
    # mapping could not be found

    for mid in tqdm.tqdm(mids, desc="Parsing wikidata mappings"):
        uri = URIRef("http://rdf.freebase.com/ns/m." + mid[3:])
        wikidata_uri = None
        for s, p, o in g.triples((uri, None, None)):
            wikidata_uri = o
            uris[mid] = wikidata_uri.n3()[1:-1]

    async def lookup_mid(mid, session, sem):
        async with sem:
            async with session.get(uris[mid]) as response:
                if response.status != 200:
                    return
                entity = await response.json()
                entities_json = entity["entities"]
                wid = str(next(iter(entities_json.keys())))

                if "en" in entities_json[wid]["labels"]:
                    name = entities_json[wid]["labels"]["en"]["value"]
                    mid2name.update({mid: name})

                if "en" in entities_json[wid]["descriptions"]:
                    desc = entities_json[wid]["descriptions"]["en"]["value"]
                    mid2desc.update({mid: desc})

                if "P31" in entities_json[wid]["claims"]:
                    for claim in entities_json[wid]["claims"]["P31"]:
                        wid = claim["mainsnak"]["datavalue"]["value"]["id"]
                        if wid in wid2label:
                            mid2class.append((mid, wid2label[wid]))
                            print(wid2label[wid])
                        else:
                            class_page = f"https://www.wikidata.org/entity/{wid}"
                            async with session.get(class_page) as response:
                                if response.status != 200:
                                    return
                                class_entity = await response.json()
                                class_entity_json = class_entity["entities"]
                                if "en" in class_entity_json[wid]["labels"]:
                                    class_name = class_entity_json[wid]["labels"]["en"][
                                        "value"
                                    ]
                                    wid2label.update({wid: class_name})
                                    mid2class.append((mid, class_name))
                                    print(class_name)
                                else:
                                    print(f"no label for class {wid}")

    async def lookup_wikidata_uris():
        print("Retrieving names and descriptions from Wikidata..")
        sem = asyncio.Semaphore(10)
        connector = aiohttp.TCPConnector(limit=2)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [lookup_mid(mid, session, sem) for mid in uris]
            for task in tqdm.asyncio.tqdm.as_completed(tasks):
                await task

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(lookup_wikidata_uris())

    with open(mid2name_file, "w", encoding="utf8") as file:
        for mid in tqdm.tqdm(mid2name, desc="Writing mid2name file"):
            file.write(f"{mid}\t{mid2name[mid]}\n")

    with open(mid2desc_file, "w", encoding="utf8") as file:
        for mid in tqdm.tqdm(mid2desc, desc="Writing mid2desc file"):
            file.write(f"{mid}\t{mid2desc[mid]}\n")

    with open(mid2class_file, "w", encoding="utf8") as file:
        for mid_class in tqdm.tqdm(mid2class, desc="Writing mid2class file"):
            file.write(f"{mid_class[0]}\t{mid_class[1]}\n")

    if mids_not_mapped:
        print("Couldn't find a mapping for MIDs:")
        for mid in mids_not_mapped:
            print(mid)

    print("Done")
