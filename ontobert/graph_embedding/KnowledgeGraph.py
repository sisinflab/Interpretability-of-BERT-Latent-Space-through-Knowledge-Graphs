import numpy as np


class KnowledgeGraph():
    name: str
    raw_triples: np.ndarray
    entities: np.array  # array di label (string)
    relations: np.array  # array di label
    entity_names: np.array = None  # nomi delle entità (ordinati come entities)
    entity_descriptions: np.array = None  # descrizioni testuali delle entità
    relation_names: np.array = None
    relation_descriptions: np.array = None

    #    entity_types (TODO)

    def __init__(
        self,
        name,
        raw_triples,
        entity_names_map={},
        entity_descriptions_map={},
        entity_classes: np.ndarray=None,
        relation_names_map={},
        relation_descriptions_map={},
        filter_out_missing_names=True,  # TODO: maybe we need this for both entities and relations
        filter_out_missing_descriptions=True,  # TODO: maybe we need this for both entities and relations
    ):
        """
        Parameters:

        * name: a string that identifies the KG;

        * raw_triples: a numpy array of shape (N, 3), containing the triples of
        the KG. The entities and relations in raw_triples do not have to be
        encoded in any specific format. Entities and relations will be
        extracted from this KG to form an inner vocabulary.

        * entity_names_map: a dictionary in the form of "entity: name";  the
        dictionary does not have to contain an entry for every possible entity;
        the keys have to follow the naming convention used in raw_triples.

        * entity_descriptions_map: a dictionary in the form of "entity:
        description"; the dictionary does not have to contain an entry for
        every possible entity; the keys have to follow the naming convention
        used in raw_triples.

        * filter_out_missing_names: all triples containing unlabeled entities
        are filtered from the KG;

        * filter_out_missing_descriptions: all triples containing undescripted
        entities are filtered from the KG;

        Notes:

        - If an entity does not have a name, its name will be the empty string
          by default;

        - All the entities that do not have an associated descriptions will
          inherit their name as their description;

        """

        self.name = name
        self.raw_triples = raw_triples
        self.entities = np.unique(raw_triples[:, [0, 2]])
        self.relations = np.unique(raw_triples[:, 1])
        self.entity_classes = entity_classes

        unnamed_entities = []
        undescripted_entities = []

        entity_names = []
        entity_descriptions = []

        for entity in self.entities:
            if entity in entity_names_map:
                entity_name = entity_names_map[entity]
            else:
                entity_name = ""
                unnamed_entities.append(entity)

            entity_names.append(entity_name)

            if entity in entity_descriptions_map:
                entity_descriptions.append(entity_descriptions_map[entity])
            else:
                entity_descriptions.append(entity_name)
                undescripted_entities.append(entity)

        self.entity_names = np.array(entity_names)
        self.entity_descriptions = np.array(entity_descriptions)

        if filter_out_missing_names:
            self.filter_out(entities=unnamed_entities)

        if filter_out_missing_descriptions:
            self.filter_out(entities=undescripted_entities)

        unnamed_relations = []
        undescripted_relations = []

        relation_names = []
        relation_descriptions = []

        for relation in self.relations:
            if relation in relation_names_map:
                relation_name = relation_names_map[relation]
            else:
                relation_name = relation
            #                unnamed_relations.append(relation) # TODO: relations without a name are named with their id for now

            relation_names.append(relation_name)

            if relation in relation_descriptions_map:
                relation_descriptions.append(relation_descriptions_map[relation])
            else:
                relation_descriptions.append(relation_name)
                undescripted_relations.append(relation)

        self.relation_names = np.array(relation_names)
        self.relation_descriptions = np.array(relation_descriptions)

    #        if filter_out_missing_names:
    #            self.filter_out(relations=unnamed_relations)
    #
    #        if filter_out_missing_descriptions:
    #            self.filter_out(relations=undescripted_relations)

    def filter_out(self, entities: np.array = None, relations: np.array = None):
        """
        Deletes all the triples in self.raw_triples that contain any of the
        labels in argument "entities" either as head or tail, and any of the
        labels in argument "relations" as relation. The inner
        entities/relations "vocabularies" are also updated in order to remove
        all the filtered ones.
        """

        if entities:
            mask = np.isin(self.entities, entities, invert=True)
            self.entities = self.entities[mask]

            if self.entity_names is not None:
                self.entity_names = self.entity_names[mask]
            if self.entity_descriptions is not None:
                self.entity_descriptions = self.entity_descriptions[mask]

            self.raw_triples = self.raw_triples[np.logical_and(
                np.isin(self.raw_triples[:, 0], entities, invert=True),
                np.isin(self.raw_triples[:, 2], entities, invert=True),
            )]

            if self.entity_classes is not None:
                self.entity_classes = self.entity_classes[
                    np.isin(self.entity_classes[:, 0], entities, invert=True)
                ]

        if relations:
            mask = np.isin(self.relations, relations, invert=True)
            self.relations = self.relations[mask]
            if self.relation_names is not None:
                self.relation_names = self.relation_names[mask]
            if self.relation_descriptions is not None:
                self.relation_descriptions = self.relation_descriptions[mask]


            self.raw_triples = self.raw_triples[
                np.isin(self.raw_triples[:, 1], relations, invert=True)
            ]
