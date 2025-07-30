"""
Base knowledge graph classes and interfaces
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class Triple:
    """Basic RDF-like triple structure"""
    subject: str
    predicate: str
    object: str

    def __str__(self):
        return f"<{self.subject}> <{self.predicate}> <{self.object}>"

    def __hash__(self):
        return hash((self.subject, self.predicate, self.object))

    def __eq__(self, other):
        if not isinstance(other, Triple):
            return False
        return (self.subject == other.subject and
                self.predicate == other.predicate and
                self.object == other.object)

class BaseKnowledgeGraph(ABC):
    """Abstract base class for knowledge graphs"""

    def __init__(self):
        self.triples: List[Triple] = []
        self.namespaces: Dict[str, str] = {}
        self.instances: Dict[str, Dict[str, Any]] = {}

    @abstractmethod
    def add_triple(self, subject: str, predicate: str, obj: str):
        """Add a triple to the knowledge graph"""
        pass

    @abstractmethod
    def add_instance(self, instance_id: str, class_type: str, properties: Dict[str, Any] = None):
        """Add an instance of a class"""
        pass

    @abstractmethod
    def export_turtle(self) -> str:
        """Export as Turtle/TTL format"""
        pass

    def add_namespace(self, prefix: str, uri: str):
        """Add a namespace"""
        self.namespaces[prefix] = uri

    def uri(self, prefix: str, local_name: str) -> str:
        """Create full URI from prefix and local name"""
        if prefix not in self.namespaces:
            raise ValueError(f"Unknown namespace prefix: {prefix}")
        return f"{self.namespaces[prefix]}{local_name}"

    def get_triples_by_subject(self, subject: str) -> List[Triple]:
        """Get all triples with given subject"""
        return [t for t in self.triples if t.subject == subject]

    def get_triples_by_predicate(self, predicate: str) -> List[Triple]:
        """Get all triples with given predicate"""
        return [t for t in self.triples if t.predicate == predicate]

    def get_triples_by_object(self, obj: str) -> List[Triple]:
        """Get all triples with given object"""
        return [t for t in self.triples if t.object == obj]

    def remove_triple(self, subject: str, predicate: str, obj: str):
        """Remove a triple from the graph"""
        triple_to_remove = Triple(subject, predicate, obj)
        self.triples = [t for t in self.triples if t != triple_to_remove]

    def clear(self):
        """Clear all triples and instances"""
        self.triples.clear()
        self.instances.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get basic statistics about the knowledge graph"""
        return {
            "triple_count": len(self.triples),
            "instance_count": len(self.instances),
            "namespace_count": len(self.namespaces),
            "unique_subjects": len(set(t.subject for t in self.triples)),
            "unique_predicates": len(set(t.predicate for t in self.triples)),
            "unique_objects": len(set(t.object for t in self.triples))
        }

class QueryableKnowledgeGraph(BaseKnowledgeGraph):
    """Knowledge graph with basic querying capabilities"""

    def __init__(self):
        super().__init__()
        self._subject_index: Dict[str, List[Triple]] = {}
        self._predicate_index: Dict[str, List[Triple]] = {}
        self._object_index: Dict[str, List[Triple]] = {}

    def add_triple(self, subject: str, predicate: str, obj: str):
        """Add a triple and update indices"""
        triple = Triple(subject, predicate, obj)

        if triple not in self.triples:
            self.triples.append(triple)

            # Update indices
            if subject not in self._subject_index:
                self._subject_index[subject] = []
            self._subject_index[subject].append(triple)

            if predicate not in self._predicate_index:
                self._predicate_index[predicate] = []
            self._predicate_index[predicate].append(triple)

            if obj not in self._object_index:
                self._object_index[obj] = []
            self._object_index[obj].append(triple)

    def get_triples_by_subject(self, subject: str) -> List[Triple]:
        """Get all triples with given subject (indexed)"""
        return self._subject_index.get(subject, [])

    def get_triples_by_predicate(self, predicate: str) -> List[Triple]:
        """Get all triples with given predicate (indexed)"""
        return self._predicate_index.get(predicate, [])

    def get_triples_by_object(self, obj: str) -> List[Triple]:
        """Get all triples with given object (indexed)"""
        return self._object_index.get(obj, [])

    def query_pattern(self, subject: Optional[str] = None,
                     predicate: Optional[str] = None,
                     obj: Optional[str] = None) -> List[Triple]:
        """Query triples matching a pattern (None = wildcard)"""
        results = self.triples.copy()

        if subject is not None:
            results = [t for t in results if t.subject == subject]

        if predicate is not None:
            results = [t for t in results if t.predicate == predicate]

        if obj is not None:
            results = [t for t in results if t.object == obj]

        return results

    def get_instance_properties(self, instance_uri: str) -> Dict[str, List[str]]:
        """Get all properties of an instance"""
        properties = {}
        triples = self.get_triples_by_subject(instance_uri)

        for triple in triples:
            if triple.predicate not in properties:
                properties[triple.predicate] = []
            properties[triple.predicate].append(triple.object)

        return properties

    def get_instances_of_type(self, class_type: str) -> List[str]:
        """Get all instances of a specific type"""
        rdf_type = self.uri("rdf", "type") if "rdf" in self.namespaces else "rdf:type"
        type_triples = self.query_pattern(predicate=rdf_type, obj=class_type)
        return [t.subject for t in type_triples]

    def clear(self):
        """Clear all triples, instances, and indices"""
        super().clear()
        self._subject_index.clear()
        self._predicate_index.clear()
        self._object_index.clear()

class ExportableKnowledgeGraph(QueryableKnowledgeGraph):
    """Knowledge graph with multiple export formats"""

    def export_turtle(self) -> str:
        """Export as Turtle/TTL format"""
        turtle = ""

        # Add namespace prefixes
        for prefix, uri in self.namespaces.items():
            turtle += f"@prefix {prefix}: <{uri}> .\n"
        turtle += "\n"

        # Add triples
        for triple in self.triples:
            s = self._format_uri_for_turtle(triple.subject)
            p = self._format_uri_for_turtle(triple.predicate)
            o = self._format_uri_for_turtle(triple.object)
            turtle += f"{s} {p} {o} .\n"

        return turtle

    def export_n_triples(self) -> str:
        """Export as N-Triples format"""
        n_triples = ""
        for triple in self.triples:
            n_triples += f"<{triple.subject}> <{triple.predicate}> <{triple.object}> .\n"
        return n_triples

    def export_json_ld(self) -> Dict[str, Any]:
        """Export as JSON-LD format"""
        context = {}
        for prefix, uri in self.namespaces.items():
            context[prefix] = uri

        graph = []

        # Group triples by subject
        subjects = {}
        for triple in self.triples:
            if triple.subject not in subjects:
                subjects[triple.subject] = {"@id": triple.subject}

            pred = self._uri_to_compact(triple.predicate)
            obj = triple.object

            # Handle object formatting
            if obj.startswith('"'):
                # Literal
                if '^^' in obj:
                    value, datatype = obj.strip('"').split('^^')
                    obj = {"@value": value, "@type": datatype}
                else:
                    obj = obj.strip('"')
            else:
                # Resource reference
                obj = {"@id": obj}

            if pred in subjects[triple.subject]:
                # Multiple values - convert to array
                if not isinstance(subjects[triple.subject][pred], list):
                    subjects[triple.subject][pred] = [subjects[triple.subject][pred]]
                subjects[triple.subject][pred].append(obj)
            else:
                subjects[triple.subject][pred] = obj

        graph = list(subjects.values())

        return {
            "@context": context,
            "@graph": graph
        }

    def _format_uri_for_turtle(self, uri: str) -> str:
        """Format URI for Turtle export"""
        if uri.startswith('"'):
            return uri  # Already a literal

        # Try to use namespace prefixes
        for prefix, namespace in self.namespaces.items():
            if uri.startswith(namespace):
                return f"{prefix}:{uri[len(namespace):]}"

        return f"<{uri}>"

    def _uri_to_compact(self, uri: str) -> str:
        """Convert URI to compact form"""
        for prefix, namespace in self.namespaces.items():
            if uri.startswith(namespace):
                return f"{prefix}:{uri[len(namespace):]}"
        return uri