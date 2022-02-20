import numpy as np
import pytest
import networkx as nx
from .literal import Literal
from .clause import Clause
from .constraint import Constraint
from .constraints_group import ConstraintsGroup


class ClausesGroup:
    def __init__(self, clauses):
        # ClausesGroup([Clause])
        self.clauses = frozenset(clauses)
        self.clauses_list = clauses

    @classmethod
    def from_constraints_group(cls, group):
        return cls([Clause.from_constraint(cons) for cons in group])

    def __len__(self):
        return len(self.clauses)

    def __eq__(self, other):
        if not isinstance(other, ClausesGroup): return False
        return self.clauses == other.clauses

    def __str__(self):
        return '\n'.join([str(clause) for clause in self.clauses])

    def __hash__(self):
        return hash(self.clauses)

    def __iter__(self):
        return iter(self.clauses)
    
    @classmethod 
    def random(cls, max_clauses, num_classes, coherent_with=np.array([])):
        clauses_count = np.random.randint(low=1, high=max_clauses)
        clauses = [Clause.random(num_classes) for i in range(clauses_count)]

        if len(coherent_with) > 0:
            keep = cls(clauses).coherent_with(coherent_with).all(axis=0)
            clauses = np.array(clauses)[keep]

        return cls(clauses)

    def compacted(self):
        clauses = list(self.clauses)
        clauses.sort(reverse=True, key=len)
        compacted = [] 

        for clause in clauses:
            compacted = [c for c in compacted if not clause.is_subset(c)]
            compacted.append(clause)

        #print(f"compacted {len(clauses) - len(compacted)} out of {len(clauses)}")
        return ClausesGroup(compacted)

    def resolution(self, atom):
        pos = Literal(atom, True)
        neg = Literal(atom, False)

        # Split clauses in three categories
        pos_clauses, neg_clauses, other_clauses = set(), set(), set()
        for clause in self.clauses:
            if pos in clause:
                pos_clauses.add(clause)
            elif neg in clause:
                neg_clauses.add(clause)
            else:
                other_clauses.add(clause)

        # Apply resolution on positive and negative clauses
        resolution_clauses = [c1.resolution(c2, literal=pos) for c1 in pos_clauses for c2 in neg_clauses]
        resolution_clauses = {clause for clause in resolution_clauses if clause != None}
        next_clauses = ClausesGroup(other_clauses.union(resolution_clauses)).compacted()

        # Compute constraints 
        pos_constraints = [clause.fix_head(pos) for clause in pos_clauses]
        neg_constraints = [clause.fix_head(neg) for clause in neg_clauses]
        constraints = ConstraintsGroup(pos_constraints + neg_constraints)

        return constraints, next_clauses

    def graph(self):
        G = nx.Graph()
        G.add_nodes_from(self.atoms(), kind='atom')
        G.add_nodes_from(self.clauses, kind='clause')

        for clause in self.clauses:
            for lit in clause:
                G.add_edge(clause, lit.atom)

        return G

    @staticmethod 
    def centrality_measures():
        return ['degree', 'eigenvector', 'katz', 'closeness', 'betweenness']

    def difficulty(self, centrality):
        G = self.graph() 
        if centrality == 'degree':
            return nx.algorithms.centrality.degree_centrality(G)
        elif centrality == 'eigenvector':
            return nx.algorithms.centrality.eigenvector_centrality_numpy(G)
        elif centrality == 'katz':
            return nx.algorithms.centrality.katz_centrality_numpy(G)
        elif centrality == 'closeness':
            return nx.algorithms.centrality.closeness_centrality(G)
        elif centrality == 'betweenness':
            return nx.algorithms.centrality.betweenness_centrality(G)
        else:
            raise Exception(f"Unknown centrality {centrality}")

    def stratify(self, centrality):
        # Centrality guides the inferrence order  
        if not isinstance(centrality, str):
            atoms = centrality
        else:
            difficulty = self.difficulty(centrality)
            atoms = list(self.atoms())
            atoms.sort(key=lambda x: difficulty[x])

        # Apply resolution repeatedly
        atoms = atoms[::-1]
        group = ConstraintsGroup([])
        clauses = self

        for atom in atoms:
            #print(f"Eliminating %{atom} from %{len(clauses)} clauses\n")
            constraints, clauses = clauses.resolution(atom)
            group = group + constraints

        if len(clauses):
            raise Exception("Unsatisfiable set of clauses")

        return group.stratify()

    def coherent_with(self, preds):
        answer = [clause.coherent_with(preds) for clause in self.clauses_list]
        return np.array(answer).transpose()

    def atoms(self):
        result = set() 
        for clause in self.clauses:
            result = result.union(clause.atoms())
        return result

def test_eq():
    c1 = Clause('1 n2 3')
    c2 = Clause('1 n2 n3 n2')
    c3 = Clause('1 3 n3')
    assert ClausesGroup([c1, c2, c3]) == ClausesGroup([c3, c2, c1, c1])
    assert ClausesGroup([c1, c2]) != ClausesGroup([c3, c2, c1, c1])

def test_resolution():
    c1 = Clause('1 2 3')
    c2 = Clause('1 n2 4')
    c3 = Clause('n1 4 n5')
    c4 = Clause('n1 2 6')
    c5 = Clause('2 n3 4')
    constraints, clauses = ClausesGroup([c1, c2, c3, c4, c5]).resolution(1) 
    print(clauses)

    assert constraints == ConstraintsGroup([
        Constraint('1 :- n2 n3'),
        Constraint('1 :- 2 n4'),
        Constraint('n1 :- n4 5'),
        Constraint('n1 :- n2 n6')
    ])

    assert clauses == ClausesGroup([
        Clause('2 3 4 n5'),
        Clause('2 3 6'),
        Clause('n2 4 n5'),
        Clause('4 2 n3')
    ])

def test_stratify():
    constraints = ClausesGroup([
        Clause('n0 n1'),
        Clause('n1 2'),
        Clause('1 n2')
    ]).stratify([0, 1, 2])
    assert len(constraints) == 2
    assert constraints[0] == ConstraintsGroup([
        Constraint('n1 :- 0')
    ])
    assert constraints[1] == ConstraintsGroup([
        Constraint('2 :- 1'),
        Constraint('n2 :- n1')
    ])

def test_coherent_with():
    clauses = ClausesGroup([ 
        Clause('0 1 n2 n3'),
        Clause('n0 1'),
        Clause('0 n1'),
        Clause('3 n3'),
        Clause('n2 n3')
    ])

    preds = np.array([ 
        [0.1, 0.2, 0.6, 0.7],
        [0.4, 0.7, 0.2, 0.3],
        [0.7, 0.2, 0.9, 0.8]
    ])

    assert (clauses.coherent_with(preds) == [ 
        [False, True, True, True, False],
        [True, True, False, True, True],
        [True, False, True, True, False]
    ]).all()

def test_empty_resolution():
    clauses = ClausesGroup([
        Clause('0 2'),
        Clause('n0 2'),
        Clause('1 n2'),
        Clause('n1 n2')
    ])

    with pytest.raises(Exception):
        clauses.stratify([0, 1, 2])


def test_random():
    num_classes = 10
    max_clauses = 30

    requirements = np.random.randint(low=0, high=2, size=(3, num_classes))
    clauses = ClausesGroup.random(max_clauses=max_clauses, num_classes=num_classes, coherent_with=requirements)
    assert len(clauses) <= max_clauses
    assert clauses.coherent_with(requirements).all()

def test_compacted():
    clauses = ClausesGroup([
        Clause('n1 n3'),
        Clause('2 n3 5'),
        Clause('1 n3'),
        Clause('1 2 n3 4'),
        Clause('n3 4'),
        Clause('2 5')
    ])
    
    correct = ClausesGroup([
        Clause('n1 n3'),
        Clause('1 n3'),
        Clause('n3 4'),
        Clause('2 5')
    ])

    assert clauses.compacted() == correct

def test_atoms():
    clauses = ClausesGroup([ 
        Clause('1 2 n3 4'),
        Clause('3 4 5 n6'),
        Clause('n6 n7 n8 9')
    ])

    assert clauses.atoms() == set(range(1, 10))

def test_graph():
    clauses = ClausesGroup([
        Clause('0 1 n2'),
        Clause('n1 2 n3'),
        Clause('n0 2')
    ])

    G = clauses.graph()
    assert len(G.nodes()) == 7 
    assert nx.algorithms.is_bipartite(G)
    assert len(G.edges()) == 8

def test_difficulty():
    clauses = ClausesGroup([
        Clause('0 2'),
        Clause('1 2'),
        Clause('2 3 4')
    ])

    assert np.allclose(list(clauses.difficulty('degree').items())[0:5], [
        (0, 1/7), 
        (1, 1/7), 
        (2, 3/7), 
        (3, 1/7), 
        (4, 1/7)
    ])
    assert np.allclose(list(clauses.difficulty('eigenvector').items())[0:5], [
        (0, 0.16827838529538847), 
        (1, 0.16827838529538852), 
        (2, 0.5745383453297614), 
        (3, 0.23798157473898407), 
        (4, 0.23798157473898351)
    ])
    assert np.allclose(list(clauses.difficulty('katz').items())[0:5], [
        (0, 0.3243108798877643),
        (1, 0.3243108798877643),
        (2, 0.39975054223505035),
        (3, 0.3276201745804966),
        (4, 0.3276201745804966)
    ])
    assert np.allclose(list(clauses.difficulty('closeness').items())[0:5], [
        (0, 0.3333333333333333),
        (1, 0.3333333333333333),
        (2, 0.6363636363636364),
        (3, 0.3684210526315789),
        (4, 0.3684210526315789)
    ])
    assert np.allclose(list(clauses.difficulty('betweenness').items())[0:5], [
        (0, 0.),
        (1, 0.),
        (2, 0.7619047619047619),
        (3, 0.),
        (4, 0.)
    ])



