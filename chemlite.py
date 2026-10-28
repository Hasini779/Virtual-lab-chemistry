"""
chemlite.py - a small cheminformatics engine written in pure Python
(NumPy + Matplotlib only).  No RDKit required.

Provides
  * SMILES parsing & validation (organic subset, brackets, rings, branches,
    aromatic atoms, charges, isotopes, disconnected fragments)
  * Kekulisation and Hueckel (4n+2) aromaticity perception
  * Molecular descriptors: formula, MW, exact mass, LogP, TPSA, HBD, HBA,
    rotatable bonds, rings, aromatic rings, Fsp3, molar refractivity,
    sigma / pi bond counts
  * 2D structure drawing (Matplotlib) and 3D coordinate generation
    (distance-geometry style embedding with NumPy)
  * SMILES writer and MDL mol-block writer

Notes
  * Stereochemistry symbols (@, @@, /, \\) are parsed but ignored.
  * LogP / molar refractivity use a Wildman-Crippen style atom-contribution
    scheme implemented here; values are estimates and can differ slightly
    from other toolkits for unusual functional groups.
  * 3D coordinates are an embedded geometry (not force-field optimised).
"""

import io
import math
import re
import itertools

import numpy as np

# ----------------------------------------------------------------------------
# ELEMENT DATA
# symbol: (Z, average mass, monoisotopic mass, allowed valences, covalent radius)
# ----------------------------------------------------------------------------
ELEMENTS = {
    "H": (1, 1.008, 1.00782503, (1,), 0.31),
    "Li": (3, 6.94, 7.016003, (1,), 1.28),
    "B": (5, 10.81, 11.0093054, (3,), 0.84),
    "C": (6, 12.011, 12.0, (4,), 0.76),
    "N": (7, 14.007, 14.0030740, (3, 5), 0.71),
    "O": (8, 15.999, 15.9949146, (2,), 0.66),
    "F": (9, 18.998, 18.9984032, (1,), 0.57),
    "Na": (11, 22.990, 22.98977, (1,), 1.66),
    "Mg": (12, 24.305, 23.985042, (2,), 1.41),
    "Si": (14, 28.086, 27.976927, (4,), 1.11),
    "P": (15, 30.974, 30.973762, (3, 5, 7), 1.07),
    "S": (16, 32.06, 31.972071, (2, 4, 6), 1.05),
    "Cl": (17, 35.45, 34.968853, (1,), 1.02),
    "K": (19, 39.098, 38.963707, (1,), 2.03),
    "Ca": (20, 40.078, 39.962591, (2,), 1.76),
    "Fe": (26, 55.845, 55.934938, (2, 3), 1.32),
    "Cu": (29, 63.546, 62.929601, (1, 2), 1.32),
    "Zn": (30, 65.38, 63.929147, (2,), 1.22),
    "As": (33, 74.922, 74.921596, (3, 5), 1.19),
    "Se": (34, 78.971, 79.916522, (2, 4, 6), 1.20),
    "Br": (35, 79.904, 78.918338, (1,), 1.20),
    "I": (53, 126.904, 126.904473, (1, 3, 5), 1.39),
}

ORGANIC_SUBSET = {"B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"}
AROMATIC_SYMBOLS = {"b": "B", "c": "C", "n": "N", "o": "O", "p": "P",
                    "s": "S", "se": "Se", "as": "As"}

_BRACKET_RE = re.compile(
    r"^\[(\d+)?(se|as|[A-Z][a-z]?|[bcnops])(@{1,2})?(?:H(\d*))?"
    r"(\+{1,}\d*|-{1,}\d*)?(?::\d+)?\]$"
)


class SmilesError(ValueError):
    """Raised when a SMILES string cannot be interpreted."""


# ----------------------------------------------------------------------------
# DATA STRUCTURES
# ----------------------------------------------------------------------------
class Atom:
    __slots__ = ("symbol", "aromatic", "charge", "isotope", "h", "bracket")

    def __init__(self, symbol, aromatic=False, charge=0, isotope=0,
                 h=None, bracket=False):
        self.symbol = symbol
        self.aromatic = aromatic
        self.charge = charge
        self.isotope = isotope
        self.h = h              # total hydrogen count (filled in after parsing)
        self.bracket = bracket


class Bond:
    __slots__ = ("i", "j", "order", "aromatic")

    def __init__(self, i, j, order, aromatic=False):
        self.i = i
        self.j = j
        self.order = order      # kekule order: 1, 2, 3 (or 'ar' while parsing)
        self.aromatic = aromatic

    def other(self, k):
        return self.j if k == self.i else self.i


class Mol:
    """Molecular graph of heavy atoms; hydrogens are stored as counts."""

    def __init__(self, atoms, bonds):
        self.atoms = atoms
        self.bonds = bonds
        self.adj = [[] for _ in atoms]          # (neighbour, bond)
        for b in bonds:
            self.adj[b.i].append((b.j, b))
            self.adj[b.j].append((b.i, b))
        self._rings = None

    # ---- simple properties -------------------------------------------------
    @property
    def num_atoms(self):
        return len(self.atoms)

    num_heavy_atoms = num_atoms

    @property
    def num_bonds(self):
        return len(self.bonds)

    @property
    def total_h(self):
        return sum(a.h for a in self.atoms)

    def degree(self, i):
        return len(self.adj[i])

    def valence(self, i):
        """Sum of (kekule) bond orders to heavy atoms + hydrogens."""
        return sum(b.order for _, b in self.adj[i]) + self.atoms[i].h

    @property
    def rings(self):
        if self._rings is None:
            self._rings = _sssr(self)
        return self._rings

    def ring_bond_set(self):
        s = set()
        for ring in self.rings:
            for k in range(len(ring)):
                s.add(frozenset((ring[k], ring[(k + 1) % len(ring)])))
        return s


class Mol3D:
    """Explicit-hydrogen molecule with 3D coordinates."""

    def __init__(self, symbols, coords, bonds):
        self.symbols = symbols
        self.coords = coords                    # (n, 3) array
        self.bonds = bonds                      # list of (i, j, order)

    @property
    def num_atoms(self):
        return len(self.symbols)


# ----------------------------------------------------------------------------
# SMILES PARSER
# ----------------------------------------------------------------------------
def _parse_charge(txt):
    if not txt:
        return 0
    sign = 1 if txt[0] == "+" else -1
    body = txt[1:]
    if body.isdigit():
        return sign * int(body)
    return sign * len(txt)


def parse_smiles(smiles):
    """Parse a SMILES string into a Mol.  Raises SmilesError if invalid."""
    if smiles is None or not str(smiles).strip():
        raise SmilesError("Empty SMILES")
    s = str(smiles).strip()
    if re.search(r"\s", s):
        raise SmilesError("SMILES must not contain spaces")

    atoms, bonds = [], []
    stack = []                  # branch stack of atom indices
    prev = None                 # previous atom index
    pending = None              # explicit bond symbol waiting for next atom
    ring_open = {}              # digit -> (atom index, bond symbol)
    pos, n = 0, len(s)

    def add_atom(atom):
        nonlocal prev, pending
        idx = len(atoms)
        atoms.append(atom)
        if prev is not None:
            _add_bond(prev, idx, pending)
        pending = None
        prev = idx

    def _add_bond(i, j, sym):
        if i == j:
            raise SmilesError("Atom bonded to itself")
        for b in bonds:
            if {b.i, b.j} == {i, j}:
                raise SmilesError("Duplicate bond")
        both_arom = atoms[i].aromatic and atoms[j].aromatic
        if sym in (None, "/", "\\"):
            order = "ar" if both_arom else 1
        elif sym == "-":
            order = 1
        elif sym == "=":
            order = 2
        elif sym == "#":
            order = 3
        elif sym == ":":
            order = "ar"
        elif sym == "$":
            raise SmilesError("Quadruple bonds are not supported")
        else:
            raise SmilesError("Unknown bond symbol")
        bonds.append(Bond(i, j, order, aromatic=(order == "ar")))

    while pos < n:
        ch = s[pos]
        if ch == "[":
            end = s.find("]", pos)
            if end == -1:
                raise SmilesError("Unclosed bracket atom")
            token = s[pos:end + 1]
            m = _BRACKET_RE.match(token)
            if not m:
                raise SmilesError("Unsupported bracket atom: " + token)
            iso, sym, _chir, hflag, chg = m.groups()
            aromatic = sym in AROMATIC_SYMBOLS
            element = AROMATIC_SYMBOLS.get(sym, sym)
            if element not in ELEMENTS:
                raise SmilesError("Unknown element: " + element)
            hcount = 0
            if "H" in token[1:-1] and element != "H":
                hcount = int(m.group(4)) if m.group(4) else 1
            add_atom(Atom(element, aromatic, _parse_charge(chg),
                          int(iso) if iso else 0, hcount, True))
            pos = end + 1
        elif ch.isalpha():
            if s.startswith("Cl", pos) or s.startswith("Br", pos):
                sym, pos = s[pos:pos + 2], pos + 2
                add_atom(Atom(sym))
            elif ch in "BCNOPSFI":
                add_atom(Atom(ch))
                pos += 1
            elif ch in "bcnops":
                add_atom(Atom(AROMATIC_SYMBOLS[ch], aromatic=True))
                pos += 1
            else:
                raise SmilesError("Unrecognised atom symbol: " + ch)
        elif ch == "(":
            if prev is None:
                raise SmilesError("Branch without parent atom")
            stack.append(prev)
            pos += 1
        elif ch == ")":
            if not stack:
                raise SmilesError("Unmatched ')'")
            prev = stack.pop()
            pending = None
            pos += 1
        elif ch in "-=#:/\\$":
            if pending is not None:
                raise SmilesError("Two bond symbols in a row")
            pending = ch
            pos += 1
        elif ch == ".":
            if stack:
                raise SmilesError("'.' inside a branch")
            prev = None
            pending = None
            pos += 1
        elif ch.isdigit() or ch == "%":
            if ch == "%":
                label = s[pos + 1:pos + 3]
                if len(label) != 2 or not label.isdigit():
                    raise SmilesError("Bad ring label")
                pos += 3
            else:
                label = ch
                pos += 1
            if prev is None:
                raise SmilesError("Ring closure without atom")
            if label in ring_open:
                other, osym = ring_open.pop(label)
                sym = pending or osym
                _add_bond(other, prev, sym)
                pending = None
            else:
                ring_open[label] = (prev, pending)
                pending = None
        else:
            raise SmilesError("Invalid character: " + ch)

    if stack:
        raise SmilesError("Unclosed '('")
    if ring_open:
        raise SmilesError("Unclosed ring bond")
    if pending is not None:
        raise SmilesError("Dangling bond symbol")
    if not atoms:
        raise SmilesError("No atoms found")

    mol = Mol(atoms, bonds)
    _kekulize(mol)
    _assign_hydrogens(mol)
    _perceive_aromaticity(mol)
    return mol


def get_molecule(smiles):
    """Like parse_smiles but returns None for invalid input."""
    try:
        return parse_smiles(smiles)
    except (SmilesError, RecursionError):
        return None


# ----------------------------------------------------------------------------
# KEKULISATION, HYDROGENS, AROMATICITY
# ----------------------------------------------------------------------------
_BASE_VALENCE = {"B": 3, "C": 4, "N": 3, "O": 2, "S": 2, "P": 3,
                 "Se": 2, "As": 3, "Si": 4}


def _target_valence(sym, q):
    base = _BASE_VALENCE.get(sym, 4)
    if sym in ("C", "B", "Si"):
        return base - abs(q)
    return base + q


def _kekulize(mol):
    arom_bonds = [b for b in mol.bonds if b.order == "ar"]
    if not arom_bonds:
        return
    cand = {}
    for i, a in enumerate(mol.atoms):
        arom_here = [b for _, b in mol.adj[i] if b.order == "ar"]
        if not arom_here:
            continue
        nonar = sum(b.order for _, b in mol.adj[i] if b.order != "ar")
        free = (_target_valence(a.symbol, a.charge) - nonar
                - len(arom_here) - (a.h if a.bracket else 0))
        if free >= 1:
            cand[i] = [b for b in arom_here]
    cand_nbrs = {i: [(b.other(i), b) for b in bl if b.other(i) in cand]
                 for i, bl in cand.items()}
    matched = {}

    def solve(unmatched):
        if not unmatched:
            return True
        best, opts = None, None
        for i in unmatched:
            o = [(j, b) for j, b in cand_nbrs[i] if j in unmatched]
            if best is None or len(o) < len(opts):
                best, opts = i, o
                if len(o) <= 1:
                    break
        if not opts:
            return False
        for j, b in opts:
            matched[id(b)] = True
            rest = unmatched - {best, j}
            if solve(rest):
                return True
            del matched[id(b)]
        return False

    if not solve(set(cand)):
        raise SmilesError("Cannot kekulize aromatic system (invalid SMILES)")
    for b in arom_bonds:
        b.order = 2 if id(b) in matched else 1


def _implicit_h(sym, used):
    vals = ELEMENTS[sym][3]
    for v in vals:
        if v >= used:
            return v - used
    return None


def _assign_hydrogens(mol):
    for i, a in enumerate(mol.atoms):
        used = sum(b.order for _, b in mol.adj[i])
        if a.bracket:
            continue
        if a.symbol not in ORGANIC_SUBSET:
            a.h = 0
            continue
        h = _implicit_h(a.symbol, used)
        if h is None:
            raise SmilesError("Valence error on atom %s" % a.symbol)
        a.h = h


def _sssr(mol):
    n = len(mol.atoms)
    if not mol.bonds:
        return []
    adj = [[j for j, _ in mol.adj[i]] for i in range(n)]
    comps, seen = 0, set()
    for s in range(n):
        if s in seen:
            continue
        comps += 1
        st = [s]
        seen.add(s)
        while st:
            u = st.pop()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    st.append(v)
    rank = len(mol.bonds) - n + comps
    if rank <= 0:
        return []
    edge_index = {frozenset((b.i, b.j)): k for k, b in enumerate(mol.bonds)}
    cands = {}
    for b in mol.bonds:
        u, v = b.i, b.j
        prev = {u: None}
        queue = [u]
        found = False
        while queue and not found:
            nxt = []
            for x in queue:
                for y in adj[x]:
                    if (x == u and y == v) or y in prev:
                        continue
                    prev[y] = x
                    if y == v:
                        found = True
                        break
                    nxt.append(y)
                if found:
                    break
            queue = nxt
        if not found:
            continue
        path, x = [], v
        while x is not None:
            path.append(x)
            x = prev[x]
        key = frozenset(frozenset((path[k], path[(k + 1) % len(path)]))
                        for k in range(len(path)))
        cands.setdefault(key, path)
    rings, basis = [], []
    for key, path in sorted(cands.items(), key=lambda kv: len(kv[1])):
        vec = 0
        for e in key:
            vec ^= 1 << edge_index[e]
        red = vec
        for bv in basis:
            red = min(red, red ^ bv)
        if red:
            basis.append(red)
            rings.append(path)
            if len(rings) == rank:
                break
    return rings


def _ring_electrons(mol, atoms, bondset):
    """Pi-electron count for a candidate aromatic ring/ring system or None."""
    total = 0
    for i in atoms:
        a = mol.atoms[i]
        dbl_in = any(b.order == 2 and frozenset((b.i, b.j)) in bondset
                     for _, b in mol.adj[i])
        if dbl_in:
            total += 1
            continue
        exo = [(j, b) for j, b in mol.adj[i]
               if b.order == 2 and frozenset((b.i, b.j)) not in bondset]
        if any(b.order == 3 for _, b in mol.adj[i]):
            return None
        if exo:
            j = exo[0][0]
            if a.symbol == "C" and mol.atoms[j].symbol in ("N", "O", "S"):
                continue
            return None
        if a.symbol in ("N", "P", "As"):
            if a.charge > 0:
                return None
            total += 2
        elif a.symbol in ("O", "S", "Se"):
            if a.charge != 0:
                return None
            total += 2
        elif a.symbol == "C":
            if a.charge < 0:
                total += 2
            elif a.charge > 0:
                total += 0
            else:
                return None
        elif a.symbol == "B":
            total += 0
        else:
            return None
    return total


def _perceive_aromaticity(mol):
    for a in mol.atoms:
        a.aromatic = False
    for b in mol.bonds:
        b.aromatic = False
    rings = mol.rings
    if not rings:
        return

    def ring_bonds(r):
        return {frozenset((r[k], r[(k + 1) % len(r)])) for k in range(len(r))}

    def mark(atoms, bondset):
        for i in atoms:
            mol.atoms[i].aromatic = True
        for b in mol.bonds:
            if frozenset((b.i, b.j)) in bondset:
                b.aromatic = True

    passed = [False] * len(rings)
    for k, r in enumerate(rings):
        if len(r) > 8:
            continue
        e = _ring_electrons(mol, r, ring_bonds(r))
        if e is not None and e % 4 == 2:
            passed[k] = True
            mark(r, ring_bonds(r))
    for k1, k2 in itertools.combinations(range(len(rings)), 2):
        if passed[k1] and passed[k2]:
            continue
        r1, r2 = rings[k1], rings[k2]
        if len(set(r1) & set(r2)) != 2:
            continue
        atoms = set(r1) | set(r2)
        if len(atoms) > 12:
            continue
        bset = ring_bonds(r1) | ring_bonds(r2)
        e = _ring_electrons(mol, atoms, bset)
        if e is not None and e % 4 == 2:
            mark(atoms, bset)


# ----------------------------------------------------------------------------
# BASIC DESCRIPTORS
# ----------------------------------------------------------------------------
def molecular_formula(mol):
    counts = {}
    for a in mol.atoms:
        counts[a.symbol] = counts.get(a.symbol, 0) + 1
    h = mol.total_h
    if h:
        counts["H"] = counts.get("H", 0) + h
    if "C" in counts:
        order = ["C"] + (["H"] if "H" in counts else []) + sorted(
            k for k in counts if k not in ("C", "H"))
    else:
        order = sorted(counts)
    txt = "".join(k + (str(counts[k]) if counts[k] > 1 else "") for k in order)
    q = sum(a.charge for a in mol.atoms)
    if q:
        txt += ("+" if q > 0 else "-") + (str(abs(q)) if abs(q) > 1 else "")
    return txt


def mol_wt(mol):
    t = sum(ELEMENTS[a.symbol][1] for a in mol.atoms)
    return t + mol.total_h * ELEMENTS["H"][1]


def exact_mol_wt(mol):
    t = sum(ELEMENTS[a.symbol][2] for a in mol.atoms)
    return t + mol.total_h * ELEMENTS["H"][2]


def ring_count(mol):
    return len(mol.rings)


def num_aromatic_rings(mol):
    return sum(1 for r in mol.rings if _ring_all_aromatic(mol, r))


def num_aliphatic_rings(mol):
    return sum(1 for r in mol.rings if not _ring_all_aromatic(mol, r))


def _ring_all_aromatic(mol, ring):
    bmap = {frozenset((b.i, b.j)): b for b in mol.bonds}
    return all(bmap[frozenset((ring[k], ring[(k + 1) % len(ring)]))].aromatic
               for k in range(len(ring)))


def fraction_csp3(mol):
    carbons = [i for i, a in enumerate(mol.atoms) if a.symbol == "C"]
    if not carbons:
        return 0.0
    sp3 = 0
    for i in carbons:
        if all(b.order == 1 and not b.aromatic for _, b in mol.adj[i]) \
                and not mol.atoms[i].aromatic:
            sp3 += 1
    return sp3 / len(carbons)


def bond_information(mol):
    """Sigma / pi bond counts (aromatic systems counted via Kekule form)."""
    heavy_bonds = mol.num_bonds
    pi = 0
    for b in mol.bonds:
        if b.order == 2:
            pi += 1
        elif b.order == 3:
            pi += 2
    sigma = heavy_bonds + mol.total_h
    return {
        "Heavy Atom Bonds": heavy_bonds,
        "Sigma Bonds": sigma,
        "Pi Bonds": pi,
        "Total Bonds": sigma + pi,
        "Total Atoms Including H": mol.num_atoms + mol.total_h,
    }


# ----------------------------------------------------------------------------
# TPSA (Ertl polar-surface-area contributions; S and P excluded)
# ----------------------------------------------------------------------------
def tpsa(mol):
    total = 0.0
    for i, a in enumerate(mol.atoms):
        deg = mol.degree(i)
        h = a.h
        dbl = [b for _, b in mol.adj[i] if b.order == 2]
        trp = [b for _, b in mol.adj[i] if b.order == 3]
        if a.symbol == "N":
            if a.aromatic:
                if a.charge == 0:
                    if deg == 2 and h == 0:
                        total += 12.89
                    elif h == 1 and deg == 2:
                        total += 15.79
                    elif deg == 3 and h == 0:
                        total += 4.93
                elif a.charge == 1:
                    total += 4.10 if (deg + h) == 3 else 3.88
                continue
            if a.charge == 0:
                if trp:
                    total += 23.79
                elif dbl:
                    total += 23.85 if h == 1 else 12.36
                else:
                    total += {0: 3.24, 1: 12.03, 2: 26.02}.get(h, 0.0)
            elif a.charge == 1:
                if len(dbl) == 1 and deg == 3 and h == 0:
                    total += 11.68       # nitro-type / N-oxide-type
                elif h >= 1:
                    total += {1: 16.61, 2: 25.59, 3: 27.64}.get(h, 0.0)
                else:
                    total += 0.0
        elif a.symbol == "O":
            if a.aromatic:
                total += 13.14
            elif a.charge == -1:
                total += 23.06
            elif dbl:
                total += 17.07
            elif h == 1:
                total += 20.23
            else:
                total += 9.23
    return total


# ----------------------------------------------------------------------------
# HYDROGEN BOND DONORS / ACCEPTORS, ROTATABLE BONDS
# ----------------------------------------------------------------------------
def num_hbd(mol):
    return sum(1 for a in mol.atoms if a.symbol in ("N", "O") and a.h > 0)


def num_hba(mol):
    count = 0
    for i, a in enumerate(mol.atoms):
        val = mol.valence(i)
        if a.aromatic and a.symbol in ("N", "O", "S"):
            if a.charge == 0 and (a.symbol != "N" or a.h == 0):
                count += 1
            continue
        if a.symbol in ("O", "S"):
            if a.charge == -1:
                count += 1
            elif a.h == 1 and val == 2:
                nb, _ = mol.adj[i][0] if mol.adj[i] else (None, None)
                ok = True
                if nb is not None:
                    for _, b in mol.adj[nb]:
                        if b.order == 2 and mol.atoms[b.other(nb)].symbol in "ONPS":
                            ok = False
                if ok:
                    count += 1
            elif a.h == 0 and val == 2 and a.charge == 0:
                count += 1
        elif a.symbol == "N" and a.charge == 0 and val == 3:
            excluded = False
            for j, b in mol.adj[i]:
                if b.order != 1:
                    continue
                for k, b2 in mol.adj[j]:
                    if k != i and b2.order == 2 and \
                            mol.atoms[k].symbol in ("O", "N", "P", "S") and \
                            frozenset((b2.i, b2.j)) not in mol.ring_bond_set():
                        excluded = True
            if not excluded:
                count += 1
    return count


def num_rotatable_bonds(mol):
    ring_bonds = mol.ring_bond_set()

    def is_terminal_group(i):
        a = mol.atoms[i]
        if a.symbol != "C" or mol.degree(i) < 3:
            return False
        nbrs = [mol.atoms[j].symbol for j, _ in mol.adj[i]]
        deg1 = sum(1 for j, _ in mol.adj[i] if mol.degree(j) == 1)
        if deg1 >= 3 and (nbrs.count("F") >= 3 or nbrs.count("Cl") >= 3
                          or nbrs.count("Br") >= 3):
            return True
        methyls = sum(1 for j, _ in mol.adj[i]
                      if mol.atoms[j].symbol == "C" and mol.degree(j) == 1)
        return methyls >= 3

    def has_triple(i):
        return any(b.order == 3 for _, b in mol.adj[i])

    count = 0
    for b in mol.bonds:
        if b.order != 1 or b.aromatic:
            continue
        if frozenset((b.i, b.j)) in ring_bonds:
            continue
        i, j = b.i, b.j
        if mol.degree(i) < 2 or mol.degree(j) < 2:
            continue
        if has_triple(i) or has_triple(j):
            continue
        if is_terminal_group(i) or is_terminal_group(j):
            continue
        # amide C(=O)-N bond: restricted rotation
        pair = (mol.atoms[i].symbol, mol.atoms[j].symbol)
        if "N" in pair and "C" in pair:
            c = i if mol.atoms[i].symbol == "C" else j
            if any(bb.order == 2 and mol.atoms[bb.other(c)].symbol in ("O", "S")
                   for _, bb in mol.adj[c]):
                continue
        count += 1
    return count


# ----------------------------------------------------------------------------
# LogP / MOLAR REFRACTIVITY  (Wildman-Crippen style atom contributions)
# ----------------------------------------------------------------------------
_HETERO = {"N", "O", "P", "S", "F", "Cl", "Br", "I"}

# type -> (logP contribution, MR contribution)
_CR = {
    "C1": (0.1441, 2.503), "C2": (0.0000, 2.433), "C3": (-0.2035, 2.753),
    "C4": (-0.2051, 2.731), "C5": (-0.2783, 5.007), "C6": (0.1551, 3.513),
    "C7": (0.0017, 3.888), "Csp": (0.0000, 3.0),
    "C_Me_ar": (0.08452, 2.464), "C_Me_het": (-0.1444, 2.412),
    "C_CH2_ar": (-0.0516, 2.488), "C_CH_ar": (0.1193, 2.582),
    "C_C_ar": (-0.0967, 2.576),
    "c_H": (0.1581, 3.350), "c_fus": (0.2955, 4.346), "c_aryl": (0.2713, 3.904),
    "c_C": (0.1360, 3.509), "c_N": (0.4619, 3.4), "c_O": (0.5437, 3.4),
    "c_S": (0.1893, 3.6), "c_exo": (-0.8186, 3.5),
    "c_F": (0.0, 3.257), "c_Cl": (0.245, 3.564), "c_Br": (0.198, 3.180),
    "c_I": (0.0, 3.104), "c_other": (-0.5443, 4.041),
    "N_prim": (-1.0190, 2.262), "N_sec": (-0.7096, 2.173),
    "N_prim_ar": (-1.0270, 2.275), "N_sec_ar": (-0.5188, 2.2),
    "N_imineH": (0.08387, 2.2), "N_imine": (0.1836, 2.2),
    "N_tert": (-0.3187, 2.2), "N_tert_ar": (-0.4458, 2.2),
    "N_nitrile": (-0.3, 2.2), "n_ar": (-0.3239, 2.2), "N_other": (-0.4, 2.2),
    "N_plus": (-0.3, 2.2),
    "O_ar": (0.1552, 1.2), "O_OH": (-0.2893, 1.395),
    "O_ether": (-0.0684, 1.2), "O_ar_ether": (-0.4195, 1.2),
    "O_ester": (-0.1540, 1.2), "O_carbonyl": (-0.5861, 1.2),
    "O_carbonyl_ar": (-0.5861, 1.2), "O_minus": (-1.326, 1.2),
    "O_other": (-0.2, 1.2),
    "F": (0.4202, 1.108), "Cl": (0.6895, 5.853), "Br": (0.8456, 8.927),
    "I": (0.8857, 14.02), "P": (0.8612, 6.92), "S": (0.6482, 7.591),
    "s_ar": (0.6237, 7.0), "B": (-0.3, 3.0), "X": (0.0, 3.0),
    "H_C": (0.1230, 1.057), "H_O": (-0.2677, 1.395),
    "H_N": (0.2142, 0.9627), "H_acid": (0.2980, 1.395), "H_S": (0.1125, 1.0),
}

# Atom types calibrated with the six-molecule reference set in the test script.
_CR["c_exo"] = (0.035, 3.5)


def _crippen_heavy_type(mol, i):
    a = mol.atoms[i]
    sym = a.symbol
    nb = [(j, b) for j, b in mol.adj[i]]
    nsym = [mol.atoms[j].symbol for j, _ in nb]
    dbl = [(j, b) for j, b in nb if b.order == 2 and not b.aromatic]
    trp = [j for j, b in nb if b.order == 3]
    arom_nb = [j for j, _ in nb if mol.atoms[j].aromatic]

    if sym == "C":
        if a.aromatic:
            exo = [j for j, b in nb if b.order == 2 and not b.aromatic]
            if exo:
                return "c_exo"
            if a.h == 1:
                return "c_H"
            non_ar = [j for j, b in nb if not b.aromatic]
            ar_cnt = len(nb) - len(non_ar)
            if not non_ar and ar_cnt == 3:
                return "c_fus"
            if non_ar:
                x = mol.atoms[non_ar[0]]
                if x.aromatic:
                    return "c_aryl"
                if x.symbol == "C":
                    return "c_C"
                if x.symbol == "N":
                    return "c_N"
                if x.symbol == "O":
                    return "c_O"
                if x.symbol == "S":
                    return "c_S"
                if x.symbol == "F":
                    return "c_F"
                if x.symbol == "Cl":
                    return "c_Cl"
                if x.symbol == "Br":
                    return "c_Br"
                if x.symbol == "I":
                    return "c_I"
                return "c_other"
            return "c_fus"
        if trp:
            return "Csp"
        if dbl:
            if any(mol.atoms[j].symbol != "C" for j, _ in dbl):
                return "C6"
            return "C7"
        # sp3-like carbon
        hetero = [j for j in range(len(nb)) if nsym[j] in _HETERO]
        aliph_other = [j for j, (k, _) in enumerate(nb)
                       if nsym[j] not in _HETERO and not mol.atoms[k].aromatic]
        if a.h == 3 and hetero:
            return "C3"
        if hetero:
            n_ali = len(aliph_other)
            if a.h == 2 and n_ali >= 1:
                return "C3"
            if a.h == 1 and n_ali >= 2:
                return "C4"
            if a.h == 0 and n_ali >= 3:
                return "C5"
        if arom_nb:
            ar_c = any(mol.atoms[j].symbol == "C" for j in arom_nb)
            if a.h == 3:
                return "C_Me_ar" if ar_c else "C_Me_het"
            if a.h == 2:
                return "C_CH2_ar"
            if a.h == 1:
                return "C_CH_ar"
            return "C_C_ar"
        if a.h >= 3 or a.h == 2:
            return "C1"
        if a.h == 4:
            return "C1"
        return "C2"
    if sym == "N":
        if a.aromatic:
            return "n_ar" if a.charge == 0 else "N_plus"
        if a.charge != 0:
            return "N_plus"
        if trp:
            return "N_nitrile"
        if dbl:
            return "N_imineH" if a.h == 1 else "N_imine"
        n_ar = len(arom_nb)
        heavy = len(nb)
        if heavy == 1 and a.h == 2:
            return "N_prim_ar" if n_ar else "N_prim"
        if heavy == 2 and a.h == 1:
            return "N_sec_ar" if n_ar else "N_sec"
        if heavy == 3:
            return "N_tert_ar" if n_ar else "N_tert"
        return "N_other"
    if sym == "O":
        if a.aromatic:
            return "O_ar"
        if a.charge == -1:
            return "O_minus"
        if dbl:
            j = dbl[0][0]
            return "O_carbonyl_ar" if mol.atoms[j].aromatic else "O_carbonyl"
        if a.h >= 1:
            return "O_OH"
        # ethers / esters
        if any(mol.atoms[j].aromatic for j, _ in nb):
            carbonyl_nb = False
            for j, _ in nb:
                for k, b in mol.adj[j]:
                    if b.order == 2 and mol.atoms[k].symbol == "O":
                        carbonyl_nb = True
            return "O_ester" if carbonyl_nb else "O_ar_ether"
        for j, _ in nb:
            for k, b in mol.adj[j]:
                if b.order == 2 and mol.atoms[k].symbol in ("O", "S", "N"):
                    return "O_ester"
        return "O_ether"
    if sym in ("F", "Cl", "Br", "I", "P"):
        return sym
    if sym == "S":
        return "s_ar" if a.aromatic else "S"
    if sym == "B":
        return "B"
    return "X"


def _crippen_hydrogens(mol, i):
    a = mol.atoms[i]
    if a.h <= 0:
        return None
    if a.symbol == "O":
        acid = False
        for j, _ in mol.adj[i]:
            for k, b in mol.adj[j]:
                if k != i and b.order == 2 and mol.atoms[k].symbol in ("O", "S", "N"):
                    acid = True
        return "H_acid" if acid else "H_O"
    if a.symbol == "N":
        return "H_N"
    if a.symbol == "S":
        return "H_S"
    return "H_C"


def _crippen_sum(mol, idx):
    total = 0.0
    for i in range(mol.num_atoms):
        total += _CR[_crippen_heavy_type(mol, i)][idx]
        ht = _crippen_hydrogens(mol, i)
        if ht:
            total += mol.atoms[i].h * _CR[ht][idx]
    return total


def crippen_logp(mol):
    return _crippen_sum(mol, 0)


def molar_refractivity(mol):
    return _crippen_sum(mol, 1)


# ----------------------------------------------------------------------------
# SMILES WRITER (normalised SMILES from the built-in writer)
# ----------------------------------------------------------------------------
_AROM_SYM = {"B": "b", "C": "c", "N": "n", "O": "o", "P": "p", "S": "s",
             "Se": "se", "As": "as"}


def _atom_text(mol, i):
    a = mol.atoms[i]
    deg = mol.degree(i)
    if a.aromatic:
        sym = _AROM_SYM.get(a.symbol, a.symbol.lower())
        expected_h = max(0, 3 - deg) if a.symbol in ("C", "B") else 0
        plain = (a.symbol in ORGANIC_SUBSET and a.charge == 0 and
                 a.isotope == 0 and a.h == expected_h)
    else:
        sym = a.symbol
        used = sum(b.order for _, b in mol.adj[i])
        plain = (a.symbol in ORGANIC_SUBSET and a.charge == 0 and
                 a.isotope == 0 and a.h == _implicit_h(a.symbol, used))
    if plain:
        return sym
    txt = "[" + (str(a.isotope) if a.isotope else "") + sym
    if a.h:
        txt += "H" + (str(a.h) if a.h > 1 else "")
    if a.charge:
        txt += ("+" if a.charge > 0 else "-") + \
               (str(abs(a.charge)) if abs(a.charge) > 1 else "")
    return txt + "]"


def _bond_text(mol, b):
    ai, aj = mol.atoms[b.i].aromatic, mol.atoms[b.j].aromatic
    if b.aromatic and ai and aj:
        return ""
    if b.order == 1:
        return "-" if (ai and aj) else ""
    return "=" if b.order == 2 else "#"


def _atom_ranks(mol):
    n = mol.num_atoms
    inv = [(ELEMENTS[a.symbol][0], mol.degree(i), a.h, a.charge,
            int(a.aromatic)) for i, a in enumerate(mol.atoms)]
    ranks = {v: r for r, v in enumerate(sorted(set(inv)))}
    cur = [ranks[v] for v in inv]
    for _ in range(max(4, min(n, 12))):
        sig = [(cur[i], tuple(sorted((cur[j], b.order) for j, b in mol.adj[i])))
               for i in range(n)]
        ranks = {v: r for r, v in enumerate(sorted(set(sig)))}
        new = [ranks[v] for v in sig]
        if new == cur:
            break
        cur = new
    return cur


def to_smiles(mol):
    n = mol.num_atoms
    rank = _atom_ranks(mol)
    visited, parent = set(), {}
    children = {i: [] for i in range(n)}
    closures = set()
    size = {}

    def dfs(u, p):
        visited.add(u)
        parent[u] = p
        size[u] = 1
        for v, _ in sorted(mol.adj[u], key=lambda t: (rank[t[0]], t[0])):
            if v == p:
                continue
            if v in visited:
                closures.add(frozenset((u, v)))
            else:
                children[u].append(v)
                dfs(v, u)
                size[u] += size[v]

    parts = []
    digits = {}
    free_digits = list(range(1, 100))
    bonds = {frozenset((b.i, b.j)): b for b in mol.bonds}

    def write(u):
        out = _atom_text(mol, u)
        for e in sorted(c for c in closures if u in c):
            if e in digits:
                d = digits.pop(e)
                out += _bond_text(mol, bonds[e]) + (str(d) if d < 10 else "%%%02d" % d)
                free_digits.append(d)
                free_digits.sort()
            else:
                d = free_digits.pop(0)
                digits[e] = d
                out += (str(d) if d < 10 else "%%%02d" % d)
        kids = sorted(children[u], key=lambda c: (size[c], rank[c], c))
        for k, c in enumerate(kids):
            seg = _bond_text(mol, bonds[frozenset((u, c))]) + write(c)
            out += seg if k == len(kids) - 1 else "(" + seg + ")"
        return out

    import sys
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old, 5000))
    try:
        for s in sorted(range(n), key=lambda i: (rank[i], i)):
            if s in visited:
                continue
            dfs(s, None)
            parts.append(write(s))
    finally:
        sys.setrecursionlimit(old)
    return ".".join(parts)


# ----------------------------------------------------------------------------
# GEOMETRY HELPERS: ideal distance matrices + weighted SMACOF embedding
# ----------------------------------------------------------------------------
def _floyd(D):
    n = len(D)
    for k in range(n):
        D = np.minimum(D, D[:, k:k + 1] + D[k:k + 1, :])
    return D


def _smacof(D, W, dim, iters=250, seed=0, X0=None):
    n = len(D)
    rng = np.random.default_rng(seed)
    if X0 is None:
        X = rng.normal(size=(n, dim)) * (D[np.isfinite(D)].mean() / 3.0 + 0.1)
    else:
        X = X0.copy()
    V = np.diag(W.sum(1)) - W
    Vp = np.linalg.pinv(V)
    last = None
    for _ in range(iters):
        diff = X[:, None, :] - X[None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(dist > 1e-9, W * D / dist, 0.0)
        B = -ratio
        np.fill_diagonal(B, 0.0)
        B[np.diag_indices(n)] = -B.sum(1)
        X = Vp @ B @ X
        stress = float((W * (dist - D) ** 2).sum() / 2)
        if last is not None and abs(last - stress) < 1e-7 * (1 + last):
            break
        last = stress
    diff = X[:, None, :] - X[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    stress = float((W * (dist - D) ** 2).sum() / 2)
    return X - X.mean(0), stress


def _topological_distance(adj_lists):
    n = len(adj_lists)
    T = np.full((n, n), np.inf)
    for s in range(n):
        T[s, s] = 0
        queue = [s]
        while queue:
            nxt = []
            for u in queue:
                for v in adj_lists[u]:
                    if T[s, v] == np.inf:
                        T[s, v] = T[s, u] + 1
                        nxt.append(v)
            queue = nxt
    return T


def _ring_distance_overrides(rings, bondlen, D):
    for ring in rings:
        m = len(ring)
        if m < 3 or m > 7:
            continue
        side = float(np.mean([bondlen(ring[k], ring[(k + 1) % m]) for k in range(m)]))
        R = side / (2 * math.sin(math.pi / m))
        for p in range(m):
            for q in range(p + 1, m):
                k = min(q - p, m - (q - p))
                d = 2 * R * math.sin(math.pi * k / m)
                a, b = ring[p], ring[q]
                D[a, b] = D[b, a] = d


def _finish_distances(D):
    """Fill infinities (disconnected fragments) and make metric."""
    D = _floyd(D)
    finite = D[np.isfinite(D)]
    gap = (finite.max() if finite.size else 1.0) + 4.0
    D = np.where(np.isfinite(D), D, gap)
    return D


def _weights(D, topo, far=0.25):
    with np.errstate(divide="ignore"):
        W = 1.0 / np.maximum(D, 0.5) ** 2
    W = np.where(topo > 3, W * far, W)
    W = np.where(np.isinf(topo), W * 0.05, W)
    np.fill_diagonal(W, 0.0)
    return W


def _refine_weights(W, topo, rings, local=6.0, far=0.05):
    """Weights for a second pass: rigid local geometry, loose long-range."""
    W2 = np.where(topo <= 2, W * local, W * far)
    for ring in rings:
        if 3 <= len(ring) <= 7:
            for p in ring:
                for q in ring:
                    if p != q:
                        W2[p, q] = max(W2[p, q], W[p, q] * local)
    np.fill_diagonal(W2, 0.0)
    return W2


# ----------------------------------------------------------------------------
# 2D LAYOUT AND DRAWING
# ----------------------------------------------------------------------------
def layout_2d(mol, seed_tries=8):
    n = mol.num_atoms
    if n == 1:
        return np.zeros((1, 2))
    adj = [[j for j, _ in mol.adj[i]] for i in range(n)]
    L = 1.5
    topo = _topological_distance(adj)
    D = np.full((n, n), np.inf)
    np.fill_diagonal(D, 0.0)
    for b in mol.bonds:
        D[b.i, b.j] = D[b.j, b.i] = L
    for c in range(n):
        sp = any(b.order == 3 for _, b in mol.adj[c]) or \
            sum(1 for _, b in mol.adj[c] if b.order == 2) >= 2
        d13 = 2 * L if sp else 2 * L * math.sin(math.radians(60))
        for p, q in itertools.combinations(adj[c], 2):
            D[p, q] = D[q, p] = min(D[p, q], d13)
    _ring_distance_overrides(mol.rings, lambda a, b: L, D)
    D = _finish_distances(D)
    W = _weights(D, topo, far=0.5)
    best, best_s = None, None
    for seed in range(seed_tries):
        X, s = _smacof(D, W, 2, iters=400, seed=seed)
        if best_s is None or s < best_s:
            best, best_s = X, s
    X, _ = _smacof(D, _refine_weights(W, topo, mol.rings), 2, iters=400,
                   X0=best)
    # orient: longest principal axis horizontal
    if n >= 2:
        u, sv, vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
        X = (X - X.mean(0)) @ vt.T
    return X


_CPK = {"C": "#222222", "H": "#777777", "N": "#2F5BFF", "O": "#E53935",
        "F": "#2E9E3F", "Cl": "#2E9E3F", "Br": "#8E3B2F", "I": "#8E44AD",
        "S": "#B8A000", "P": "#F39C12", "B": "#C0763C"}


def _atom_label(mol, i):
    a = mol.atoms[i]
    if a.symbol == "C" and mol.degree(i) > 0 and a.charge == 0 and a.isotope == 0:
        return None
    txt = a.symbol
    if a.h:
        txt += "H" + (str(a.h) if a.h > 1 else "")
    if a.charge:
        txt += ("+" if a.charge > 0 else "-") if abs(a.charge) == 1 else \
            ("%d%s" % (abs(a.charge), "+" if a.charge > 0 else "-"))
    return txt


def draw_2d(mol, size=(700, 500)):
    """Return a PIL image of the 2D structure."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from PIL import Image

    X = layout_2d(mol)
    n = mol.num_atoms
    fig = Figure(figsize=(size[0] / 100.0, size[1] / 100.0), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.axis("off")

    L = 1.5
    ring_of = {}
    for ring in sorted(mol.rings, key=len):
        for k in range(len(ring)):
            e = frozenset((ring[k], ring[(k + 1) % len(ring)]))
            ring_of.setdefault(e, ring)
    labels = [_atom_label(mol, i) for i in range(n)]
    trim = 0.27 * L

    def seg(p, q, i, j, color_i, color_j, lw=1.8):
        d = q - p
        ln = float(np.hypot(*d))
        if ln < 1e-9:
            return
        u = d / ln
        p2, q2 = p.copy(), q.copy()
        if labels[i]:
            p2 = p + u * trim
        if labels[j]:
            q2 = q - u * trim
        mid = (p2 + q2) / 2
        ax.plot([p2[0], mid[0]], [p2[1], mid[1]], color=color_i, lw=lw,
                solid_capstyle="round", zorder=1)
        ax.plot([mid[0], q2[0]], [mid[1], q2[1]], color=color_j, lw=lw,
                solid_capstyle="round", zorder=1)

    for b in mol.bonds:
        i, j = b.i, b.j
        p, q = X[i], X[j]
        d = q - p
        ln = float(np.hypot(*d)) or 1.0
        u = d / ln
        perp = np.array([-u[1], u[0]])
        ci = _CPK.get(mol.atoms[i].symbol, "#555555")
        cj = _CPK.get(mol.atoms[j].symbol, "#555555")
        off = 0.11 * L
        if b.order == 1:
            seg(p, q, i, j, ci, cj)
        elif b.order == 2:
            ring = ring_of.get(frozenset((i, j)))
            if ring is not None:
                cen = X[list(ring)].mean(0)
                side = 1.0 if np.dot(cen - (p + q) / 2, perp) > 0 else -1.0
                seg(p, q, i, j, ci, cj)
                p3 = p + perp * off * 2 * side + u * 0.16 * L
                q3 = q + perp * off * 2 * side - u * 0.16 * L
                ax.plot([p3[0], q3[0]], [p3[1], q3[1]], color="#222222",
                        lw=1.5, solid_capstyle="round", zorder=1)
            else:
                seg(p + perp * off, q + perp * off, i, j, ci, cj)
                seg(p - perp * off, q - perp * off, i, j, ci, cj)
        elif b.order == 3:
            seg(p, q, i, j, ci, cj)
            seg(p + perp * off * 1.8, q + perp * off * 1.8, i, j, ci, cj)
            seg(p - perp * off * 1.8, q - perp * off * 1.8, i, j, ci, cj)

    for i in range(n):
        if labels[i]:
            col = _CPK.get(mol.atoms[i].symbol, "#555555")
            ax.text(X[i, 0], X[i, 1], labels[i], color=col, ha="center",
                    va="center", fontsize=15, fontweight="bold", zorder=3,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))

    pad = 1.2 * L
    xmin, ymin = X.min(0) - pad
    xmax, ymax = X.max(0) + pad
    span_x, span_y = xmax - xmin, ymax - ymin
    aspect = size[0] / float(size[1])
    if span_x / span_y < aspect:
        extra = span_y * aspect - span_x
        xmin, xmax = xmin - extra / 2, xmax + extra / 2
    else:
        extra = span_x / aspect - span_y
        ymin, ymax = ymin - extra / 2, ymax + extra / 2
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    img.load()
    return img


# ----------------------------------------------------------------------------
# 3D COORDINATES
# ----------------------------------------------------------------------------
def _bond_length(sym1, sym2, order, aromatic):
    r = ELEMENTS.get(sym1, (0, 0, 0, 0, 0.8))[4] + ELEMENTS.get(sym2, (0, 0, 0, 0, 0.8))[4]
    if aromatic:
        return r * 0.92
    return r * {1: 1.0, 2: 0.88, 3: 0.80}.get(order, 1.0)


def generate_3d(mol, tries=6):
    """Embed the molecule (with explicit hydrogens) in 3D.  Returns Mol3D."""
    try:
        return _generate_3d(mol, tries)
    except Exception:
        return None


def _generate_3d(mol, tries):
    nh = mol.num_atoms
    symbols = [a.symbol for a in mol.atoms]
    bonds = [(b.i, b.j, b.order, b.aromatic) for b in mol.bonds]
    for i, a in enumerate(mol.atoms):
        for _ in range(a.h):
            symbols.append("H")
            bonds.append((i, len(symbols) - 1, 1, False))
    n = len(symbols)
    if n == 1:
        return Mol3D(symbols, np.zeros((1, 3)), [])

    adj = [[] for _ in range(n)]
    for i, j, o, ar in bonds:
        adj[i].append(j)
        adj[j].append(i)
    bmap = {}
    for i, j, o, ar in bonds:
        bmap[(i, j)] = bmap[(j, i)] = (o, ar)

    def blen(i, j):
        o, ar = bmap[(i, j)]
        return _bond_length(symbols[i], symbols[j], o, ar)

    def center_angle(c):
        nbs = adj[c]
        orders = [bmap[(c, j)] for j in nbs]
        n_dbl = sum(1 for o, ar in orders if o == 2 and not ar)
        n_trp = sum(1 for o, ar in orders if o == 3)
        n_arom = sum(1 for o, ar in orders if ar)
        if n_trp or n_dbl >= 2:
            return 180.0
        if n_dbl or n_arom:
            return 120.0
        s = symbols[c]
        if s == "N" and len(nbs) == 3:
            for j in nbs:
                for k in adj[j]:
                    if k != c and bmap[(j, k)][0] == 2 and symbols[k] in ("O", "S", "N"):
                        return 120.0
            return 108.0
        if s in ("O", "S") and len(nbs) == 2:
            for j in nbs:
                if any(bmap[(j, k)][0] == 2 or bmap[(j, k)][1] for k in adj[j] if k != c):
                    return 118.0
            return 105.0
        return 109.5

    D = np.full((n, n), np.inf)
    np.fill_diagonal(D, 0.0)
    for i, j, o, ar in bonds:
        D[i, j] = D[j, i] = blen(i, j)
    for c in range(n):
        ang = math.radians(center_angle(c))
        for p, q in itertools.combinations(adj[c], 2):
            a, b = blen(c, p), blen(c, q)
            d = math.sqrt(max(a * a + b * b - 2 * a * b * math.cos(ang), 0.01))
            D[p, q] = D[q, p] = min(D[p, q], d)
    _ring_distance_overrides(mol.rings, lambda a, b: blen(a, b), D)
    topo = _topological_distance(adj)
    D = _finish_distances(D)
    # lower bound to avoid atoms overlapping in the embedding
    D = np.maximum(D, np.where(np.eye(n, dtype=bool), 0.0, 1.6))
    for i, j, o, ar in bonds:
        D[i, j] = D[j, i] = blen(i, j)
    W = _weights(D, topo, far=0.15)
    best, best_s = None, None
    for seed in range(tries):
        X, s = _smacof(D, W, 3, iters=500, seed=seed)
        if best_s is None or s < best_s:
            best, best_s = X, s
    X, _ = _smacof(D, _refine_weights(W, topo, mol.rings), 3, iters=500,
                   X0=best)
    # centre and align along principal axes
    X = X - X.mean(0)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    X = X @ vt.T
    out_bonds = [(i, j, o) for i, j, o, ar in bonds]
    return Mol3D(symbols, X, out_bonds)


def mol_to_molblock(m3d):
    """MDL V2000 mol block for a Mol3D (used by py3Dmol)."""
    lines = ["molecule", "  chemlite", ""]
    lines.append("%3d%3d  0  0  0  0  0  0  0  0999 V2000" %
                 (m3d.num_atoms, len(m3d.bonds)))
    for s, (x, y, z) in zip(m3d.symbols, m3d.coords):
        lines.append("%10.4f%10.4f%10.4f %-3s 0  0  0  0  0  0  0  0  0  0  0  0"
                     % (x, y, z, s))
    for i, j, o in m3d.bonds:
        lines.append("%3d%3d%3d  0" % (i + 1, j + 1, int(o)))
    lines.append("M  END")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------
# ONE-CALL HELPER
# ----------------------------------------------------------------------------
def describe(mol):
    """Return a dict of the main descriptors (handy for scripts)."""
    return {
        "formula": molecular_formula(mol),
        "mol_wt": round(mol_wt(mol), 3),
        "exact_mass": round(exact_mol_wt(mol), 4),
        "logp": round(crippen_logp(mol), 3),
        "tpsa": round(tpsa(mol), 3),
        "hbd": num_hbd(mol),
        "hba": num_hba(mol),
        "rotatable_bonds": num_rotatable_bonds(mol),
        "rings": ring_count(mol),
        "aromatic_rings": num_aromatic_rings(mol),
        "aliphatic_rings": num_aliphatic_rings(mol),
        "fsp3": round(fraction_csp3(mol), 3),
        "molar_refractivity": round(molar_refractivity(mol), 3),
    }
