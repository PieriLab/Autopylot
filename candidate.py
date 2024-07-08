import itertools as it
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import io_utils as io


@dataclass
class CandidateListGenerator:
    calc_type: str
    nelec: int
    charge: int
    n_singlets: int
    n_triplets: int
    method: Optional[list[str]] = field(default_factory=lambda: [])
    fon_temperature: Optional[list[float]] = field(default_factory=lambda: [])
    rc_w: Optional[list[float]] = field(default_factory=lambda: [])
    active_space: Optional[list[list[int]]] = field(default_factory=lambda: [])

    def create_candidate_list(self):
        attribute_list = [self.method, self.fon_temperature, self.rc_w, self.active_space]
        arg_prod = [x if x else [[]] for x in attribute_list]
        candidate_list = []
        for met, fon, rcw, acs in it.product(*arg_prod):
            d = {'calc_type': self.calc_type, 'nelec': self.nelec, 'charge': self.charge, 'n_singlets': self.n_singlets, 'n_triplets': self.n_triplets, 'method': met, 'active_space': acs, 'rc_w': rcw, 'fon_temperature': fon}
            cand = Candidate(**d)
            candidate_list.append(cand)
        return candidate_list


@dataclass
class Candidate:
    calc_type: str
    nelec: int
    charge: int
    n_singlets: int
    n_triplets: int
    method: Optional[str] = None
    fon_temperature: Optional[float] = None
    rc_w: Optional[float] = None
    active_space: Optional[list[int]] = None

    @property
    def orbitals(self):
        if self.active_space:
            return self.active_space[1]
        else:
            return None

    @property
    def electrons(self):
        if self.active_space:
            return self.active_space[0]
        else:
            return None

    @property
    def folder_name(self):
        name = f'_{self.method}' if self.method else ""
        name += f'_T{self.fon_temperature}' if self.fon_temperature else ""
        name += f'_w{self.rc_w}' if self.rc_w else ""
        name += f'_AS{self.electrons}{self.orbitals}' if self.active_space else ""
        return name
    
    @property
    def full_method(self):
        full_method = self.calc_type
        if self.method:
            full_method += f'_{self.method}'
        if self.fon_temperature:
            full_method += f'_T{self.fon_temperature}'
        if self.rc_w:
            full_method += f'_w{self.rc_w}'
        if self.active_space:
            full_method += f'_AS{self.electrons}{self.orbitals}'
        return full_method

    def validate_as(self):
        if self.active_space:
            acs1 = self.electrons
            acs2 = self.orbitals
            assert ((acs1 % 2) == 0), f"You want {acs1} electrons in your active space? I don't think so."
            assert ((acs1 >= 2) and ((acs1/2) < acs2)), f"How can I put {acs1} electrons in {acs2} orbitals? This is stupid."

    @property
    def calc_settings(self):
        if self.calc_type == 'casscf_fomo':
            new_settings = {
                'run': 'energy',
                'method': 'rhf',
                'threall': '1.1e-14',
                'convthre': '1.0e-6',
                'precision': 'mixed',
                'casci': 'yes',
                'fon': 'yes',
                'fon_temperature': self.fon_temperature,
                'closed': int((self.nelec/2)-(self.electrons/2)),
                'active': self.orbitals,
                'cassinglets': self.n_singlets,
                'castriplets': self.n_triplets,
                'gpus': '2'
                }
        if self.calc_type == 'casscf':
            new_settings = {
                'run': 'energy',
                'guess': 'generate',
                'method': 'rhf',
                'threall': '1.1e-14',
                'convthre': '1.0e-6',
                'precision': 'mixed',
                'fon': 'yes',
                'fon_temperature': '0.2',
                'fon_target': '0.2',
                'fon_anneal': 'no',
                'casscf': 'yes',
                'closed': int((self.nelec/2)-(self.electrons/2)),
                'active': self.orbitals,
                'cassinglets': self.n_singlets,
                'castriplets': self.n_triplets,
                'gpus': '2'
                }
        if self.calc_type == 'casdft':
            new_settings = {
                'run': 'energy',
                'threall': '1.1e-14',
                'convthre': '1.0e-6',
                'precision': 'mixed',
                'method': self.method,
                'rc_w': self.rc_w,
                'fon': 'yes',
                'fon_temperature': self.fon_temperature,
                'casci': 'yes',
                'cphftol': '1.0e-6',
                'cphfiter': '1000',
                'cphfalgorithm': 'inc_diis',
                'closed': int((self.nelec/2)-(self.electrons/2)),
                'active': self.orbitals,
                'cassinglets': self.n_singlets,
                'castriplets': self.n_triplets,
                'gpus': '2'
                }
        if self.calc_type == 'hhtda':
            new_settings = {
                'run': 'energy',
                'charge': int(self.charge-2),
                'threall': '1.1e-14',
                'convthre': '1.0e-6',
                'precision': 'mixed',
                'hhtda': 'yes',
                'method': self.method,
                'rc_w': self.rc_w,
                'scf': 'diis+a',
                'cphftol': '1.0e-6',
                'cphfiter': '1000',
                'cphfalgorithm': 'inc_diis',
                'cismax': '300',
                'cismaxiter': '500',
                'cisconvtol': '1.0e-6',
                'cisnumstates': int(self.n_singlets + self.n_triplets),
                'hhtdasinglets': self.n_singlets,
                'hhtdatriplets': self.n_triplets,
                'gpus': '2'
                }
        if self.calc_type == 'hhtda_fomo':
            new_settings = {
                'run': 'energy',
                'threall': '1.1e-14',
                'convthre': '1.0e-6',
                'precision': 'mixed',
                'hhtda': 'yes',
                'method': self.method,
                'rc_w': self.rc_w,
                'fon': 'yes',
                'fon_temperature': self.fon_temperature,
                'cphftol': '1.0e-6',
                'cphfiter': '1000',
                'cphfalgorithm': 'inc_diis',
                'scf': 'diis+a',
                'cismax': '300',
                'cismaxiter': '500',
                'cisconvtol': '1.0e-6',
                'cisnumstates': int(self.n_singlets + self.n_triplets),
                'closed': 0,
                'active': int((self.nelec/2)+1),
                'hhtdasinglets': self.n_singlets,
                'hhtdatriplets': self.n_triplets,
                'gpus': '2'
                }
        return new_settings


def main():
    fn = Path('test_input.yaml')
    settings = io.yload(fn)
    a = CandidateListGenerator(**settings['candidates']['fomo'])
    print(f'{a=}')
    b = a.create_candidate_list()
    print(f'{b=}')
    c = b[0]
    c.validate_as()
    print(c.folder_name)


if __name__ == "__main__":
    main()
