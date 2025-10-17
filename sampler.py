import chem
import itertools
import random

import numpy as np

import reaction
import process
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


class Sampler: # Considering only bond types of the reactions ...

    def __init__(self,
                 reactivity_model=None, 
                 ea_estimator=None, 
                 classifier=None,
                 uff_energy_criteria=None):
        self.reactivity_model = reactivity_model
        self.ea_estimator = ea_estimator
        self.classifier = classifier
        self.n_criteria = 20
        self.ea_criteria = 10000
        self.chg_criteria = 0
        self.min_form = 0
        self.min_break = 0
        self.chg_diff = 0 
        self.max_bo_diff = 0
        self.radius = 1 # Radius for model reaction ...
        self.model_type = 'adj'
        self.remove_repetition = True
        self.solvent_infos = [] # [(molecule, donor, acceptor)]
        if uff_energy_criteria is None:
            uff_energy_criteria = 1000
        enumerator = None
        if reactivity_model is None:
            enumerator = reaction.RxnEnumerator(2,2)
        self.enumerator = enumerator

    def set_n_form(self,n_form):
        self.enumerator.set_max_form(n_form)

    def set_n_break(self,n_break):
        self.enumerator.set_max_break(n_break)
    
    def sample_rxns(self,reactant,num_sample = 200):
        
        # Sample rxns ...
        """
        if n > self.n_criteria:
            if self.reactivity_model is None:
                rxns = self.random_sample(reactant,10*num_sample)
            else:
                rxns = self.reactivity_model(reactant,10*num_sample)
        else:
            # Directly sample all ...
            rxns = self.enumerator.enumerate_rxns(reactant,remove_repetition=True)
        """
        rxns = self.enumerator.enumerate_rxns(reactant, remove_repetition=self.remove_repetition)
        r_id = reactant.get_id()
        bf_dict = dict()
        atom_list = reactant.atom_list
        n = len(atom_list)
        non_h_keys = []
        h_keys = []
        solvent_added_rxns = []


        for rxn in rxns:
            num_break = len(rxn.bond_breaks)
            num_form = len(rxn.bond_forms)
            if num_form < self.min_form:
                continue
            if num_break < self.min_break:
                continue                        

            additional_rxns = self.get_solvent_added_rxns(rxn) # Initially add solvent added rxns ...
            solvent_added_rxns += additional_rxns
            put_in = True
            bf_string = rxn.get_bf_string()
            # Check repeating ...
            ts_molecule = rxn.get_ts_molecule()
            if len(ts_molecule.get_molecule_list()) > 1: # Spectating exists ...
                continue
            rxn_id = rxn.get_id()
            if put_in:
                rxn.rxn_id = rxn_id
                if bf_string not in bf_dict:
                    bf_dict[bf_string] = [rxn]
                else:
                    bf_dict[bf_string].append(rxn)
                if 'H' in bf_string:
                    if bf_string not in h_keys:
                        h_keys.append(bf_string)
                else:
                    if bf_string not in non_h_keys:
                        non_h_keys.append(bf_string)    
                    
        num_dict = len(bf_dict)
        if num_dict == 0:
            return []
        num_remain = num_sample
        num_equal = int(num_remain/(2*num_dict)) + 1 
        sampled_rxn_ids = set([])

        non_h_rxns = []

        # Sample reactions from non-H reactions ...
        final_bf_dict = dict()
        for bf_string in non_h_keys:
            final_bf_dict[bf_string] = []
            # Sample num_equal randomlly within each bf_dict
            sub_rxns = bf_dict[bf_string]
            # If less reaction, sample all ...
            if len(sub_rxns) < num_equal:
                for rxn in sub_rxns:
                    rxn_id = rxn.rxn_id
                    if rxn_id in sampled_rxn_ids:
                        continue
                    is_screened, screening_log = self.check_screening(rxn)
                    #print (is_screened, screening_log,rxn.bond_forms, rxn.bond_breaks)
                    if not is_screened:
                        final_bf_dict[bf_string].append(rxn)
                        sampled_rxn_ids.add(rxn_id)
                        num_remain -= 1

            else:
                # If many, randomly sample ...
                indices = list(range(len(sub_rxns)))
                while len(final_bf_dict[bf_string]) < num_equal and len(indices) > 0:
                    index = random.sample(indices,1)[0]
                    rxn = sub_rxns[index]
                    rxn_id = rxn.rxn_id
                    indices.remove(index)
                    if rxn_id in sampled_rxn_ids:
                        continue
                    is_screened, screening_log = self.check_screening(rxn)
                    #print (is_screened, screening_log,rxn.bond_forms, rxn.bond_breaks)
                    if not is_screened:
                        final_bf_dict[bf_string].append(rxn)
                        sampled_rxn_ids.add(rxn_id)
                        num_remain -= 1
            # Adjust num_equal dynamically ...
            num_equal = int(num_remain/(2*num_dict)) + 1 

        for bf in final_bf_dict:
            non_h_rxns += final_bf_dict[bf]
        if len(non_h_rxns) > num_sample/2:
            random.shuffle(non_h_rxns)
            non_h_rxns = non_h_rxns[:int(num_sample/2)+1]

        h_rxns = []

        # Sample reactions from Hydrogen related reactions ...
        final_bf_dict = dict()
        for bf_string in h_keys:
            final_bf_dict[bf_string] = []
            # Sample num_equal randomlly within each bf_dict
            sub_rxns = bf_dict[bf_string]
            # If less reaction, sample all ...
            if len(sub_rxns) < num_equal:
                for rxn in sub_rxns:
                    rxn_id = rxn.rxn_id
                    if rxn_id in sampled_rxn_ids:
                        continue
                    is_screened, screening_log = self.check_screening(rxn)
                    value = 0
                    #print (is_screened, screening_log,rxn.bond_forms, rxn.bond_breaks)
                    if not is_screened:
                        final_bf_dict[bf_string].append(rxn)
                        sampled_rxn_ids.add(rxn_id)
                        num_remain = -1                            
            else:
                # If many, randomly sample ...
                indices = list(range(len(sub_rxns)))
                while len(final_bf_dict[bf_string]) < num_equal and len(indices) > 0:
                    index = random.sample(indices,1)[0]
                    rxn = sub_rxns[index]
                    rxn_id = rxn.rxn_id
                    indices.remove(index)
                    if rxn_id in sampled_rxn_ids:
                        continue
                    is_screened, screening_log = self.check_screening(rxn)
                    #print (is_screened, screening_log,rxn.bond_forms, rxn.bond_breaks)
                    if not is_screened:
                        final_bf_dict[bf_string].append(rxn)
                        sampled_rxn_ids.add(rxn_id)
                        num_remain -= 1
            # Adjust num_equal dynamically ...
            num_equal = int(num_remain/(2*num_dict)) + 1 

        for bf in final_bf_dict:
            h_rxns += final_bf_dict[bf]
        if len(h_rxns) > num_sample/2:
            random.shuffle(h_rxns)
            h_rxns = h_rxns[:int(num_sample/2)+1]

        # Add new reaction, for solvent added reactions ...
        bf_dict = dict()
        #print ('solvent',solvent_added_rxns)
        for rxn in solvent_added_rxns:
            num_break = len(rxn.bond_breaks)
            num_form = len(rxn.bond_forms)
            #if num_form < self.min_form:
            #    continue
            #if num_break < self.min_break:
            #    continue                        
            put_in = True
            bf_string = rxn.get_bf_string()
            # Check repeating ...
            ts_molecule = rxn.get_ts_molecule()
            if len(ts_molecule.get_molecule_list()) > 1: # Spectating exists ...
                #print ('screened',rxn.bond_forms, rxn.bond_breaks)
                continue
            #chg = ts_molecule.get_chg()
            #molecule_id = ts_molecule.get_connectivity_id() * (chg + 13)
            #formula_id = ts_molecule.get_formula_id()
            #rxn_id = f'{r_id}_{molecule_id}' # r_id + ts_id (only connectivity, since formula id is contained in r_id
            rxn_id = rxn.get_id()
            if put_in:
                rxn.rxn_id = rxn_id
                if bf_string not in bf_dict:
                    bf_dict[bf_string] = [rxn]
                else:
                    bf_dict[bf_string].append(rxn)
            #print (rxn.bond_forms, rxn.bond_breaks)

        num_dict = len(bf_dict)
        #print (bf_dict)
        if num_dict == 0:
            return non_h_rxns + h_rxns
        num_remain = num_sample
        num_equal = int(num_remain/num_dict) + 1 
        sampled_rxn_ids = set([])
        final_bf_dict = dict()

        for bf_string in bf_dict:
            final_bf_dict[bf_string] = []
            sub_rxns = bf_dict[bf_string]
            if len(sub_rxns) < num_equal:
                for rxn in sub_rxns:
                    rxn_id = rxn.rxn_id
                    if rxn_id in sampled_rxn_ids:
                        continue
                    is_screened, screening_log = self.check_screening(rxn)
                    if not is_screened:

                        final_bf_dict[bf_string].append(rxn)
                        sampled_rxn_ids.add(rxn_id)
                        num_remain -= 1
            else:
                # If many, randomly sample ...
                indices = list(range(len(sub_rxns)))
                while len(final_bf_dict[bf_string]) < num_equal and len(indices) > 0:
                    index = random.sample(indices,1)[0]
                    rxn = sub_rxns[index]
                    rxn_id = rxn.rxn_id
                    indices.remove(index)
                    if rxn_id in sampled_rxn_ids:
                        continue
                    is_screened, screening_log = self.check_screening(rxn)
                    #print (is_screened, screening_log,rxn.bond_forms, rxn.bond_breaks)
                    if not is_screened:
                        final_bf_dict[bf_string].append(rxn)
                        sampled_rxn_ids.add(rxn_id)
                        num_remain -= 1
            # Adjust num equal dynamically ...
            num_equal = int(num_remain/num_dict) + 1
        
        solvent_added_rxns = []
        for bf_string in final_bf_dict:
            solvent_added_rxns += final_bf_dict[bf_string]

        return non_h_rxns + h_rxns + solvent_added_rxns


    def random_sample(self,reactant,num_sample = 200):
        # Sample as autoregressive manner 
        pass


    def apply_graph_screening(self,rxn):
        pass


    def apply_ts_screening(self,rxn):
        pass


    def check_screening(self,rxn):
        '''
        Currently, we'll consider the following ...
        1. Additional heuritics (optional, but quite necessary ...)
        2. If product is None
        3. All atoms are neutral
        3. bo_diff < -2
        4. Single atom 
        5. Reactions with spectator molecules removed
        6. TS-like structure generation and/or energy
        '''
        # If bond change is 4, pass ...
        bond_forms = rxn.bond_forms
        bond_breaks = rxn.bond_breaks
        #if len(bond_forms) + len(bond_breaks) > 3:
        #    return True
        # Check ts adj matrix
        ts_adj_matrix = np.copy(rxn.reactant.get_adj_matrix())
        for bond_form in bond_forms:
            s, e = bond_form
            ts_adj_matrix[s][e] += 1
            ts_adj_matrix[e][s] += 1
        molecule_groups = process.group_molecules(ts_adj_matrix)
        if len(molecule_groups) > 1: # Spectating molecule exists 
            return True,'Spectating'
        r_bo_matrix = rxn.reactant.get_bo_matrix()
        products = rxn.get_products()
        if len(products) == 0:
            return True,'No well sanitized'

        product = products[0] # Most stabilized ???
        rxn.product = product

        # If R==P, ignore reaction ...
        if rxn.reactant == product:
            return True, 'R==P'

        p_bo_matrix = product.get_bo_matrix()
        r_bo_sum = np.sum(r_bo_matrix,axis=1)
        p_bo_sum = np.sum(p_bo_matrix,axis=1)
        
        # Check bo sum ...        
        #if np.sum(p_bo_sum) - np.sum(r_bo_sum) <= -self.max_bo_diff:
        #    return True
       
        # Check octet score difference ...
        r_score = reaction.get_octet_score(rxn.reactant)
        p_score = reaction.get_octet_score(product)
        #if p_score - r_score <= -self.max_bo_diff:
        if p_score > r_score + self.max_bo_diff:
            return True, 'octet'

        # Always, compare charge between R and P ...
        chg_list = product.get_chg_list()
        r_score = reaction.get_chg_score(rxn.reactant)
        p_score = reaction.get_chg_score(product)
        
        #overcharged_indices = np.where(np.abs(chg_list) > self.chg_criteria)[0].tolist()
        #print (self.chg_diff,p_score, r_score)
        if p_score > r_score + self.chg_diff:
            #print (overcharged_indices, chg_list)
            return True, 'Charge'
        
        # Remove single atom that are not halogen ...
        z_list = rxn.reactant.get_z_list()
        single_atom_indices = np.where(p_bo_sum == 0)[0].tolist()
        halogens = [9,17,35,53]
        for index in single_atom_indices:
            if z_list[index] not in halogens:
                return True, 'single atom'
        
        # Try creating 3D TS-like structure
        # and check UFF energy with R connectivity
        original_z_list = np.copy(rxn.reactant.get_z_list())
        adj_matrix = np.copy(rxn.reactant.get_matrix("adj"))
        for bond_form in bond_forms:
            start, end = bond_form
            adj_matrix[start][end] += 1
            adj_matrix[end][start] += 1

        # Sample one conformer with valid molecule
        ts_molecule = chem.Molecule([original_z_list, adj_matrix, rxn.reactant.get_chg(), rxn.reactant.get_multiplicity()])
        ts_molecule = ts_molecule.get_valid_molecule()
        ts_bo = ts_molecule.get_bo_matrix()
        ts_chg_list = ts_molecule.get_chg_list()
        
        try:
            ts_molecules = ts_molecule.sample_conformers(1)
        except:
            print ('Conformer generation failed ...')
            print ('generated ts molecule:',ts_molecule.get_smiles('ace'))
            return True, 'No TS Gen'
        
        if len(ts_molecules) == 0:
            return True, 'No TS Gen'        
        #print (ts_molecule.get_smiles('ace'))
        #ts_molecules[0].print_coordinate_list()
        return False, ''


    # Add protic solvent for hydrogen transfer containing reactions ...
    def get_solvent_added_rxns(self,rxn):
        # First, check whether rxn contains hydrogen transfer
        bond_forms = rxn.bond_forms
        reactant = rxn.reactant
        atom_list = reactant.atom_list
        num_atom = len(atom_list)
        considered_bond_forms = [] # [(new_bond_forms, bond)]
        for bond in bond_forms:
            start, end = bond
            z1 = atom_list[start].get_atomic_number()
            z2 = atom_list[end].get_atomic_number()
            if z1 == 1:
                new_bond_forms = bond_forms.copy()
                new_bond_forms.remove(bond)
                considered_bond_forms.append((new_bond_forms, start, end))
            if z2 == 1:
                new_bond_forms = bond_forms.copy()
                new_bond_forms.remove(bond)
                considered_bond_forms.append((new_bond_forms, end, start))

        # Now, make new reactions ...
        new_rxns = []
        for solvent_info in self.solvent_infos:
            solvent, solvent_donor_index, solvent_acceptor_index = solvent_info
            neighbors = solvent.get_neighbors(solvent_donor_index) 
            solvent_broken_index = neighbors[0] # Must be only one ...
            if solvent_acceptor_index is None:
                # Get neighbor of donor index ...
                solvent_acceptor_index = solvent_broken_index
            solvent_atom_list = solvent.atom_list

            # Modify atom list and atom indices for each molecule
            for info in considered_bond_forms:
                new_bond_forms, donor_index, acceptor_index = info
                # Make new bond forms/breaks
                final_acceptor_index = solvent_acceptor_index + num_atom
                final_donor_index = solvent_donor_index + num_atom
                new_bond_forms = new_bond_forms.copy()
                new_bond_forms += [(donor_index, final_acceptor_index),(acceptor_index,final_donor_index)]
                new_bond_breaks = rxn.bond_breaks.copy()
                new_bond_breaks.append((solvent_broken_index + num_atom, final_donor_index))  

                # Make new reactant
                new_reactant = reactant.copy()
                new_reactant.multiplicity = None
                new_reactant.chg = None
                new_reactant.adj_matrix = None

                # Copy molecule list
                new_molecule_list = new_reactant.get_molecule_list().copy()
                new_atom_indices_for_each_molecule = reactant.get_atom_indices_for_each_molecule().copy()
                
                new_molecule_list.append(solvent)
                solvent_indices = list(range(num_atom,num_atom + len(solvent_atom_list)))
                new_atom_indices_for_each_molecule.append(solvent_indices)
                new_reactant.molecule_list = new_molecule_list
                new_reactant.atom_indices_for_each_molecule = new_atom_indices_for_each_molecule

                # Modify atom_list/chg_list/bo_matrix
                new_atom_list = [atom.copy() for atom in atom_list] + [atom.copy() for atom in solvent_atom_list]
                m = len(new_atom_list)
                new_bo_matrix = np.zeros((m,m))
                new_chg_list = np.zeros((m))
                #print (new_atom_indices_for_each_molecule)
                for i,atom_indices in enumerate(new_atom_indices_for_each_molecule):
                    reduce_function = np.ix_(atom_indices, atom_indices)
                    new_bo_matrix[reduce_function] = new_molecule_list[i].get_bo_matrix()
                    new_chg_list[atom_indices] = new_molecule_list[i].get_chg_list()

                new_reactant.atom_list = new_atom_list
                new_reactant.atom_feature = {'chg':new_chg_list}
                new_reactant.bo_matrix = new_bo_matrix
               
                # Make new rxn
                new_rxn = reaction.Rxn(new_reactant, new_bond_forms, new_bond_breaks)
                new_rxns.append(new_rxn)

        return new_rxns


    def random_sample(self,reactant,num_sample = 200):
        # Sample as autoregressive manner 
        pass


    def apply_graph_screening(self,rxn):
        pass


    def apply_ts_screening(self,rxn):
        pass


    def check_screening(self,rxn):
        '''
        Currently, we'll consider the following ...
        1. Additional heuritics (optional, but quite necessary ...)
        2. If product is None
        3. All atoms are neutral
        3. bo_diff < -2
        4. Single atom 
        5. Reactions with spectator molecules removed
        6. TS-like structure generation and/or energy
        '''
        # If bond change is 4, pass ...
        bond_forms = rxn.bond_forms
        bond_breaks = rxn.bond_breaks
        #if len(bond_forms) + len(bond_breaks) > 3:
        #    return True
        # Check ts adj matrix
        ts_adj_matrix = np.copy(rxn.reactant.get_adj_matrix())
        for bond_form in bond_forms:
            s, e = bond_form
            ts_adj_matrix[s][e] += 1
            ts_adj_matrix[e][s] += 1
        molecule_groups = process.group_molecules(ts_adj_matrix)
        if len(molecule_groups) > 1: # Spectating molecule exists 
            return True,'Spectating'
        r_bo_matrix = rxn.reactant.get_bo_matrix()
        products = rxn.get_products()
        if len(products) == 0:
            return True,'No well sanitized'

        product = products[0] # Most stabilized ???
        rxn.product = product

        # If R==P, ignore reaction ...
        #if rxn.reactant == product:
        #    return True, 'R==P'

        p_bo_matrix = product.get_bo_matrix()
        r_bo_sum = np.sum(r_bo_matrix,axis=1)
        p_bo_sum = np.sum(p_bo_matrix,axis=1)
        
        # Check bo sum ...        
        #if np.sum(p_bo_sum) - np.sum(r_bo_sum) <= -self.max_bo_diff:
        #    return True
       
        # Check octet score difference ...
        r_score = reaction.get_octet_score(rxn.reactant)
        p_score = reaction.get_octet_score(product)
        #if p_score - r_score <= -self.max_bo_diff:
        if p_score > r_score + self.max_bo_diff:
            return True, 'octet'

        # Always, compare charge between R and P ...
        chg_list = product.get_chg_list()
        r_score = reaction.get_chg_score(rxn.reactant)
        p_score = reaction.get_chg_score(product)
        
        #overcharged_indices = np.where(np.abs(chg_list) > self.chg_criteria)[0].tolist()
        #print (self.chg_diff,p_score, r_score)
        if p_score > r_score + self.chg_diff:
            #print (overcharged_indices, chg_list)
            return True, 'Charge'
        
        # Remove single atom that are not halogen ...
        z_list = rxn.reactant.get_z_list()
        single_atom_indices = np.where(p_bo_sum == 0)[0].tolist()
        halogens = [9,17,35,53]
        for index in single_atom_indices:
            if z_list[index] not in halogens:
                return True, 'single atom'
        
        # Try creating 3D TS-like structure
        # and check UFF energy with R connectivity
        original_z_list = np.copy(rxn.reactant.get_z_list())
        adj_matrix = np.copy(rxn.reactant.get_matrix("adj"))
        for bond_form in bond_forms:
            start, end = bond_form
            adj_matrix[start][end] += 1
            adj_matrix[end][start] += 1

        # Sample one conformer with valid molecule
        ts_molecule = chem.Molecule([original_z_list, adj_matrix, rxn.reactant.get_chg(), rxn.reactant.get_multiplicity()])
        ts_molecule = ts_molecule.get_valid_molecule()
        ts_bo = ts_molecule.get_bo_matrix()
        ts_chg_list = ts_molecule.get_chg_list()
        
        try:
            ts_molecules = ts_molecule.sample_conformers(1)
        except:
            print ('Conformer generation failed ...')
            print ('generated ts molecule:',ts_molecule.get_smiles('ace'))
            return True, 'No TS Gen'
        
        if len(ts_molecules) == 0:
            return True, 'No TS Gen'        
        #print (ts_molecule.get_smiles('ace'))
        #ts_molecules[0].print_coordinate_list()
        return False, ''


    # Add protic solvent for hydrogen transfer containing reactions ...
    def get_solvent_added_rxns(self,rxn):
        # First, check whether rxn contains hydrogen transfer
        bond_forms = rxn.bond_forms
        reactant = rxn.reactant
        atom_list = reactant.atom_list
        num_atom = len(atom_list)
        considered_bond_forms = [] # [(new_bond_forms, bond)]
        for bond in bond_forms:
            start, end = bond
            z1 = atom_list[start].get_atomic_number()
            z2 = atom_list[end].get_atomic_number()
            if z1 == 1:
                new_bond_forms = bond_forms.copy()
                new_bond_forms.remove(bond)
                considered_bond_forms.append((new_bond_forms, start, end))
            if z2 == 1:
                new_bond_forms = bond_forms.copy()
                new_bond_forms.remove(bond)
                considered_bond_forms.append((new_bond_forms, end, start))

        # Now, make new reactions ...
        new_rxns = []
        for solvent_info in self.solvent_infos:
            solvent, solvent_donor_index, solvent_acceptor_index = solvent_info
            neighbors = solvent.get_neighbors(solvent_donor_index) 
            solvent_broken_index = neighbors[0] # Must be only one ...
            if solvent_acceptor_index is None:
                # Get neighbor of donor index ...
                solvent_acceptor_index = solvent_broken_index
            solvent_atom_list = solvent.atom_list

            # Modify atom list and atom indices for each molecule
            for info in considered_bond_forms:
                new_bond_forms, donor_index, acceptor_index = info
                # Make new bond forms/breaks
                final_acceptor_index = solvent_acceptor_index + num_atom
                final_donor_index = solvent_donor_index + num_atom
                new_bond_forms = new_bond_forms.copy()
                new_bond_forms += [(donor_index, final_acceptor_index),(acceptor_index,final_donor_index)]
                new_bond_breaks = rxn.bond_breaks.copy()
                new_bond_breaks.append((solvent_broken_index + num_atom, final_donor_index))  

                # Make new reactant
                new_reactant = reactant.copy()
                new_reactant.multiplicity = None
                new_reactant.chg = None
                new_reactant.adj_matrix = None

                # Copy molecule list
                new_molecule_list = new_reactant.get_molecule_list().copy()
                new_atom_indices_for_each_molecule = reactant.get_atom_indices_for_each_molecule().copy()
                
                new_molecule_list.append(solvent)
                solvent_indices = list(range(num_atom,num_atom + len(solvent_atom_list)))
                new_atom_indices_for_each_molecule.append(solvent_indices)
                new_reactant.molecule_list = new_molecule_list
                new_reactant.atom_indices_for_each_molecule = new_atom_indices_for_each_molecule

                # Modify atom_list/chg_list/bo_matrix
                new_atom_list = [atom.copy() for atom in atom_list] + [atom.copy() for atom in solvent_atom_list]
                m = len(new_atom_list)
                new_bo_matrix = np.zeros((m,m))
                new_chg_list = np.zeros((m))
                #print (new_atom_indices_for_each_molecule)
                for i,atom_indices in enumerate(new_atom_indices_for_each_molecule):
                    reduce_function = np.ix_(atom_indices, atom_indices)
                    new_bo_matrix[reduce_function] = new_molecule_list[i].get_bo_matrix()
                    new_chg_list[atom_indices] = new_molecule_list[i].get_chg_list()

                new_reactant.atom_list = new_atom_list
                new_reactant.atom_feature = {'chg':new_chg_list}
                new_reactant.bo_matrix = new_bo_matrix
               
                # Make new rxn
                new_rxn = reaction.Rxn(new_reactant, new_bond_forms, new_bond_breaks)
                new_rxns.append(new_rxn)

        return new_rxns




if __name__ == '__main__':
    import sys
    smiles = sys.argv[1]

    try:
        num_rxn = int(sys.argv[2])
    except:
        num_rxn = 10

    # Test for solvent addition reactions
    intermediate = chem.Intermediate(smiles)
    #rxn = reaction.Rxn(intermediate,[(1,7)],[(2,7)])
    sampler = Sampler()
    sampler.chg_criteria = 0
    #new_rxns = sampler.get_solvent_added_rxns(rxn)

    rxns = sampler.sample_rxns(intermediate,num_rxn)

    for rxn in rxns:
        reactant = rxn.reactant
        product = rxn.get_products()[0]   
        #r_smiles = reactant.get_smiles('ace')
        #p_smiles = product.get_smiles('ace')  
        #print ('bond info:',rxn.bond_forms, rxn.bond_breaks)
        #print (reactant.get_adj_matrix(), reactant.get_z_list())
        #print (product.get_adj_matrix(), product.get_z_list())

        r_smiles = reactant.get_smiles_with_rdkit(atom_mapping = True)
        p_smiles = product.get_smiles_with_rdkit(atom_mapping = True)

        print (f'{r_smiles}>>{p_smiles}')

        #product = new_rxn.get_products()[0]
        #print (reactant.get_smiles('ace'),product.get_smiles('ace'))
        #print (rxn.get_ts_molecule().get_smiles('ace')) 




