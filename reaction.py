# -*- coding: utf-8 -*-
'''
--enumeration.py--
Generate new 2D graph intermeidates for given reactant
Note that if method starts with apply, it directly modifies the input
If not, it will return something that you want (Ex. Propagate_one_IM)

It contains routines that
 - Read a xyz file and save the geometry
 - Find bonds in molecules, make adjacency matrices and bond order matrices
 - Distinguish molecules by defining Coulomb matrices
and so on..
'''

import numpy as np
import itertools
import multiprocessing
import time

import chem
import process

from utils import frag

def sanitize(intermediate,method = 'SumofFragments'):
    atom_indices_for_each_molecule = intermediate.get_atom_indices_for_each_molecule()
    molecule_list = intermediate.get_molecule_list()
    n = len(molecule_list)
    m = len(intermediate.atom_list)
    bo_matrix = np.zeros((m,m))
    #print (atom_indices_for_each_molecule)
    well_sanitized = True
    smiles_list = []
    for i in range(n):
        molecule = molecule_list[i]
        atom_indices = atom_indices_for_each_molecule[i]
        # Later, we'll add new condition ...
        chg_list = molecule.get_chg_list()
        molecule_chg = np.sum(chg_list)
        z_list = molecule.get_z_list()
        #new_bo_matrix = process.get_bo_matrix_from_adj_matrix(molecule,molecule_chg)
        #new_chg_list = process.get_chg_list_from_bo_matrix(molecule,molecule_chg,new_bo_matrix)            
        molecule.bo_matrx = None
        molecule.atom_feature['chg'] = None
        new_chg_list, new_bo_matrix = process.get_chg_and_bo(molecule,molecule_chg,method=method)
        if new_chg_list is None:
            return False, intermediate
        if np.sum(new_chg_list) != np.sum(chg_list):
            well_sanitized = False
        molecule.atom_feature['chg'] = new_chg_list
        molecule.bo_matrix = new_bo_matrix
        #print ('chgchg',chg_list)
        intermediate.atom_feature['chg'][atom_indices] = new_chg_list
        reduce_function = np.ix_(atom_indices,atom_indices)
        bo_matrix[reduce_function] = new_bo_matrix
        #smiles_list.append(molecule.get_smiles())
    intermediate.bo_matrix = bo_matrix
    return well_sanitized, intermediate


def get_octet_score(intermediate):
    period_list, group_list = intermediate.get_period_group_list()
    n = len(intermediate.atom_list)
    bo_matrix = intermediate.get_bo_matrix()
    b = np.sum(bo_matrix,axis=1)
    g = group_list
    c = intermediate.get_chg_list()
    score = 0 
    for i in range(n):
        o = 8 # Valence elecctron ...
        p = period_list[i]
        if p == 1:
            o = 2
        s = o-(g[i]+b[i]-c[i])
        if s < 0:
            if p < 3: # Octet violation ...
                return None
            else:
                s = 0 # Octet expansion rule ...
        # Make exception on carbene species ...
        if s > 0:
            # Consider neighbor, whether it can make resonance ...
            neighbors = intermediate.get_neighbors(i)
            for neighbor in neighbors:
                # Only resonance for single bond ... (TODO: Later, this needs to be removed ...)
                if bo_matrix[i][neighbor] == 1: 
                    # If it has lone pair electrons ...
                    l = g[neighbor] - b[neighbor] - c[neighbor]
                    if l > 0: # Lone pair donation for octet deficient atom
                        s = 0
                        break
        score += s ** 2
    return score


def get_chg_score(intermediate):
    chg_list = intermediate.get_chg_list()
    if chg_list is None:
        return None
    else:
        return np.sum(chg_list ** 2)


def get_bo_score(intermediate):
    bo_matrix = intermediate.get_bo_matrix()
    if bo_matrix is None:
        return None
    else:
        return np.sum(bo_matrix)


class Rxn:

    def __init__(self,reactant = None,bond_forms = [],bond_breaks = []):
        self.reactant = reactant # chem.Intermediate
        self.bond_forms = bond_forms # [(s,e),...]
        self.bond_breaks = bond_breaks # [(s,e),...]
        self.product = None
        self.ts = None
        self.Ea = None # Kinetics
        self.dE = None # Thermodynamics
        self.rxn_id = None
        self.reaction_type = 'hetero'
        self.smiles = None

    def get_reactant(self):
        return self.reactant


    def get_product(self): # Just return product with new connectivity ...
        bond_forms = self.bond_forms
        bond_breaks = self.bond_breaks

        if self.product is not None:
            return self.product
        
        elif bond_forms is not None and bond_breaks is not None:
            reactant = self.reactant
            product = reactant.copy()
            product.bo_matrix = None
            p_adj_matrix = np.copy(reactant.get_adj_matrix()) # From R
            for bond_form in bond_forms:
                start,end = bond_form
                p_adj_matrix[start][end] += 1
                p_adj_matrix[end][start] += 1
            for bond_break in bond_breaks:
                start,end = bond_break
                p_adj_matrix[start][end] -= 1
                p_adj_matrix[end][start] -= 1
            product.adj_matrix = p_adj_matrix
            return product
        
        else:
            print ('There are no reaction information!')
            return None
    

    def get_products(self,method='SumofFragments'):
        '''
        Get product that fulfills the following objectives: 
        1. octet rule maximizing product ...
        2. If tie, minimizing the charge separation ...
        3. If tie, maximizing the bond orders
        4. If still many, return all of them ... (May be use quantum chemical calculation to screen them ...)
        '''
        bond_forms = self.bond_forms
        bond_breaks = self.bond_breaks
        if self.product is not None:
            return [self.product]
        elif self.bond_forms is not None and self.bond_breaks is not None:
            reactant = self.reactant
            chg_list = np.copy(reactant.get_chg_list())
            # Check whether formal charges are well given ...
            if chg_list is None:
                print ('Formal charge should be provided !!!')
                return None
            p_adj_matrix = np.copy(reactant.get_adj_matrix())
            for bond_form in bond_forms:
                start,end = bond_form
                p_adj_matrix[start][end] += 1
                p_adj_matrix[end][start] += 1
            for bond_break in bond_breaks:
                start,end = bond_break
                p_adj_matrix[start][end] -= 1
                p_adj_matrix[end][start] -= 1
            molecule_groups = process.group_molecules(p_adj_matrix)
            #print (molecule_groups, bond_forms, bond_breaks)
            # Enumerate charge flow ...
            if self.reaction_type == 'homo':
                chg_flows = [[0]*len(bond_breaks)]
            else:
                chg_lists = [[-1,1]] * len(bond_breaks)
                chg_flows = list(itertools.product(*chg_lists))
            # Make product including sanitization ...            
            candidate_products = []
            chg_distributions = []
            for chg_flow in chg_flows:
                product = reactant.copy()
                product.bo_matrix = None
                product.adj_matrix = np.copy(p_adj_matrix)
                chg_list = product.get_chg_list()
                for i,bond_break in enumerate(bond_breaks):
                    start, end = bond_break
                    chg_list[start] += chg_flow[i]
                    chg_list[end] -= chg_flow[i]
                molecule_chgs = tuple([np.sum(chg_list[atom_indices])] for atom_indices in molecule_groups)
                if molecule_chgs not in chg_distributions:
                    candidate_products.append(product)
                    chg_distributions.append(molecule_chgs)

            # Get distinct number of products ...
            products = []
            for product in candidate_products:
                well_sanitized,product = sanitize(product,method)
                #print (product.get_smiles('ace'))
                if well_sanitized:
                    #print (bond_forms, bond_breaks, molecule_groups, chg_flow, chg_list)
                    products.append(product)
 
            # Rank products by the objective function (1. octet, 2. chg, 3. bond order)
            if len(products) == 0:
                return []
            # Check octet rules
            scores = []
            for product in products:
                score = get_octet_score(product)
                if score is None:
                    score = 10000
                scores.append(score)
            min_value = min(scores)
            if min_value > 1000:
                return []
            minimal_indices = [index for index, value in enumerate(scores) if value == min_value]
            products = [products[index] for index in minimal_indices]
            if len(products) == 1:
                return products
            
            # Check charge separation ...
            scores = []
            for product in products:
                score = get_chg_score(product)
                if score is None:
                    score = 10000
                scores.append(score)
            min_value = min(scores)
            minimal_indices = [index for index, value in enumerate(scores) if value == min_value]
            products = [products[index] for index in minimal_indices]
            if len(products) == 1:
                return products

            # Check bond order maximization ...
            scores = []
            for product in products:
                score = get_bo_score(product)
                if score is None:
                    score = 10000
                scores.append(score)
            min_value = min(scores)
            minimal_indices = [index for index, value in enumerate(scores) if value == min_value]
            products = [products[index] for index in minimal_indices]
            return products

        else:
            print ('There are no reaction information!')
            return None


    def get_reaction_info(self):
        reaction_info = {'f':[],'b':[]}
        for start,end,c in self.bond_forms:
            reaction_info['f'].append((start,end))
        for start,end,c in self.bond_breaks:
            reaction_info['b'].append((start,end))
        return reaction_info
 
  
    def get_reduced_rxn(self):
        # Find reacting atoms ..
        ts_molecule = self.get_ts_molecule()
        ts_atom_indices = ts_molecule.get_atom_indices_for_each_molecule()
        reduce_function = dict()
        # Participating indices
        participating_indices = set([])
        
        for bond_form in self.bond_forms:
            start,end,charge = bond_form
            for atom_indices in ts_atom_indices:
                if start in atom_indices:
                    participating_indices = participating_indices | set(atom_indices)
                if end in atom_indices:
                    participating_indices = participating_indices | set(atom_indices)

        for bond_break in self.bond_breaks:
            start,end,charge = bond_break
            for atom_indices in ts_atom_indices:
                if start in atom_indices:
                    participating_indices = participating_indices | set(atom_indices)
                if end in atom_indices:
                    participating_indices = participating_indices | set(atom_indices)

        participating_indices = list(participating_indices)
        participating_indices.sort()
        for i, index in enumerate(participating_indices):
            reduce_function[index] = i

        new_bond_forms = []
        new_bond_breaks = []
        for bond_form in self.bond_forms:
            start,end,charge = bond_form
            new_bond_forms.append((reduce_function[start],reduce_function[end],charge))
        for bond_break in self.bond_breaks:
            start,end,charge = bond_break
            new_bond_breaks.append((reduce_function[start],reduce_function[end],charge))
        reduced_reactant = process.get_reduced_intermediate(self.reactant,reduce_function)
        return Rxn(reduced_reactant,new_bond_forms,new_bond_breaks)

        #Rxn()

    def get_id(self):
        reactant = self.get_reactant()
        chg = reactant.get_chg()
        product = self.get_product()        
        ts = self.get_ts_molecule()
        formula_id = reactant.get_formula_id()
        r_id = reactant.get_molecule_id()
        ts_id = ts.get_molecule_id()
        p_id = product.get_molecule_id()
        return f'{formula_id}_{r_id}_{ts_id}_{p_id}'


    def get_bf_string(self):
        atom_list = self.get_reactant().atom_list
        b_content = []
        f_content = []
        for bond_form in self.bond_forms:
            s, e = bond_form
            s = atom_list[s].get_element()
            e = atom_list[e].get_element()
            if s > e:
                b_string = f'{s}_{e}'
            else:
                b_string = f'{e}_{s}'
            b_content.append(b_string)

        for bond_break in self.bond_breaks:
            s, e = bond_break
            s = atom_list[s].get_element()
            e = atom_list[e].get_element()
            if s > e:
                f_string = f'{s}_{e}'
            else:
                f_string = f'{e}_{s}'
            f_content.append(f_string)
        
        b_content.sort()
        f_content.sort()
        b_string = '.'.join(b_content)
        f_string = '.'.join(f_content)
        bf_string = f"b{b_string}f{f_string}"
        return bf_string


    # Truncated reaction ...
    def get_model_reaction(self,model_type='adj',k=1): # k=1: radius
        reactant = self.reactant
        adj_matrix = reactant.get_adj_matrix()
 
        # Get reactive atoms (reaction participating atoms)
        reactive_atoms = set([])
        
        # If adj ...
        if model_type == 'adj':
            bond_forms = self.bond_forms
            bond_breaks = self.bond_breaks

        # If bo ...
        elif model_type == 'bo':
            products = self.get_products()
            if len(products) == 0:
                return None
            product = products[0]
            r_bo_matrix = reactant.get_bo_matrix()
            p_bo_matrix = product.get_bo_matrix() 
            bond_forms = np.stack(np.where(r_bo_matrix < p_bo_matrix),axis=1)
            bond_breaks = np.stack(np.where(r_bo_matrix > p_bo_matrix),axis=1)

        for bond_form in bond_forms:
            s, e = bond_form
            reactive_atoms.add(s)
            reactive_atoms.add(e)
        for bond_break in bond_breaks:
            s, e = bond_break
            reactive_atoms.add(s)
            reactive_atoms.add(e)      

        considering_atoms = reactive_atoms.copy()
        total_considering_atoms = reactive_atoms.copy()
    
        for i in range(k):
            new_considering_atoms = set([])
            for j in considering_atoms:
                neighbor_list = np.where(adj_matrix[j]>0)[0].tolist()
                for neighbor in neighbor_list:
                    new_considering_atoms.add(neighbor)
                new_considering_atoms = new_considering_atoms - considering_atoms
            considering_atoms = new_considering_atoms
            total_considering_atoms = total_considering_atoms | new_considering_atoms
            if len(considering_atoms) == 0:
                break
        
        # Last indices are different (Change into hydrogen atoms ...)
        last_indices = set([])
        #'''
        for i in total_considering_atoms:
            neighbor_list = np.where(adj_matrix[i]>0)[0].tolist()
            for neighbor in neighbor_list:
                last_indices.add(neighbor)
        last_indices = last_indices - total_considering_atoms
        #'''

        atom_indices = list(total_considering_atoms | last_indices)
        atom_indices.sort()

        # Prepare new atom list, chg, multiplicity
        new_atom_list = []
        for i in atom_indices:
            atom = reactant.atom_list[i]
            if i in total_considering_atoms:
                new_atom_list.append(atom.copy())
            else: # In last indices ...
                z = atom.get_atomic_number()
                if z == 1 or z == 6:
                    new_atom_list.append(chem.Atom('H'))
                else:
                    new_atom_list.append(atom.copy())

        new_reactant = chem.Intermediate()
        new_reactant.atom_list = new_atom_list
        new_reactant.chg = reactant.get_chg()
        new_reactant.multiplicity = reactant.get_multiplicity()

        # Prepare new bond forms new breaks
        reduce_function = {atom_indices[i]:i for i in range(len(atom_indices))}
        new_bond_forms = []
        new_bond_breaks = []
        for bond_form in bond_forms:
            s, e = bond_form
            s_new = reduce_function[s]
            e_new = reduce_function[e]
            if s_new > e_new:
                bond = (e_new, s_new)
            else:
                bond = (s_new,e_new)
            new_bond_forms.append(bond)
        
        for bond_break in bond_breaks:
            s, e = bond_break
            s_new = reduce_function[s]
            e_new = reduce_function[e]
            if s_new > e_new:
                bond = (e_new, s_new)
            else:
                bond = (s_new,e_new)
            new_bond_breaks.append(bond)

        # Prepare new adj matrix
        reduce_function = np.ix_(atom_indices,atom_indices)
        new_adj_matrix = adj_matrix[reduce_function]
        new_reactant.adj_matrix = new_adj_matrix        

        model_rxn = Rxn(new_reactant, new_bond_forms, new_bond_breaks)
        return model_rxn


    def solve_equation(self):
        pass

    def setEa(self,Ea):
        self.Ea = Ea

    def setdE(self,dE):
        self.dE = dE

    def set_name(self,name):
        self.name = name


    def get_ts_molecule(self):
        ts = self.reactant.copy()
        adj_matrix = np.copy(self.reactant.get_adj_matrix())
        for bond_flow in self.bond_forms:
            start,end = bond_flow
            adj_matrix[start][end] += 1
            adj_matrix[end][start] += 1
        ts.adj_matrix = adj_matrix
        ts.bo_matrix = None
        return ts

    # TODO: Need to convert rdkit with only explicit H aded ...
    def get_smiles(self,method='ace'):
        return ''        


    def __eq__(self,rxn):
        # Must compare three, R, TS, P ...
        reactant = self.get_reactant()
        reactant_prime = rxn.get_reactant()
        if not (reactant == reactant_prime):
            return False

        ts = self.get_ts_molecule()
        ts_prime = rxn.get_ts_molecule()

        if not (ts == ts_prime):
            return False        

        product = self.get_product()
        product_prime = rxn.get_product()
        if not (product == product_prime):
            return False

        return True
    


class RxnEnumerator:
    
    def __init__(self,max_form,max_break,reaction_type='hetero',max_valency_info = dict()):
        self.enumerator = AdjEnumerator(max_form,max_break)
        self.max_valency_info = max_valency_info
        self.reaction_type = reaction_type

    def set_max_form(self,n_form):
        self.enumerator.max_form = n_form

    def set_max_break(self,n_break):
        self.enumerator.max_break = n_break

    def enumerate_rxns(self,reactant,reactive_atoms=None,remove_repetition=False): # f,b
        if type(reactant) is list:
            intermediate = chem.Intermediate(reactant)
        else:
            intermediate = reactant
        atom_list = intermediate.atom_list
        n = len(atom_list)
        max_valency_list = np.zeros((n))
        for i in range(n):
            z = atom_list[i].get_atomic_number()
            if z not in self.max_valency_info:
                max_valency_list[i] = atom_list[i].get_max_valency()
            else:
                max_valency_list[i] = self.max_valency_info[z]
       
        #print (max_valency_list,reactant.get_z_list())
        bond_changes = self.enumerator.get_all_possible_bond_changes(intermediate,reactive_atoms,None,max_valency_list)
        rxns = []
               
        # Remove reactions that have share same TS connectivity ...
        molecule_list = intermediate.get_molecule_list()
        atom_indices_for_each_molecule = intermediate.get_atom_indices_for_each_molecule()
        total_atom_id_list = [None] * n
        for i, molecule in enumerate(molecule_list):
            atom_id_list = molecule.get_atom_id_list()
            for j,index in enumerate(atom_indices_for_each_molecule[i]):
                total_atom_id_list[index] = atom_id_list[j]
        
        rxn_dict = dict()
        index = 0
        cnt = 1

        for bond_change in bond_changes:
            #print (bond_change,len(bond_changes))
            formed_bonds = bond_change[0]
            broken_bonds = bond_change[1]
            reactant_copy = intermediate.copy()
            rxn = Rxn(reactant_copy,formed_bonds,broken_bonds)
            rxn.reaction_type = self.reaction_type            
            if remove_repetition:
                formed_score_tuples = []
                broken_score_tuples = []
                for s, e in formed_bonds:
                    form_score = [total_atom_id_list[s], total_atom_id_list[e]]
                    form_score.sort()
                    formed_score_tuples.append(tuple(form_score))
                for s, e in broken_bonds:
                    broken_score = [total_atom_id_list[s], total_atom_id_list[e]]
                    broken_score.sort()
                    broken_score_tuples.append(tuple(broken_score))
                
                formed_score_tuples = tuple(sorted(formed_score_tuples))
                broken_score_tuples = tuple(sorted(broken_score_tuples))
                check_tuple = (formed_score_tuples,broken_score_tuples)
                if check_tuple in rxn_dict:
                    cnt += 1
                    # Check ts molecule ...
                    sub_rxns = rxn_dict[check_tuple]
                    put_in = True
                    for sub_rxn in sub_rxns:
                        if rxn == sub_rxn:
                            # print (product1.get_smiles('ace'))
                            # print (product2.get_smiles('ace'))
                            put_in = False
                            break
                    if put_in:
                        rxn_dict[check_tuple].append(rxn)
                else:
                    rxn_dict[check_tuple] = [rxn]
            else:
                rxns.append(rxn)
               
        for check_tuple in rxn_dict:
            #if len(rxn_dict[check_tuple]) > 1:
            #    print (check_tuple)
            #    for rxn in rxn_dict[check_tuple]:
            #        print (rxn.bond_forms, rxn.bond_breaks)
            rxns += rxn_dict[check_tuple]
        return rxns


class AdjEnumerator:
    
    def __init__(self,max_form,max_break,max_order = 100):
        self.max_form = max_form
        self.max_break = max_break
        self.screener = None        
        self.max_order = max_order # If max_order = 2, only allow uni/bimolecular reactions ...

    def get_all_possible_bond_changes(self,intermediate,active_atom_indices = None,active_bonds = None,max_valency_list = None):
        bo_matrix = intermediate.get_matrix('bo')
        if bo_matrix is None:
            print ('bo cannot be found! Cannot enumerate reactions!!! ')
            return []
        adj_matrix = np.where(bo_matrix>0,1,0)
        injection_function = dict()
        if active_atom_indices == None:
            active_atom_indices = list(range(len(intermediate.atom_list)))
        active_atom_indices.sort()
        for i in range(len(active_atom_indices)):
            injection_function[i] = active_atom_indices[i]
        reduce_idx = np.ix_(active_atom_indices,active_atom_indices)
        reduced_bo_matrix = bo_matrix[reduce_idx]
        broken_set = [()]
        formed_set = [()]
        max_break = self.max_break
        max_form = self.max_form
        # Only break non-conjugated bonds  
        possible_bond_break_set = np.stack(np.where(reduced_bo_matrix == 1),axis=1).tolist()
        possible_bond_form_set = np.stack(np.where(reduced_bo_matrix + np.diag([1]*len(reduced_bo_matrix))==0),axis=1).tolist()
        possible_reactions = []
        original_valency_list = np.sum(adj_matrix,axis=1)
        if max_valency_list is None: 
            max_valency_list = intermediate.get_max_valency_list()
        #print (max_valency_list,intermediate.get_z_list())
        remain_valency_list = max_valency_list - original_valency_list
        self.remain_valency_list = remain_valency_list[active_atom_indices]
        #print (remain_valency_list)
        #s = time.time()
        index = 0
        while index<len(possible_bond_break_set):
            bond = possible_bond_break_set[index]
            reduced_start = possible_bond_break_set[index][0]
            reduced_end = possible_bond_break_set[index][1]
            start = injection_function[reduced_start]
            end = injection_function[reduced_end]
            if start > end:
                del(possible_bond_break_set[index])
            else:
                #possible_bond_break_set[index][0] = start
                #possible_bond_break_set[index][1] = end
                possible_bond_break_set[index] = tuple(possible_bond_break_set[index])
                if active_bonds is not None and (start,end) in active_bonds:
                    del(possible_bond_break_set[index])
                else:
                    index += 1
        index = 0
        while index<len(possible_bond_form_set):
            bond = possible_bond_form_set[index]
            reduced_start = possible_bond_form_set[index][0]
            reduced_end = possible_bond_form_set[index][1]
            start = injection_function[reduced_start]
            end = injection_function[reduced_end]
            if start>end:
                del(possible_bond_form_set[index])
            else:
                #possible_bond_form_set[index][0] = start
                #possible_bond_form_set[index][1] = end
                possible_bond_form_set[index] = tuple(possible_bond_form_set[index])
                if active_bonds is not None and (start,end) in active_bonds:
                    del(possible_bond_form_set[index])
                else:
                    index += 1
        #print ('z',reactant.get_z_list())
        #print ('f',possible_bond_form_set)
        #print ('b',possible_bond_break_set)
        #print ('v',max_valency_list)
        #print (injection_function)

        # Identify neighbor for reducing complexity for over octet cases
        possible_bond_break_info = dict()
        possible_bond_form_info = dict()
        inverse_bond_break = dict()
        inverse_bond_form = dict()
        for i in range(len(active_atom_indices)):
            possible_bond_break_info[i] = []
            possible_bond_form_info[i] = []
        for i in range(len(possible_bond_break_set)):
            bond_break = possible_bond_break_set[i]
            start = bond_break[0]
            end = bond_break[1]
            possible_bond_break_info[start].append(i)
            possible_bond_break_info[end].append(i)
            inverse_bond_break[bond_break] = i
        for i in range(len(possible_bond_form_set)):
            bond_form = possible_bond_form_set[i]
            start = bond_form[0]
            end = bond_form[1]
            possible_bond_form_info[start].append(i)
            possible_bond_form_info[end].append(i)
            inverse_bond_form[bond_form] = i
        self.possible_bond_form_set = possible_bond_form_set
        self.possible_bond_break_set = possible_bond_break_set
        self.possible_bond_break_info = possible_bond_break_info
        self.possible_bond_form_info = possible_bond_form_info
        self.inverse_bond_break = inverse_bond_break
        self.inverse_bond_form = inverse_bond_form
        self.injection_function = injection_function
        #print (possible_bond_form_info)
        #print (possible_bond_form_set)
        #print ('possible form: ',possible_bond_form_set)
        #print ('possible break: ',possible_bond_break_set)
        #print ('inverse form',inverse_bond_form)
        #print ('inverse break',inverse_bond_break)
        cnt = 0 # Count generated intermediates
        current_reaction_list = [[[],[]]]
        total_possible_reactions = self.enumerate_additional_bond_break(current_reaction_list[0])
        # Enumeration w.r.t bond formation starts here ...
        for i in range(self.max_form):
            additional_reaction_list = []
            next_reaction_list = []
            for reaction in current_reaction_list:
                next_reaction_list += self.enumerate_bond_form(reaction)
            feasible_reaction_list = []
            cnt += len(next_reaction_list)
            #print ('cnt after form',cnt)
            #print (next_reaction_list)
            for reaction in next_reaction_list:
                basic_feasible_list = self.get_over_valence_resolved_reaction(reaction)
                for new_reaction in basic_feasible_list:
                    is_valid = self.is_valid_reaction(new_reaction)
                    if is_valid:
                        feasible_reaction_list.append(new_reaction)
                #feasible_reaction_list += self.get_over_valence_resolved_reaction(reaction)                
            for reaction in feasible_reaction_list:
                additional_reaction_list += self.enumerate_additional_bond_break(reaction)
            total_possible_reactions += (feasible_reaction_list+additional_reaction_list)
            current_reaction_list = feasible_reaction_list
        
        # Recover into reaction with injection function
        final_possible_reactions = []
        for reaction in total_possible_reactions:
            formed_indices = reaction[0]
            broken_indices = reaction[1]
            original_reaction = [[],[]]
            for formed_index in formed_indices:
                formed_bond = possible_bond_form_set[formed_index]
                start = injection_function[formed_bond[0]]
                end = injection_function[formed_bond[1]]
                if start<end:
                    original_reaction[0].append((start,end))
                else:
                    original_reaction[0].append((end,start))
            for broken_index in broken_indices:
                broken_bond = possible_bond_break_set[broken_index]
                start = injection_function[broken_bond[0]]
                end = injection_function[broken_bond[1]]
                if start<end:
                    original_reaction[1].append((start,end))
                else:
                    original_reaction[1].append((end,start))
            final_possible_reactions.append(original_reaction)

        return final_possible_reactions
    

    def enumerate_additional_bond_break(self,reaction): # Add break for minimal break intermediates
        formed_indices = reaction[0]
        broken_indices = reaction[1]
        num_broken = len(broken_indices)
        remain = self.max_break - num_broken
        current_broken_list = [[]]
        total_enumerated_reaction_list = []
        for k in range(remain):
            new_broken_list = []
            for current_broken in current_broken_list:
                new_broken_list += self.enumerate_single_additional_bond_break(reaction,current_broken)
            current_broken_list = new_broken_list
            for broken_indices in new_broken_list:
                new_reaction = [reaction[0].copy(),reaction[1].copy()+broken_indices]
                new_reaction[1].sort()
                total_enumerated_reaction_list.append(new_reaction)
        return total_enumerated_reaction_list


    def enumerate_single_additional_bond_break(self,reaction,new_broken): # Add single break for minimal break intermdiates
        broken_indices = reaction[1]
        num_broken = len(broken_indices)
        remain = self.max_break - num_broken
        if remain <= 0:
            return []
        else:
            last_index = -1
            if len(new_broken) > 0:
                last_index = new_broken[-1]
            new_broken_list = []
            n = len(self.possible_bond_break_set)
            possible_indices = list(range(last_index+1,n))
            #if remain == 1:
            #    print ('reaction',reaction)
            for index in possible_indices:
                if index not in broken_indices:
                    new_broken_list.append(new_broken+[index])
            return new_broken_list

    def enumerate_bond_form(self,reaction): # Add (single) form for minimal break intermediates
        new_reaction_list = []
        formed_indices = reaction[0]
        broken_indices = reaction[1]
        possible_bond_break_info = self.possible_bond_break_info
        possible_bond_form_info = self.possible_bond_form_info
        n = len(self.possible_bond_form_set)

        #print ('ff',[self.possible_bond_form_set[idx] for idx in formed_indices])

        if len(formed_indices)>=self.max_form:
            return []
        remain = self.max_break - len(broken_indices)
        if remain < 0:
            print ('something wrong')
            return []
        last_index = -1
        if len(formed_indices) > 0:
            last_index = formed_indices[-1]
        
        un_filled_indices,filled_indices = self.split_indices(reaction)
        #print (un_filled_indices, filled_indices, remain)
        # Only consider unfilled indices
        if remain == 0:
            # Match uf and uf: Only use uf and uf and check the index
            m = len(un_filled_indices)
            for j in range(m):
                for k in range(1,m-j):
                    bond = (un_filled_indices[j],un_filled_indices[j+k])
                    if bond not in self.inverse_bond_form:
                        continue
                    bond_index = self.inverse_bond_form[bond]
                    if bond_index > last_index:
                        new_reaction = [reaction[0].copy()+[bond_index],reaction[1].copy()]
                        new_reaction_list.append(new_reaction)
        elif remain == 1:
            # Match uf and random: All can go inside!
            #print (possible_bond_form_info)
            for un_filled_index in un_filled_indices:
                possible_indices = possible_bond_form_info[un_filled_index]
                for index in possible_indices:
                    if index > last_index:
                        new_reaction = [reaction[0].copy()+[index],reaction[1].copy()]
                        new_reaction_list.append(new_reaction)
        else:
            # Match random and random: Use all enumeration
            possible_indices = list(range(last_index+1,n))
            for index in possible_indices:
                new_reaction = [reaction[0].copy()+[index],reaction[1].copy()]
                new_reaction_list.append(new_reaction)
        #for new_reaction in new_reaction_list:
        #    print ('new',[self.possible_bond_form_set[idx] for idx in new_reaction[0]]) 
        return new_reaction_list

    def split_indices(self,reaction):
        remain_valency_list = self.remain_valency_list
        possible_bond_break_set = self.possible_bond_break_set
        possible_bond_form_set = self.possible_bond_form_set
        valency_list = np.copy(remain_valency_list)
        formed_indices = reaction[0]
        broken_indices = reaction[1]
        for formed_index in formed_indices:
            formed_bond = possible_bond_form_set[formed_index]
            start = formed_bond[0]
            end = formed_bond[1]
            valency_list[start] -= 1
            valency_list[end] -= 1
        for broken_index in broken_indices:
            broken_bond = possible_bond_break_set[broken_index]
            start = broken_bond[0]
            end = broken_bond[1]
            valency_list[start] += 1
            valency_list[end] += 1
        un_filled_indices = np.where(valency_list>0)[0].tolist()
        filled_indices = np.where(valency_list == 0)[0].tolist()
        un_filled_indices.sort()
        filled_indices.sort()
        return un_filled_indices,filled_indices        

    def get_over_valence_resolved_reaction(self,reaction): # Maximal two, since step by step
        over_valence_indices = self.find_over_valence(reaction)
        possible_bond_break_info = self.possible_bond_break_info
        broken_indices = reaction[1]
        last_index = -1
        #print ('ff',[self.possible_bond_form_set[idx] for idx in reaction[0]])
        #print ('over',over_valence_indices)
        if len(broken_indices) > 0:
            last_index = broken_indices[-1]
        if len(over_valence_indices) > self.max_break - len(broken_indices):
            return []
        if len(over_valence_indices) == 0:
            return [reaction]
        possible_combination = []
        new_reaction_list = []
        for index in over_valence_indices: # Max over valence value is 1, since step-by-step
            possible_bond_break = possible_bond_break_info[index]
            possible_combination.append(possible_bond_break)
        possible_combination = list(itertools.product(*possible_combination))
        for possible_reaction in possible_combination:
            if len(possible_reaction) > 1:
                #print (reaction)
                #print (possible_reaction)
                if possible_reaction[0] not in reaction[1] and possible_reaction[1] not in reaction[1]:
                    new_reaction = [reaction[0].copy(),reaction[1].copy()+[possible_reaction[0],possible_reaction[1]]]
                else:
                    continue
            else: 
                if possible_reaction[0] not in reaction[1]:
                    new_reaction = [reaction[0].copy(),reaction[1].copy()+[possible_reaction[0]]]
                else:
                    continue
            new_reaction[1].sort()
            new_reaction_list.append(new_reaction)
        return new_reaction_list

    def find_over_valence(self,reaction):
        remain_valency_list = self.remain_valency_list
        possible_bond_form_set = self.possible_bond_form_set
        possible_bond_break_set = self.possible_bond_break_set
        formed_bonds = [possible_bond_form_set[index] for index in reaction[0]]
        broken_bonds = [possible_bond_break_set[index] for index in reaction[1]]
        valency_list = np.copy(remain_valency_list)
        for formed_bond in formed_bonds:
            valency_list[formed_bond[0]] -= 1
            valency_list[formed_bond[1]] -= 1
        for broken_bond in broken_bonds:
            valency_list[broken_bond[0]] += 1
            valency_list[broken_bond[1]] += 1
        indices = np.where(valency_list<0)[0].tolist()
        return indices

    def is_valid_reaction(self,rxn):
        if rxn == [(),()]:
            return False
        screener = self.screener
        possible_bond_form_set = self.possible_bond_form_set
        possible_bond_break_set = self.possible_bond_break_set
        formed_bonds = [possible_bond_form_set[index] for index in rxn[0]]
        broken_bonds = [possible_bond_break_set[index] for index in rxn[1]]
        reaction = [formed_bonds,broken_bonds]
        if screener is None:
            return True
        else:
            self.ring += 1
            injection_function = self.injection_function
            reaction = [[],[]]
            for formed_bond in formed_bonds:
                start = injection_function[formed_bond[0]]
                end = injection_function[formed_bond[1]]
                reaction[0].append((start,end))
            for broken_bond in broken_bonds:
                start = injection_function[broken_bond[0]]
                end = injection_function[broken_bond[1]]
                reaction[1].append((start,end))
            return screener.check_ring(reaction)
        return True


    def make_possible_adj(self, reaction):
        """
        helper function for get_possible_reactions()

        tells that the reaction (reactions_list[index]) creates possible adj

        ### possible adj for each element
        H : 1
        period 2 elements| C: 4, N: 4, O: 3
        period 3~ elements| equals to last digit of their group number
        """
        remain_valency_list = self.remain_valency_list
        if len(reaction[0]) == 0 and len(reaction[1]) == 0: 
            return False
        v_list = np.copy(remain_valency_list)
        for formed_bond in reaction[0]:
            v_list[formed_bond[0]] -= 1
            v_list[formed_bond[1]] -= 1
        for broken_bond in reaction[1]:
            v_list[broken_bond[0]] += 1
            v_list[broken_bond[1]] += 1
        return np.all(v_list>=0)


def test_enumeration():
    reactant = chem.Intermediate(sys.argv[1])
    reactant.initialize()
    print (reactant.get_smiles())
    #try:
    #    enumerator_type = sys.argv[2]
    #except:
    #    enumerator_type = 'new'
    try:
        num_form = int(sys.argv[2])
    except:
        num_form = 2
    try:
        num_break = int(sys.argv[3])
    except:
        num_break = 2
    enumerator = RxnEnumerator(num_form,num_break)
    print ('start enumeration:',reactant.get_smiles())
    rxns = enumerator.enumerate_rxns(reactant,remove_repetition = True)
    cnt = 0
    print (len(rxns))
    r_score = get_octet_score(reactant)
    for rxn in rxns:
        products = rxn.get_products()
        if len(products) == 0:
            continue
        p_score = get_octet_score(products[0])
        if p_score > r_score:
            continue
        for product in products:
            chg_list = product.get_chg_list()
            non_chg_indices = np.where(np.abs(chg_list)>0)[0].tolist()
            ts_molecule = rxn.get_ts_molecule()
            n = len(ts_molecule.get_molecule_list())
            if len(non_chg_indices) == 0 and n == 1:
            #if n == 1:
                print (product.get_smiles('ace'),rxn.bond_forms,rxn.bond_breaks)
                cnt += 1
            
    print (cnt)


def test_model_reaction():
    reactant = chem.Intermediate(sys.argv[1])
    reactant.initialize()
    print (reactant.get_smiles())
    try:
        num_form = int(sys.argv[2])
    except:
        num_form = 2
    try:
        num_break = int(sys.argv[3])
    except:
        num_break = 2
    model_type = 'adj'
    enumerator = RxnEnumerator(num_form,num_break)
    print ('new enumeration:',reactant.get_smiles())
    rxns = enumerator.enumerate_rxns(reactant,remove_repetition = True)
    cnt = 0

    rxn_info = dict()
    for rxn in rxns:
        model_rxn = rxn.get_model_reaction(model_type = model_type,k=0)
        model_rxn_id = model_rxn.get_id()
        if model_rxn_id in rxn_info:
            rxn_info[model_rxn_id].append(rxn)
        else:
            rxn_info[model_rxn_id] = [rxn]
    
    for rxn_id in rxn_info:
        print (rxn_id, len(rxn_info[rxn_id]))

    #rxn_id = list(rxn_info.keys())[0]
    for rxn_id in list(rxn_info.keys()):
        for rxn in rxn_info[rxn_id]:
            products = rxn.get_products()
            if len(products) == 0:
                continue
            else:
                product = products[0]
                p_score = get_octet_score(products[0])
                r_score = get_octet_score(rxn.reactant)
                if p_score > r_score:
                    continue
                p_score = get_chg_score(products[0])
                r_score = get_chg_score(rxn.reactant)
                if p_score > r_score:
                    continue
              
                print (rxn_id,rxn.bond_forms, rxn.bond_breaks)
                model_rxn = rxn.get_model_reaction(model_type = model_type, k = 0)
                #print (model_rxn.reactant.get_z_list())
                #print (model_rxn.bond_forms, model_rxn.bond_breaks)
                #print (model_rxn.get_ts_molecule().get_atom_id_list())
                #print (products[0].get_smiles('ace'), model_rxn.get_smiles('ace'))

def test_model_reaction2():
    import sys
    rxn = reaction_from_string(sys.argv[1])
    model_rxn = rxn.get_model_reaction(model_type='bo',k=0)
    print (rxn.bond_forms, rxn.bond_breaks)
    print (model_rxn.reactant.get_z_list())
    print (model_rxn.reactant.get_formula(return_type='str'))


if __name__ == '__main__':
    import sys
    #test_enumeration()
    test_model_reaction2()

