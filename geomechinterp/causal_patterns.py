
import numpy as np
import random
from itertools import product, permutations


def sample_hasse_diagram_binary(reflections: list[str | int]):
    idx = list(range(len(reflections)))
    
    causal_edge_list = []
    # causal sampling
    # example for [0,1,2]
    # (1, [])
    # (0, [1])
    # (2, [1,0])
    fixed = []
    remaining = idx
    for i in range(len(idx)):
        # fix attribute
        att_idx = random.sample(remaining, 1)[0]
        # select control attributes from already fixed
        num_controls = np.random.choice(len(fixed)+1) if len(fixed) else 0
        controls = random.sample(fixed, num_controls)
        
        fixed.append(att_idx)
        remaining = [idx for idx in remaining if idx != att_idx]
        causal_edge_list.append((att_idx, controls))

    return causal_edge_list


def generate_binary_masks(n):
    return list(product([0, 1], repeat=n))


def get_all_possible_hasse_diagrams(n: int) -> np.array:   
    """Note that masks are `abstract` causal patterns, 
    they still need to be "initialized" by a particular permutation of attributes.""" 
    assert n < 10, 'lets not blow up for now'
    # size of the causal part of adjacency matrix
    # 0 + 1 + 2 + ... + (n-1)
    s = (n**2-n)//2

    # get all possible binary masks
    masks = np.array(generate_binary_masks(s))
    return masks


def flattened_to_causal_triangle(flat_vector):
    """Example usage
    flattened_to_causal_triangle([1, 0, 1, 0, 1, 0])
    >>> [(1,), (0, 1), (0, 1, 0)]
    """
    # infer n
    s = len(flat_vector)
    n = (-1+np.sqrt(1+4*2*s))/2 
    assert round(n) == n, 'flat vector must come from causal triangle (3,6,10,15,21,...)'
    causal_triangle = []
    index = 0
    for i in range(int(n)+1):
        causal_triangle.append(tuple([flat_vector[index + j] for j in range(i)]))
        index += i
    return causal_triangle


def get_all_possible_causal_patterns_abstract(n: int):
    causal_masks = get_all_possible_hasse_diagrams(n)
    all_causal_edge_lists = []
    for hasse_flat in causal_masks:
        hasse_triangle = tuple(flattened_to_causal_triangle(hasse_flat))
        all_causal_edge_lists.append(hasse_triangle)
    return all_causal_edge_lists


def get_all_possible_causal_patterns_labeled(reflections: list[str | int]):
    n = len(reflections)
    idx = list(range(n))
    causal_masks = get_all_possible_hasse_diagrams(n)
    all_permutations = list(permutations(idx))
    # final causal patterns are direct product of the two
    all_causal_edge_lists = {}
    total_count = len(causal_masks) * len(all_permutations)
    for hasse_flat in causal_masks:
        hasse_triangle = tuple(flattened_to_causal_triangle(hasse_flat))
        all_causal_edge_lists[hasse_triangle] = {}
        for perm in all_permutations:
            perm_str = tuple([reflections[i] for i in perm])
            perm_specific_causal_graph = []
            for i,k in enumerate(perm):
                # ('a','b')
                control_attrs = tuple(sorted([reflections[perm[j]] for j,mask in enumerate(hasse_triangle[i]) if mask]))
                # ('b', ('a',))
                # ('c', ('a','b'))
                fixed_and_controls = (reflections[idx[k]], control_attrs)
                perm_specific_causal_graph.append(fixed_and_controls)
            # again, sorted needed for subsequent deduplication
            # ('a', 'b', 'c): (('a', ()), ('b', ('a',), ('c', ('a','b')))
            all_causal_edge_lists[hasse_triangle][perm_str] = tuple(sorted(perm_specific_causal_graph))
    
    # deduplicated, flattened version
    deduplicated_dict = {}
    for hasse_triangle in all_causal_edge_lists:
        for perm, perm_specific_causal_graph in all_causal_edge_lists[hasse_triangle].items():
            # perm allows to avoid reconstruction of causal sorting
            # (('a', ()), ('b', ('a',), ('c', ('a','b'))): ('a', 'b', 'c')
            deduplicated_dict[perm_specific_causal_graph] = perm

    print(f'Total Count: {total_count}, Deduplicated Count: {len(deduplicated_dict)}')

    deduplicated_by_perm = {}
    for perm_idx in all_permutations:
        perm_str1 = tuple([reflections[i] for i in perm_idx])
        deduplicated_by_perm[perm_str1] = []
        for perm_specific_causal_graph, perm_str2 in deduplicated_dict.items():
            # we sort causal graph according to permutation, just for simplicity of reading
            sorted_perm_sp_causal_graph = []
            if perm_str2 == perm_str1:
                for refl in perm_str1:
                    sele = [val for val in perm_specific_causal_graph if val[0]==refl][0]
                    sorted_perm_sp_causal_graph.append(sele)
            if sorted_perm_sp_causal_graph:
                deduplicated_by_perm[perm_str1].append(sorted_perm_sp_causal_graph)

    return all_causal_edge_lists, deduplicated_by_perm


def print_causal_structure(causal_structure):
    for pair in causal_structure:
        print("[")
        for feature, controls in pair:
            controls_str = " -> ".join(controls) if controls else "free"
            print(f"  {feature}: {controls_str}")
        print("]")


import torch
from itertools import product

def generate_truth_tables(N, exclude_non_causal=True):
    num_combinations = 2 ** N
    num_functions = 2 ** num_combinations
    input_combinations = list(product([0, 1], repeat=N))
    input_combinations = torch.tensor(input_combinations, dtype=torch.int8)
    
    # Generate all possible truth tables
    truth_tables = torch.zeros((num_functions, num_combinations), dtype=torch.int8)
    
    for i in range(num_functions):
        binary_string = f'{i:0{num_combinations}b}'
        truth_tables[i] = torch.tensor([int(bit) for bit in binary_string], dtype=torch.int8)

    if not exclude_non_causal:
        return truth_tables
    
    # Filtering to keep only causal functions
    causal_truth_tables = []

    for table in truth_tables:
        is_causal = True
        for var_idx in range(N):
            input_combinations_flipped = input_combinations.clone()
            input_combinations_flipped[:, var_idx] = 1 - input_combinations_flipped[:, var_idx]
            output_changed = False
            for i, original_input in enumerate(input_combinations):
                flipped_index = (input_combinations == input_combinations_flipped[i]).all(dim=1).nonzero(as_tuple=True)[0].item()
                if table[i] != table[flipped_index]:
                    output_changed = True
                    break
            if not output_changed:
                is_causal = False
                break
        
        if is_causal:
            causal_truth_tables.append(table)
    
    return torch.stack(causal_truth_tables)


def apply_truth_table(truth_table, inputs):
    """
    Apply a truth table to a set of inputs.

    Parameters:
    - truth_table: A tensor representing the truth table.
    - inputs: A tensor of shape (M, N) where M is the number of input sets and N is the number of inputs.
    
    Returns:
    - outputs: A tensor of shape (M,) containing the outputs for each input set.
    """
    # Determine the index of each input in the lexicographical order
    indices = torch.sum(inputs * (2 ** torch.arange(inputs.size(1) - 1, -1, -1)), dim=1).long()
    # Use the indices to get the output from the truth table
    return truth_table[indices]

# # Example usage
# N = 2
# causal_truth_tables = generate_truth_tables(N, exclude_non_causal=True)

# # Select the first causal truth table to apply
# truth_table = causal_truth_tables[0]

# print(f"Sampled Causal Operator:\n{truth_table}")

# # New input sets to evaluate (M, N)
# new_inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.int8)

# # Apply the selected truth table to the new inputs
# outputs = apply_truth_table(truth_table, new_inputs)
# print(f"Inputs:\n{new_inputs}")
# print(f"Outputs:\n{outputs}")