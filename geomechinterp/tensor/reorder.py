import torch


def sort_matrix_by_row_and_column_sums(matrix):
    """
    Takes a torch matrix and returns:
    - Row permutation vector (indices that sort rows by descending sum)
    - Column permutation vector (indices that sort columns by descending sum)
    - Ordered matrix with rows and columns sorted by descending sum

    Args:
        matrix (torch.Tensor): Input 2D tensor (matrix)

    Returns:
        tuple: (row_perm, col_perm, ordered_matrix)
    """
    if len(matrix.shape) != 2:
        raise ValueError("Input must be a 2D matrix.")

    # Calculate row and column sums
    row_sums = matrix.sum(dim=1)
    col_sums = matrix.sum(dim=0)

    # Sort rows and columns by descending sums
    row_perm = torch.argsort(row_sums, descending=True)
    col_perm = torch.argsort(col_sums, descending=True)

    # Permute rows and columns
    sorted_rows = matrix[row_perm, :]
    ordered_matrix = sorted_rows[:, col_perm]

    return row_perm, col_perm, ordered_matrix


# Example usage
matrix = torch.tensor([[5, 2, 3], [1, 9, 6], [7, 4, 8]])
row_perm, col_perm, ordered_matrix = sort_matrix_by_row_and_column_sums(matrix)

print("Row permutation:", row_perm)
print("Column permutation:", col_perm)
print("Ordered matrix:")
print(ordered_matrix)
