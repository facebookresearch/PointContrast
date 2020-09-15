from scipy.sparse import csr_matrix
import torch


class SparseMM(torch.autograd.Function):
  """
  Sparse x dense matrix multiplication with autograd support.
  Implementation by Soumith Chintala:
  https://discuss.pytorch.org/t/
  does-pytorch-support-autograd-on-sparse-matrix/6156/7
  """

  def forward(self, matrix1, matrix2):
    self.save_for_backward(matrix1, matrix2)
    return torch.mm(matrix1, matrix2)

  def backward(self, grad_output):
    matrix1, matrix2 = self.saved_tensors
    grad_matrix1 = grad_matrix2 = None

    if self.needs_input_grad[0]:
        grad_matrix1 = torch.mm(grad_output, matrix2.t())

    if self.needs_input_grad[1]:
        grad_matrix2 = torch.mm(matrix1.t(), grad_output)

    return grad_matrix1, grad_matrix2


def sparse_float_tensor(values, indices, size=None):
  """
  Return a torch sparse matrix give values and indices (row_ind, col_ind).
  If the size is an integer, return a square matrix with side size.
  If the size is a torch.Size, use it to initialize the out tensor.
  If none, the size is inferred.
  """
  indices = torch.stack(indices).int()
  sargs = [indices, values.float()]
  if size is not None:
    # Use the provided size
    if isinstance(size, int):
      size = torch.Size((size, size))
    sargs.append(size)
  if values.is_cuda:
    return torch.cuda.sparse.FloatTensor(*sargs)
  else:
    return torch.sparse.FloatTensor(*sargs)


def diags(values, size=None):
  values = values.view(-1)
  n = values.nelement()
  size = torch.Size((n, n))
  indices = (torch.arange(0, n), torch.arange(0, n))
  return sparse_float_tensor(values, indices, size)


def sparse_to_csr_matrix(tensor):
  tensor = tensor.cpu()
  inds = tensor._indices().numpy()
  vals = tensor._values().numpy()
  return csr_matrix((vals, (inds[0], inds[1])), shape=[s for s in tensor.shape])


def csr_matrix_to_sparse(mat):
  row_ind, col_ind = mat.nonzero()
  return sparse_float_tensor(
      torch.from_numpy(mat.data),
      (torch.from_numpy(row_ind), torch.from_numpy(col_ind)),
      size=torch.Size(mat.shape))
