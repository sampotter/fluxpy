import flux.config
import scipy.sparse.linalg


class IndentedPrinter(object):
    indent = -1
    def print(self, *args, **kwargs):
        if flux.config.DEBUG:
            print('    ' * IndentedPrinter.indent, end='')
            print(*args, **kwargs)
    def __enter__(self):
        IndentedPrinter.indent += 1
        return self
    def __exit__(self, type, value, traceback):
        IndentedPrinter.indent -= 1


class DebugLinearOperator(scipy.sparse.linalg.LinearOperator):
    def __init__(self, spmat):
        self._spmat = spmat
        self._spmat_adjoint = self._spmat.T
        self.matvec_count = 0
        self.rmatvec_count = 0
        self.matmat_count = 0
        self.rmatmat_count = 0

    def _matvec(self, x):
        self.matvec_count += 1
        return self._spmat@x

    def _matmat(self, A):
        self.matmat_count += 1
        return self._spmat@A

    def _rmatvec(self, x):
        self.rmatvec_count += 1
        return self._spmat_adjoint@x

    def _rmatmat(self, A):
        self.rmatmat_count += 1
        return self._spmat_adjoint@A

    def debug_print(self):
        with IndentedPrinter() as _:
            _.print('matvecs: %d' % self.matvec_count)
            _.print('matmats: %d' % self.matmat_count)
            _.print('rmatvecs: %d' % self.rmatvec_count)
            _.print('rmatmats: %d' % self.rmatmat_count)

    @property
    def dtype(self):
        return self._spmat.dtype

    @property
    def nnz(self):
        return self._spmat.nnz

    @property
    def shape(self):
        return self._spmat.shape
