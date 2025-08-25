import torch
import tntorch as tn
import numpy as np
from typing import List, Optional, Union
from datetime import datetime
from pathlib import Path
import re
from .tools import (
    fractional_matrix_power,
    generate_random_TT_vector,
    generate_sftucker_shapes,
    generate_cores_shapes,
    transposed_khatri_rao,
)


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r"(\d+)", text)]


class TTSFTuckerTensor:
    """
    Special class for representation of a Tucker decomposition
    with shared factors and Tucker core decomposed into Tensor Train
    decomposition
        tucker_factors -- tucker factors (the last one gets shared if needed)
        tt_cores -- custom tt cores
        shared_factors_amount -- amount of Tucker factors to simulate shared Tucker factor
    """

    tucker_factors: List[torch.Tensor]
    tt_cores: List[torch.Tensor]
    _shared_factors_amount: int
    device: torch.device
    _orthogonalization_tt: Optional[int] = None
    _orthogonalization_tucker: bool = False

    def __init__(
        self,
        shared_factors_amount: Optional[int] = None,
        non_shared_factors_amount: Optional[int] = None,
        main_dimensions: Optional[np.ndarray] = None,
        tt_cores: Optional[List[torch.Tensor]] = None,
        tucker_factors: Optional[List[torch.Tensor]] = None,
        dir_name: Optional[Path] = None,
        device: torch.device = "cpu",
    ):
        """
        Creates a tensor in a Tucker decomposition with its Tucker core
        decomposed into TT. Tucker factors are stored as a list of len
        (`non_shared_factors_amount` + 1) and the last factor should be
        mathematically repeated `shared_factors_amount` times, but stored only
        as one tensor, simulating Shared Factor Tucker decomposition. Therefore
        the whole TTSFTuckerTensor consists of
        d = (`non_shared_factors_amount` + `shared_factors_amount`) TT cores (forming Tucker Core)
        and (`non_shared_factors_amount` + 1) Tucker Factors.

        INPUT:
            shared_factors_amount -- amount of Tucker factors to simulate shared Tucker factor
            non_shared_factors_amount -- amount of vanilla Tucker factors
            main_dimensions --  array of dimensions of full tensor represented by TTSFTucker format
            tt_cores -- custom tt cores
            tucker_factors -- custom tucker factors
        """

        if (
            (tucker_factors is not None)
            and (tt_cores is not None)
            and (shared_factors_amount is not None)
        ):
            assert shared_factors_amount >= 0
            self.tucker_factors = tucker_factors
            self._shared_factors_amount = shared_factors_amount
            self.tt_cores = tt_cores

        elif not (non_shared_factors_amount is None) and not (
            shared_factors_amount is None
        ):
            assert shared_factors_amount >= 0
            assert non_shared_factors_amount >= 0
            assert non_shared_factors_amount + shared_factors_amount > 0

            self.tucker_factors = []
            self.tt_cores = []
            self._shared_factors_amount = shared_factors_amount

            # generate tucker ranks
            sf_tucker_shapes = generate_sftucker_shapes(
                non_shared_factors_amount,
                shared_factors_amount,
                main_dimensions=main_dimensions,
            )
            ranks = sf_tucker_shapes[:, 1]
            for factor_shape in sf_tucker_shapes:
                self.tucker_factors.append(
                    torch.linalg.qr(torch.randn(*factor_shape, device=device)).Q
                )
            d_t = len(sf_tucker_shapes[:, 0]) - (1 if shared_factors_amount else 0)
            tt_cores_shapes = generate_cores_shapes(
                d=-1,
                p=1,
                main_dimensions=np.concatenate(
                    [
                        ranks[:d_t],
                        np.ones(shared_factors_amount).astype(int) * ranks[-1],
                    ]
                ),
            )
            self.tt_cores = generate_random_TT_vector(tt_cores_shapes, device=device)
            self.round()
        elif dir_name is not None:
            self.load(dir_name=dir_name, device=device)
        else:
            raise NotImplementedError()

    @property
    def shared_factors_amount(self) -> int:
        return self._shared_factors_amount

    @property
    def N(self) -> int:
        return len(self.main_dimensions)

    @property
    def orthogonalization(self) -> Optional[int]:
        if self._orthogonalization_tucker:
            return self._orthogonalization_tt
        else:
            return None

    @property
    def non_ortho_cores(self) -> List[torch.Tensor]:
        t = self.clone()
        non_orthos = []
        for i in range(self.N):
            t.orthogonalize(i)
            non_orthos.append(t.tt_cores[i].clone())
        return non_orthos

    @property
    def tt_ranks(self) -> np.ndarray:
        """
        returns ranks of `tt_cores`; for tt cores of shape:

            `[1, r^{tucker}_1, r^{tt}_1], ..., [r^{tt}_{d_t - 1}, r^{tucker}_{d_t}, r^{tt}_{d_t}],
            [r^{tt}_{d_t}, r^{tucker}_s, r^{tt}_{d_t + 1}], ..., [r^{tt}_{d_t + d_s}, r^{tucker}_s, 1]`

        returns `[r^{tt}_1, ..., r^{tt}_{d_t + d_s}]` where `d_t` -- amount of vanilla Tucker cores
        and `d_s` amount of shared Tucker cores.
        """
        return np.asarray([core.shape[2] for core in self.tt_cores[:-1]])

    @property
    def tucker_ranks(self) -> np.ndarray:
        """
        returns ranks of `tucker_cores`; for tucker cores of shape:

            `[n_1, r^{tucker}_1], ..., [n_{d_t}, r^{tucker}_{d_t}],
            [n_{s}, r^{tucker}_s], ..., [n_{s}, r^{tucker}_s]`

        returns `[r^{tucker}_1, ..., r^{tucker}_{d_t}, r^{tucker}_s]`
        where `d_t` -- amount of vanilla Tucker cores and `d_s` amount of shared Tucker cores.
        """
        # len(tucker_factors) = d_t + 1, because shared factor is stored only once
        return np.asarray([factor.shape[1] for factor in self.tucker_factors])

    @property
    def main_dimensions(self) -> np.ndarray:
        """
        returns main dimensions of tensor represented by `TTSFTuckerTensor`;
        for tucker cores of shape:

            `[n_1, r^{tucker}_1], ..., [n_{d_t}, r^{tucker}_{d_t}],
            [n_{s}, r^{tucker}_s], ..., [n_{s}, r^{tucker}_s]`

        returns `[n_1, ..., n_{d_t}, n_{s}, ..., n_{s}]`
        where `d_t` -- amount of vanilla Tucker cores and `d_s` amount of shared Tucker cores.
        """
        d_t = len(self.tt_cores) - self.shared_factors_amount
        main_dims = np.asarray(
            [factor.shape[0] for factor in self.tucker_factors[:d_t]]
        )
        if self.shared_factors_amount:
            main_dims = np.concatenate(
                [
                    main_dims,
                    np.ones(self.shared_factors_amount)
                    * self.tucker_factors[-1].shape[0],
                ]
            )

        return main_dims.astype(int)

    @property
    def params_amount(self) -> int:
        """
        returns amount of elements in all cores and factors
        """
        am = 0
        d_t = len(self.main_dimensions) - self.shared_factors_amount
        for core in self.tt_cores:
            am += np.prod(core.shape)

        for factor in self.tucker_factors[:d_t]:
            am += np.prod(factor.shape)
        if self.shared_factors_amount:
            am += np.prod(self.tucker_factors[-1].shape)
        return am

    @property
    def device(self) -> torch.device:
        return self.tt_cores[0].device

    @property
    def batch_size(self) -> int:
        return self.tt_cores[-1].shape[-1]

    def __add__(self, other: "TTSFTuckerTensor"):
        if isinstance(other, TTSFTuckerTensor):
            assert np.all(self.main_dimensions == other.main_dimensions)
            assert np.all(self.shared_factors_amount == other.shared_factors_amount)
            assert self.batch_size == other.batch_size
            assert self.device == other.device

            N = len(self.main_dimensions)
            sum_tucker_factors = [
                torch.hstack([A, B])
                for A, B in zip(self.tucker_factors, other.tucker_factors)
            ]
            # suboptimal better to use pad
            sum_tt_cores = []
            for i, cores in enumerate(zip(self.tt_cores, other.tt_cores)):
                A, B = cores
                dim_A = np.asarray(A.shape)
                dim_B = np.asarray(B.shape)
                sum_shape = dim_A + dim_B
                if i == 0:
                    sum_shape[0] = 1
                    sum_core = torch.zeros(tuple(sum_shape), device=self.device)
                    sum_core[:, : dim_A[1], : dim_A[2]] = A
                    sum_core[:, dim_A[1] :, dim_A[2] :] = B
                elif i == N - 1:
                    # handle last tt cores last batchsize dimension

                    # merge it
                    sum_shape[1] = sum_shape[1] * (sum_shape[2] // 2)
                    sum_shape[2] = 1
                    dim_A = [dim_A[0], dim_A[1] * dim_A[2], 1]
                    dim_B = [dim_B[0], dim_B[1] * dim_B[2], 1]

                    A_r = A.reshape(dim_A)
                    B_r = B.reshape(dim_B)

                    # perform sum as usual
                    sum_core = torch.zeros(tuple(sum_shape), device=self.device)
                    sum_core[: dim_A[0], : dim_A[1], :] = A_r
                    sum_core[dim_A[0] :, dim_A[1] :, :] = B_r

                    # unmerge
                    sum_core = sum_core.reshape(
                        sum_shape[0],
                        sum_shape[1] // self.batch_size,
                        self.batch_size,
                    )
                else:
                    sum_core = torch.zeros(tuple(sum_shape), device=self.device)
                    sum_core[: dim_A[0], : dim_A[1], : dim_A[2]] = A
                    sum_core[dim_A[0] :, dim_A[1] :, dim_A[2] :] = B

                sum_tt_cores.append(sum_core)

            return TTSFTuckerTensor(
                shared_factors_amount=self.shared_factors_amount,
                tucker_factors=sum_tucker_factors,
                tt_cores=sum_tt_cores,
            )
        else:
            raise NotImplementedError(
                f"Operation __add__ is not defined for {type(self)} and {type(other)}"
            )

    def __sub__(self, other: "TTSFTuckerTensor"):

        return self + (-1) * other

    def __rmul__(self, other):
        return self * other

    def __mul__(self, other):
        """
        Elementwise multiplication of `TTSFTuckerTensor` by scalar
        INPUT:
            other -- Scalar value
        OUTPUT:
            new `TTSFTuckerTensor` multiplied by `other`
        """
        if isinstance(other, (int, float, torch.Tensor)):
            t = self.clone()
            t.tt_cores[0] *= other
            t._orthogonalization_tt = None

            return t
        else:
            raise NotImplementedError(
                f"Operation __mul__ is not defined for {type(self)} and {type(other)}"
            )

    def __imul__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            self.tt_cores[0] *= other
            self._orthogonalization_tt = None
            return self
        else:
            raise NotImplementedError(
                f"Operation __imul__ is not defined for {type(self)} and {type(other)}"
            )

    def __neg__(self):
        return (-1) * self

    def __truediv__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            t = self.clone()
            t.tt_cores[0] /= other
            t._orthogonalization_tt = None
            return t
        else:
            raise NotImplementedError(
                f"Operation __truediv__ is not defined for {type(self)} and {type(other)}"
            )

    def __itruediv__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            self.tt_cores[0] /= other
            self._orthogonalization_tt = None
            return self
        else:
            raise NotImplementedError(
                f"Operation __itruediv__ is not defined for {type(self)} and {type(other)}"
            )

    def __matmul__(self, other):
        if isinstance(other, TTSFTuckerTensor):
            return tt_sf_tucker_scalar_prod(self, other)
        elif (
            isinstance(other, torch.Tensor)
            and len(other.shape) == 2
            and self.batch_size == other.shape[0]
        ):
            t = self.clone()
            t.tt_cores[-1] = torch.einsum("abc,cd->abd", t.tt_cores[-1], other)
            return t
        else:
            raise TypeError(
                f"Operation __matmul__ is not defined for {type(self)} and {type(other)}"
            )

    def hadamard_product(self, other: "TTSFTuckerTensor"):
        assert self.shared_factors_amount == other.shared_factors_amount
        assert self.N == other.N
        res_tt_cores = [
            torch.kron(s_c.contiguous(), o_c.contiguous())
            for s_c, o_c in zip(self.tt_cores, other.tt_cores)
        ]
        res_tucker_factors = [
            transposed_khatri_rao(s_f, o_f)
            for s_f, o_f in zip(self.tucker_factors, other.tucker_factors)
        ]
        return TTSFTuckerTensor(
            shared_factors_amount=self.shared_factors_amount,
            tucker_factors=res_tucker_factors,
            tt_cores=res_tt_cores,
        )

    def save(self, dir_name: Optional[Path] = None):
        """
        Saves all cores to given dir
        """
        if dir_name is None:
            dir_name = Path(
                "saves/"
                + "save_"
                + datetime.today().strftime("%Y-%m-%d %H:%M:%S")
                + "/"
            )

        dir_name.mkdir(parents=True, exist_ok=True)

        for i, core in enumerate(self.tt_cores):
            torch.save(core, dir_name / (str(i) + "_tt_core.pt"))
        for i, factor in enumerate(self.tucker_factors):
            torch.save(factor, dir_name / (str(i) + "_tucker_factor.pt"))

        torch.save(
            torch.Tensor(data=[self.shared_factors_amount]),
            dir_name / "shared_factors_amount.pt",
        )

    def load(self, dir_name: str, device: torch.device = "cpu"):
        """
        Loads all cores from given dir
        """
        self.tt_cores = []
        names = [n.name for n in Path.glob(dir_name, pattern="*[0-9]_tt_core.pt")]
        names.sort(key=natural_keys)
        for name in names:
            self.tt_cores.append(torch.load(dir_name / name, map_location=device))

        self.tucker_factors = []
        names = [n.name for n in Path.glob(dir_name, pattern="*[0-9]_tucker_factor.pt")]
        names.sort(key=natural_keys)
        for name in names:
            self.tucker_factors.append(torch.load(dir_name / name, map_location=device))

        for n in Path.glob(dir_name, pattern="shared_factors_amount.pt"):
            self._shared_factors_amount = int(
                torch.load(dir_name / n.name, map_location=device)
            )
        self._orthogonalization_tt = None
        self._orthogonalization_tucker = False

    def norm(
        self, mu: Optional[int] = None, fast: bool = False, honest: bool = False
    ) -> torch.Tensor:
        """
        Computes the Frobenius norm of a tensor.
        mu -- if mu \in [0, d-1] A, B  treated as TT-mu-orthogonal
              with orthogonal Tucker Factors
        """
        if fast and (self.orthogonalization is not None):
            if self.batch_size != 1:
                raise NotImplementedError()
            return torch.linalg.norm(self.tt_cores[self.orthogonalization])
        else:
            sq_norm = self.sq_norm(mu, fast, honest=honest)
            if len(sq_norm.shape) == 2:
                return fractional_matrix_power(sq_norm, 1 / 2)
            else:
                return torch.sqrt(torch.clamp(sq_norm, min=0))

    def sq_norm(
        self, mu: Optional[int] = None, fast: bool = False, honest: bool = False
    ) -> torch.Tensor:
        """
        Computes the squared Frobenius norm of a tensor.
        mu -- if mu \in [0, d-1] A, B  treated as TT-mu-orthogonal
              with orthogonal Tucker Factors
        """
        return tt_sf_tucker_scalar_prod(self, self, mu, honest=honest)

    def clone_tt_cores(self) -> List[torch.Tensor]:
        """
        create a copy of `tt_cores` of TTSFTucker tensor
        rm

        """
        return [core.clone() for core in self.tt_cores]

    def clone_tucker_factors(self) -> List[torch.Tensor]:
        """
        create a copy of `tucker_factors` of TTSFTucker tensor
        """
        return [core.clone() for core in self.tucker_factors]

    def clone(self) -> "TTSFTuckerTensor":
        """
        Creates exact same TTSFTucker tensor (copy)
        INPUT:
            None
        OUTPUT:
            TTSFTuckerTensor -- a copy of `self`
        """
        new_t = TTSFTuckerTensor(
            tt_cores=self.clone_tt_cores(),
            tucker_factors=self.clone_tucker_factors(),
            shared_factors_amount=self.shared_factors_amount,
        )
        new_t._orthogonalization_tt = self._orthogonalization_tt
        new_t._orthogonalization_tucker = self._orthogonalization_tucker
        return new_t

    def to_dense(self) -> torch.Tensor:
        """
        Decompresses the tensor into torch Tensor
        """
        tt_cores = self.clone_tt_cores()
        N = len(self.tt_cores)
        d_t = N - self.shared_factors_amount
        for i in range(d_t):
            tt_cores[i] = torch.einsum(
                "abc,vb->avc", self.tt_cores[i], self.tucker_factors[i]
            )
        for j in range(d_t, N):
            tt_cores[j] = torch.einsum(
                "abc,vb->avc", self.tt_cores[j], self.tucker_factors[-1]
            )

        return tn.Tensor(tt_cores).torch()

    def convert_to_tt(self) -> List[torch.Tensor]:
        tt_cores = self.clone_tt_cores()
        N = len(self.tt_cores)
        d_t = N - self.shared_factors_amount

        for i in range(d_t):
            tt_cores[i] = torch.einsum(
                "ar,crd->cad", self.tucker_factors[i], tt_cores[i]
            )
        for i in range(d_t, N):
            tt_cores[i] = torch.einsum(
                "ar,crd->cad", self.tucker_factors[-1], tt_cores[i]
            )

        return tt_cores

    def _orthogonalize_tt_cores(self, mu: int):
        """
        Works inplace. Orthogonolizes `tt_cores` into `mu`-orthogonalization
        """
        N = len(self.main_dimensions)
        mu = mu % N
        if self.orthogonalization == mu:
            return
        tn_tt_cores = tn.Tensor(self.tt_cores)
        tn_tt_cores.orthogonalize(mu)
        self.tt_cores = tn_tt_cores.cores
        self._orthogonalization_tt = mu

    def _orthogonalize_tucker_factors(self):
        """
        Works inplace. Orthogonolizes `tucker_factors`
        """
        if self._orthogonalization_tucker:
            return
        else:
            N = len(self.tt_cores)
            d_t = N - self.shared_factors_amount
            for mu, factor in enumerate(self.tucker_factors[:d_t]):
                qf, rf = torch.linalg.qr(factor)
                self.tt_cores[mu] = torch.einsum(
                    "abc,db->adc",
                    self.tt_cores[mu],
                    rf,
                )
                self.tucker_factors[mu] = qf
            if self.shared_factors_amount:
                qf, rf = torch.linalg.qr(self.tucker_factors[-1])
                self.tucker_factors[-1] = qf
                for i, core in enumerate(self.tt_cores[d_t:], start=d_t):
                    self.tt_cores[i] = torch.einsum("abc,db->adc", core, rf)
            self._orthogonalization_tucker = True
            self._orthogonalization_tt = None

    def orthogonalize(self, mu: int):
        """
        Works inplace. Orthogonolizes all factors
        """
        self._orthogonalize_tucker_factors()
        self._orthogonalize_tt_cores(mu=mu)

    def round_tt_cores(self, eps: float = 1e-12, tt_ranks: Optional[np.ndarray] = None):
        """
        Rounds `tt_cores` up to the `eps` relative error or up to the given `tt_ranks`
        Rounding is being applyed in TT format, not forming the whole tucker core out of
        `tt_cores`.
        INPUT:
            eps -- rounding error, will not be exceeded
            tt_ranks -- all ranks should be rmax at most (default: no limit)
        OUTPUT:
            rounded tt cores `[U_1, ..., U_{d_t + d_s}]`; `U_mu` as it would be
            in mu-orthogonalized Tensor (after `TTSFTucker.orthogonalize(mu)`)
        """
        tn_tt_cores = tn.Tensor(self.tt_cores)
        tn_tt_cores.round_tt(eps=eps, rmax=tt_ranks)
        self.tt_cores = tn_tt_cores.cores
        self._orthogonalization_tt = 0

    def round(
        self,
        eps: float = 1e-10,
        tt_ranks: Optional[np.ndarray] = None,
        sf_tucker_ranks: Optional[np.ndarray] = None,
    ):
        """
        Rounds `TTSFTuckerTensor` up to the `eps` relative error or up to the given `tt_ranks` and
        `sf_tucker_ranks`
        Rounding is being applyed in TTSFTucker format, not forming the whole tensor out of
        `tt_cores` and `tucker_factors`.
        INPUT:
            eps -- rounding error, will not be exceeded
            tt_ranks -- all ranks of `tt_cores` should be rmax at most (default: no limit)
            sf_tucker_ranks -- all ranks of `tucker_factors` should be rmax at most (default: no limit)
        OUTPUT:
            None
        """
        N = len(self.tt_cores)
        d_s = self.shared_factors_amount
        d_t = N - d_s
        if not hasattr(sf_tucker_ranks, "__len__"):
            sf_tucker_ranks = [sf_tucker_ranks] * (d_t + (1 if d_s else 0))
        if not hasattr(tt_ranks, "__len__"):
            tt_ranks = [tt_ranks] * (N - 1)
        self._orthogonalize_tucker_factors()
        self.round_tt_cores(eps=eps / 2, tt_ranks=tt_ranks)

        non_ortho_cores = [self.tt_cores[0].clone()]
        non_ortho_cores[0] = non_ortho_cores[0].reshape(self.tt_cores[0].shape[1], -1)
        for i in range(1, N):
            self.orthogonalize(i)
            non_ortho_cores.append(torch.einsum("abc->bac", self.tt_cores[i].clone()))
            non_ortho_cores[-1] = non_ortho_cores[-1].reshape(
                self.tt_cores[i].shape[1], -1
            )

        for i in range(d_t + (1 if self.shared_factors_amount else 0)):
            # vanilla TT Tucker
            if i == d_t:
                # Shared Factor TT Tucker
                nn_o = torch.hstack(non_ortho_cores[d_t:])
                Y, _ = tn.truncated_svd(
                    nn_o,
                    delta=None,
                    eps=eps / 2,
                    rmax=sf_tucker_ranks[-1],
                    left_ortho=True,
                    algorithm="svd",
                    verbose=False,
                    batch=False,
                )
                for j in range(d_t, N):
                    self.tt_cores[j] = torch.einsum("yr,ayb->arb", Y, self.tt_cores[j])
            else:
                Y, _ = tn.truncated_svd(
                    non_ortho_cores[i],
                    eps=eps / 2,
                    rmax=sf_tucker_ranks[i],
                    left_ortho=True,
                    algorithm="svd",
                    verbose=False,
                    batch=False,
                )
                # tt_cores[i] \times_k Y^\intercal
                self.tt_cores[i] = torch.einsum("yr,ayb->arb", Y, self.tt_cores[i])
            self.tucker_factors[i] = self.tucker_factors[i] @ Y
        self._orthogonalization_tt = None
        self.orthogonalize(-1)


def tt_sf_tucker_scalar_prod(
    A: TTSFTuckerTensor,
    B: TTSFTuckerTensor,
    mu: Optional[int] = None,
    honest: bool = True,
) -> torch.Tensor:
    """
    INPUT:
        A -- a TTSFTuckerTensor
        B -- a TTSFTuckerTensor
        mu -- if mu \in [0, d-1] A, B  treated as TT-mu-orthogonal
              with orthogonal Tucker Factors
    OUTPUT:
        <A, B> \in R
    """
    assert np.all(A.main_dimensions == B.main_dimensions)
    N = len(A.main_dimensions)
    d_s = A.shared_factors_amount
    d_t = N - d_s
    if honest:
        assert B.shared_factors_amount == A.shared_factors_amount
        B_cl = B.clone()
        for i, a_factor in enumerate(A.tucker_factors[:d_t]):
            B_cl.tucker_factors[i] = torch.einsum(
                "nb,na->ab", B_cl.tucker_factors[i], a_factor
            )
        if A.shared_factors_amount != 0:
            B_cl.tucker_factors[-1] = torch.einsum(
                "nb,na->ab", B_cl.tucker_factors[-1], A.tucker_factors[-1]
            )

        def _tt_scalar_prod(x_tt, y_tt) -> float:
            start_core = torch.einsum("abx,aby->xy", x_tt[0], y_tt[0])
            for x_core, y_core in zip(x_tt[1:], y_tt[1:]):
                start_core = torch.einsum("xy,xac,yad->cd", start_core, x_core, y_core)

            return (
                start_core
                if x_tt[-1].shape[-1] != 1 or y_tt[-1].shape[-1] != 1
                else start_core.squeeze()
            )

        return _tt_scalar_prod(B_cl.convert_to_tt(), A.clone_tt_cores())
    else:
        if mu is None:
            mu = N - 1
            A_clone = A.clone()
            A_clone.orthogonalize(mu)
            if A is B:
                B = A_clone
            else:
                B_clone = B.clone()
                B_clone.orthogonalize(mu)

            A = A_clone

        mu = mu % N
        scalar_prod = torch.einsum("abd,abc->dc", A.tt_cores[mu], B.tt_cores[mu])
        return (
            scalar_prod
            if scalar_prod.shape[1] != 1 or scalar_prod.shape[-1] != 1
            else scalar_prod.squeeze()
        )


class TTSFTuckerTangentVector:
    """
    Special class for representation of tangent vector at a point of
    `TTSFTuckerTensor`
        point: a point from manifold, a foot of tangent vector
        delta_regular_factors: deltas of non shared Tucker cores
        delta_shared_factor: deltas of shared Tucker core
        delta_tt_cores: deltas of shared Tucker core
    """

    point: TTSFTuckerTensor
    delta_regular_factors: List[torch.Tensor]
    delta_shared_factor: torch.Tensor
    delta_tt_cores: List[torch.Tensor]
    device: torch.device

    def __init__(
        self,
        manifold_point: Optional[TTSFTuckerTensor] = None,
        delta_regular_factors: Optional[List[torch.Tensor]] = None,
        delta_shared_factor: Optional[torch.Tensor] = None,
        delta_tt_cores: Optional[List[torch.Tensor]] = None,
        dir_name: Optional[Path] = None,
        requires_grad: bool = False,
        device: torch.device = "cpu",
    ):
        """
        Creates a tangent vector for `TTSFTuckerTensor` `manifold_point`
        which is in a Tucker decomposition with its Tucker core
        decomposed into TT.
        INPUT:
            point: a point from manifold, a foot of tangent vector
            delta_regular_factors: deltas of non shared Tucker cores
            delta_shared_factor: deltas of shared Tucker core
            delta_tt_cores: deltas of shared Tucker core
        """
        if dir_name is not None:
            self.load(dir_name=dir_name, device=device, requires_grad=requires_grad)
        elif manifold_point is not None:
            assert manifold_point.shared_factors_amount > 0

            self.point = manifold_point.clone()
            N = len(self.point.main_dimensions)
            d_s = self.point.shared_factors_amount
            d_t = N - d_s
            self.point.orthogonalize(-1)

            self.delta_tt_cores = []
            if delta_tt_cores is None:
                for core in self.point.tt_cores[:-1]:
                    self.delta_tt_cores.append(
                        torch.zeros_like(core, requires_grad=requires_grad)
                    )
                last_core = self.point.tt_cores[-1].clone().detach()
                last_core.requires_grad = requires_grad
                self.delta_tt_cores.append(last_core)
            else:
                for core in delta_tt_cores:
                    core = core.clone().detach()
                    core.requires_grad = requires_grad
                    self.delta_tt_cores.append(core)

            self.delta_regular_factors = []
            if delta_regular_factors is None:
                self.delta_regular_factors = [
                    torch.zeros_like(factor, requires_grad=requires_grad)
                    for factor in self.point.tucker_factors[:d_t]
                ]
            else:
                for factor in delta_regular_factors:
                    factor = factor.clone().detach()
                    factor.requires_grad = requires_grad
                    self.delta_regular_factors.append(factor)

            if delta_shared_factor is None:
                self.delta_shared_factor = torch.zeros_like(
                    self.point.tucker_factors[-1], requires_grad=requires_grad
                )
            else:
                _delta_shared_factor = delta_shared_factor.clone().detach()
                _delta_shared_factor.requires_grad = requires_grad
                self.delta_shared_factor = _delta_shared_factor
        else:
            raise NotImplementedError()

    @property
    def shared_factors_amount(self) -> int:
        return self.point.shared_factors_amount

    @property
    def N(self) -> int:
        return self.point.N

    @property
    def tt_ranks(self) -> np.ndarray:
        """
        returns ranks of `tt_cores` of `manifold_point`
        """
        return self.point.tt_ranks

    @property
    def tucker_ranks(self) -> np.ndarray:
        """
        returns ranks of `tucker_cores` of `manifold_point`
        """
        # len(tucker_factors) = d_t + 1, because shared factor is stored only once
        return self.point.tucker_ranks

    @property
    def main_dimensions(self) -> np.ndarray:
        """
        returns main dimensions of `manifold_point`
        """
        return self.point.main_dimensions

    @property
    def device(self) -> torch.device:
        return self.point.device

    @property
    def batch_size(self) -> int:
        return self.point.batch_size

    def __matmul__(self, other):
        if isinstance(other, TTSFTuckerTangentVector):
            assert self.batch_size == other.batch_size
            assert len(self.main_dimensions) == len(other.main_dimensions)
            sq_norm = 0
            for c1, c2 in zip(self.delta_tt_cores, other.delta_tt_cores):
                sq_norm += torch.einsum("abc,abc->", c1, c2)
            return sq_norm
        elif (
            isinstance(other, torch.Tensor)
            and len(other.shape) == 2
            and self.batch_size == other.shape[0]
        ):
            t = self.clone()
            t.point.tt_cores[-1] = torch.einsum(
                "abc,cd->abd", t.point.tt_cores[-1], other
            )
            t.delta_tt_cores[-1] = torch.einsum(
                "abc,cd->abd", t.delta_tt_cores[-1], other
            )
            return t
        else:
            raise TypeError(
                f"Operation __matmul__ is not defined for {type(self)} and {type(other)}"
            )

    def __add__(self, other: "TTSFTuckerTangentVector") -> "TTSFTuckerTangentVector":
        assert np.all(self.main_dimensions == other.main_dimensions)
        assert np.all(self.shared_factors_amount == other.shared_factors_amount)
        assert self.device == other.device
        sum_delta_regular_factors = [
            a + b
            for a, b in zip(self.delta_regular_factors, other.delta_regular_factors)
        ]
        sum_delta_shared_factor = self.delta_shared_factor + other.delta_shared_factor
        sum_delta_tt_cores = [
            a + b for a, b in zip(self.delta_tt_cores, other.delta_tt_cores)
        ]

        return TTSFTuckerTangentVector(
            manifold_point=self.point.clone(),
            delta_regular_factors=sum_delta_regular_factors,
            delta_shared_factor=sum_delta_shared_factor,
            delta_tt_cores=sum_delta_tt_cores,
        )

    def __sub__(self, other: "TTSFTuckerTangentVector") -> "TTSFTuckerTangentVector":
        return self + (-1) * other

    def __rmul__(self, a: Union[int, float]) -> "TTSFTuckerTangentVector":
        """
        Elementwise multiplication of `TTSFTuckerTangentVector` by scalar
        INPUT:
            a -- Scalar value
        OUTPUT:
            new `TTSFTuckerTangentVector` multiplied by `a`
        """
        t = self.clone()

        for i in range(len(t.delta_regular_factors)):
            t.delta_regular_factors[i] *= a

        for i in range(len(self.delta_tt_cores)):
            t.delta_tt_cores[i] *= a

        t.delta_shared_factor *= a
        return t

    def __neg__(self) -> "TTSFTuckerTangentVector":
        return (-1) * self

    def save(self, dir_name: Optional[Path] = None):
        """
        Saves all cores to given dir
        """
        if dir_name is None:
            dir_name = Path(
                "saves/"
                + "save_"
                + datetime.today().strftime("%Y-%m-%d %H:%M:%S")
                + "/"
            )

        dir_name.mkdir(parents=True, exist_ok=True)
        self.point.save(dir_name=dir_name)

        for i, core in enumerate(self.delta_tt_cores):
            torch.save(core, dir_name / (str(i) + "_delta_tt_core.pt"))
        for i, factor in enumerate(self.delta_regular_factors):
            torch.save(factor, dir_name / (str(i) + "_delta_regular_factor.pt"))

        torch.save(
            self.delta_shared_factor,
            dir_name / "delta_shared_factor.pt",
        )

    def load(
        self, dir_name: str, device: torch.device = "cpu", requires_grad: bool = False
    ):
        """
        Loads all cores from given dir
        """
        self.point = TTSFTuckerTensor(dir_name=dir_name, device=device)
        self.delta_tt_cores = []
        names = [n.name for n in Path.glob(dir_name, pattern="*[0-9]_delta_tt_core.pt")]
        names.sort(key=natural_keys)
        for name in names:
            delta_tt_core = torch.load(dir_name / name, map_location=device)
            delta_tt_core.requires_grad = requires_grad
            self.delta_tt_cores.append(delta_tt_core)

        self.delta_regular_factors = []
        names = [
            n.name
            for n in Path.glob(dir_name, pattern="*[0-9]_delta_regular_factor.pt")
        ]
        names.sort(key=natural_keys)
        for name in names:
            delta_regular_factor = torch.load(dir_name / name, map_location=device)
            delta_regular_factor.requires_grad = requires_grad
            self.delta_regular_factors.append(delta_regular_factor)

        for n in Path.glob(dir_name, pattern="delta_shared_factor.pt"):
            delta_shared_factor = torch.load(dir_name / n.name, map_location=device)
            delta_shared_factor.requires_grad = requires_grad

            self.delta_shared_factor = delta_shared_factor

    def norm(self) -> torch.Tensor:
        """
        Computes the Frobenius norm of a tangent vector
        """
        sq_norm = self @ self
        N = len(self.main_dimensions)
        non_orthos = self.point.non_ortho_cores
        for i in range(N):
            cur_delta = (
                self.delta_regular_factors[i]
                if i < N - self.shared_factors_amount
                else self.delta_shared_factor
            )
            sq_norm += torch.einsum(
                "abc,db,de,aec->",
                non_orthos[i],
                cur_delta,
                cur_delta,
                non_orthos[i],
            )
        return torch.sqrt(torch.clamp(sq_norm, min=0))

    def clone_delta_tt_cores(self) -> List[torch.Tensor]:
        """
        create a copy of `delta_tt_cores` of TTSFTuckerTangentVector
        """
        return [core.clone() for core in self.delta_tt_cores]

    def clone_delta_regular_factors(self) -> List[torch.Tensor]:
        """
        create a copy of `delta_regular_factors` of TTSFTuckerTangentVector
        """
        return [core.clone() for core in self.delta_regular_factors]

    def clone_delta_shared_factor(self) -> torch.Tensor:
        """
        create a copy of `delta_shared_facto` of TTSFTuckerTangentVector
        """
        return self.delta_shared_factor.clone()

    def clone_point(self) -> TTSFTuckerTensor:
        """
        create a copy of `point` of TTSFTuckerTangentVector
        """
        return self.point.clone()

    def clone(self) -> "TTSFTuckerTangentVector":
        """
        Creates exact same TTSFTuckerTangentVector tensor (copy)
        INPUT:
            None
        OUTPUT:
            TTSFTuckerTangentVector -- a copy of `self`
        """
        return TTSFTuckerTangentVector(
            manifold_point=self.clone_point(),
            delta_regular_factors=self.clone_delta_regular_factors(),
            delta_shared_factor=self.clone_delta_shared_factor(),
            delta_tt_cores=self.clone_delta_tt_cores(),
        )

    def construct(self) -> TTSFTuckerTensor:
        """
        Decompresses the tensor into TTSF
        """

        N = self.N
        d_s = self.shared_factors_amount
        d_t = N - d_s
        point = self.point.clone()
        non_orthos = self.point.non_ortho_cores
        point.orthogonalize(-1)
        point_lortho_tt_cores = point.clone_tt_cores()

        tucker_factors = [
            torch.hstack([A, B])
            for A, B in zip(point.tucker_factors[:d_t], self.delta_regular_factors)
        ] + (
            [torch.hstack([point.tucker_factors[-1], self.delta_shared_factor])]
            if d_s != 0
            else []
        )

        point.orthogonalize(0)
        point_rortho_tt_cores = point.clone_tt_cores()

        tt_cores = []
        for i, cores in enumerate(
            zip(self.delta_tt_cores, point_lortho_tt_cores, point_rortho_tt_cores)
        ):
            delta, lcore, rcore = cores
            dim_A = np.asarray(delta.shape)
            sum_shape = 2 * dim_A
            if i == 0:
                sum_shape[0] = 1
                sum_core = torch.zeros(tuple(sum_shape), device=self.device)
                sum_core[:, : dim_A[1], : dim_A[-1]] = delta
                sum_core[:, dim_A[1] :, : dim_A[-1]] = non_orthos[0]
                sum_core[:, : dim_A[1], dim_A[-1] :] = lcore
            elif i == N - 1:
                # merge
                sum_shape[1] = sum_shape[1] * (sum_shape[2] // 2)
                sum_shape[2] = 1
                dim_A = [dim_A[0], dim_A[1] * dim_A[2], 1]

                delta_r = delta.reshape(dim_A)
                rcore_r = rcore.reshape(dim_A)
                non_ortho_r = non_orthos[-1].reshape(dim_A)

                # perform sum as usual

                sum_core = torch.zeros(tuple(sum_shape), device=self.device)
                sum_core[dim_A[0] :, : dim_A[1], :] = delta_r
                sum_core[: dim_A[0], : dim_A[1], :] = rcore_r
                sum_core[dim_A[0] :, dim_A[1] :, :] = non_ortho_r

                # unmerge
                sum_core = sum_core.reshape(
                    sum_shape[0],
                    sum_shape[1] // self.batch_size,
                    self.batch_size,
                )
            else:
                sum_core = torch.zeros(tuple(sum_shape), device=self.device)
                sum_core[dim_A[0] :, : dim_A[1], : dim_A[-1]] = delta
                sum_core[dim_A[0] :, dim_A[1] :, : dim_A[-1]] = non_orthos[i]
                sum_core[: dim_A[0], : dim_A[1], : dim_A[-1]] = rcore
                sum_core[dim_A[0] :, : dim_A[1], dim_A[-1] :] = lcore

            tt_cores.append(sum_core)
        return TTSFTuckerTensor(
            shared_factors_amount=d_s, tt_cores=tt_cores, tucker_factors=tucker_factors
        )

    def to_dense(self) -> torch.Tensor:
        """
        Decompresses the tangent vector into torch Tensor
        """
        t = self.construct()
        t.round()
        return t.to_dense()
