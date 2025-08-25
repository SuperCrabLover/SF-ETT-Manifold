import torch
import tntorch
import copy
import numpy as np
from typing import List, Optional, Tuple, Union, Callable


def transposed_khatri_rao(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    INPUT:
        A: matrix_1
        B: matrix_2
    OUTPUT:
        [[A[0, :] kron B[0, :]],
            ...
        [A[-1, :] kron B[-1, :]]]
    """
    return torch.vstack([torch.kron(A[k, :], B[k, :]) for k in range(A.shape[0])])


def henon_heiles(x: torch.Tensor) -> torch.Tensor:
    """
    INPUT:
        x: input states \in R^{p \times N}, p --- amount of states (batch size)
    OUTPUT:
        Henon Heiles potential for given state
        hh(x) =
        $\frac{1}{2}\sum_{k = 1}^d q_k^2 +
        \lambda \sum\limits_{k = 1}^{d-1} \left(q_k^2q_{k+1} - \frac{1}{3}q_{k+1}^3\right)$
    """
    l: float = 0.111803
    sq_x = x * x
    sh_x = torch.roll(x, -1, dims=-1)
    sh_x[:, -1] *= 0
    return torch.sum(sq_x, dim=-1) / 2 + l * torch.sum(
        sq_x * sh_x - (sh_x * sh_x * sh_x) / 3, dim=-1
    )


def generate_laplace_TT(n: int, d: int, alpha=1, device="cpu") -> List[torch.Tensor]:
    """
    INPUT:
        n: size of core slices
        d: amount of cores
        deivce -- cpu or cuda
    OUTPUT:
        TT vector with given core shapes
    """
    e = torch.eye(n, device=device)
    # [A I] \babochka [I 0] \babochka ... \babochka [I]
    #                 [A I]                         [A]
    # A = tridiag(-1, 2, -1)
    a = (
        torch.diag(torch.ones(n, device=device) * (-2), 0)
        + torch.diag((1) * torch.ones(n - 1, device=device), 1)
        + torch.diag((1) * torch.ones(n - 1, device=device), -1)
    )
    a = (-1) * a * alpha  # * (n + 1) ** 2
    cores = []

    core1 = torch.zeros((1, n, n, 2), device=device)
    core1[0, :, :, 0] = a
    core1[0, :, :, 1] = e
    cores.append(core1)

    for k in range(1, d - 1):
        core = torch.zeros((2, n, n, 2), device=device)
        core[0, :, :, 0] = e
        core[1, :, :, 0] = a
        core[1, :, :, 1] = e
        cores.append(core)

    cored = torch.zeros((2, n, n, 1), device=device)
    cored[0, :, :, 0] = e
    cored[1, :, :, 0] = a
    cores.append(cored)
    return cores


def generate_random_TT_vector(
    cores_shapes: np.ndarray[Tuple[int, int, int]], device: str = "cpu"
) -> List[torch.Tensor]:
    """
    INPUT:
        cores_shapes -- [[1, n_1, r_1], [r_1, n_2, r_2], ..., [r_{d-1}, n_d, p]]
                        s.t. r_i <= r_max for all i

        deivce -- cpu or cuda
    OUTPUT:
        TT vector with given core shapes
    """
    X = [
        torch.rand(core_shape[0], core_shape[1], core_shape[2], device=device)
        for core_shape in cores_shapes
    ]

    return X


def generate_random_TT_matrix(
    cores_shapes: np.ndarray[Tuple[int, int, int]],
    device: str = "cpu",
    symmetric: bool = False,
    pos_semidef: bool = False,
) -> List[torch.Tensor]:
    """
    INPUT:
        cores_shapes: [[1, n_1, r_1], [r_1, n_2, r_2], ..., [r_{d-1}, n_d, p]]
                        s.t. r_i <= r_max for all i

        deivce: cpu or cuda
        symmetric: generate symmetric TT-matrix
        semidef: generate positive semidefinite TT-matrix
    OUTPUT:
        TT matrix with given core shapes [[1, n_1, n_1, r_1], [r_1, n_2, n_2, r_2], ..., [r_{d-1}, n_d, n_d, 1]]
    """
    # p == 1
    cores_shapes[-1][-1] = 1
    X = [
        torch.rand(
            core_shape[0],
            core_shape[1],
            core_shape[1],
            core_shape[2],
            device=device,
        )
        for core_shape in cores_shapes
    ]
    if pos_semidef:
        X_t = [core.transpose(1, 2) for core in X]
        XX_t = []
        for core, core_t in zip(X, X_t):
            cs = core.shape
            cs_t = core_t.shape
            final_shape = (cs[0] * cs_t[0], cs[1], cs_t[2], cs[-1] * cs_t[-1])
            XX_t.append(
                torch.einsum("abcd, ecgh -> aebgdh", core, core_t).reshape(final_shape)
            )
        return XX_t
    if symmetric:
        X = [core + torch.transpose(core, 1, 2) for core in X]
        return X
    return X


def hstack_tt(*ts: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    INPUT:
        ts: a sequence of TT-vectors of length k to concat
            via last core's last dimension
    OUTPUT:
        Concatination of k TT vectors:
        1st cores: [1, n_1, r^1_1] [r^1_1, n_2, r^1_2] ... [r^1_{d-1}, n_d, p^1]
        2nd cores: [1, n_1, r^2_1] [r^2_1, n_2, r^2_2] ... [r^2_{d-1}, n_d, p^2]
                                ......
        kth cores: [1, n_1, r^k_1] [r^k_1, n_2, r^k_2] ... [r^k_{d-1}, n_d, p^k]

        result: [1, n_1, R_1] [R_1, n_2, R_2] ... [R_{d-1}, n_d, p^1 + p^2 + ... + p^k]
    """
    dim = -1
    # Convert to tn torch
    ts = tuple([tntorch.Tensor(elem) for elem in ts])
    if hasattr(ts[0], "__len__"):
        ts = ts[0]
    if len(ts) == 1:
        return ts[0].clone()
    if any(
        [
            any(
                [
                    t.shape[n] != ts[0].shape[n]
                    for n in np.delete(range(ts[0].dim()), dim)
                ]
            )
            for t in ts[1:]
        ]
    ):
        raise ValueError(
            "To concatenate tensors, all must have the same shape along all but the given dim"
        )
    device = ts[0].cores[0].device
    shapes = np.array([t.cores[dim].shape[dim] for t in ts])
    sumshapes = np.concatenate([np.array([0]), np.cumsum(shapes)])
    for i in range(len(ts)):
        t = ts[i].clone()
        if t.cores[dim].dim() == 2:
            t.cores[dim] = torch.zeros(t.cores[dim].shape[-1], sumshapes[-1]).to(device)
        else:
            t.cores[dim] = torch.zeros(
                t.cores[dim].shape[0], t.cores[dim].shape[1], sumshapes[2]
            ).to(device)

        t.cores[dim][..., sumshapes[i] : sumshapes[i + 1]] += ts[i].cores[dim]
        if i == 0:
            result = t
        else:
            result = sum_tntorch_vectors(result, t)
    return result.cores


def sum_tntorch_vectors(
    A_tn: tntorch.Tensor,
    B_tn: tntorch.Tensor,
    tt_ranks: Optional[tuple] = None,
) -> tntorch.Tensor:
    """
    INPUT:
        A_tn, B_tn: block TT vectors in tntorch that need to be summed
                    (last core's last dimension is p >= 1)
        tt_ranks:   ranks of fixed rank manifold to project the sum onto
                    if needed
    OUTPUT:
        the sum:    tntorch block TT vector A_tn + B_tn
    """
    q = A_tn.cores[-1].shape[-1]
    assert q == B_tn.cores[-1].shape[-1]

    merge_last_core(A_tn)
    merge_last_core(B_tn)

    sum_tn = A_tn + B_tn

    split_last_core(A_tn, q)
    split_last_core(B_tn, q)
    split_last_core(sum_tn, q)

    if tt_ranks is not None:
        sum_tn.round_tt(rmax=tt_ranks)

    return sum_tn


def get_eigenvalues_bilinear(
    X_sol: List[torch.Tensor], A: List[torch.Tensor]
) -> torch.Tensor:
    """
    INPUT:
        X: block tensor train (last core's last dimension is p >= 1),
            solution X for tr(X.T @ A @ X) -> min_X task
        A: matrix in TT matrix format (cores with shape of (r_{i - 1}, n_i, n_i, r_i))

    OUTPUT:
        (eig_val_1, ..., eig_val_p)

    !!!USES bilinear_form function!!!
    """
    # Make TT-Matrix from TT-Vector
    x_sol = [core.unsqueeze(dim=2) for core in X_sol]
    return torch.diagonal(bilinear_form(A, x_sol, x_sol)).cpu()


def lowdim_grq(
    X: torch.Tensor, A: torch.Tensor, M: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    generalized Rayleigh quotient
    INPUT:
        X: matrix of p-columns,
        A: eigenproblem matrix
        M: gram matrix for columns if None --> eye
    OUTPUT:
       R(X) = tr{(X.T @ M @ X)^{-1} @ (X.T @ A @ X)}
    """
    if M is None:
        scalar_prod = X.T @ X
    else:
        scalar_prod = X.T @ M @ X
    return torch.trace(torch.linalg.inv(scalar_prod) @ (X.T @ A @ X))


def tt_grq(
    X: List[torch.Tensor],
    A: List[torch.Tensor],
    M: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Computes generalized Rayleigh quotient for TT vectors
    INPUT:
        X: block tensor train (last core's last dimension is p >= 1),
        M: matrix in TT matrix format (cores with shape of (r_{i - 1}, n_i, n_i, r_i))
           gram matrix for columns if None --> eye
        A: matrix in TT matrix format (cores with shape of (r_{i - 1}, n_i, n_i, r_i))
           eigenproblem matrix
    OUTPUT:
       R(X) = tr{(X.T @ M @ X)^{-1} @ (X.T @ A @ X)}
    !!!USES bilinear_form function!!!
    """
    if M is None:
        M = generate_tt_identity_matrix(
            tuple([core.shape[1] for core in X]), device=X[0].device
        )

    x = [core.unsqueeze(dim=2) for core in X]
    return torch.trace(
        torch.linalg.inv(bilinear_form(M, x, x)) @ (bilinear_form(A, x, x))
    )


def generate_sftucker_shapes(
    d_t: int,
    d_s: int,
    low: int = 60,
    low_r: int = 20,
    high: int = 100,
    high_r: int = 40,
    main_dimensions: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Returns (d_t + d_s) cores shapes
    [[r_1, n_1], ..., [r_{d_t}, n_{d_t}], [r_s, n], ..., [r_s, n]]
    such that
    r := [r_{1}, ..., r_{d_t}, r_{s}, ..., r_{s}]
    r_i <= prod(r.pop(i)), for all i in 1,...,(d_t + d_s)
    INPUT:
        d_t: amount of vanilla Tucker cores
        d_s: amount of shared Tucker cores
        low, high: n_i in [low, high]
        high_r: r_i in [low_r, high_r]
        main_dimensions: if given d_t is ignored,
    OUTPUT:
        np.array of shape
        [d_t + 1, 2]:  [[r_1, n_1], ..., [r_{d_t}, n_{d_t}], [r_s, n]]
    """
    assert d_t >= 0 and d_s >= 0
    assert d_t + d_s > 0
    if main_dimensions is None:
        main_dimensions = np.random.randint(low, high + 1, d_t + (1 if d_s else 0))
    else:
        if d_s:
            main_dimensions = np.concatenate(
                [main_dimensions[:d_t], main_dimensions[-1:]]
            )
    ranks = np.random.randint(low_r, np.minimum(main_dimensions, high_r) + 1)
    log_ranks = np.log(ranks)
    sum_log_ranks = np.sum(log_ranks[:d_t]) + log_ranks[-1] * d_s
    for i in range(d_t):
        if log_ranks[i] > 0.5 * sum_log_ranks:
            ranks[i] = np.min(
                [
                    (np.prod(ranks[:-1]) // ranks[i]) * ranks[-1] ** d_s,
                    main_dimensions[i],
                ]
            ).astype(int)
    return np.hstack([main_dimensions.reshape(-1, 1), ranks.reshape(-1, 1)])


def generate_cores_shapes(
    d: int,
    p: int,
    low: int = 2,
    low_r: int = 10,
    high: int = 5,
    high_r: int = 15,
    main_dimensions: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Returns (d) cores shapes with last core last
    dimension equal to (p)
    [1, n_1, r_1] [r_1, n_2, r_2] ... [r_{d-1}, n_d, p]
    core shapes are ensured to be minimal (see is_fullrank)
    INPUT:
        d: amount of TT cores
        p: last core last dimension
        low, high: n_i in [low, high]
        high_r: r_i in [low_r, high_r]
        main_dimensions: if given d, low, high are ignored

    OUTPUT:
        np.array of shape [d, 3]: [[1, n_1, r_1], [r_1, n_2, r_2], ..., [r_{d-1}, n_d, p]]
    """
    is_random_dims: bool = False
    if main_dimensions is None:
        assert low >= 2 and low <= high and d >= 3 and low_r >= 2
        assert np.ceil(p / high).astype(int) <= high_r
        main_dimensions = np.random.randint(low=low, high=high + 1, size=d)
        is_random_dims = True
    else:
        d = main_dimensions.shape[0]
        low = np.min(main_dimensions)
        high = np.max(main_dimensions) + 1

    ranks_p = np.zeros(d + 1).astype(int)
    ranks_p[0] = 1
    ranks_p[d] = p

    def get_possible_bounds(pos: int) -> Tuple[int]:
        """
        generates bound for r_pos in case [prev_rank, curr_dim, r_pos] according to
        full rank cores conditions (aka "minimal" condition)

        INPUT:
            pos: in [0, 1, ..., i, ..., d-1, d] -> [1, r_1, ..., r_i, ..., r_{d-1}, p]
        OUTPUT:
            r_min, r_max
        """
        if pos == 0:
            raise ValueError(
                f"Impossible to build cores_shapes with d = {d},"
                f"p = {p}, low = {low}, high = {high}, high_r = {high_r}"
            )
        prev_rank = ranks_p[pos - 1]
        curr_dim = main_dimensions[pos - 1]
        return max(low_r, np.ceil(prev_rank / curr_dim).astype(int)), min(
            curr_dim * prev_rank, high_r
        )

    def decrease_ranks(current_pos: int, prev_lower_bound: int):
        """
        Decreases ranks from current_pos till the end of array of ranks
        for the sake of full rank cores conditions (aka "minimal" condition)
        works inplace
        INPUT:
            current_pos: in [1, ..., i, ..., d-1] -> [r_1, ..., r_i, ..., r_{d-1}]
            prev_lower_bound: computed from situation:
                              [r_{current_pos}, n_{current_pos+1}, r_{current_pos+1}]
                              r_{current_pos} <= n_{current_pos+1} * r_{current_pos+1} =: prev_lower_bound
        OUTPUT:
            void
        """
        assert current_pos < d
        if ranks_p[current_pos] > prev_lower_bound:
            if current_pos == 0:
                raise ValueError(
                    f"Impossible to build cores_shapes with d = {d},"
                    f"p = {p}, low = {low}, high = {high}, high_r = {high_r}"
                )
            ranks_p[current_pos] = prev_lower_bound
            decrease_ranks(
                current_pos - 1, prev_lower_bound * main_dimensions[current_pos - 1]
            )
        else:
            return

    def increase_ranks(current_pos: int, prev_upper_bound: int):
        """
        Increases ranks from current_pos till the end of array of ranks
        for the sake of full rank cores conditions (aka "minimal" condition)
        works inplace
        INPUT:
            current_pos: in [1, ..., i, ..., d-1] -> [r_1, ..., r_i, ..., r_{d-1}]
            prev_upper_bound: computed from situation:
                              [r_{current_pos}, n_{current_pos+1}, r_{current_pos+1}]
                              prev_upper_bound := r_{current_pos+1} / n_{current_pos+1} <= r_{current_pos}
        OUTPUT:
            void
        """
        assert current_pos < d
        if ranks_p[current_pos] < prev_upper_bound:
            if current_pos == 0:
                raise ValueError(
                    f"Impossible to build cores_shapes with d = {d},"
                    f"p = {p}, low = {low}, high = {high}, high_r = {high_r}"
                )
            if prev_upper_bound > high_r:
                if not is_random_dims:
                    raise ValueError(
                        f"Impossible to build cores_shapes with d = {d},"
                        f"p = {p}, low = {low}, high = {high}, high_r = {high_r}"
                    )
                main_dimensions[current_pos] = high
                ranks_p[current_pos] = max(
                    low_r, np.ceil(ranks_p[current_pos + 1] / high).astype(int)
                )
            else:
                ranks_p[current_pos] = prev_upper_bound
            increase_ranks(
                current_pos - 1,
                max(
                    low_r,
                    np.ceil(
                        ranks_p[current_pos] / main_dimensions[current_pos - 1]
                    ).astype(int),
                ),
            )
        else:
            return

    def fix_last_core():
        """
        checks if last rank r_{d-1} in needed bounds
        """
        pos_lowb, pos_upb = get_possible_bounds(d - 1)
        last_dim = main_dimensions[d - 1]
        p_lowb, p_upb = max(low_r, np.ceil(p / last_dim).astype(int)), min(
            last_dim * p, high_r
        )
        if p_upb < pos_lowb:
            # no intersection p_low ... p_up ... pos_low ... pos_up
            # decrease ranks
            decrease_ranks(d - 1, p_upb)
        elif pos_upb < p_lowb:
            # no intersection pos_low ... pos_up ... p_low ... p_up
            # increase ranks
            increase_ranks(d - 1, p_lowb)
        else:
            # intersection exists:
            pos_set = set(range(pos_lowb, pos_upb + 1))
            p_set = set(range(p_lowb, p_upb + 1))
            ranks_p[d - 1] = np.random.choice(
                np.array(list(pos_set.intersection(p_set)))
            )
        return

    # for i, main_dim in enumerate(main_dimensions[0:-2], start=1):
    for i in range(1, d):
        curr_low_r, curr_high_r = get_possible_bounds(i)
        new_rank = np.random.randint(low=curr_low_r, high=curr_high_r + 1)
        ranks_p[i] = new_rank
    fix_last_core()
    cores_shapes = np.vstack([ranks_p[:-1], main_dimensions, ranks_p[1:]]).T
    return cores_shapes


def compute_cores_shapes(
    A: List[torch.Tensor], max_rank: int, p: int
) -> np.ndarray[np.ndarray[int]]:
    """
    INPUT:
        A: matrix in TT matrix format (cores with shape of (r_{i - 1}, n_i, n_i, r_i))
        max_rank: a desired maximum of all ranks r_i
        p: last core tensor last dimension (amount of eigenvalues/ eigenvectors to search)

    OUTPUT:
        [[r_0, n_1, r_1], [r_1, n_2, r_2], ..., [r_{d-1}, n_d, p]]
        s.t. r_i <= r_max for all i
    """

    cores_shapes = []
    core_am = len(A)
    n = A[0].shape[1]
    # m is amount of cores to get closest to max_rank with n**m <= max_rank
    m = int(np.floor(np.log(max_rank) / np.log(n)))
    assert m < core_am
    assert n * n**m >= p
    for i in range(1, m + 1):
        cores_shapes.append([n ** (i - 1), n, n ** (i)])
    n_pow_m = n**m
    cores_shapes += [[n_pow_m, n, n_pow_m]] * (core_am - m - 1)
    cores_shapes.append([n_pow_m, n, p])
    return np.array(cores_shapes)


def fractional_matrix_power(A: torch.Tensor, power: float) -> torch.Tensor:
    """
    https://discuss.pytorch.org/t/raising-a-tensor-to-a-fractional-power/93655
    INPUT:
        A -- matrix for computing A ** power SYMMETRIC
        power -- float for computing A ** power
    OUPUT:
        A ** power

    WARNING: if A is ill-conditioned the computation is instable
    """
    device = A.device
    if power == 0.0:
        return torch.eye(A.shape[0]).to(device)
    else:
        evals, evecs = torch.linalg.eigh(A)  # get eigendecomposition
        evpow = evals ** (power)
        Apow = torch.matmul(
            evecs, torch.matmul(torch.diag(evpow), torch.inverse(evecs))
        )

        return Apow


def max_eye(square_matrix: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """
    INPUT:
        square_matrix -- element of R^{n \times n}
        device -- cuda or cpu

    OUTPUT:
        tensor -- max absolute element of vectorise(square_matrix) - vectorise(identical_matrix)

    """
    return torch.max(
        torch.abs(
            square_matrix.to(device) - torch.eye(square_matrix.shape[0]).to(device)
        )
    )


def generate_tt_identity_matrix(
    shape: Tuple[int, ...],
    device: Optional[str] = None,
    requires_grad: bool = None,
) -> List[torch.Tensor]:
    """
    Generates an identity matrix in TT-matrix format
    INPUT:
        shape: (n_1, n_2, ..., n_d), d -- amount of tt-cores
        device to save tensors on
        requires_grad of tensors
    OUTPUT:
        tt_cores of identity matrix
    """
    return [
        torch.eye(n, device=device, requires_grad=requires_grad)
        .unsqueeze(0)
        .unsqueeze(-1)
        for n in shape
    ]


def tensorize(
    matrix: torch.Tensor, reshape_size: Tuple[int, ...], p: int = 0
) -> torch.Tensor:
    """
    INPUT:
        matrix -- element of R^{n \times m}
        reshape_size -- tuple of new shape
        p -- additional multiplier of last dimension
    OUTPUT:
        tensor -- IF p <= 0 : element of R^{reshape_size[0] \times reshape_size[1] \times \dotsc \times reshape_size[N - 1]}
                  ELSE : element of R^{reshape_size[0] \times reshape_size[1] \times \dotsc \times reshape_size[N - 1] * p}
    """

    if p <= 0:
        return matrix.reshape(*reshape_size)
    else:
        reshape_size = *reshape_size[:-1], reshape_size[-1] * p
        return matrix.reshape(*reshape_size)


def matricize(
    tensor: tntorch.tensor.Tensor, reshape_size: Tuple[int, int], inplace: bool = False
) -> torch.Tensor:
    """
    INPUT:
        tensor -- element of R^{n_1 \times n_2 \times \dots \times n_d}
        reshape_size -- tuple of new shape
    OUTPUT:
        tensor -- element of R^{reshape_size[0] \times reshape_size[1]}


    If p > 0: tntorch.tensor.Tensor.torch() method cannot be called
    because the last core has shape of (a, b, p) instead of (a, b, 1), so
    this function firstly makes the shape of the last core be (a, b * p, 1)
    and then invokes tntorch.tensor.Tensor.torch() method to get
    a torch.tensor to make a reshape into 'reshape_size' needed
    """
    if not inplace:
        tensor = tensor.clone()

    merge_last_core(tensor)

    return tensor.torch().reshape(*reshape_size)


def ttmat_to_mat(tt_mat: List[torch.Tensor]) -> torch.Tensor:
    """
    INPUT:
        tt_mat -- matrix in TT matrix format (cores with shape of (r_{i - 1}, n_i, m_i, r_i))
    OUTPUT:
        tensor -- element of R^{\prod n_i \times \prod m_i}
    """
    X = []
    shapes = []
    n, m = 1, 1
    for core in tt_mat:
        n *= core.shape[1]
        m *= core.shape[2]
        shapes.append(core.shape[1])
        shapes.append(core.shape[2])
        X.append(
            core.reshape(core.shape[0], core.shape[1] * core.shape[2], core.shape[3])
        )
    X = tntorch.Tensor(X).torch().reshape(shapes)

    dims_am = len(shapes) // 2
    permutation = []
    for x, y in zip(range(dims_am), range(dims_am, 2 * dims_am)):
        permutation = permutation + [x, y]
    X = X.permute(*permutation).contiguous()

    return X.reshape(n, m)


def merge_last_core(tensor: tntorch.tensor.Tensor) -> None:
    """
    INPUT:
        tensor -- element of R^{n_1 \times n_2 \times \dots \times n_d \times p},
        with last core of shape [r_{d-1}, n_d, p]
    OUTPUT:
        None

    !!!INPLACE OPERATION!!!

    Created to undo split_last_core and for `tools.matricize` func (if you do
    not understand how this function works, read `tools.matricize` docstr)
    """
    p = tensor.cores[-1].shape[-1]
    if p > 1:
        last_core = tensor.cores[-1]
        lc_shape = last_core.shape
        last_core = last_core.reshape(lc_shape[0], lc_shape[1] * lc_shape[2], 1)
        tensor.cores[-1] = last_core


def split_last_core(tensor: tntorch.tensor.Tensor, p: int):
    """
    INPUT:
        tensor -- element of R^{n_1 \times n_2 \times \dots \times n_d * p},
        with last core of shape [r_{d-1}, n_d * p, 1]
        p --- the last core third dimension (last core second dimension should be
        divisible by p)
    OUTPUT:
        None

    !!!INPLACE OPERATION!!!

    Created to undo merge_last_core

    """
    if tensor.cores[-1].shape[-1] == 1:
        last_core = tensor.cores[-1]
        lc_shape = last_core.shape

        assert lc_shape[1] % p == 0

        last_core = last_core.view(lc_shape[0], lc_shape[1] // p, p)
        tensor.cores[-1] = last_core


def objective_bilinear(
    X: tntorch.Tensor,
    H: List[torch.Tensor],
    Y: Optional[tntorch.Tensor] = None,
) -> torch.Tensor:
    """
    INPUT:
        X: block tensor train (last core's last dimension is p >= 1),
        Y: block tensor train (last core's last dimension is p >= 1) or None,
        H: matrix in TT matrix format (cores with shape of (r_{i - 1}, n_i, n_i, r_i))

    OUTPUT:
        tr(X.T @ H @ Y) if Y is not None else r(X.T @ H @ X)

    !!!USES bilinear_form function!!!

    Implements TT calcualtion of tr(X.T @ H @ Y)
    If Y is not passed --> tr(X.T @ H @ X)
    """

    if Y is not None:
        assert len(Y.cores) == len(X.cores)
        assert X.cores[-1].shape[-1] == Y.cores[-1].shape[-1]
        # make Y in TT-matrix representation
        y = [core.unsqueeze(dim=2) for core in Y.cores]
    else:
        y = None
    # make X in TT-matrix representation
    x = [core.unsqueeze(dim=2) for core in X.cores]
    mat = bilinear_form(H, x, x if y is None else y)
    if len(mat.shape) == 0:
        return mat
    else:
        return torch.trace(mat)


def objective_mat_by_vec(
    X: tntorch.tensor.Tensor,
    H: List[torch.Tensor],
    Y: Optional[tntorch.tensor.Tensor] = None,
) -> torch.Tensor:
    """
    INPUT:
        X: block tensor train (last core's last dimension is p >= 1),
        Y: block tensor train (last core's last dimension is p >= 1) or None,
        H: matrix in TT matrix format (cores with shape of (r_{i - 1}, n_i, n_i, r_i))

    OUTPUT:
        tr(X.T @ H @ Y) if Y is not None else r(X.T @ H @ X)

    !!!USES vector_by_vector_TT and matrix_by_vector_TT functions!!!

    Implements TT calcualtion of tr(X.T @ H @ Y)
    If Y is not passed --> tr(X.T @ H @ X)
    """

    anw = []
    if Y is not None:
        assert len(Y.cores) == len(X.cores)
        assert X.cores[-1].shape[-1] == Y.cores[-1].shape[-1]

    for j in range(X.cores[-1].shape[-1]):
        vec1 = X.cores[:-1] + [X.cores[-1][:, :, j].unsqueeze(dim=-1)]
        if Y is not None:
            vec2 = Y.cores[:-1] + [Y.cores[-1][:, :, j].unsqueeze(dim=-1)]
        else:
            vec2 = vec1
        anw.append(vector_by_vector_TT(matrix_by_vector_TT(H, vec1), vec2))
    return torch.sum(torch.stack(anw))


def objective_matricize(
    X: Union[tntorch.Tensor, torch.Tensor],
    H: torch.Tensor,
    reshape_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """
    INPUT:
        X: block tensor train (last core's last dimension is p >= 1),
        H: matrix in common matrix format \in R^{n \times n}

    OUTPUT:
        tr(X.T @ H @ X)

    !!!USES matricize functions!!!

    Implements simple calcualtion of tr(X.T @ H @ Y)
    """

    if reshape_size is None:
        return (X.T @ H @ X).trace()
    else:
        X = matricize(tensor=copy.deepcopy(X), reshape_size=reshape_size)
        return (X.T @ H @ X).trace()


def vector_by_vector_TT(
    vector1_cores: List[torch.Tensor], vector2_cores: List[torch.Tensor]
) -> torch.Tensor:
    """
    INPUT:
        vector1_cores --- a vector in TT format
        vector2_cores --- a vector in TT format

    OUTPUT:
        y = <x1, x2> --- a number

    This code implements Algo 4 from
    Oseledets, Ivan V. "Tensor-train decomposition."
    SIAM Journal on Scientific Computing 33.5 (2011): 2295-2317.

    !WARNING: THE FOLLOWING CODE IS A SUBOPTIMAL IMPLEMENTATION!
    """
    v = torch.kron(
        vector1_cores[0][:, 0, :].contiguous(), vector2_cores[0][:, 0, :].contiguous()
    )
    for j in range(1, vector1_cores[0].shape[1]):
        v += torch.kron(
            vector1_cores[0][:, j, :].contiguous(),
            vector2_cores[0][:, j, :].contiguous(),
        )

    for k in range(1, len(vector1_cores)):
        p_k = v @ torch.kron(
            vector1_cores[k][:, 0, :].contiguous(),
            vector2_cores[k][:, 0, :].contiguous(),
        )
        for j in range(1, vector1_cores[k].shape[1]):
            p_k += v @ torch.kron(
                vector1_cores[k][:, j, :].contiguous(),
                vector2_cores[k][:, j, :].contiguous(),
            )
        v = p_k
    return v


def matrix_by_vector_TT(
    matrix_cores: List[torch.Tensor], vector_cores: List[torch.Tensor]
) -> List[torch.Tensor]:
    """
    INPUT:
        matrix_cores --- a matrix in matrix-TT format with kernels of
        shape (r_{i - 1} x n_i x n_i x r_{i})

        vector_cores --- a vector in TT format

    OUTPUT:
        y = Mx --- a vector in TT format

    This code implements Algo 5 from
    Oseledets, Ivan V. "Tensor-train decomposition."
    SIAM Journal on Scientific Computing 33.5 (2011): 2295-2317.

    !WARNING: THE FOLLOWING CODE IS A SUBOPTIMAL IMPLEMENTATION!
    """
    Y = []
    for k in range(len(matrix_cores)):
        # k-th TT kernel of Y with dim of (r_{k-1} x n_k x r_k):
        Y_k = []
        mck_shape = matrix_cores[k].shape
        # for l in range of 0 ... n_k - 1
        for l in range(mck_shape[1]):
            temp_y = torch.kron(
                matrix_cores[k][:, l, 0, :].contiguous(),
                vector_cores[k][:, 0, :].contiguous(),
            )
            for j in range(1, mck_shape[2]):
                temp_y += torch.kron(
                    matrix_cores[k][:, l, j, :].contiguous(),
                    vector_cores[k][:, j, :].contiguous(),
                )

            Y_k.append(temp_y)

        # Y_k is (n_k x r_{k - 1} x r_{k}, so
        # transpose makes it correct core-tensor
        Y.append(torch.stack(Y_k).transpose(1, 0))

    return Y


def tt_scalar_prod(
    A: Optional[List[torch.Tensor]],
    T: List[torch.Tensor],
    K: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    """
    INPUT:
        A -- a TT-matrix
        T -- a bundle of q TT tensors (last core shape (r_{d-1}, n_d, q), q >= 1)
        K -- a bundle of q TT tensors (last core shape (r_{d-1}, n_d, p), p >= 1)
             (if None K = T)
    OUTPUT:
        T.transpose @ A @ K

    Computes T.T @ A @ K
    If K is None -> K = T
    """
    if A is None:
        A = generate_tt_identity_matrix(tuple([core.shape[1] for core in T]))
    if K is None:
        # If K = T compute lower diag
        b = [T[k].unsqueeze(dim=2) for k in range(len(T))]
        c = b
    else:
        b = [T[k].unsqueeze(dim=2) for k in range(len(T))]
        c = [K[k].unsqueeze(dim=2) for k in range(len(K))]
    obj_mat = bilinear_form(A=A, b=b, c=c)
    return obj_mat


def bilinear_form(
    A: List[torch.Tensor], b: List[torch.Tensor], c: List[torch.Tensor]
) -> torch.Tensor:
    """
    Bilinear form b^t A c; A is a TT-matrix, b and c can be batches.

    INPUT:
        A: matrix in TT matrix format (cores with shape of (r_{i - 1}, n_i, n_i, r_i))
        b: block tensor train (last core's last dimension is p >= 1) in
           FORMAT OF TT-MATRIX -- (r_{i - 1}, n_i, n_i, r_i))!,
        c: block tensor train (last core's last dimension is q >= 1) in
           FORMAT OF TT-MATRIX -- (r_{i - 1}, n_i, n_i, r_i))!,

    OUTPUT:
        A matrix of b.t @ A @ c \in R^{p \times q}

    """

    ndims = len(A)
    curr_core_1 = b[0]
    curr_core_2 = c[0]
    curr_matrix_core = A[0]
    # We enumerate the dummy dimension (that takes 1 value) with `k`.
    # You may think that using two different k would be faster, but in my
    # experience it's even a little bit slower (but neglectable in general).
    einsum_str = "aikb,cijd,ejkf->bdf"
    res = torch.einsum(einsum_str, curr_core_1, curr_matrix_core, curr_core_2)

    for core_idx in range(1, ndims):
        curr_core_1 = b[core_idx]
        curr_core_2 = c[core_idx]
        curr_matrix_core = A[core_idx]
        einsum_str = "ace,aikb,cijd,ejkf->bdf"
        res = torch.einsum(einsum_str, res, curr_core_1, curr_matrix_core, curr_core_2)

    # Squeeze to make the result a number instead of 1 x 1 for NON batch case
    # and to make the result a tensor of size
    #   batch_size
    # instead of
    #   batch_size x 1 x 1
    # in the batch case.
    return torch.squeeze(res)


def make_tridiagonal_TT_matrix(
    alpha: float, beta: float, gamma: float, l: int, device="cpu"
) -> List[torch.Tensor]:
    """
    This code implements idea from
    chapter "The QTT structure of tridiagonal Toeplitz matrices" from
    https://doi.org/10.1137/100820479

    INPUT:
        alpha: main diag value
        beta: super diag value
        gamma: sub diag value
        l >= 2: tridiag mat should be 2^l x 2^l shape (l >= 2!)
        device -- cpu or cuda
    OUTPUT:
        TT matrix representation of tridiag matrix 2^l x 2^l. Output
        cores with shape of (r_{i - 1}, n_i, n_i, r_i)
    """
    assert l >= 2

    I = torch.eye(2, device=device)
    J = torch.tensor([[0.0, 1.0], [0.0, 0.0]], device=device)

    cores = []
    # shape (1, 2, 2, 3)
    cores.append(torch.stack([I, J.T, J], dim=-1).unsqueeze(dim=0))

    # First row
    middle_core_rows = [cores[0]]
    # Second
    middle_core_rows.append(
        torch.stack(
            [torch.zeros(2, 2, device=device), J, torch.zeros(2, 2, device=device)],
            dim=-1,
        ).unsqueeze(dim=0)
    )
    # Third
    middle_core_rows.append(
        torch.stack(
            [torch.zeros(2, 2, device=device), torch.zeros(2, 2, device=device), J.T],
            dim=-1,
        ).unsqueeze(dim=0)
    )
    # shape (3, 2, 2, 3)
    middle_core = torch.vstack(middle_core_rows)

    for i in range(l - 2):
        cores.append(middle_core)

    cores.append(
        torch.stack(
            [alpha * I + beta * J + gamma * J.T, gamma * J, beta * J.T], dim=0
        ).unsqueeze(dim=-1)
    )

    return cores
