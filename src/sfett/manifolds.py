import torch
from typing import Callable, List, Tuple, Optional, Union
from abc import ABC, abstractmethod
from .tensors import TTSFTuckerTensor, TTSFTuckerTangentVector, tt_sf_tucker_scalar_prod



class BaseManifold(ABC):
    @abstractmethod
    def project(self, X: torch.Tensor) -> torch.Tensor:
        """
        INPUT:
            X -- a tensor to project onto the Manifold
        OUTPUT:
            Y -- a projection of X onto the Manifold
        """
        pass


class BaseDiffManifold(BaseManifold):
    @abstractmethod
    def project_on_tangent(self, X: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        """
        INPUT:
            X -- a tensor to project
            point -- point from Manifold where tangent space is computed

        OUTPUT:
            Y -- a projection of X onto a Manifold's
            tangent space at given point
        """
        pass

class TTSFTuckerManifold(BaseDiffManifold):

    def project(self, X: torch.Tensor) -> TTSFTuckerTensor:
        """
        INPUT:
            X -- a tensor to project onto the Manifold
        OUTPUT:
            Y -- a projection of X onto the Manifold
        """
        raise NotImplementedError()

    def project_on_tangent(
        self,
        X: Union[torch.Tensor, TTSFTuckerTensor, TTSFTuckerTangentVector],
        point: TTSFTuckerTensor,
    ) -> TTSFTuckerTangentVector:
        """
        INPUT:
            X -- a tensor to project
            point -- point from Manifold where tangent space is computed

        OUTPUT:
            Y -- a projection of X onto a Manifold's
            tangent space at given point
        """

        if isinstance(X, TTSFTuckerTensor):
            X_cl = X.clone()
            X_cl.orthogonalize(-1)

            def _f(_x: TTSFTuckerTensor) -> float:
                return tt_sf_tucker_scalar_prod(X_cl, _x)

            return self.grad(_f)(point)
        else:
            raise NotImplementedError()

    def _enforce_grad_deltas_tt_gauge_conditions(
        self, manifold_point: TTSFTuckerTensor, grad_tt_deltas: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Project tt grad deltas that define tangent space vec onto the gauge conditions
        """
        proj_deltas = []
        N = len(manifold_point.tt_cores)
        point = manifold_point.clone()
        point.orthogonalize(-1)
        left = point.clone_tt_cores()
        for i in range(N):
            left_i = left[i]
            right_r = left_i.shape[-1]
            q = left_i.reshape((-1, right_r))
            if i < N - 1:
                proj_delta = grad_tt_deltas[i]
                proj_delta = proj_delta.reshape((-1, right_r))
                proj_delta -= q @ (q.T @ proj_delta)
                proj_delta = proj_delta.reshape(left_i.shape)
            else:
                proj_delta = grad_tt_deltas[i]
            proj_deltas.append(proj_delta)
        return proj_deltas

    def _enforce_grad_regular_factors_gauge_conditions(
        self,
        manifold_point: TTSFTuckerTensor,
        grad_regular_factors_deltas: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Project tucker regular factors grad deltas that define tangent space vec onto the gauge conditions
        """
        proj_deltas = []
        N = len(manifold_point.main_dimensions)
        point = manifold_point.clone()
        for i, (delta_factor, factor) in enumerate(
            zip(grad_regular_factors_deltas, point.tucker_factors)
        ):
            point.orthogonalize(i)
            non_ortho_core = torch.einsum(
                "abc,adc->bd", point.tt_cores[i], point.tt_cores[i]
            )
            proj_delta = delta_factor
            proj_delta -= factor @ (factor.T @ proj_delta)
            proj_delta = proj_delta @ torch.linalg.inv(non_ortho_core)
            proj_deltas.append(proj_delta)
        return proj_deltas

    def _enforce_grad_shared_factor_gauge_conditions(
        self,
        manifold_point: TTSFTuckerTensor,
        grad_shared_factor_delta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project tucker regular factors grad deltas that define tangent space vec onto the gauge conditions
        """
        N = len(manifold_point.main_dimensions)
        d_s = manifold_point.shared_factors_amount
        r_s = manifold_point.tucker_ranks[-1]
        d_t = N - d_s

        point = manifold_point.clone()
        sum_non_ortho_cores = torch.zeros((r_s, r_s), device=point.device)
        for i in range(d_t, N):
            point.orthogonalize(i)
            non_ortho_core = torch.einsum(
                "abc,adc->bd", point.tt_cores[i], point.tt_cores[i]
            )
            sum_non_ortho_cores += non_ortho_core

        shared_factor = point.tucker_factors[-1]
        proj_delta = grad_shared_factor_delta
        proj_delta -= shared_factor @ (shared_factor.T @ grad_shared_factor_delta)
        proj_delta = proj_delta @ torch.linalg.inv(sum_non_ortho_cores)
        return proj_delta

    def grad(
        self, func: Callable[[TTSFTuckerTensor], float]
    ) -> Callable[[TTSFTuckerTensor], TTSFTuckerTangentVector]:
        """
        INPUT:
            func: function that takes `TTSFTuckerTensor` object as
            input and outputs a `float`.
        OUTPUT:
            Function that computes Riemannian gradient of `func` at a given point.

        Riemannian autodiff: decorator to compute gradient projected on tangent space.

        Returns a function which at a point X computes projection of the euclidian gradient
        df/dx onto the tangent space of `TTSFTucker`-tensors at point x.
        """

        def _grad(manifold_point: TTSFTuckerTensor) -> TTSFTuckerTangentVector:
            m_point = manifold_point.clone()
            m_point.round()
            manif_point_as_tan_diff = TTSFTuckerTangentVector(
                manifold_point=m_point, requires_grad=True
            )
            manif_point_diff = manif_point_as_tan_diff.construct()
            function_value = func(manif_point_diff)
            function_value.backward()

            grad_tt_cores = [
                delta_tt_core.grad
                for delta_tt_core in manif_point_as_tan_diff.delta_tt_cores
            ]
            delta_tt_cores = self._enforce_grad_deltas_tt_gauge_conditions(
                manifold_point=m_point, grad_tt_deltas=grad_tt_cores
            )

            grad_regular_factors = [
                delta_regular_factor.grad
                for delta_regular_factor in manif_point_as_tan_diff.delta_regular_factors
            ]
            delta_regular_factors = self._enforce_grad_regular_factors_gauge_conditions(
                manifold_point=m_point,
                grad_regular_factors_deltas=grad_regular_factors,
            )

            grad_shared_factor = manif_point_as_tan_diff.delta_shared_factor.grad
            delta_shared_factor = self._enforce_grad_shared_factor_gauge_conditions(
                manifold_point=m_point,
                grad_shared_factor_delta=grad_shared_factor,
            )
            return TTSFTuckerTangentVector(
                manifold_point=m_point,
                delta_tt_cores=delta_tt_cores,
                delta_regular_factors=delta_regular_factors,
                delta_shared_factor=delta_shared_factor,
                requires_grad=False,
                device=m_point.device,
            )

        return _grad

