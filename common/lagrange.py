from omnisafe.common.lagrange import Lagrange

class Lag(Lagrange):

    def update_lagrange_multiplier(self, Jc: float) -> None:

        # 重写更新函数，不让lag乘子归零
        self.lambda_optimizer.zero_grad()
        lambda_loss = self.compute_lambda_loss(Jc)
        lambda_loss.backward()
        self.lambda_optimizer.step()
        self.lagrangian_multiplier.data.clamp_(
            1e-10,
            self.lagrangian_upper_bound,
        )