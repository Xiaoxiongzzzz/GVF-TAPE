import torch
import torch.nn as nn


# Time = 0 corresponds to prior distribution(Nomal Gaussian), Time = 1 corresponds to data distribution
class FlowIKModel(nn.Module):
    def __init__(
        self,
        resnet,
        device,
        output_dim=7,
        sub_task_dim=12,
        hidden_dim=256,
        loss_type="L2",
    ):
        """
        Input:
            resnet: nn.module
            output_dim: the dim we need (e.g. 7dims for robot action)
        """

        super(FlowIKModel, self).__init__()
        self.resnet = resnet
        self.output_dim = output_dim
        self.device = device
        self.loss_type = loss_type

        assert loss_type in [
            "L1",
            "L2",
        ], "Invalid loss type, only support L1 and L2 loss"

        self.pred_head = nn.Sequential(
            nn.Linear(512 + output_dim + 32, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.time_embed_layer = nn.Linear(1, 32)
        self.regularization_head = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, sub_task_dim),
        )

    def train_loss(self, x, cond, regularization_term=None):
        """
        Input:
            x: action [b, 7]
            cond: e.g. image
        """

        noise = torch.randn_like(x).to(self.device)

        t = torch.rand(x.shape[0], 1).to(self.device)
        noisy_x = (1 - t) * noise + t * x

        time_embed = self.time_embed_layer(t)

        cond = self.resnet(cond)
        cond_input = torch.cat([cond, noisy_x, time_embed], dim=1)

        pred_velocity = self.pred_head(cond_input)
        loss_dict = {}
        if self.loss_type == "L2":
            loss_dict["velocity_loss"] = torch.mean((pred_velocity - (x - noise)) ** 2)
        if self.loss_type == "L1":
            loss_dict["velocity_loss"] = torch.mean(
                torch.abs(pred_velocity - (x - noise))
            )

        if regularization_term is not None:
            loss_dict["regularization_loss"] = torch.mean(
                (self.regularization_head(cond) - regularization_term) ** 2
            )

        return loss_dict

    def sample(self, cond, sample_step):
        B = cond.shape[0]
        noise = torch.randn([B, self.output_dim]).to(self.device)
        noisy_x = noise

        delta_t = 1 / sample_step
        time_step_list = [i * delta_t for i in range(sample_step)]

        cond = self.resnet(cond)
        for t in time_step_list:
            t = torch.Tensor([t] * B).view(-1, 1).to(self.device)
            time_embed = self.time_embed_layer(t)
            cond_input = torch.cat([cond, noisy_x, time_embed], dim=1)
            pred_velocity = self.pred_head(cond_input)
            noisy_x = noisy_x + pred_velocity * delta_t

        x_1 = noisy_x

        return x_1

    def sub_task_predict(self, cond):
        cond = self.resnet(cond)
        sub_task_pred = self.regularization_head(cond)
        return sub_task_pred


# if __name__ == "__main__":
#     from policy.ik_model.resnet import ResNet18
#     device = torch.device("cuda")
#     resnet = ResNet18(input_dim=6).to(device)
#     ik_model = FlowIKModel(resnet=resnet, device=device).to(device)
#     x = torch.randn(10, 7).to(device)
#     cond = torch.randn(10, 6, 128, 128).to(device)
#     loss = ik_model.train_loss(x, cond)
#     x = ik_model.sample(cond, sample_step=10)
