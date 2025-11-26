import numpy as np
from collections import deque
import robomimic.utils.tensor_utils as TensorUtils
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torchvision.transforms as T

from einops import rearrange, repeat

from .model import *
from .model.track_patch_embed import TrackPatchEmbed
from .vilt_modules.transformer_modules import *
from .vilt_modules.rgb_modules import *
from .vilt_modules.language_modules import *
from .vilt_modules.extra_state_modules import ExtraModalityTokens
from .vilt_modules.policy_head import *
from .utils.flow_utils import ImageUnNormalize, sample_double_grid, tracks_to_video

###############################################################################
#
# A ViLT Policy without additional input
#
###############################################################################


class VanillaBCViLTPolicy(nn.Module):
    """
    Input: (o_{t-H}, ... , o_t)
    Output: a_t or distribution of a_t
    """

    def __init__(
        self,
        obs_cfg,
        img_encoder_cfg,
        language_encoder_cfg,
        extra_state_encoder_cfg,
        spatial_transformer_cfg,
        temporal_transformer_cfg,
        policy_head_cfg,
        feature_encoder_cfg,
        load_path=None,
    ):
        super().__init__()

        self._process_obs_shapes(**obs_cfg)

        # 1. encode image and features
        self._setup_image_encoder(**img_encoder_cfg)

        # 2. define spatial positional embeddings, modality embeddings, and spatial token for summary
        self._setup_spatial_positional_embeddings()

        # 3. define spatial transformer
        self._setup_spatial_transformer(**spatial_transformer_cfg)
        self.spatial_language_encoder = self._setup_language_encoder(
            output_size=self.spatial_embed_size, **language_encoder_cfg
        )

        # 4. encode extra information (e.g. gripper, joint_state)
        self.extra_encoder = self._setup_extra_state_encoder(
            extra_embedding_size=self.temporal_embed_size, **extra_state_encoder_cfg
        )

        # 5. define temporal transformer
        self._setup_temporal_transformer(**temporal_transformer_cfg)
        self.temporal_language_encoder = self._setup_language_encoder(
            output_size=self.temporal_embed_size, **language_encoder_cfg
        )

        # 6. define policy head
        self._setup_policy_head(**policy_head_cfg)

    def _process_obs_shapes(
        self, obs_shapes, num_views, extra_states, img_mean, img_std, max_seq_len
    ):
        self.img_normalizer = T.Normalize(img_mean, img_std)
        self.img_unnormalizer = ImageUnNormalize(img_mean, img_std)
        self.obs_shapes = obs_shapes
        self.num_views = num_views
        self.extra_state_keys = extra_states
        self.max_seq_len = max_seq_len

    def _setup_image_encoder(
        self, network_name, patch_size, embed_size, no_patch_embed_bias
    ):
        self.spatial_embed_size = embed_size
        self.image_encoders = []
        for _ in range(self.num_views):
            input_shape = self.obs_shapes["rgb"]
            self.image_encoders.append(
                eval(network_name)(
                    input_shape=input_shape,
                    patch_size=patch_size,
                    embed_size=self.spatial_embed_size,
                    no_patch_embed_bias=no_patch_embed_bias,
                )
            )
        self.image_encoders = nn.ModuleList(self.image_encoders)

        self.img_num_patches = sum([x.num_patches for x in self.image_encoders])

    def _setup_spatial_positional_embeddings(self):
        # setup positional embeddings
        spatial_token = nn.Parameter(
            torch.randn(1, 1, self.spatial_embed_size)
        )  # SPATIAL_TOKEN
        img_patch_pos_embed = nn.Parameter(
            torch.randn(1, self.img_num_patches, self.spatial_embed_size)
        )
        modality_embed = nn.Parameter(
            torch.randn(1, len(self.image_encoders) + 1, self.spatial_embed_size)
        )  # IMG_PATCH_TOKENS + LANGUAGE_TOKEN

        self.register_parameter("spatial_token", spatial_token)
        self.register_parameter("img_patch_pos_embed", img_patch_pos_embed)
        self.register_parameter("modality_embed", modality_embed)

        # for selecting modality embed
        modality_idx = []
        for i, encoder in enumerate(self.image_encoders):
            modality_idx += [i] * encoder.num_patches
        modality_idx += [modality_idx[-1] + 1]  # for language token
        self.modality_idx = torch.LongTensor(modality_idx)

    def _setup_language_encoder(self, network_name, **language_encoder_kwargs):
        return eval(network_name)(**language_encoder_kwargs)

    def _setup_extra_state_encoder(self, **extra_state_encoder_cfg):
        if len(self.extra_state_keys) == 0:
            return None
        else:
            return ExtraModalityTokens(
                use_joint=("joint_states" in self.extra_state_keys),
                use_gripper=("gripper_states" in self.extra_state_keys),
                use_ee=("ee_states" in self.extra_state_keys),
                **extra_state_encoder_cfg
            )

    def _setup_spatial_transformer(
        self,
        num_layers,
        num_heads,
        head_output_size,
        mlp_hidden_size,
        dropout,
        spatial_downsample,
        spatial_downsample_embed_size,
        use_language_token=True,
    ):
        self.spatial_transformer = TransformerDecoder(
            input_size=self.spatial_embed_size,
            num_layers=num_layers,
            num_heads=num_heads,
            head_output_size=head_output_size,
            mlp_hidden_size=mlp_hidden_size,
            dropout=dropout,
        )

        if spatial_downsample:
            self.temporal_embed_size = spatial_downsample_embed_size
            self.spatial_downsample = nn.Linear(
                self.spatial_embed_size, self.temporal_embed_size
            )
        else:
            self.temporal_embed_size = self.spatial_embed_size
            self.spatial_downsample = nn.Identity()

        self.spatial_transformer_use_text = use_language_token

    def _setup_temporal_transformer(
        self,
        num_layers,
        num_heads,
        head_output_size,
        mlp_hidden_size,
        dropout,
        use_language_token=True,
    ):
        self.temporal_position_encoding_fn = SinusoidalPositionEncoding(
            input_size=self.temporal_embed_size
        )

        self.temporal_transformer = TransformerDecoder(
            input_size=self.temporal_embed_size,
            num_layers=num_layers,
            num_heads=num_heads,
            head_output_size=head_output_size,
            mlp_hidden_size=mlp_hidden_size,
            dropout=dropout,
        )
        self.temporal_transformer_use_text = use_language_token

        action_cls_token = nn.Parameter(torch.zeros(1, 1, self.temporal_embed_size))
        nn.init.normal_(action_cls_token, std=1e-6)
        self.register_parameter("action_cls_token", action_cls_token)

    def _setup_policy_head(self, network_name, **policy_head_kwargs):
        self.late_fusion = policy_head_kwargs.pop("late_fusion")
        policy_head_kwargs["input_size"] = self.temporal_embed_size

        action_shape = policy_head_kwargs["output_size"]
        self.act_shape = action_shape
        self.out_shape = np.prod(action_shape)
        policy_head_kwargs["output_size"] = self.out_shape
        self.policy_head = eval(network_name)(**policy_head_kwargs)

    @torch.no_grad()
    def __normalize_img__(self, rgb):
        rgb = self.img_normalizer(rgb / 255.0)
        return rgb

    def spatial_encode(self, obs, extra_states, task_emb):
        """
        Encode the images separately in the videos along the spatial axis.
        Args:
            obs: (b, view, time, channel, height, width)
            extra_states: {k: b t n}
            task_emb: b e
        Returns: out: (b t 2+num_extra c)
        """
        # 1. encode image
        img_encoded = []
        for view_idx in range(self.num_views):
            img_encoded.append(
                rearrange(
                    TensorUtils.time_distributed(
                        obs[:, view_idx, ...], self.image_encoders[view_idx]
                    ),
                    "b t c h w -> b t (h w) c",
                )
            )  # (b, t, num_patches, c)

        img_encoded = torch.cat(img_encoded, -2)  # (b, t, 2*num_patches, c)
        img_encoded += self.img_patch_pos_embed.unsqueeze(0)  # (b, t, 2*num_patches, c)
        B, T = img_encoded.shape[:2]

        # 2. append language token and add modality embeddings
        text_encoded = self.spatial_language_encoder(task_emb)
        text_encoded = text_encoded.view(B, 1, 1, -1).expand(-1, T, -1, -1)

        encoded = torch.cat([img_encoded, text_encoded], -2)
        encoded += self.modality_embed[None, :, self.modality_idx[:], :]

        # 3. add spatial token
        spatial_token = self.spatial_token.unsqueeze(0).expand(
            B, T, -1, -1
        )  # (b, t, 1, c)
        encoded = torch.cat(
            [spatial_token, img_encoded], -2
        )  # (b, t, 2*num_img_patch + 2, c)

        # 4. pass through transformer
        encoded = rearrange(
            encoded, "b t n c -> (b t) n c"
        )  # (b*t, 2*num_img_patch + num_feature_patch, c)
        out = self.spatial_transformer(encoded)
        out = out[:, 0]  # extract spatial token as summary at o_t
        out = self.spatial_downsample(out).view(B, T, 1, -1)  # (b, t, 1, c')

        # 5. encode extra states
        if self.extra_encoder is None:
            extra = None
        else:
            extra = self.extra_encoder(extra_states)  # (B, T, num_extra, c')

        # 6. add action token and language token
        text_encoded_ = self.temporal_language_encoder(task_emb)
        text_encoded_ = text_encoded_.view(B, 1, 1, -1).expand(-1, T, -1, -1)
        action_cls_token = self.action_cls_token.unsqueeze(0).expand(
            B, T, -1, -1
        )  # (b, t, 1, c')
        out_seq = [action_cls_token, text_encoded_, out]

        # 7. add extra state token
        if self.extra_encoder is not None:
            out_seq.append(extra)
        output = torch.cat(out_seq, -2)  # (b, t, 2 or 3 + num_extra, c')

        return output

    def temporal_encode(self, x):
        """
        Args:
            x: b, t, num_modality, c
        Returns:
        """
        pos_emb = self.temporal_position_encoding_fn(x)  # (t, c)
        x = x + pos_emb.unsqueeze(1)  # (b, t, 2+num_extra, c)
        sh = x.shape
        self.temporal_transformer.compute_mask(x.shape)

        x = TensorUtils.join_dimensions(x, 1, 2)  # (b, t*num_modality, c)
        x = self.temporal_transformer(x)
        x = x.reshape(*sh)  # (b, t, num_modality, c)
        return x[:, :, 0]  # (b, c)

    def forward(self, obs, extra_states, task_embed):
        """
        Return feature and info.
        Args:
            obs: b v t c h w        (0-1)
            extra_states: {k: b t e}
        """
        x = self.spatial_encode(
            obs, extra_states, task_embed
        )  # x: (b, t, 2+num_extra, c)
        x = self.temporal_encode(x)  # (b, t, c)

        dist = self.policy_head(
            x
        )  # only use the current timestep feature to predict action
        return dist

    def forward_loss(self, obs, extra_states, task_embed, action):
        """
        Args:
            obs: (b, views, times, channel, height, width)
            extra_states: {k: (b, time, state_dim)}
            action: (b, time, act_dim)
        """
        obs = self.__normalize_img__(obs)
        dist = self.forward(obs, extra_states, task_embed)
        loss = self.policy_head.loss_fn(dist, action, reduction="mean")

        ret_dict = {
            "bc_loss": loss.sum().item(),
        }

        if not self.policy_head.deterministic:
            # pseudo loss
            sampled_action = dist.sample().detach()
            mse_loss = F.mse_loss(sampled_action, action)
            ret_dict["pseudo_sampled_action_mse_loss"] = mse_loss.sum().item()

        ret_dict["loss"] = ret_dict["bc_loss"]
        return loss.sum(), ret_dict

    def act(self, obs, extra_states, task_embed):
        """
        Args:
            obs: (b, views,  channel, height, width)
            extra_states: {k: (b, state_dim,)}
            task_embed: (b, task_embed_dim)
        Return:
            action: [b, act_dim] numpy type
        """
        self.eval()
        B = obs.shape[0]

        # 1. preprosess
        obs = self.__normalize_img__(obs)
        obs = obs.unsqueeze(2)  # (b, views, 1, channel, height, width)
        extra_states = {
            k: v.unsqueeze(1) for k, v in extra_states.items()
        }  # {k: (b, 1, n)}

        # 2. update latent buffer and inference the action
        with torch.no_grad():
            latent = self.spatial_encode(obs, extra_states, task_embed)
            self.latent_buffer = self.push_to_buffer(self.latent_buffer, latent)
            temporal_input = self.get_input_from_buffer(self.latent_buffer)
            action_token = self.temporal_encode(temporal_input)[
                :, -1
            ]  # the latest timestep action token
            action = (
                self.policy_head(action_token).detach().cpu()
            )  # project to robot control
            action = torch.clamp(action, -1, 1)
        return action.float().numpy()

    def push_to_buffer(self, buffer: list, element):
        if len(buffer) > self.max_seq_len:
            buffer.pop(0)  # remove the earliest element
        buffer.append(element)  # add the latest element

        return buffer

    def get_input_from_buffer(self, latent_buffer: list):
        """
        Args:
        latent_buffer: list of [b, 1, modality, c]
        Return:
        temporal_input: [b, max_seq_len, modality, c]
        """
        temporal_input = torch.concat(latent_buffer, dim=1)

        return temporal_input

    def reset(self, device):
        # define latent buffer and initialize it with all 0 paddings
        # self.latent_buffer = [torch.zeros(1, 1, 4, self.temporal_embed_size).to(device) for i in range(self.max_seq_len)]
        self.latent_buffer = []

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"))
