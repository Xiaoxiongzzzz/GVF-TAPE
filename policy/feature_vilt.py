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
# A ViLT Policy
#
###############################################################################


class FeatureViLTPolicy(nn.Module):
    """
    Input: feature (b, time, 18, 48, 64)
    Output: a_t or distribution of a_t
    """

    def __init__(
        self,
        obs_cfg,
        img_encoder_cfg,
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
        self._setup_feature_encoder(**feature_encoder_cfg)

        # 3. define spatial positional embeddings, modality embeddings, and spatial token for summary
        self._setup_spatial_positional_embeddings()

        # 4. define spatial transformer
        self._setup_spatial_transformer(**spatial_transformer_cfg)

        ### 5. encode extra information (e.g. gripper, joint_state)
        self.extra_encoder = self._setup_extra_state_encoder(
            extra_embedding_size=self.temporal_embed_size, **extra_state_encoder_cfg
        )

        # 7. define temporal transformer
        self._setup_temporal_transformer(**temporal_transformer_cfg)

        # 8. define policy head
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

    def _setup_feature_encoder(
        self, network_name, patch_size, embed_size, no_patch_embed_bias
    ):
        input_shape = self.obs_shapes["feature"]

        self.feature_encoder = eval(network_name)(
            input_shape=input_shape,
            patch_size=patch_size,
            embed_size=self.spatial_embed_size,
            no_patch_embed_bias=no_patch_embed_bias,
        )

        self.feature_num_patches = self.feature_encoder.num_patches

    def _setup_spatial_positional_embeddings(self):
        # setup positional embeddings
        spatial_token = nn.Parameter(
            torch.randn(1, 1, self.spatial_embed_size)
        )  # SPATIAL_TOKEN
        feature_patch_pos_embed = nn.Parameter(
            torch.randn(1, self.feature_num_patches, self.spatial_embed_size)
        )
        # TODO:
        modality_embed = nn.Parameter(
            torch.randn(1, 1, self.spatial_embed_size)
        )  # FEATURE_TOKENS

        self.register_parameter("spatial_token", spatial_token)
        self.register_parameter("feature_patch_pos_embed", feature_patch_pos_embed)
        self.register_parameter("modality_embed", modality_embed)

        # for selecting modality embed
        modality_idx = [0] * self.feature_num_patches  # for feature modality embeding
        self.modality_idx = torch.LongTensor(modality_idx)

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
        if self.late_fusion:
            policy_head_kwargs["input_size"] = (
                self.temporal_embed_size
                + self.feature_num_patches * self.spatial_embed_size
            )
        else:
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

    def spatial_encode(self, feature, extra_states, return_feature=False, debug=False):
        """
        Encode the images separately in the videos along the spatial axis.
        Args:
            feature: (b,times, channel, height, width)
            extra_states: {k: b t n}
        Returns: out: (b t 2+num_extra c), feature: (b, patches, channel)
        """
        B, T = feature.shape[:2]
        # 1. encode feature
        feature = rearrange(feature, "b t c h w -> (b t) c h w")
        feature_encoded = self.feature_encoder(feature)
        feature_encoded = rearrange(
            feature_encoded, "(b t) c h w ->  b t (h w) c", b=B, t=T
        )  # （b, patch, channel）
        feature_encoded += self.feature_patch_pos_embed.unsqueeze(0)

        # 2. add spatial [cls] token
        spatial_token = self.spatial_token.unsqueeze(0).expand(
            B, T, -1, -1
        )  # (b, t, 1, c)
        encoded = torch.cat(
            [spatial_token, feature_encoded], -2
        )  # (b, 1+num_feature_patch, c)

        # 3. pass through spatial transformer
        encoded = rearrange(encoded, "b t n c -> (b t) n c")
        out = self.spatial_transformer(encoded)
        out = out[:, 0]  # extract spatial token as summary at o_t
        out = self.spatial_downsample(out).view(B, T, 1, -1)  # (b, t, 1, c')

        # 4. encode extra states
        if self.extra_encoder is None:
            extra = None
        else:
            extra = self.extra_encoder(extra_states)  # (b, t, num_extra, c')

        # 5. add action token
        action_cls_token = self.action_cls_token.unsqueeze(0).expand(
            B, T, -1, -1
        )  # (b, 1, c')
        out_seq = [action_cls_token, out]

        # 6. add extra state token
        if self.extra_encoder is not None:
            out_seq.append(extra)
        output = torch.cat(out_seq, -2)  # (b, t, 1 + 1 + num_extra, c')

        if return_feature:
            output = (output, feature_encoded)  # feature: b, patches, c

        return output

    def temporal_encode(self, x):
        """
        Args:
            x: b, t, n, c
        Returns:
        """
        pos_emb = self.temporal_position_encoding_fn(x)  # (t, c)
        x = x + pos_emb.unsqueeze(1)  # (b, t, n, c )
        sh = x.shape
        self.temporal_transformer.compute_mask(x.shape)

        x = TensorUtils.join_dimensions(x, 1, 2)  # b, t*n, c
        x = self.temporal_transformer(x)
        x = x.reshape(*sh)
        return x[:, :, 0]  # action token: b, t, c

    def forward(self, feature, extra_states):
        """
        Return feature and info.
        Args:
            feature: b t c h w
            extra_states: {k: b t e}
        Return:
            dist: b, t, action_dim
        """
        x, feature_encoded = self.spatial_encode(
            feature, extra_states, return_feature=True
        )  # x: (b, t, 2+num_extra, c)
        x = self.temporal_encode(x)  # (b, t, c)

        if self.late_fusion:
            feature_encoded = torch.flatten(feature_encoded, start_dim=-2)  # b, t, p#c
            x = torch.cat([x, feature_encoded], dim=-1)  # (b, channel)

        dist = self.policy_head(x)  # project to robot control b, t, action_dim
        return dist

    def forward_loss(self, feature, extra_states, action):
        """
        Args:
            feature: (b, t, channel, h, w)
            extra_states: {k: (b, time, state_dim)}
            action: (b act_dim)
        """
        dist = self.forward(feature, extra_states)
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

    def act(self, feature, extra_states):
        """
        Args:
            obs: (b, views, height, width, channel)
            feature: (b, channel, h, w)
            extra_states: {k: (b, state_dim,)}
        Return:
            action: [b, act_dim] numpy type
        """
        self.eval()
        B = feature.shape[0]

        # 1. update buffer and get input
        self.extra_state_buffer = self.push_to_buffer(
            self.extra_state_buffer, extra_states
        )

        _, extra_state_input = self.get_input_from_buffer(
            self.visual_observation_buffer, self.extra_state_buffer
        )

        # 2. inference the action
        with torch.no_grad():
            action = (
                self.forward(feature, extra_state_input).detach().cpu()
            )  # (b, act_dim)

        return action.float().numpy()

    def push_to_buffer(self, buffer: list, element):
        buffer.pop(0)  # remove the earliest element
        buffer.append(element)  # add the latest element

        return buffer

    def get_input_from_buffer(self, visual_obs_buffer: list, extra_state_buffer: list):
        """
        Args:
        visual_obs_buffer: list of [b, v, c, h, w]
        extra_state_buffer: list of {k: b, n}
        Return:
        visual_obs_input: [b, v, t, c, h, w]
        extra_state_input: {k: b, t, n}
        """
        visual_obs_input = torch.stack(visual_obs_buffer, dim=2)
        extra_state_input = {
            k: torch.stack([element[k] for element in extra_state_buffer], dim=1)
            for k in extra_state_buffer[0].keys()
        }

        return visual_obs_input, extra_state_input

    def reset(self, device):
        # define buffer queue for visual observation and extra_states, initialize them with all padding
        self.visual_observation_buffer = [
            torch.zeros(
                1,
                self.num_views,
                self.obs_shapes["rgb"][0],
                self.obs_shapes["rgb"][1],
                self.obs_shapes["rgb"][2],
            ).to(device)
            for i in range(self.max_seq_len)
        ]
        extra_state_padding = {
            "joint_states": torch.zeros(
                1,
                7,
            ),
            "gripper_states": torch.zeros(
                1,
                2,
            ),
        }
        extra_state_padding = {k: v.to(device) for k, v in extra_state_padding.items()}
        self.extra_state_buffer = [extra_state_padding for i in range(self.max_seq_len)]

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"))
