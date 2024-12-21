import functools
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from concurrent.futures import ThreadPoolExecutor


################################################################################
# Example configuration with reduced complexity and shorter chunks
################################################################################
class ExampleConfig:
    def __init__(self):
        # Basic model config
        self.use_causal_conv = False
        self.pad_mode = "zero"
        self.norm_type = "time_group_norm"

        # Overlapping-chunk config
        self.chunk_length_s = 0.32  # e.g., 0.32s chunks
        self.overlap = 0.15        # e.g., 15% overlap

        # Audio / model dimension config
        self.audio_channels = 1
        self.sampling_rate = 24000
        self.num_filters = 32  # reduced from bigger numbers like 64 or 128
        self.kernel_size = 5
        self.dilation_growth_rate = 2
        self.residual_kernel_size = 3
        self.num_residual_layers = 2  # fewer residual layers
        self.compress = 4

        # LSTM + hidden dimension
        self.num_lstm_layers = 1     # fewer LSTM layers
        self.hidden_size = 256       # reduced from e.g. 512
        self.last_kernel_size = 3

        # Upsampling ratios (decoder/encoder)
        # These define how many times we downsample/upsample
        self.upsampling_ratios = [2, 2, 2]  # example: 8x total

        # Transpose conv trim
        self.trim_right_ratio = 0.0

        # Normalization
        self.normalize = True

        # Bandwidth settings
        self.target_bandwidths = [6.0]  # e.g., 6kbps
        # (If you want multiple possible bandwidths, e.g., [6.0, 12.0, 24.0])

        # Codebook
        self.codebook_size = 1024
        self.codebook_dim = 128

        # We set half precision to speed up on Apple Silicon if your library supports it
        self.precision = 16


################################################################################
# Custom LSTM Metal kernel (unchanged)
################################################################################
_lstm_kernel = mx.fast.metal_kernel(
    name="lstm",
    input_names=["x", "h_in", "cell", "hidden_size", "time_step", "num_time_steps"],
    output_names=["hidden_state", "cell_state"],
    header="""
    template <typename T>
    T sigmoid(T x) {
        auto y = 1 / (1 + metal::exp(-metal::abs(x)));
        return (x < 0) ? 1 - y : y;
    }
    """,
    source="""
        uint b = thread_position_in_grid.x;
        uint d = hidden_size * 4;

        uint elem = b * d + thread_position_in_grid.y;
        uint index = elem;
        uint x_index = b * num_time_steps * d + time_step * d + index;

        auto i = sigmoid(h_in[index] + x[x_index]);
        index += hidden_size;
        x_index += hidden_size;
        auto f = sigmoid(h_in[index] + x[x_index]);
        index += hidden_size;
        x_index += hidden_size;
        auto g = metal::precise::tanh(h_in[index] + x[x_index]);
        index += hidden_size;
        x_index += hidden_size;
        auto o = sigmoid(h_in[index] + x[x_index]);

        cell_state[elem] = f * cell[elem] + i * g;
        hidden_state[elem] = o * metal::precise::tanh(cell_state[elem]);
    """,
)

def lstm_custom(x, h_in, cell, time_step):
    """
    x: shape (B, T, 4*H)
    h_in: shape (B, 4*H)
    cell: shape (B, H)
    time_step: int
    """
    assert x.ndim == 3, "Input to LSTM must have 3 dimensions."
    out_shape = cell.shape
    return _lstm_kernel(
        inputs=[x, h_in, cell, out_shape[-1], time_step, x.shape[-2]],
        output_shapes=[out_shape, out_shape],
        output_dtypes=[h_in.dtype, h_in.dtype],
        grid=(x.shape[0], h_in.size // 4, 1),
        threadgroup=(256, 1, 1),
    )


################################################################################
# Simple LSTM module
################################################################################
class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.Wx = mx.zeros((4 * hidden_size, input_size))
        self.Wh = mx.zeros((4 * hidden_size, hidden_size))
        self.bias = mx.zeros((4 * hidden_size,)) if bias else None

    def __call__(self, x, hidden=None, cell=None):
        """
        x: shape (B, T, input_size)
        hidden: shape (B, 4*H), cell: shape (B, H)
        """
        if self.bias is not None:
            x = mx.addmm(self.bias, x, self.Wx.T)  # shape (B, T, 4*H)
        else:
            x = x @ self.Wx.T

        all_hidden = []
        B = x.shape[0]
        cell = cell or mx.zeros((B, self.hidden_size), x.dtype)
        for t in range(x.shape[-2]):
            if hidden is None:
                hidden = mx.zeros((B, self.hidden_size * 4), x.dtype)
            else:
                hidden = hidden @ self.Wh.T
            hidden, cell = lstm_custom(x, hidden, cell, t)
            all_hidden.append(hidden)

        return mx.stack(all_hidden, axis=-2)


################################################################################
# EncodecConv1d, EncodecConvTranspose1d
################################################################################
class EncodecConv1d(nn.Module):
    """Conv1d with asymmetric or causal padding and normalization."""
    def __init__(
        self,
        config,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        self.config = config
        self.causal = config.use_causal_conv
        self.pad_mode = config.pad_mode
        self.norm_type = config.norm_type

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, dilation=dilation
        )
        if self.norm_type == "time_group_norm":
            self.norm = nn.GroupNorm(1, out_channels, pytorch_compatible=True)

        self.stride = stride
        # Effective kernel size with dilations
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.padding_total = kernel_size - stride

    def _get_extra_padding_for_conv1d(self, hidden_states: mx.array) -> mx.array:
        length = hidden_states.shape[1]
        n_frames = (length - self.kernel_size + self.padding_total) / self.stride + 1
        n_frames = int(math.ceil(n_frames)) - 1
        ideal_length = n_frames * self.stride + self.kernel_size - self.padding_total
        return ideal_length - length

    def _pad1d(
        self,
        hidden_states: mx.array,
        paddings: Tuple[int, int],
        mode: str = "zero",
        value: float = 0.0,
    ):
        if mode != "reflect":
            return mx.pad(
                hidden_states, paddings, mode="constant", constant_values=value
            )

        length = hidden_states.shape[1]
        prefix = hidden_states[:, 1 : paddings[0] + 1][:, ::-1]
        suffix = hidden_states[:, max(length - (paddings[1] + 1), 0) : -1][:, ::-1]
        return mx.concatenate([prefix, hidden_states, suffix], axis=1)

    def __call__(self, hidden_states):
        # Optional half-precision for conv input
        if getattr(self.config, "precision", 32) == 16:
            hidden_states = hidden_states.astype(mx.float16)

        extra_padding = self._get_extra_padding_for_conv1d(hidden_states)

        if self.causal:
            # Left padding for causal
            hidden_states = self._pad1d(
                hidden_states, (self.padding_total, extra_padding), mode=self.pad_mode
            )
        else:
            # Asymmetric padding for odd strides
            padding_right = self.padding_total // 2
            padding_left = self.padding_total - padding_right
            hidden_states = self._pad1d(
                hidden_states,
                (padding_left, padding_right + extra_padding),
                mode=self.pad_mode,
            )

        hidden_states = self.conv(hidden_states)

        if self.norm_type == "time_group_norm":
            hidden_states = self.norm(hidden_states)

        return hidden_states


class EncodecConvTranspose1d(nn.Module):
    """ConvTranspose1d with asymmetric or causal padding and normalization."""
    def __init__(
        self,
        config,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
    ):
        super().__init__()
        self.config = config
        self.causal = config.use_causal_conv
        self.trim_right_ratio = config.trim_right_ratio
        self.norm_type = config.norm_type

        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride)
        if config.norm_type == "time_group_norm":
            self.norm = nn.GroupNorm(1, out_channels, pytorch_compatible=True)

        self.padding_total = kernel_size - stride

    def __call__(self, hidden_states):
        # Optional half-precision
        if getattr(self.config, "precision", 32) == 16:
            hidden_states = hidden_states.astype(mx.float16)

        hidden_states = self.conv(hidden_states)

        if self.norm_type == "time_group_norm":
            hidden_states = self.norm(hidden_states)

        if self.causal:
            padding_right = math.ceil(self.padding_total * self.trim_right_ratio)
        else:
            padding_right = self.padding_total // 2

        padding_left = self.padding_total - padding_right
        end = hidden_states.shape[1] - padding_right
        hidden_states = hidden_states[:, padding_left:end, :]
        return hidden_states


################################################################################
# EncodecLSTM and EncodecResnetBlock
################################################################################
class EncodecLSTM(nn.Module):
    def __init__(self, config, dimension):
        super().__init__()
        self.config = config
        self.lstm = [LSTM(dimension, dimension) for _ in range(config.num_lstm_layers)]

    def __call__(self, hidden_states):
        if getattr(self.config, "precision", 32) == 16:
            hidden_states = hidden_states.astype(mx.float16)

        h = hidden_states
        for lstm in self.lstm:
            h = lstm(h)
        return h + hidden_states


class EncodecResnetBlock(nn.Module):
    """
    Residual block from SEANet model as used by EnCodec.
    """
    def __init__(self, config, dim: int, dilations: List[int]):
        super().__init__()
        self.config = config
        kernel_sizes = (config.residual_kernel_size, 1)
        if len(kernel_sizes) != len(dilations):
            raise ValueError("Number of kernel sizes should match number of dilations")

        hidden = dim // config.compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [nn.ELU()]
            block += [
                EncodecConv1d(config, in_chs, out_chs, kernel_size, dilation=dilation)
            ]
        self.block = block

        if getattr(config, "use_conv_shortcut", True):
            self.shortcut = EncodecConv1d(config, dim, dim, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def __call__(self, hidden_states):
        if getattr(self.config, "precision", 32) == 16:
            hidden_states = hidden_states.astype(mx.float16)

        residual = hidden_states
        for layer in self.block:
            hidden_states = layer(hidden_states)

        return self.shortcut(residual) + hidden_states


################################################################################
# EncodecEncoder, EncodecDecoder
################################################################################
class EncodecEncoder(nn.Module):
    """SEANet encoder as used by EnCodec."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        model = [
            EncodecConv1d(
                config, config.audio_channels, config.num_filters, config.kernel_size
            )
        ]
        scaling = 1

        # Downsample stages
        for ratio in reversed(config.upsampling_ratios):
            current_scale = scaling * config.num_filters
            for j in range(config.num_residual_layers):
                model += [
                    EncodecResnetBlock(
                        config, current_scale, [config.dilation_growth_rate**j, 1]
                    )
                ]
            model += [nn.ELU()]
            model += [
                EncodecConv1d(
                    config,
                    current_scale,
                    current_scale * 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                )
            ]
            scaling *= 2

        # LSTM mid-block
        model += [EncodecLSTM(config, scaling * config.num_filters)]
        model += [nn.ELU()]

        # Final conv to get to "hidden_size"
        model += [
            EncodecConv1d(
                config,
                scaling * config.num_filters,
                config.hidden_size,
                config.last_kernel_size,
            )
        ]
        self.layers = model

    def __call__(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class EncodecDecoder(nn.Module):
    """SEANet decoder as used by EnCodec."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        scaling = int(2 ** len(config.upsampling_ratios))
        model = [
            EncodecConv1d(
                config,
                config.hidden_size,
                scaling * config.num_filters,
                config.kernel_size,
            )
        ]
        model += [EncodecLSTM(config, scaling * config.num_filters)]

        # Upsampling stages
        for ratio in config.upsampling_ratios:
            current_scale = scaling * config.num_filters
            model += [nn.ELU()]
            model += [
                EncodecConvTranspose1d(
                    config,
                    current_scale,
                    current_scale // 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                )
            ]
            for j in range(config.num_residual_layers):
                model += [
                    EncodecResnetBlock(
                        config, current_scale // 2, (config.dilation_growth_rate**j, 1)
                    )
                ]
            scaling //= 2

        model += [nn.ELU()]
        model += [
            EncodecConv1d(
                config,
                config.num_filters,
                config.audio_channels,
                config.last_kernel_size,
            )
        ]
        self.layers = model

    def __call__(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


################################################################################
# EncodecVectorQuantization, EncodecEuclideanCodebook
################################################################################
class EncodecEuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = mx.zeros((config.codebook_size, config.codebook_dim))

    def quantize(self, hidden_states: mx.array):
        # hidden_states: (N, D)
        if getattr(self.config, "precision", 32) == 16:
            hidden_states = hidden_states.astype(mx.float16)

        embed = self.embed.T  # (D, K)
        scaled_states = hidden_states.square().sum(axis=1, keepdims=True)
        dist = -(
            scaled_states
            - 2 * hidden_states @ embed
            + embed.square().sum(axis=0, keepdims=True)
        )
        embed_ind = dist.argmax(axis=-1)
        return embed_ind

    def encode(self, hidden_states: mx.array):
        shape = hidden_states.shape
        hidden_states = hidden_states.reshape((-1, shape[-1]))
        embed_ind = self.quantize(hidden_states)
        embed_ind = embed_ind.reshape(*shape[:-1])
        return embed_ind

    def decode(self, embed_ind: mx.array):
        return self.embed[embed_ind]


class EncodecVectorQuantization(nn.Module):
    """Vector quantization implementation."""
    def __init__(self, config):
        super().__init__()
        self.codebook = EncodecEuclideanCodebook(config)

    def encode(self, hidden_states: mx.array):
        return self.codebook.encode(hidden_states)

    def decode(self, embed_ind: mx.array):
        return self.codebook.decode(embed_ind)


################################################################################
# EncodecResidualVectorQuantizer
################################################################################
class EncodecResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.codebook_size = config.codebook_size
        hop_length = np.prod(config.upsampling_ratios)
        self.frame_rate = math.ceil(config.sampling_rate / hop_length)
        self.num_quantizers = int(
            1000 * config.target_bandwidths[-1] // (self.frame_rate * 10)
        )
        self.layers = [
            EncodecVectorQuantization(config) for _ in range(self.num_quantizers)
        ]

    def get_num_quantizers_for_bandwidth(
        self, bandwidth: Optional[float] = None
    ) -> int:
        bw_per_q = math.log2(self.codebook_size) * self.frame_rate
        num_quantizers = self.num_quantizers
        if bandwidth is not None and bandwidth > 0.0:
            num_quantizers = int(max(1, math.floor(bandwidth * 1000 / bw_per_q)))
        return num_quantizers

    def encode(self, embeddings: mx.array, bandwidth: Optional[float] = None) -> mx.array:
        num_quantizers = self.get_num_quantizers_for_bandwidth(bandwidth)
        residual = embeddings
        all_indices = []
        for layer in self.layers[:num_quantizers]:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            # In-place subtraction if safe
            residual -= quantized
            all_indices.append(indices)
        out_indices = mx.stack(all_indices, axis=1)
        return out_indices

    def decode(self, codes: mx.array) -> mx.array:
        quantized_out = None
        for i, indices in enumerate(codes.split(codes.shape[1], axis=1)):
            layer = self.layers[i]
            quantized = layer.decode(indices.squeeze(1))
            if quantized_out is None:
                quantized_out = quantized
            else:
                quantized_out += quantized
        return quantized_out


################################################################################
# EncodecModel with chunk-based parallelism, half-precision, etc.
################################################################################
class EncodecModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = EncodecEncoder(config)
        self.decoder = EncodecDecoder(config)
        self.quantizer = EncodecResidualVectorQuantizer(config)

        # We'll create a cached fade weight for overlap-add if needed
        self._weight_cache = {}

    @property
    def channels(self):
        return self.config.audio_channels

    @property
    def sampling_rate(self):
        return self.config.sampling_rate

    @property
    def chunk_length(self):
        if self.config.chunk_length_s is None:
            return None
        else:
            return int(self.config.chunk_length_s * self.config.sampling_rate)

    @property
    def chunk_stride(self):
        if self.config.chunk_length_s is None or self.config.overlap is None:
            return None
        else:
            # E.g., if overlap=0.15, stride is 85% of chunk_length
            return max(1, int((1.0 - self.config.overlap) * self.chunk_length))

    def _create_fade_weights(self, frame_length: int, dtype) -> mx.array:
        """
        Creates or retrieves cached fade weights of shape (frame_length, 1).
        Weighted overlap-add to reduce boundary artifacts.
        """
        if frame_length in self._weight_cache:
            return self._weight_cache[frame_length]

        time_vec = mx.linspace(0, 1, frame_length + 2, dtype=dtype)[1:-1]
        fade = 0.5 - (time_vec - 0.5).abs()  # shape (frame_length,)
        fade = fade[:, None]  # shape (frame_length, 1)
        self._weight_cache[frame_length] = fade
        return fade

    def _linear_overlap_add(self, frames: List[mx.array], stride: int) -> mx.array:
        if len(frames) == 0:
            raise ValueError("`frames` cannot be an empty list.")

        dtype = frames[0].dtype
        N, frame_length, C = frames[0].shape
        total_size = stride * (len(frames) - 1) + frames[-1].shape[1]

        weight = self._create_fade_weights(frame_length, dtype)
        sum_weight = mx.zeros((total_size, 1), dtype=dtype)
        out = mx.zeros((N, total_size, C), dtype=dtype)

        offset = 0
        for frame in frames:
            flen = frame.shape[1]
            out[:, offset : offset + flen] += weight[:flen] * frame
            sum_weight[offset : offset + flen] += weight[:flen]
            offset += stride

        return out / sum_weight

    def _encode_frame(
        self, input_values: mx.array, bandwidth: float, padding_mask: mx.array
    ) -> Tuple[mx.array, Optional[mx.array]]:
        # Optional: half-precision
        if getattr(self.config, "precision", 32) == 16:
            input_values = input_values.astype(mx.float16)

        if self.config.normalize:
            input_values = input_values * padding_mask[..., None]
            # Compute scale factor
            mono = mx.sum(input_values, axis=2, keepdims=True) / input_values.shape[2]
            scale = mono.square().mean(axis=1, keepdims=True).sqrt() + 1e-8
            input_values = input_values / scale
        else:
            scale = None

        embeddings = self.encoder(input_values)
        codes = self.quantizer.encode(embeddings, bandwidth)
        return codes, scale

    def encode(
        self,
        input_values: mx.array,
        padding_mask: mx.array = None,
        bandwidth: Optional[float] = None,
    ) -> Tuple[mx.array, List[Optional[mx.array]]]:
        """
        Encodes the input audio waveform into discrete codes (chunk-based parallel).
        """
        if bandwidth is None:
            bandwidth = self.config.target_bandwidths[0]
        if bandwidth not in self.config.target_bandwidths:
            raise ValueError(
                f"This model doesn't support the bandwidth {bandwidth}. "
                f"Select one of {self.config.target_bandwidths}."
            )

        B, input_length, channels = input_values.shape
        if channels < 1 or channels > 2:
            raise ValueError(
                f"Number of audio channels must be 1 or 2, but got {channels}"
            )

        chunk_length = self.chunk_length
        if chunk_length is None:
            # Single-chunk scenario
            chunk_length = input_length
            stride = input_length
        else:
            stride = self.chunk_stride

        if padding_mask is None:
            padding_mask = mx.ones(input_values.shape[:2], dtype=mx.bool_)

        step = chunk_length - stride
        if (input_length % stride) != step:
            raise ValueError(
                "The input length is not properly padded for chunked encoding. "
                "Make sure to pad the input correctly."
            )

        # Prepare chunk slices
        chunks = []
        for offset in range(0, input_length - step, stride):
            mask = padding_mask[:, offset : offset + chunk_length].astype(mx.bool_)
            frame = input_values[:, offset : offset + chunk_length]
            chunks.append((frame, mask))

        # Parallel chunk encoding
        def encode_chunk(args):
            f, m = args
            return self._encode_frame(f, bandwidth, m)

        encoded_frames_list = []
        scales_list = []

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(encode_chunk, chunks))

        for code, scale in results:
            encoded_frames_list.append(code)
            scales_list.append(scale)

        # shape: (num_chunks, B, num_quantizers, chunk_time, ...)
        encoded_frames = mx.stack(encoded_frames_list)
        return (encoded_frames, scales_list)

    def _decode_frame(self, codes: mx.array, scale: Optional[mx.array]) -> mx.array:
        embeddings = self.quantizer.decode(codes)

        # Half-precision if configured
        if getattr(self.config, "precision", 32) == 16:
            embeddings = embeddings.astype(mx.float16)

        outputs = self.decoder(embeddings)
        if scale is not None:
            outputs = outputs * scale
        return outputs

    def decode(
        self,
        audio_codes: mx.array,
        audio_scales: Union[mx.array, List[Optional[mx.array]]],
        padding_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Decodes quantized codes back to waveforms (chunk-based parallel + overlap-add).
        """
        chunk_length = self.chunk_length
        if chunk_length is None:
            # Single chunk decode
            if audio_codes.shape[0] != 1:
                raise ValueError(f"Expected one frame, got {len(audio_codes)}")
            audio_values = self._decode_frame(audio_codes[0], audio_scales[0])
        else:
            # Parallel chunk decode
            def decode_chunk(args):
                code, sc = args
                return self._decode_frame(code, sc)

            with ThreadPoolExecutor() as executor:
                frames = list(executor.map(decode_chunk, zip(audio_codes, audio_scales)))

            # Weighted overlap-add
            audio_values = self._linear_overlap_add(frames, self.chunk_stride or 1)

        # Truncate if leftover
        if padding_mask is not None and padding_mask.shape[1] < audio_values.shape[1]:
            audio_values = audio_values[:, : padding_mask.shape[1]]
        return audio_values

    @classmethod
    def from_pretrained(cls, path_or_repo: str):
        """
        Loads a model from a local path or huggingface_hub repository.
        """
        from huggingface_hub import snapshot_download

        path = Path(path_or_repo)
        if not path.exists():
            path = Path(
                snapshot_download(
                    repo_id=path_or_repo,
                    allow_patterns=["*.json", "*.safetensors", "*.model"],
                )
            )

        with open(path / "config.json", "r") as f:
            config = SimpleNamespace(**json.load(f))

        model = EncodecModel(config)
        model.load_weights(str(path / "model.safetensors"))
        processor = functools.partial(
            preprocess_audio,
            sampling_rate=config.sampling_rate,
            chunk_length=model.chunk_length,
            chunk_stride=model.chunk_stride,
        )
        mx.eval(model)
        return model, processor


################################################################################
# Audio Preprocessing Helper
################################################################################
def preprocess_audio(
    raw_audio: Union[mx.array, List[mx.array]],
    sampling_rate: int = 24000,
    chunk_length: Optional[int] = None,
    chunk_stride: Optional[int] = None,
):
    """
    Prepares raw audio (list or single array) for the model by padding to
    chunk-friendly sizes, returning (inputs, masks).
    """
    if not isinstance(raw_audio, list):
        raw_audio = [raw_audio]

    # Ensure shape is (time, channels)
    raw_audio = [x[..., None] if x.ndim == 1 else x for x in raw_audio]

    max_length = max(array.shape[0] for array in raw_audio)
    if chunk_length is not None and chunk_stride is not None:
        remainder = max_length % chunk_stride
        if remainder != 0:
            max_length += (chunk_stride - remainder)

    inputs = []
    masks = []
    for x in raw_audio:
        length = x.shape[0]
        mask = mx.ones((length,), dtype=mx.bool_)
        difference = max_length - length
        if difference > 0:
            mask = mx.pad(mask, (0, difference))
            x = mx.pad(x, ((0, difference), (0, 0)))
        inputs.append(x)
        masks.append(mask)

    # final shape: (B, T, C)
    return mx.stack(inputs), mx.stack(masks)


################################################################################
# Example usage if you want to create and run the model
################################################################################
if __name__ == "__main__":
    # 1. Instantiate the config with all the performance tweaks
    config = ExampleConfig()

    # 2. Create the model
    model = EncodecModel(config)

    # 3. Suppose you have some raw audio samples
    # e.g., a random dummy signal for demonstration:
    dummy_audio = mx.randn((48000,))  # 2 seconds @ 24000 Hz, single channel
    inputs, masks = preprocess_audio(
        dummy_audio, 
        sampling_rate=config.sampling_rate,
        chunk_length=model.chunk_length,
        chunk_stride=model.chunk_stride
    )

    # 4. Encode
    codes, scales = model.encode(inputs, masks, bandwidth=6.0)

    # 5. Decode
    outputs = model.decode(codes, scales, masks)

    print("Input shape:", inputs.shape)
    print("Codes shape:", codes.shape)
    print("Output shape:", outputs.shape)
    # Then measure iteration speed as needed (timeit, etc.).

