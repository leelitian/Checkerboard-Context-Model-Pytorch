import torch

from compressai.models.google import JointAutoregressiveHierarchicalPriors
from layers import CheckerboardMaskedConv2d
from modules import Demultiplexer, Multiplexer

class CheckerboardAutogressive(JointAutoregressiveHierarchicalPriors):
    def __init__(self, N=192, M=192, **kwargs):
        super().__init__(N, M, **kwargs)

        self.context_prediction = CheckerboardMaskedConv2d(
            M, 2 * M, kernel_size=5, padding=2, stride=1
        )

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )

        # set non_anchor to 0
        y_half = y_hat.clone()
        y_half[:, :, 0::2, 0::2] = 0
        y_half[:, :, 1::2, 1::2] = 0

        # set anchor's ctx to 0, otherwise there will be a bias
        ctx_params = self.context_prediction(y_half)
        ctx_params[:, :, 0::2, 1::2] = 0
        ctx_params[:, :, 1::2, 0::2] = 0

        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        # Notion: in compressai, the means must be subtracted before quantification.
        # In order to get y_half, we need subtract y_anchor's means and then quantize,
        # to get y_anchor's means, we have to go through 'gep' here
        N, _, H, W = z_hat.shape
        zero_ctx_params = torch.zeros([N, 2 * self.M, H * 4, W * 4]).to(z_hat.device)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, zero_ctx_params), dim=1)
        )
        _, means_hat = gaussian_params.chunk(2, 1)
        y_hat = self.gaussian_conditional.quantize(y, "dequantize", means=means_hat)

        # set non_anchor to 0
        y_half = y_hat.clone()
        y_half[:, :, 0::2, 0::2] = 0
        y_half[:, :, 1::2, 1::2] = 0

        # set anchor's ctx to 0, otherwise there will be a bias
        ctx_params = self.context_prediction(y_half)
        ctx_params[:, :, 0::2, 1::2] = 0
        ctx_params[:, :, 1::2, 0::2] = 0

        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )

        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        y_anchor, y_non_anchor = Demultiplexer(y)
        scales_hat_anchor, scales_hat_non_anchor = Demultiplexer(scales_hat)
        means_hat_anchor, means_hat_non_anchor = Demultiplexer(means_hat)

        indexes_anchor = self.gaussian_conditional.build_indexes(scales_hat_anchor)
        indexes_non_anchor = self.gaussian_conditional.build_indexes(scales_hat_non_anchor)

        anchor_strings = self.gaussian_conditional.compress(y_anchor, indexes_anchor, means=means_hat_anchor)
        non_anchor_strings = self.gaussian_conditional.compress(y_non_anchor, indexes_non_anchor, means=means_hat_non_anchor)

        return {
            "strings": [anchor_strings, non_anchor_strings, z_strings],
            "shape": z.size()[-2:],
        }
    
    def decompress(self, strings, shape):
        """
        See Figure 5. Illustration of the proposed two-pass decoding.
        """
        assert isinstance(strings, list) and len(strings) == 3
        z_hat = self.entropy_bottleneck.decompress(strings[2], shape)
        params = self.h_s(z_hat)

        # PASS 1: anchor
        N, _, H, W = z_hat.shape
        zero_ctx_params = torch.zeros([N, 2 * self.M, H * 4, W * 4]).to(z_hat.device)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, zero_ctx_params), dim=1)
        )

        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        scales_hat_anchor, _ = Demultiplexer(scales_hat)
        means_hat_anchor, _ = Demultiplexer(means_hat)

        indexes_anchor = self.gaussian_conditional.build_indexes(scales_hat_anchor)
        y_anchor = self.gaussian_conditional.decompress(strings[0], indexes_anchor, means=means_hat_anchor)     # [1, 384, 8, 8]
        y_anchor = Multiplexer(y_anchor, torch.zeros_like(y_anchor))    # [1, 192, 16, 16]
        
        # PASS 2: non-anchor
        ctx_params = self.context_prediction(y_anchor)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )

        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, scales_hat_non_anchor = Demultiplexer(scales_hat)
        _, means_hat_non_anchor = Demultiplexer(means_hat)

        indexes_non_anchor = self.gaussian_conditional.build_indexes(scales_hat_non_anchor)
        y_non_anchor = self.gaussian_conditional.decompress(strings[1], indexes_non_anchor, means=means_hat_non_anchor)     # [1, 384, 8, 8]
        y_non_anchor = Multiplexer(torch.zeros_like(y_non_anchor), y_non_anchor)    # [1, 192, 16, 16]

        # gather
        y_hat = y_anchor + y_non_anchor
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {
            "x_hat": x_hat,
        }


if __name__ == "__main__":
    x = torch.randn([1, 3, 256, 256])
    model = CheckerboardAutogressive()
    model.update(force=True)

    out = model.compress(x)
    rec = model.decompress(out["strings"], out["shape"])
