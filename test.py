"""Run ImageEmbed and attention tests on CPU and available MPS: python test.py."""

import unittest

import torch


class DeviceTestCase(unittest.TestCase):
    device = torch.device('cpu')

    def setUp(self):
        super().setUp()
        torch.manual_seed(0)
        # All models, inputs, and expected tensors are created on this device.
        self.enterContext(self.device)


class TestImageEmbed(DeviceTestCase):
    @classmethod
    def setUpClass(cls):
        from model import ImageEmbed, ViTConfig

        cls.ImageEmbed = ImageEmbed
        cls.ViTConfig = ViTConfig

    def make_embed(self, image_size=(3, 8, 12), patch_size=4, ebd_dim=4, dropout=0.0):
        in_channels, height, width = image_size
        cfg = self.ViTConfig(
            image_size=(height, width), in_channels=in_channels,
            ebd_dim=ebd_dim, n_head=2, head_dim=ebd_dim // 2,
            patch_size=patch_size, n_layer=1, ffn_dim=16, cls_num=3,
            dropout_img_ebd=dropout,
        )
        return self.ImageEmbed(cfg)

    def test_output_shape(self):
        for batch_size, image_size in ((1, (3, 8, 8)), (3, (3, 8, 12)), (2, (1, 12, 8))):
            with self.subTest(batch_size=batch_size, image_size=image_size):
                embed = self.make_embed(image_size)
                x = torch.ones(batch_size, *image_size)
                output = embed(x)
                patch_count = (image_size[1] // 4) * (image_size[2] // 4)
                self.assertEqual(output.shape, (batch_size, patch_count + 1, 4))
                self.assertEqual(output.dtype, x.dtype)
                self.assertEqual(output.device, x.device)

    def test_cls_token_is_first_for_every_image(self):
        embed = self.make_embed()
        expected_cls = torch.tensor([1.0, 2.0, 3.0, 4.0])
        with torch.no_grad():
            embed.cls_token.copy_(expected_cls.reshape(1, 1, 4))
        x = torch.arange(3 * 3 * 8 * 12, dtype=torch.float32).reshape(3, 3, 8, 12)
        output = embed(x)
        for image in output:
            torch.testing.assert_close(image[0], expected_cls)

    def test_patch_projection_and_order(self):
        embed = self.make_embed(image_size=(1, 4, 6), patch_size=2, ebd_dim=2)
        with torch.no_grad():
            embed.conv.weight[0].fill_(1.0)
            embed.conv.weight[1].fill_(2.0)
            embed.conv.bias.copy_(torch.tensor([0.0, -1.0]))
            embed.pos_ebd.zero_()
        x = torch.arange(24, dtype=torch.float32).reshape(1, 1, 4, 6)
        # Each 2x2 patch produces [sum(patch), 2 * sum(patch) - 1].
        expected = torch.tensor([
            [14.0, 27.0], [22.0, 43.0], [30.0, 59.0],
            [62.0, 123.0], [70.0, 139.0], [78.0, 155.0],
        ])
        torch.testing.assert_close(embed(x)[0, 1:], expected)

    def test_position_embeddings_are_added_to_patches(self):
        embed = self.make_embed()
        expected_positions = torch.arange(1, 25, dtype=torch.float32).reshape(6, 4)
        with torch.no_grad():
            embed.conv.weight.zero_()
            embed.conv.bias.zero_()
            embed.pos_ebd[:, 1:].copy_(expected_positions.unsqueeze(0))
        output = embed(torch.ones(2, 3, 8, 12))
        for image in output:
            torch.testing.assert_close(image[1:], expected_positions)

    def test_backward_reaches_input_and_all_parameters(self):
        embed = self.make_embed()
        with torch.no_grad():
            for parameter in embed.parameters():
                parameter.fill_(0.1)
        x = torch.ones(2, 3, 8, 12, requires_grad=True)
        embed(x).square().sum().backward()

        for name, tensor in [('input', x), *embed.named_parameters()]:
            with self.subTest(tensor=name):
                self.assertIsNotNone(tensor.grad)
                self.assertTrue(torch.isfinite(tensor.grad).all().item())
                self.assertGreater(tensor.grad.abs().sum().item(), 0.0)

        # Both images contribute to the same learnable CLS parameter.
        torch.testing.assert_close(
            embed.cls_token.grad,
            2 * x.size(0) * (
                embed.cls_token.detach() + embed.pos_ebd[:, :1].detach()
            ),
        )

    def test_rejects_image_dimensions_not_divisible_by_patch_size(self):
        for image_size in ((3, 7, 8), (3, 8, 10)):
            with self.subTest(image_size=image_size):
                with self.assertRaises((AssertionError, ValueError)):
                    self.make_embed(image_size)

    def make_dropout_embed(self):
        embed = self.make_embed(dropout=1.0)
        with torch.no_grad():
            embed.conv.weight.zero_()
            embed.conv.bias.zero_()
            embed.cls_token.fill_(1.0)
            embed.pos_ebd.fill_(1.0)
        return embed

    def test_dropout_is_applied_during_training(self):
        # p=1 makes the expected result deterministic.
        embed = self.make_dropout_embed().train()
        output = embed(torch.ones(2, 3, 8, 12))
        torch.testing.assert_close(output, torch.zeros(2, 7, 4))

    def test_dropout_is_disabled_during_evaluation(self):
        embed = self.make_dropout_embed().eval()
        output = embed(torch.ones(2, 3, 8, 12))
        expected = torch.ones(2, 7, 4)
        expected[:, 0] = 2.0
        torch.testing.assert_close(output, expected)


class TestMultiHeadSelfAttention(DeviceTestCase):
    @classmethod
    def setUpClass(cls):
        from model import ImageEmbed, MultiHeadSelfAttention, ViTConfig

        cls.ImageEmbed = ImageEmbed
        cls.MultiHeadSelfAttention = MultiHeadSelfAttention
        cls.ViTConfig = ViTConfig

    def make_attention(
        self, ebd_dim=8, n_head=2, head_dim=4,
        dropout_att_weight=0.0, dropout_att_out=0.0,
    ):
        cfg = self.ViTConfig(
            image_size=(8, 8), in_channels=3,
            ebd_dim=ebd_dim, n_head=n_head, head_dim=head_dim,
            patch_size=4, n_layer=1, ffn_dim=16, cls_num=3,
            dropout_att_weight=dropout_att_weight,
            dropout_att_out=dropout_att_out,
        )
        return self.MultiHeadSelfAttention(cfg)

    def make_reference(self, attention):
        """Use PyTorch's MHA with identical parameters and dropout disabled."""
        reference = torch.nn.MultiheadAttention(
            attention.cfg.ebd_dim, attention.cfg.n_head, batch_first=True,
        ).to(dtype=attention.Wq.weight.dtype)
        projections = (attention.Wq, attention.Wk, attention.Wv)
        with torch.no_grad():
            reference.in_proj_weight.copy_(torch.cat([p.weight for p in projections]))
            reference.in_proj_bias.copy_(torch.cat([p.bias for p in projections]))
            reference.out_proj.load_state_dict(attention.output_proj.state_dict())
        return reference

    def test_output_shape_and_noncontiguous_input(self):
        cases = ((1, 1, 4, 1, 4), (2, 5, 8, 2, 4), (3, 7, 7, 3, 2))
        for batch, tokens, channels, heads, head_dim in cases:
            with self.subTest(shape=(batch, tokens, channels), heads=heads):
                attention = self.make_attention(channels, heads, head_dim)
                x = torch.randn(batch, channels, tokens).transpose(1, 2)
                if tokens > 1:
                    self.assertFalse(x.is_contiguous())
                output = attention(x)
                self.assertEqual(output.shape, (batch, tokens, channels))
                self.assertEqual(output.dtype, x.dtype)
                self.assertEqual(output.device, x.device)
                torch.testing.assert_close(output, attention(x.contiguous()))

    def test_output_and_gradients_match_pytorch_attention(self):
        # MPS does not support float64; use appropriate float32 tolerances there.
        on_mps = self.device.type == 'mps'
        dtype = torch.float32 if on_mps else torch.float64
        rtol, atol = (1e-4, 1e-6) if on_mps else (1e-6, 1e-8)
        attention = self.make_attention().to(dtype=dtype)
        reference = self.make_reference(attention)
        x = torch.randn(2, 5, 8, dtype=dtype, requires_grad=True)
        reference_x = x.detach().clone().requires_grad_(True)

        output = attention(x)
        # need_weights=True selects the explicit attention calculation as an oracle.
        expected, _ = reference(reference_x, reference_x, reference_x, need_weights=True)
        self.assertEqual(output.device, x.device)
        torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)
        output.square().sum().backward()
        expected.square().sum().backward()

        reference_weight_grads = reference.in_proj_weight.grad.chunk(3)
        reference_bias_grads = reference.in_proj_bias.grad.chunk(3)
        comparisons = [('input', x.grad, reference_x.grad)]
        for i, name in enumerate(('Wq', 'Wk', 'Wv')):
            projection = getattr(attention, name)
            comparisons.extend([
                (f'{name}.weight', projection.weight.grad, reference_weight_grads[i]),
                (f'{name}.bias', projection.bias.grad, reference_bias_grads[i]),
            ])
        for name in ('weight', 'bias'):
            comparisons.append((
                f'output_proj.{name}',
                getattr(attention.output_proj, name).grad,
                getattr(reference.out_proj, name).grad,
            ))
        for name, actual, expected_grad in comparisons:
            with self.subTest(gradient=name):
                self.assertIsNotNone(actual)
                self.assertTrue(torch.isfinite(actual).all().item())
                self.assertEqual(actual.device, x.device)
                torch.testing.assert_close(actual, expected_grad, rtol=rtol, atol=atol)

    def test_cls_can_attend_to_later_tokens(self):
        attention = self.make_attention(ebd_dim=4, n_head=2, head_dim=2)
        with torch.no_grad():
            for projection in (attention.Wq, attention.Wk):
                projection.weight.zero_()
                projection.bias.zero_()
            for projection in (attention.Wv, attention.output_proj):
                projection.weight.copy_(torch.eye(4))
                projection.bias.zero_()
        x = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
        # Zero Q/K gives uniform attention, including from CLS to later tokens.
        expected = x.mean(dim=1, keepdim=True).expand_as(x)
        torch.testing.assert_close(attention(x), expected)

    def test_attention_dropout_during_training(self):
        attention = self.make_attention(dropout_att_weight=1.0).train()
        with torch.no_grad():
            attention.output_proj.bias.fill_(0.25)
        output = attention(torch.randn(2, 5, 8))
        # Dropped attention values are zero, but output projection still adds bias.
        torch.testing.assert_close(output, torch.full((2, 5, 8), 0.25))

    def test_output_dropout_during_training(self):
        attention = self.make_attention(dropout_att_out=1.0).train()
        with torch.no_grad():
            attention.output_proj.weight.zero_()
            attention.output_proj.bias.fill_(1.0)
        output = attention(torch.randn(2, 5, 8))
        torch.testing.assert_close(output, torch.zeros(2, 5, 8))

    def test_both_dropouts_are_disabled_during_evaluation(self):
        for weight_dropout, output_dropout in ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)):
            with self.subTest(weight=weight_dropout, output=output_dropout):
                attention = self.make_attention(
                    dropout_att_weight=weight_dropout, dropout_att_out=output_dropout,
                ).eval()
                reference = self.make_reference(attention).eval()
                x = torch.randn(2, 5, 8)
                expected, _ = reference(x, x, x, need_weights=True)
                torch.testing.assert_close(attention(x), expected)
                torch.testing.assert_close(attention(x), expected)

    def test_rejects_input_with_wrong_rank(self):
        attention = self.make_attention()
        for shape in ((5, 8), (2, 3, 5, 8)):
            with self.subTest(shape=shape):
                with self.assertRaises(AssertionError):
                    attention(torch.zeros(shape))

    def test_image_embedding_integration_and_backward(self):
        attention = self.make_attention()
        embedding_cfg = self.ViTConfig(
            image_size=(8, 12), in_channels=3,
            ebd_dim=attention.cfg.ebd_dim,
            n_head=attention.cfg.n_head,
            head_dim=attention.cfg.head_dim,
            patch_size=attention.cfg.patch_size,
            n_layer=attention.cfg.n_layer,
            ffn_dim=attention.cfg.ffn_dim,
            cls_num=attention.cfg.cls_num,
        )
        embedding = self.ImageEmbed(embedding_cfg)
        image = torch.randn(2, 3, 8, 12, requires_grad=True)
        output = attention(embedding(image))
        self.assertEqual(output.shape, (2, 7, 8))
        output[:, 0].square().mean().backward()

        tensors = [('image', image)]
        tensors.extend((f'embedding.{name}', p) for name, p in embedding.named_parameters())
        tensors.extend((f'attention.{name}', p) for name, p in attention.named_parameters())
        for name, tensor in tensors:
            with self.subTest(gradient=name):
                self.assertIsNotNone(tensor.grad)
                self.assertTrue(torch.isfinite(tensor.grad).all().item())
        self.assertGreater(image.grad.abs().sum().item(), 0.0)
        self.assertGreater(embedding.cls_token.grad.abs().sum().item(), 0.0)


@unittest.skipUnless(torch.backends.mps.is_available(), 'MPS is unavailable')
class TestImageEmbedMPS(TestImageEmbed):
    device = torch.device('mps')


@unittest.skipUnless(torch.backends.mps.is_available(), 'MPS is unavailable')
class TestMultiHeadSelfAttentionMPS(TestMultiHeadSelfAttention):
    device = torch.device('mps')


if __name__ == '__main__':
    unittest.main(verbosity=2)
