import torch
from torch import nn
import torch.nn.functional as F
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d, ResBlock3d, SPADEResnetBlock
from modules.dense_motion import DenseMotionNetwork
from modules.rppg import RPPGFusion, TemporalRPPGEncoder, SkinPhysiologyResidual


class OcclusionAwareGenerator(nn.Module):
    """
    Generator follows NVIDIA architecture.
    """

    def __init__(self, image_channel, feature_channel, num_kp, block_expansion, max_features, num_down_blocks, reshape_channel, reshape_depth,
                 num_resblocks, estimate_occlusion_map=False, dense_motion_params=None,
                 estimate_jacobian=False, physiology_params=None):
        super(OcclusionAwareGenerator, self).__init__()

        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, feature_channel=feature_channel,
                                                           estimate_occlusion_map=estimate_occlusion_map,
                                                           **dense_motion_params)
        else:
            self.dense_motion_network = None

        self.first = SameBlock2d(image_channel, block_expansion, kernel_size=(3, 3), padding=(1, 1))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        self.second = nn.Conv2d(in_channels=out_features, out_channels=max_features, kernel_size=1, stride=1)

        self.reshape_channel = reshape_channel
        self.reshape_depth = reshape_depth

        self.resblocks_3d = torch.nn.Sequential()
        for i in range(num_resblocks):
            self.resblocks_3d.add_module('3dr' + str(i), ResBlock3d(reshape_channel, kernel_size=3, padding=1))

        out_features = block_expansion * (2 ** (num_down_blocks))
        self.third = SameBlock2d(max_features, out_features, kernel_size=(3, 3), padding=(1, 1), lrelu=True)
        self.fourth = nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=1, stride=1)

        self.resblocks_2d = torch.nn.Sequential()
        for i in range(num_resblocks):
            self.resblocks_2d.add_module('2dr' + str(i), ResBlock2d(out_features, kernel_size=3, padding=1))

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = max(block_expansion, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = max(block_expansion, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.final = nn.Conv2d(block_expansion, image_channel, kernel_size=(7, 7), padding=(3, 3))
        self.estimate_occlusion_map = estimate_occlusion_map
        self.image_channel = image_channel

        self.rppg_fusion = RPPGFusion(out_features)
        physiology_params = {} if physiology_params is None else physiology_params
        condition_dim = physiology_params.get('condition_dim', 64)
        self.physiology_encoder = TemporalRPPGEncoder(
            embedding_dim=condition_dim,
            hidden_dim=physiology_params.get('hidden_dim', 64),
            num_blocks=physiology_params.get('num_blocks', 4),
        )
        self.physiology_residual = SkinPhysiologyResidual(
            feature_channels=3,
            condition_dim=condition_dim,
            hidden_channels=physiology_params.get('residual_channels', 64),
            max_delta=physiology_params.get('max_delta', 4.0 / 255.0),
            rank=physiology_params.get('rank', 2),
            free_residual_scale=physiology_params.get('free_residual_scale', 0.0),
        )

    def deform_input(self, inp, deformation):
        _, d_old, h_old, w_old, _ = deformation.shape
        _, _, d, h, w = inp.shape
        if d_old != d or h_old != h or w_old != w:
            deformation = deformation.permute(0, 4, 1, 2, 3)
            deformation = F.interpolate(deformation, size=(d, h, w), mode='trilinear')
            deformation = deformation.permute(0, 2, 3, 4, 1)
        return F.grid_sample(inp, deformation)

    @staticmethod
    def _select_physiology_condition(encoded, frame_index):
        batch_size, time, _ = encoded.shape
        if frame_index is None:
            if time != 1:
                raise ValueError("rppg_frame_index is required for multi-frame target_rppg")
            return encoded[:, 0]
        if not torch.is_tensor(frame_index):
            frame_index = torch.full((batch_size,), int(frame_index),
                                     dtype=torch.long, device=encoded.device)
        else:
            frame_index = frame_index.to(device=encoded.device, dtype=torch.long).view(-1)
            if frame_index.numel() == 1:
                frame_index = frame_index.expand(batch_size)
        if frame_index.numel() != batch_size:
            raise ValueError("rppg_frame_index must be scalar or have one value per batch item")
        frame_index = frame_index.clamp(0, time - 1)
        return encoded[torch.arange(batch_size, device=encoded.device), frame_index]

    def forward(self, source_image, kp_driving, kp_source, rppg_feat=None,
                target_rppg=None, rppg_frame_index=None, skin_mask=None,
                physiology_condition=None, physiology_value=None):
        # Encoding (downsampling) part
        out = self.first(source_image)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
        out = self.second(out)
        bs, c, h, w = out.shape
        # print(out.shape)
        feature_3d = out.view(bs, self.reshape_channel, self.reshape_depth, h ,w)
        feature_3d = self.resblocks_3d(feature_3d)

        # Transforming feature representation according to deformation and occlusion
        output_dict = {}
        if self.dense_motion_network is not None:
            dense_motion = self.dense_motion_network(feature=feature_3d, kp_driving=kp_driving,
                                                     kp_source=kp_source)
            output_dict['mask'] = dense_motion['mask']

            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']
                output_dict['occlusion_map'] = occlusion_map
            else:
                occlusion_map = None
            deformation = dense_motion['deformation']
            out = self.deform_input(feature_3d, deformation)

            bs, c, d, h, w = out.shape
            out = out.view(bs, c*d, h, w)
            out = self.third(out)
            out = self.fourth(out)

            if occlusion_map is not None:
                if out.shape[2] != occlusion_map.shape[2] or out.shape[3] != occlusion_map.shape[3]:
                    occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode='bilinear')
                out = out * occlusion_map

            # output_dict["deformed"] = self.deform_input(source_image, deformation)  # 3d deformation cannot deform 2d image

        # rPPG injection
        if rppg_feat is not None:
            out = self.rppg_fusion(out, rppg_feat)

        # Decoding part
        out = self.resblocks_2d(out)
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        base_prediction = torch.sigmoid(self.final(out))

        physiology_delta = None
        if target_rppg is not None or physiology_condition is not None:
            if physiology_condition is None:
                encoded_rppg = self.physiology_encoder(target_rppg)
                condition = self._select_physiology_condition(encoded_rppg, rppg_frame_index)
            else:
                condition = physiology_condition
            if physiology_value is None:
                if target_rppg is None:
                    raise ValueError('physiology_value is required with pre-encoded conditions')
                physiology_value = self._select_physiology_condition(
                    target_rppg.unsqueeze(-1), rppg_frame_index
                ).squeeze(-1)
            physiology_output = self.physiology_residual(
                base_prediction, condition, physiology_value
            )
            physiology_delta = physiology_output['delta']
            prediction = torch.clamp(base_prediction + physiology_delta, 0.0, 1.0)
        else:
            prediction = base_prediction

        output_dict["prediction"] = prediction
        output_dict["prediction_base"] = base_prediction
        if physiology_delta is not None:
            output_dict["physiology_delta"] = physiology_delta
            output_dict["physiology_skin_mask"] = physiology_output['skin_mask']
            output_dict["physiology_response_maps"] = physiology_output['response_maps']

        return output_dict


class SPADEDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        ic = 256
        oc = 64
        norm_G = 'spadespectralinstance'
        label_nc = 256

        self.fc = nn.Conv2d(ic, 2 * ic, 3, padding=1)
        self.G_middle_0 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_1 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_2 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_3 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_4 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_5 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.up_0 = SPADEResnetBlock(2 * ic, ic, norm_G, label_nc)
        self.up_1 = SPADEResnetBlock(ic, oc, norm_G, label_nc)
        self.conv_img = nn.Conv2d(oc, 3, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, feature):
        seg = feature
        x = self.fc(feature)
        x = self.G_middle_0(x, seg)
        x = self.G_middle_1(x, seg)
        x = self.G_middle_2(x, seg)
        x = self.G_middle_3(x, seg)
        x = self.G_middle_4(x, seg)
        x = self.G_middle_5(x, seg)
        x = self.up(x)
        x = self.up_0(x, seg)         # 256, 128, 128
        x = self.up(x)
        x = self.up_1(x, seg)         # 64, 256, 256

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        # x = torch.tanh(x)
        x = F.sigmoid(x)

        return x


class OcclusionAwareSPADEGenerator(nn.Module):

    def __init__(self, image_channel, feature_channel, num_kp, block_expansion, max_features, num_down_blocks, reshape_channel, reshape_depth,
                 num_resblocks, estimate_occlusion_map=False, dense_motion_params=None,
                 estimate_jacobian=False, physiology_params=None, physiology_fusion_mode=None):
        super(OcclusionAwareSPADEGenerator, self).__init__()

        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, feature_channel=feature_channel,
                                                           estimate_occlusion_map=estimate_occlusion_map,
                                                           **dense_motion_params)
        else:
            self.dense_motion_network = None

        self.first = SameBlock2d(image_channel, block_expansion, kernel_size=(3, 3), padding=(1, 1))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        self.second = nn.Conv2d(in_channels=out_features, out_channels=max_features, kernel_size=1, stride=1)

        self.reshape_channel = reshape_channel
        self.reshape_depth = reshape_depth

        self.resblocks_3d = torch.nn.Sequential()
        for i in range(num_resblocks):
            self.resblocks_3d.add_module('3dr' + str(i), ResBlock3d(reshape_channel, kernel_size=3, padding=1))

        out_features = block_expansion * (2 ** (num_down_blocks))
        self.third = SameBlock2d(max_features, out_features, kernel_size=(3, 3), padding=(1, 1), lrelu=True)
        self.fourth = nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=1, stride=1)

        self.estimate_occlusion_map = estimate_occlusion_map
        self.image_channel = image_channel

        self.rppg_fusion = RPPGFusion(out_features)

        self.decoder = SPADEDecoder()
        physiology_params = {} if physiology_params is None else physiology_params
        condition_dim = physiology_params.get('condition_dim', 64)
        self.physiology_encoder = TemporalRPPGEncoder(
            embedding_dim=condition_dim,
            hidden_dim=physiology_params.get('hidden_dim', 64),
            num_blocks=physiology_params.get('num_blocks', 4),
        )
        # SPADEDecoder returns RGB directly, so its bounded physiology residual
        # is conditioned on the baseline RGB prediction.
        self.physiology_residual = SkinPhysiologyResidual(
            feature_channels=3,
            condition_dim=condition_dim,
            hidden_channels=physiology_params.get('residual_channels', 64),
            max_delta=physiology_params.get('max_delta', 4.0 / 255.0),
            rank=physiology_params.get('rank', 2),
            free_residual_scale=physiology_params.get('free_residual_scale', 0.0),
        )

        from modules.physiology_motion import MotionAlignedPhysioModulator, FUSION_MODES
        self.physiology_fusion_mode = physiology_fusion_mode or physiology_params.get('fusion_mode', 'rgb_residual')
        if self.physiology_fusion_mode not in FUSION_MODES:
            raise ValueError('Unknown physiology fusion mode: ' + self.physiology_fusion_mode)
        self.checkpoint_decoder = False
        if self.physiology_fusion_mode.startswith('feature_'):
            self.physiology_modulator = MotionAlignedPhysioModulator(
                reference_channels=reshape_channel, depth=reshape_depth,
                feature_channels=out_features, condition_dim=condition_dim,
                rank=physiology_params.get('rank', 2),
                hidden_channels=physiology_params.get('basis_channels', 16),
                max_feature_delta=physiology_params.get('max_feature_delta', 0.1),
                initial_strength=physiology_params.get('initial_feature_strength', 0.1))

    def deform_input(self, inp, deformation):
        # Explicit defaults preserve the pre-existing GFVC convention. Its
        # linspace [-1,1] grid is retained, including historical border sampling.
        _, d_old, h_old, w_old, _ = deformation.shape
        _, _, d, h, w = inp.shape
        if (d_old, h_old, w_old) != (d, h, w):
            deformation = F.interpolate(deformation.permute(0,4,1,2,3),
                size=(d,h,w), mode='trilinear', align_corners=False).permute(0,2,3,4,1)
        return F.grid_sample(inp, deformation, mode='bilinear',
                             padding_mode='zeros', align_corners=False)

    _select_physiology_condition = staticmethod(OcclusionAwareGenerator._select_physiology_condition)

    def prepare_reference(self, source_image):
        out = self.first(source_image)
        for block in self.down_blocks:
            out = block(out)
        out = self.second(out)
        b, _, h, w = out.shape
        volume = self.resblocks_3d(out.view(b,self.reshape_channel,self.reshape_depth,h,w))
        # Keep the source alive while its context exists, so its allocation
        # cannot be recycled for a different sample with the same pointer.
        version = None if torch.is_inference(source_image) else source_image._version
        context = dict(source_image=source_image,source_pointer=source_image.data_ptr(), source_version=version,
                       source_snapshot=source_image.clone() if version is None else None,
                       source_shape=tuple(source_image.shape), reference_volume=volume)
        if self.physiology_fusion_mode.startswith('feature_'):
            context['reference_basis'] = self.physiology_modulator.make_basis(volume)
        return context

    def motion_context(self, source_image, kp_driving, kp_source, reference_context=None):
        reference = self.prepare_reference(source_image) if reference_context is None else reference_context
        version = None if torch.is_inference(source_image) else source_image._version
        if (reference['source_pointer'] != source_image.data_ptr() or
            reference['source_version'] != version or
            reference['source_shape'] != tuple(source_image.shape)):
            raise ValueError('Reference context belongs to another/modified source image')
        if version is None and not torch.equal(source_image,reference['source_snapshot']):
            raise ValueError('Reference context belongs to another/modified source image')
        if self.dense_motion_network is None:
            raise RuntimeError('SPADE motion-feature path requires the existing DenseMotionNetwork')
        volume = reference['reference_volume']
        dense = self.dense_motion_network(feature=volume,kp_driving=kp_driving,kp_source=kp_source)
        deformation = dense['deformation']
        warped = self.deform_input(volume,deformation)
        b,c,d,h,w = warped.shape
        features = self.fourth(self.third(warped.reshape(b,c*d,h,w)))
        occlusion = dense.get('occlusion_map')
        if occlusion is not None:
            if features.shape[2:] != occlusion.shape[2:]:
                occlusion = F.interpolate(occlusion,size=features.shape[2:],mode='bilinear',align_corners=False)
            features = features * occlusion
        return dict(features=features,deformation=deformation,occlusion_map=occlusion,
                    mask=dense['mask'],reference_context=reference)

    def decode_context(self, context, physiology_condition=None, physiology_value=None,
                       rppg_feat=None, baseline_decoder=None, prediction_base_override=None):
        mode = self.physiology_fusion_mode
        features = context['features']
        if rppg_feat is not None:
            if mode.startswith('feature_'):
                raise ValueError('Feature modulation does not accept a second legacy rppg_feat injection')
            if mode != 'none':
                features = self.rppg_fusion(features,rppg_feat)
        enabled = mode != 'none' and physiology_condition is not None
        if enabled and physiology_value is None:
            raise ValueError('physiology_value is required with encoded physiology_condition')
        decoder_base = self.decoder if baseline_decoder is None else baseline_decoder
        # Baseline serves image-space constraints; it is NOT a second trainable
        # RGB residual branch. The actual prediction always comes from SPADE.
        if enabled and mode.startswith('feature_'):
            with torch.no_grad():
                base = decoder_base(features) if prediction_base_override is None else prediction_base_override
            reference = context['reference_context']
            result = self.physiology_modulator(features,reference['reference_basis'],
                physiology_condition,physiology_value,context['occlusion_map'],
                context['deformation'],self.deform_input,mode == 'feature_motion_aligned')
            fused = features + result['feature_delta']
            if self.checkpoint_decoder and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                prediction = checkpoint(self.decoder,fused,use_reentrant=False)
            else:
                prediction = self.decoder(fused)
            skin = F.interpolate(result['skin_mask'],size=prediction.shape[2:],mode='bilinear',align_corners=False)
            gate = F.interpolate(result['gate'],size=prediction.shape[2:],mode='bilinear',align_corners=False)
            diagnostics = dict(physiology_feature_delta=result['feature_delta'],
                physiology_feature_norm=result['feature_delta'].square().mean((1,2,3)).sqrt(),
                physiology_gate=gate,physiology_skin_mask=skin,
                physiology_response_maps=result['response_maps'],
                physiology_basis_3d=result['aligned_basis'])
            delta = prediction - base
        else:
            base = decoder_base(features) if prediction_base_override is None else prediction_base_override
            if enabled:
                if self.checkpoint_decoder and torch.is_grad_enabled():
                    from torch.utils.checkpoint import checkpoint
                    result = checkpoint(self.physiology_residual,base,physiology_condition,physiology_value,use_reentrant=False)
                else:
                    result = self.physiology_residual(base,physiology_condition,physiology_value)
                delta = result['delta']
                prediction = torch.clamp(base+delta,0.,1.)
                diagnostics = dict(physiology_skin_mask=result['skin_mask'],
                                   physiology_response_maps=result['response_maps'])
            else:
                prediction = base;delta = torch.zeros_like(base)
                diagnostics = dict(physiology_skin_mask=base[:,:1]*0.,
                                   physiology_response_maps=base[:,:1]*0.)
        output = dict(prediction=prediction,prediction_base=base,physiology_delta=delta,
                      mask=context['mask'],deformation=context['deformation'])
        if context['occlusion_map'] is not None:
            output['occlusion_map'] = context['occlusion_map']
        output.update(diagnostics)
        return output

    def forward(self, source_image, kp_driving, kp_source, rppg_feat=None,
                target_rppg=None, rppg_frame_index=None, skin_mask=None,
                physiology_condition=None, physiology_value=None, reference_context=None,
                baseline_decoder=None, prediction_base_override=None):
        # skin_mask is retained for call compatibility, intentionally never used
        # as a decoder input. It is supervised outside the generator only.
        if physiology_condition is None and target_rppg is not None:
            physiology_condition = self._select_physiology_condition(
                self.physiology_encoder(target_rppg),rppg_frame_index)
        if physiology_value is None and target_rppg is not None:
            physiology_value = self._select_physiology_condition(
                target_rppg.unsqueeze(-1),rppg_frame_index).squeeze(-1)
        context = self.motion_context(source_image,kp_driving,kp_source,reference_context)
        return self.decode_context(context,physiology_condition,physiology_value,rppg_feat,baseline_decoder,prediction_base_override)
