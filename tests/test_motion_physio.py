import unittest
import torch
from torch import nn
from modules.generator import OcclusionAwareSPADEGenerator
from modules.physiology_motion import MotionAlignedPhysioModulator,load_generator_compatible
from modules.rppg import TemporalRPPGEncoder


def centers(d,h,w,b=1):
    z=2*(torch.arange(d)+.5)/d-1;y=2*(torch.arange(h)+.5)/h-1;x=2*(torch.arange(w)+.5)/w-1
    zz,yy,xx=torch.meshgrid(z,y,x,indexing='ij')
    return torch.stack((xx,yy,zz),-1).unsqueeze(0).repeat(b,1,1,1,1)


def warp(x,grid):return OcclusionAwareSPADEGenerator.deform_input(None,x,grid)


class MotionPhysioTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(42);torch.set_num_threads(2)

    def test_controlled_translation_not_random_motion(self):
        # A positive sampling-grid x offset draws from the right, moving a
        # known impulse ONE voxel to the left. Also test z-axis separately.
        x=torch.zeros(2,2,5,7,9);x[:,:,2,3,4]=1
        g=centers(5,7,9,2);m=MotionAlignedPhysioModulator(4,5,8,8,2,8)
        self.assertTrue(torch.allclose(warp(x,g),x,atol=1e-6))
        gx=g.clone();gx[...,0]+=2/9
        expected=torch.zeros_like(x);expected[:,:,2,3,3]=1
        self.assertTrue(torch.allclose(m.align_basis(x,gx,warp,True),expected,atol=1e-6))
        gz=g.clone();gz[...,2]+=2/5
        expected=torch.zeros_like(x);expected[:,:,1,3,4]=1
        self.assertTrue(torch.allclose(warp(x,gz),expected,atol=1e-6))
        self.assertTrue(torch.equal(m.align_basis(x,gx,warp,False),x))

    def test_shapes_zero_value_and_zero_occlusion(self):
        m=MotionAlignedPhysioModulator(4,3,8,8,2,8)
        ref=torch.randn(2,4,3,8,8);basis=m.make_basis(ref);f=torch.randn(2,8,8,8);c=torch.randn(2,8)
        g=centers(3,8,8,2);occ=torch.ones(2,1,8,8)
        out=m(f,basis,c,torch.tensor([.5,-.5]),occ,g,warp)
        self.assertEqual(out['aligned_basis'].shape,(2,2,3,8,8))
        self.assertEqual(out['feature_delta'].shape,f.shape)
        self.assertEqual(out['response_maps'].shape,(2,2,8,8))
        self.assertTrue(torch.all(out['feature_delta'].abs()<=m.max_feature_delta))
        for value,occlusion in [(torch.zeros(2),occ),(torch.ones(2),torch.zeros_like(occ))]:
            zero=m(f,basis,c,value,occlusion,g,warp)['feature_delta']
            self.assertTrue(torch.equal(zero,torch.zeros_like(zero)))

    def test_bvp_swap_and_all_submodule_gradients(self):
        enc=TemporalRPPGEncoder(8,8,1);m=MotionAlignedPhysioModulator(4,3,8,8,2,8)
        f=torch.randn(2,8,8,8);ref=torch.randn(2,4,3,8,8)
        wave=torch.randn(2,9);encoded=enc(wave);basis=m.make_basis(ref);grid=centers(3,8,8,2)
        a=m(f,basis,encoded[:,2],wave[:,2],torch.ones(2,1,8,8),grid,warp)
        b=m(f,basis,enc(-wave)[:,2],-wave[:,2],torch.ones(2,1,8,8),grid,warp)
        self.assertGreater(float((a['feature_delta']-b['feature_delta']).abs().max()),1e-7)
        (a['feature_delta'].square().mean()+.01*a['skin_mask'].mean()).backward()
        for name,module in [('encoder',enc),('basis',m.basis_encoder),('projection',m.depth_projection),
                            ('head',m.modulation_head),('gate',m.skin_gate)]:
            grads=[p.grad for p in module.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads),name)
            self.assertGreater(sum(float(g.abs().sum()) for g in grads),0,name)
        self.assertGreater(float(m.strength_logit.grad.abs()),0)

    def test_aligned_and_unaligned_same_parameter_schema(self):
        a=MotionAlignedPhysioModulator(4,3,8,8,2,8);b=MotionAlignedPhysioModulator(4,3,8,8,2,8)
        self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))
        b.load_state_dict(a.state_dict(),strict=True)

    def test_rank4_supported(self):
        m=MotionAlignedPhysioModulator(4,3,8,8,4,8)
        basis=m.make_basis(torch.randn(2,4,3,8,8))
        out=m(torch.randn(2,8,8,8),basis,torch.randn(2,8),torch.ones(2),
              torch.ones(2,1,8,8),centers(3,8,8,2),warp)
        self.assertEqual(out['aligned_basis'].shape,(2,4,3,8,8))
        self.assertEqual(out['feature_delta'].shape,(2,8,8,8))

    def test_reference_context_rejects_other_or_mutated_source(self):
        class Guard(nn.Module):
            motion_context=OcclusionAwareSPADEGenerator.motion_context
        model=Guard();source=torch.zeros(1,3,8,8)
        context=dict(source_image=source,source_pointer=source.data_ptr(),source_version=source._version,
                     source_shape=tuple(source.shape))
        with self.assertRaises(ValueError):model.motion_context(source.clone(),{}, {},context)
        source.add_(1.)
        with self.assertRaises(ValueError):model.motion_context(source,{}, {},context)

    def test_inference_tensor_mutation_guard(self):
        class Guard(nn.Module):
            motion_context=OcclusionAwareSPADEGenerator.motion_context
            dense_motion_network=None
        model=Guard()
        with torch.inference_mode():
            source=torch.zeros(1,3,8,8)
            context=dict(source_image=source,source_pointer=source.data_ptr(),source_version=None,
                         source_shape=tuple(source.shape),source_snapshot=source.clone())
            with self.assertRaisesRegex(RuntimeError,'requires the existing DenseMotionNetwork'):
                model.motion_context(source,{}, {},context)
            source.add_(1.)
            with self.assertRaises(ValueError):model.motion_context(source,{}, {},context)

    def test_checkpoint_loader_rejects_unknown_and_missing_trained_keys(self):
        class Toy(nn.Module):
            def __init__(self):
                super().__init__();self.backbone=nn.Linear(2,2);self.physiology_encoder=nn.Linear(2,2)
                self.physiology_fusion_mode='rgb_residual'
        m=Toy();state=m.state_dict();load_generator_compatible(m,state,declared_mode='rgb_residual')
        with self.assertRaises(RuntimeError):load_generator_compatible(m,{**state,'unknown':torch.zeros(1)})
        bad=dict(state);del bad['physiology_encoder.bias']
        with self.assertRaises(RuntimeError):load_generator_compatible(m,bad)
        with self.assertRaises(RuntimeError):load_generator_compatible(m,state,declared_mode='feature_motion_aligned')
        original={k:v for k,v in state.items() if k.startswith('backbone.')}
        report=load_generator_compatible(m,original);self.assertTrue(report['original_GFVC'])
        bad=dict(original);del bad['backbone.bias']
        with self.assertRaises(RuntimeError):load_generator_compatible(m,bad)


if __name__=='__main__':unittest.main()
