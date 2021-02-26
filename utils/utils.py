import numpy as np

from .. import dnnlib
from . import pretrained_networks
from ..dnnlib import tflib


class Build_model:
    def __init__(self, opt):

        self.opt = opt
        network_pkl = self.opt.network_pkl
        print('Loading networks from "%s"...' % network_pkl)
        _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
        self.Gs = Gs
        self.Gs_syn_kwargs = dnnlib.EasyDict()
        self.Gs_syn_kwargs.output_transform = dict(
            func=tflib.convert_images_to_uint8, nchw_to_nhwc=True
        )
        self.Gs_syn_kwargs.randomize_noise = False
        self.Gs_syn_kwargs.minibatch_size = 4
        self.noise_vars = [
            var
            for name, var in Gs.components.synthesis.vars.items()
            if name.startswith("noise")
        ]
        rnd = np.random.RandomState(0)
        tflib.set_vars(
            {var: rnd.randn(*var.shape.as_list()) for var in self.noise_vars}
        )

    def generate_im_from_random_seed(self, seed=22, truncation_psi=0.5):
        Gs = self.Gs
        seeds = [seed]
        noise_vars = [
            var
            for name, var in Gs.components.synthesis.vars.items()
            if name.startswith("noise")
        ]

        Gs_kwargs = dnnlib.EasyDict()
        Gs_kwargs.output_transform = dict(
            func=tflib.convert_images_to_uint8, nchw_to_nhwc=True
        )
        Gs_kwargs.randomize_noise = False
        if truncation_psi is not None:
            Gs_kwargs.truncation_psi = truncation_psi

        for seed_idx, seed in enumerate(seeds):
            print(
                "Generating image for seed %d (%d/%d) ..."
                % (seed, seed_idx, len(seeds))
            )
            rnd = np.random.RandomState(seed)
            z = rnd.randn(1, *Gs.input_shape[1:])  # [minibatch, component]
            tflib.set_vars(
                {var: rnd.randn(*var.shape.as_list()) for var in noise_vars}
            )  # [height, width]
            images = Gs.run(z, None, **Gs_kwargs)  # [minibatch, height, width, channel]
            # PIL.Image.fromarray(images[0], 'RGB').save(dnnlib.make_run_dir_path('seed%04d.png' % seed))
        return images

    def generate_im_from_z_space(self, z, truncation_psi=0.5):
        Gs = self.Gs

        Gs_kwargs = dnnlib.EasyDict()
        Gs_kwargs.output_transform = dict(
            func=tflib.convert_images_to_uint8, nchw_to_nhwc=True
        )
        Gs_kwargs.randomize_noise = False
        if truncation_psi is not None:
            Gs_kwargs.truncation_psi = truncation_psi  # [height, width]

        images = Gs.run(z, None, **Gs_kwargs)
        # PIL.Image.fromarray(images[0], 'RGB').save(dnnlib.make_run_dir_path('test_from_z.png'))
        return images

    def generate_im_from_w_space(self, w):

        images = self.Gs.components.synthesis.run(w, **self.Gs_syn_kwargs)
        # PIL.Image.fromarray(images[0], 'RGB').save(dnnlib.make_run_dir_path('test_from_w.png'))
        return images
