from utils import Build_model
import tensorflow as tf
import torch
from module.flow import cnf
import numpy as np
import pickle
import os
import copy


class LatentState:
    def __init__(self, w, flow_attributes, flow_lights) -> None:
        self.w = w
        self.flow_attributes = flow_attributes
        self.flow_lights = flow_lights


class GANLatentFactory:
    def __init__(self) -> None:
        DATA_ROOT = "./data"
        self.raw_w = pickle.load(
            open(os.path.join(DATA_ROOT, "sg2latents.pickle"), "rb")
        )
        self.raw_attributes = np.load(os.path.join(DATA_ROOT, "attributes.npy"))
        self.raw_lights = np.load(os.path.join(DATA_ROOT, "light.npy"))

    def get_state(self, idx) -> LatentState:
        if idx >= 1000:
            raise Exception("Latent index %i is out of bounds. Max index is 999" % idx)
        return LatentState(
            w=self.raw_w["Latent"][idx],
            flow_attributes=self.raw_attributes[idx].ravel(),
            flow_lights=self.flow_lights[idx],
        )

    def get_random_state(self) -> LatentState:
        return self.get_state(np.random.randint(1000))


class App:
    class Opt:
        def __init__(self, network_pkl) -> None:
            self.network_pkl = network_pkl

    def __init__(self) -> None:
        # Open a new TensorFlow session.
        config = tf.ConfigProto(allow_soft_placement=True)
        session = tf.Session(config=config)

        opt = self.Opt("gdrive:networks/stylegan2-ffhq-config-f.pkl")
        with session.as_default():
            model = Build_model(opt)
            w_avg = model.Gs.get_var("dlatent_avg")

        prior = cnf(512, "512-512-512-512-512", 17, 1)
        prior.load_state_dict(torch.load("flow_weight/modellarge10k.pt"))
        prior.eval()

        self.session = session
        self.gan_model = model
        self.w_avg = w_avg
        self.flow_model = prior.cpu()
        self.pca_components = np.load(
            "pca_components/stylegan2-ffhq_style_ipca_c80_n300000_w.npz"
        )["lat_comp"]

    def change_flow_attribute(
        self, prev_image_state, attr_index, new_value
    ) -> LatentState:
        new_image_state = copy.deepcopy(prev_image_state)
        new_image_state.flow_attributes[attr_index] = new_value
        prev_z = self.__flow_w_to_z(*prev_image_state)
        new_w = self.__flow_z_to_w(
            prev_z, new_image_state.flow_attributes, new_image_state.flow_lights
        )
        new_image_state.w = self.__isolate_layers(new_w, prev_image_state.w, attr_index)
        return new_image_state

    def change_pca_attribute(
        self, prev_image_state, component_index, start_layer, end_layer, increment
    ) -> LatentState:
        new_image_state = copy.deepcopy(prev_image_state)
        component_direction = self.pca_components[component_index]
        for layer in range(start_layer, end_layer):
            new_image_state.w[0][layer] += component_direction * increment
        return new_image_state

    @torch.no_grad()
    def generate_image(self, latent_state) -> np.ndarray:
        with self.session.as_default():
            img = self.model.generate_im_from_w_space(latent_state.w)[0].copy()
        return img

    @staticmethod
    def __isolate_layers(new_w, orig_w, attr_index) -> np.ndarray:
        """Used to isoalate layers for flow attributes change."""
        orig_w = torch.Tensor(orig_w)
        if attr_index == 0:
            new_w[0][8:] = orig_w[0][8:]

        elif attr_index == 1:
            new_w[0][:2] = orig_w[0][:2]
            new_w[0][4:] = orig_w[0][4:]

        elif attr_index == 2:
            new_w[0][4:] = orig_w[0][4:]

        elif attr_index == 3:
            new_w[0][4:] = orig_w[0][4:]

        elif attr_index == 4:
            new_w[0][6:] = orig_w[0][6:]

        elif attr_index == 5:
            new_w[0][:5] = orig_w[0][:5]
            new_w[0][10:] = orig_w[0][10:]

        elif attr_index == 6:
            new_w[0][0:4] = orig_w[0][0:4]
            new_w[0][8:] = orig_w[0][8:]

        elif attr_index == 7:
            new_w[0][:4] = orig_w[0][:4]
            new_w[0][6:] = orig_w[0][6:]
        return new_w

    @torch.no_grad()
    def __flow_w_to_z(self, w, attributes, lighting):
        w_cuda = torch.Tensor(w)
        att_cuda = (
            torch.from_numpy(np.asarray(attributes))
            .float()
            .unsqueeze(0)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        light_cuda = torch.Tensor(lighting)

        features = torch.cat([light_cuda, att_cuda], dim=1).clone().detach()
        zero_padding = torch.zeros(1, 18, 1)
        z = self.flow_model(w_cuda, features, zero_padding)[0].clone().detach()

        return z

    @torch.no_grad()
    def __flow_z_to_w(self, z, attributes, lighting):
        att_cuda = (
            torch.Tensor(np.asarray(attributes))
            .float()
            .unsqueeze(0)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        light_cuda = torch.Tensor(lighting)

        features = torch.cat([light_cuda, att_cuda], dim=1).clone().detach()
        zero_padding = torch.zeros(1, 18, 1)
        w = self.flow_model(z, features, zero_padding, True)[0].clone().detach().numpy()

        return w
