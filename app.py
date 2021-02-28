from .utils.utils import Build_model
import tensorflow as tf
import torch
from .flow_module.flow import cnf
import numpy as np
import pickle
import os
import copy


class LatentState:
    """Contains W latent vector, continous normalizing flows attributes 
    and lights (used only for compatability reasons)
    """

    def __init__(self, w, flow_attributes, flow_lights) -> None:
        self.w = w
        self.flow_attributes = flow_attributes
        self.flow_lights = flow_lights


class GANLatentFactory:
    """LatentState factory to get W vector alongside with cnf attributes and lights.
    """

    def __init__(self) -> None:
        scriptpath = os.path.dirname(__file__)
        self.raw_w = pickle.load(
            open(os.path.join(scriptpath, "data/sg2latents.pickle"), "rb")
        )
        self.raw_attributes = np.load(os.path.join(scriptpath, "data/attributes.npy"))
        self.raw_lights = np.load(os.path.join(scriptpath, "data/light.npy"))

    def get_state(self, idx) -> LatentState:
        """Generate LatentState from available W vector and cnf attributes.

        Args:
            idx (int):

        Raises:
            IndexError: only 1000 W vectors with cnf attributes are available

        Returns:
            LatentState:
        """
        return LatentState(
            w=self.raw_w["Latent"][idx],
            flow_attributes=self.raw_attributes[idx].ravel(),
            flow_lights=self.raw_lights[idx],
        )

    def get_random_state(self) -> LatentState:
        return self.get_state(np.random.randint(1000))


class App:
    """Main entry point. 
    Class implements two main methods of latent vector modifications:
      * attributes change using normilizing flows from StyleFlow framework
      * attributes change using main PCA components from GANSpace framework
    During initializations models are created.
    """

    class Opt:
        # dirty hack class
        def __init__(self, network_pkl) -> None:
            self.network_pkl = network_pkl

    def __init__(self) -> None:
        """Donwloads and sets stylegan2 model weights, 
        attributes' flows model weights, pca components of W latent space.
        Inits tf session.
        """
        # Open a new TensorFlow session.
        scriptpath = os.path.dirname(__file__)
        config = tf.ConfigProto(allow_soft_placement=True)
        session = tf.Session(config=config)

        opt = self.Opt("gdrive:networks/stylegan2-ffhq-config-f.pkl")
        with session.as_default():
            model = Build_model(opt)

        prior = cnf(512, "512-512-512-512-512", 17, 1)
        prior.load_state_dict(
            torch.load(os.path.join(scriptpath, "data/modellarge10k.pt"))
        )
        prior.eval()

        self.session = session
        self.gan_model = model
        self.flow_model = prior.cpu()
        self.pca_components = np.load(
            os.path.join(scriptpath, "data/stylegan2-ffhq_style_ipca_c80_n300000_w.npz")
        )["lat_comp"]

    def change_flow_attribute(
        self, prev_image_state, attr_index, new_value
    ) -> LatentState:
        """Change latent W vector using normilizing flows attribute.
        Pretrained continous normalizing flow(cnf) model is used to change the vector.
        To make change more disentangled, new vector is stripped in special way.  

        Args:
            prev_image_state (LatentState): state to change
            attr_index (int): attribute index to change
            new_value (float):

        Returns:
            LatentState:
        """
        new_image_state = copy.deepcopy(prev_image_state)
        new_image_state.flow_attributes[attr_index] = new_value
        prev_z = self.__flow_w_to_z(
            prev_image_state.w,
            prev_image_state.flow_attributes,
            prev_image_state.flow_lights,
        )
        new_w = self.__flow_z_to_w(
            prev_z, new_image_state.flow_attributes, new_image_state.flow_lights
        )
        new_image_state.w = self.__isolate_layers(new_w, prev_image_state.w, attr_index)
        return new_image_state

    def change_pca_attribute(
        self, prev_image_state, component_index, start_layer, end_layer, increment
    ) -> LatentState:
        """Change latent W vector using PCA component attribute.
        Each W vector is applied to 18 style layers during image generation.
        Each PCA component is a direction in W vector space.
        This method changes initial W vector in PCA component direction, but only on certain style layers.
        80 main PCA components are available in ascending order.

        Args:
            prev_image_state ([type]): state to change
            component_index ([type]): index of component
            start_layer ([type]): first style layer to edit (inclusive)
            end_layer ([type]): last style layer to edit (exclusive)
            increment ([type]): step into PCA component direction

        Returns:
            LatentState: [description]
        """
        new_image_state = copy.deepcopy(prev_image_state)
        component_direction = self.pca_components[component_index].squeeze()
        for layer in range(start_layer, end_layer):
            new_image_state.w[0][layer] += component_direction * increment
        return new_image_state

    @torch.no_grad()
    def generate_image(self, latent_state) -> np.ndarray:
        with self.session.as_default():
            img = self.gan_model.generate_im_from_w_space(latent_state.w)[0].copy()
        return img

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
