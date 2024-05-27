import torch
import torch.nn.functional as F
from diffusers import (
    DDIMPipeline,
    DDIMScheduler,
    DDPMPipeline,
    DDPMScheduler,
    PNDMPipeline,
    UNet2DModel,
)
from .BasePredictor import BasePredictorClass


class DiffuserUNet(BasePredictorClass):
    def __init__(self, in_channels=3, mid_channels=128, out_channels=3, dropout=0., residual=False,
                 output_type="gaussian"):

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.residual = residual
        super().__init__(output_type=output_type)
        # super(torch.nn.Module, self).__init__()
        if output_type == 'ratio2':
            self.unet_birth = UNet2DModel(
                sample_size=(32, 32),
                in_channels=in_channels,
                out_channels=out_channels,
                block_out_channels=tuple([k * mid_channels for k in (1, 2, 3, 4)]),
                dropout=dropout,
                time_embedding_type="positional"
            )
            self.unet_death = UNet2DModel(
                sample_size=(32, 32),
                in_channels=in_channels,
                out_channels=out_channels,
                block_out_channels=tuple([k * mid_channels for k in (1, 2, 3, 4)]),
                dropout=dropout,
                time_embedding_type="positional"
            )
        else:
            self.unet = UNet2DModel(
                sample_size=(32, 32),
                in_channels=in_channels,
                out_channels=out_channels,
                block_out_channels=tuple([k * mid_channels for k in (1, 2, 3, 4)]),
                dropout=dropout,
                time_embedding_type="positional"
            )

    def predict(self, x, t):
        # assert x.min() >= -1.0 and x.max() <= 1.0
        if t.ndim > 1:
            t = t.mean(dim=[1, 2, 3])
        t = t * 1000
        if not self.output_type == 'ratio2':
            output = self.unet.forward(sample=x, timestep=t)["sample"]
        else:
            output_birth = self.unet_birth.forward(sample=x, timestep=t)["sample"]
            output_death = self.unet_birth.forward(sample=x, timestep=t)["sample"]
            output = torch.concat([output_death, output_birth], dim=1)

        return output

    def from_pretrained(self, model_id="google/ddpm-cifar10-32"):
        if self.output_type == 'ratio2':
            self.unet_birth = self.unet_birth.from_pretrained(model_id)
            print(f"Loaded Pretrained Diffuser model from {model_id}")
            # print(f"{self.unet=}")
            first_in_channels = self.unet_birth.conv_in.weight.shape[1]
            if first_in_channels != self.out_channels:
                first_out_channels = self.unet_birth.conv_in.weight.shape[0]
                self.unet_birth.conv_in = torch.nn.Conv2d(in_channels=self.in_channels, out_channels=first_out_channels,
                                                          kernel_size=3, padding=1)
                print(f"Adapted last layer from {first_in_channels} to {self.in_channels}")

            last_out_channels = list(self.unet_birth.modules())[-1].weight.shape[0]
            if last_out_channels != self.out_channels:
                last_in_channels = list(self.unet_birth.modules())[-1].weight.shape[1]
                self.unet_birth.conv_out = torch.nn.Conv2d(in_channels=last_in_channels, out_channels=self.out_channels,
                                                           kernel_size=3, padding=1)
                print(f"Adapted last layer from {last_out_channels} to {self.out_channels}")

            self.unet_death = self.unet_death.from_pretrained(model_id)
            print(f"Loaded Pretrained Diffuser model from {model_id}")
            # print(f"{self.unet=}")
            first_in_channels = self.unet_death.conv_in.weight.shape[1]
            if first_in_channels != self.out_channels:
                first_out_channels = self.unet_death.conv_in.weight.shape[0]
                self.unet_death.conv_in = torch.nn.Conv2d(in_channels=self.in_channels, out_channels=first_out_channels,
                                                          kernel_size=3, padding=1)
                print(f"Adapted last layer from {first_in_channels} to {self.in_channels}")

            last_out_channels = list(self.unet_death.modules())[-1].weight.shape[0]
            if last_out_channels != self.out_channels:
                last_in_channels = list(self.unet_death.modules())[-1].weight.shape[1]
                self.unet_death.conv_out = torch.nn.Conv2d(in_channels=last_in_channels, out_channels=self.out_channels,
                                                           kernel_size=3, padding=1)
                print(f"Adapted last layer from {last_out_channels} to {self.out_channels}")
        else:
            self.unet = self.unet.from_pretrained(model_id)
            print(f"Loaded Pretrained Diffuser model from {model_id}")
            # print(f"{self.unet=}")
            first_in_channels = self.unet.conv_in.weight.shape[1]
            if first_in_channels != self.out_channels:
                first_out_channels = self.unet.conv_in.weight.shape[0]
                self.unet.conv_in = torch.nn.Conv2d(in_channels=self.in_channels, out_channels=first_out_channels,
                                                    kernel_size=3, padding=1)
                print(f"Adapted last layer from {first_in_channels} to {self.in_channels}")

            last_out_channels = list(self.unet.modules())[-1].weight.shape[0]
            if last_out_channels != self.out_channels:
                last_in_channels = list(self.unet.modules())[-1].weight.shape[1]
                self.unet.conv_out = torch.nn.Conv2d(in_channels=last_in_channels, out_channels=self.out_channels,
                                                     kernel_size=3, padding=1)
                print(f"Adapted last layer from {last_out_channels} to {self.out_channels}")

        return self


if __name__ == "__main__":
    model_id = "google/ddpm-cifar10-32"

    model = UNet2DModel(sample_size=(32, 32), in_channels=1, out_channels=1,
                        time_embedding_type="positional").from_pretrained(model_id)
    # scheduler = DDIMScheduler.from_pretrained(model_id)
    # scheduler.set_timesteps(num_inference_steps=5)

    # load model and scheduler
    # ddpm = DDPMPipeline.from_pretrained(
    # 	model_id
    # ).to(device)  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference

    # run pipeline in inference (sample random noise and denoise)
    # image = ddpm().images[0]

    print(f"{model=}")
    print(isinstance(model, torch.nn.Module))

    data = torch.randn((5, 1, 32, 32))
    t = torch.randn((5,)).abs()

    out = model(sample=data, timestep=t)

    print(out["sample"].shape)
