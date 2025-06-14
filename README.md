# Instance Segmentation + GAN Model for Generating 2D Sprites Based on Character Portraits

## Structure

![img](./img/diagram.png)

1. We make an instance segmentation model that will collect character portraits and 2D sprites to speed up data collection process.
2. img2img model to generate 2D sprites from character portraits. Start with `idle-down` generation, and then expand to creating whole set of sprites.

## Environment
setting up docker files

```sh
$ docker build -t nino ./dockerfiles
```


## Inferencing

### 1. Setup Environment

```sh
$ ./setup_models.ps1
$ ./scripts/run_mask_rcnn.ps1
```