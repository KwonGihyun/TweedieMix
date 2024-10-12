
import os
import random
from pathlib import Path
import numpy as np
import PIL
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# def preprocess(image, scale, resample):
#     image = image.resize((scale, scale), resample=resample)
#     image = np.array(image).astype(np.uint8)
#     image = (image / 127.5 - 1.0).astype(np.float32)
#     return image
def preprocess(image, scale, resample):
    image.thumbnail((scale,scale))
    image = np.array(image).astype(np.uint8)
    image = (image / 127.5 - 1.0).astype(np.float32)
    return image

def collate_fn(examples, with_prior_preservation):
    input_ids_one = [example["instance_prompt_ids_one"] for example in examples]
    input_ids_two = [example["instance_prompt_ids_two"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    mask = [example["mask"] for example in examples]
    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids_one += [example["class_prompt_ids_one"] for example in examples]
        input_ids_two += [example["class_prompt_ids_two"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]
        mask += [example["class_mask"] for example in examples]

    input_ids_one = torch.cat(input_ids_one, dim=0)
    input_ids_two = torch.cat(input_ids_two, dim=0)
    pixel_values = torch.stack(pixel_values)
    mask = torch.stack(mask)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    mask = mask.to(memory_format=torch.contiguous_format).float()

    batch = {
        "input_ids_one": input_ids_one,
        "input_ids_two": input_ids_two,
        "pixel_values": pixel_values,
        "mask": mask.unsqueeze(1)
    }
    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


class CustomDiffusionDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        concepts_list,
        tokenizer_one,
        tokenizer_two,
        size=512,
        center_crop=False,
        with_prior_preservation=False,
        num_class_images=200,
        hflip=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two
        self.interpolation = PIL.Image.BILINEAR

        self.instance_images_path = []
        self.class_images_path = []
        self.with_prior_preservation = with_prior_preservation
        for concept in concepts_list:
            inst_img_path = [(x, concept["instance_prompt"]) for x in Path(concept["instance_data_dir"]).iterdir() if x.is_file()]
            self.instance_images_path.extend(inst_img_path)

            if with_prior_preservation:
                class_data_root = Path(concept["class_data_dir"])
                if os.path.isdir(class_data_root):
                    class_images_path = list(class_data_root.iterdir())
                    class_prompt = [concept["class_prompt"] for _ in range(len(class_images_path))]
                else:
                    with open(class_data_root, "r") as f:
                        class_images_path = f.read().splitlines()
                    with open(concept["class_prompt"], "r") as f:
                        class_prompt = f.read().splitlines()

                class_img_path = [(x, y) for (x, y) in zip(class_images_path, class_prompt)]
                self.class_images_path.extend(class_img_path[:num_class_images])

        random.shuffle(self.instance_images_path)
        self.num_instance_images = len(self.instance_images_path)
        self.num_class_images = len(self.class_images_path)
        self._length = max(self.num_class_images, self.num_instance_images)
        self.flip = transforms.RandomHorizontalFlip(0.5 * hflip)

        self.image_transforms = transforms.Compose(
            [
                self.flip,
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image, instance_prompt = self.instance_images_path[index % self.num_instance_images]
        instance_image = Image.open(instance_image)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        instance_image = self.flip(instance_image)

        ##############################################################################
        #### apply resize augmentation and create a valid image region mask ##########
        ##############################################################################
#         if np.random.randint(0, 3) < 2:
#             random_scale = np.random.randint(self.size // 3, self.size+1)
#         else:
#             random_scale = np.random.randint(int(1.2*self.size), int(1.4*self.size))

#         if random_scale % 2 == 1:
#             random_scale += 1
        # random_scale = np.random.
    
        
        
        random_scale = np.random.randint(self.size // 3, self.size+1)
        
        instance_image1 = preprocess(instance_image, random_scale, self.interpolation)
        shape_x,shape_y,_ = instance_image1.shape
        # import pdb; pdb.set_trace()
        
        bias_x = np.random.randint(0,self.size-shape_x+1)
        bias_y = np.random.randint(0,self.size-shape_y+1)
        # if random_scale < 0.6*self.size:
            # add_to_caption = np.random.choice(["a far away ", "very small "])
        # add_to_caption = ""
        # instance_prompt = add_to_caption + instance_prompt
        # cx = np.random.randint(random_scale // 2, self.size - random_scale // 2 + 1)
        # cy = np.random.randint(random_scale // 2, self.size - random_scale // 2 + 1)
        instance_image1 = preprocess(instance_image, random_scale, self.interpolation)
        instance_image = np.zeros((self.size, self.size, 3), dtype=np.float32)
        instance_image[bias_x: bias_x+shape_x , bias_y: bias_y+shape_y , :] = instance_image1

        # instance_image[cx - random_scale // 2: cx + random_scale // 2  , cy - random_scale // 2 : cy + random_scale // 2 , :] = instance_image1

        mask = np.zeros((self.size // 8, self.size // 8))
        mask[(bias_x) // 8 + 1: (bias_x+shape_x) // 8 - 1, (bias_y) // 8 + 1: (bias_y+shape_y) // 8 - 1] = 1.
#         elif random_scale > self.size:
#             add_to_caption = np.random.choice(["zoomed in ", "close up "])
#             instance_prompt = add_to_caption + instance_prompt
#             cx = np.random.randint(self.size // 2, random_scale - self.size // 2 + 1)
#             cy = np.random.randint(self.size // 2, random_scale - self.size // 2 + 1)

#             instance_image = preprocess(instance_image, random_scale, self.interpolation)
#             instance_image = instance_image[cx - self.size // 2: cx + self.size // 2, cy - self.size // 2: cy + self.size // 2, :]
#             mask = np.ones((self.size // 8, self.size // 8))
#         else:
#             instance_image = preprocess(instance_image, self.size, self.interpolation)
#             mask = np.ones((self.size // 8, self.size // 8))
        ########################################################################

        example["instance_images"] = torch.from_numpy(instance_image).permute(2, 0, 1)
        example["mask"] = torch.from_numpy(mask)
        example["instance_prompt_ids_one"] = self.tokenizer_one(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer_one.model_max_length,
            return_tensors="pt",
        ).input_ids
        example["instance_prompt_ids_two"] = self.tokenizer_two(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer_two.model_max_length,
            return_tensors="pt",
        ).input_ids
        if self.with_prior_preservation:
            class_image, class_prompt = self.class_images_path[index % self.num_class_images]
            class_image = Image.open(class_image)
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_mask"] = torch.ones_like(example["mask"])
            example["class_prompt_ids_one"] = self.tokenizer_one(
                class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer_one.model_max_length,
                return_tensors="pt",
            ).input_ids
            example["class_prompt_ids_two"] = self.tokenizer_two(
                class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer_two.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example
