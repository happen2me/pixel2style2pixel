from abc import abstractmethod
import torchvision.transforms as transforms
from dataset import augmentations
import torch


class TransformsConfig(object):

	def __init__(self, opts):
		self.opts = opts

	@abstractmethod
	def get_transforms(self):
		pass


class EncodeTransforms(TransformsConfig):

	def __init__(self, opts):
		super(EncodeTransforms, self).__init__(opts)

	def get_transforms(self):
		transforms_dict = {
			'transform_gt_train': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.RandomHorizontalFlip(0.5),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_source': None,
			'transform_test': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_inference': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
		}
		return transforms_dict


class FrontalizationTransforms(TransformsConfig):

	def __init__(self, opts):
		super(FrontalizationTransforms, self).__init__(opts)

	def get_transforms(self):
		transforms_dict = {
			'transform_gt_train': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.RandomHorizontalFlip(0.5),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_source': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.RandomHorizontalFlip(0.5),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_test': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_inference': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
		}
		return transforms_dict


class SketchToImageTransforms(TransformsConfig):

	def __init__(self, opts):
		super(SketchToImageTransforms, self).__init__(opts)

	def get_transforms(self):
		transforms_dict = {
			'transform_gt_train': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_source': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor()]),
			'transform_test': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_inference': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor()]),
		}
		return transforms_dict


class SegToImageTransforms(TransformsConfig):

	def __init__(self, opts):
		super(SegToImageTransforms, self).__init__(opts)
		self.opts = opts

	def get_transforms(self):
		transforms_dict = {
			'transform_gt_train': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5] * self.opts.output_nc, [0.5] * self.opts.output_nc)]),
			'transform_source': transforms.Compose([
				transforms.ToTensor(),
				Conver2Uint8(),
				MyResize((256, 256)),
				ToOneHot(self.opts.label_nc)
				]),
			'transform_test': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5] * self.opts.output_nc, [0.5] * self.opts.output_nc)]),
			'transform_inference': transforms.Compose([
				transforms.ToTensor(),
				Conver2Uint8(),
				MyResize((256, 256)),
				ToOneHot(self.opts.label_nc)
				])
		}
		return transforms_dict


class SuperResTransforms(TransformsConfig):

	def __init__(self, opts):
		super(SuperResTransforms, self).__init__(opts)

	def get_transforms(self):
		if self.opts.resize_factors is None:
			self.opts.resize_factors = '1,2,4,8,16,32'
		factors = [int(f) for f in self.opts.resize_factors.split(",")]
		print("Performing down-sampling with factors: {}".format(factors))
		transforms_dict = {
			'transform_gt_train': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_source': transforms.Compose([
				transforms.Resize((256, 256)),
				augmentations.BilinearResize(factors=factors),
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_test': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_inference': transforms.Compose([
				transforms.Resize((256, 256)),
				augmentations.BilinearResize(factors=factors),
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
		}
		return transforms_dict


class Conver2Uint8(torch.nn.Module):
    '''
    Resize input when the target dim is not divisible by the input dim
    '''
    def __init__(self):
        super().__init__()

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        img = torch.round(torch.mul(img, 255))
        return img
    
class MyResize(torch.nn.Module):
    '''
    Resize input when the target dim is not divisible by the input dim
    '''
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        h, w = img.shape[-2], img.shape[-1]
        target_h, target_w = self.size
        assert h % target_h == 0, f"target_h({target_h}) must be divisible by h({h})"
        assert w % target_w == 0, f"target_w({target_w}) must be divisible by w({w})"
        # Resize by assigning the max value of each pixel grid
        kernel_h = h // target_h
        kernel_w = w // target_w
        img_target = torch.nn.functional.max_pool2d(img, kernel_size=(kernel_h, kernel_w), stride=(kernel_h, kernel_w))
        return img_target

class ToOneHot(torch.nn.Module):
	'''
	Convert input to one-hot encoding
	'''
	def __init__(self, num_classes):
		super().__init__()
		self.num_classes = num_classes
	
	def forward(self, img):
		"""
		Args:
			img (Tensor): Image to be scaled of shape (1, h, w).

		Returns:
			Tensor: Rescaled image.
		"""
		img = img.long()[0]
		img = torch.nn.functional.one_hot(img, num_classes=self.num_classes)
		img = img.permute(2, 0, 1)
		return img