import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class PairRandomCrop(transforms.RandomCrop):

    def __call__(self, image, label):

        if self.padding is not None:
            image = F.pad(image, self.padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and image.size[0] < self.size[1]:
            image = F.pad(image, (self.size[1] - image.size[0], 0), self.fill, self.padding_mode)
            label = F.pad(label, (self.size[1] - label.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and image.size[1] < self.size[0]:
            image = F.pad(image, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)
            label = F.pad(label, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(image, self.size)

        return F.crop(image, i, j, h, w), F.crop(label, i, j, h, w)


class PairCompose(transforms.Compose):
    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class PairRandomHorizontalFilp(transforms.RandomHorizontalFlip):
    def __call__(self, img, label):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img), F.hflip(label)
        return img, label


class PairToTensor(transforms.ToTensor):
    def __call__(self, pic, label):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic), F.to_tensor(label)


class TripleRandomCrop(transforms.RandomCrop):

    def __call__(self, image1, image2, label):

        if self.padding is not None:
            image1 = F.pad(image1, self.padding, self.fill, self.padding_mode)
            image2 = F.pad(image2, self.padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and image1.size[0] < self.size[1]:
            image1 = F.pad(image1, (self.size[1] - image1.size[0], 0), self.fill, self.padding_mode)
            image2 = F.pad(image2, (self.size[1] - image2.size[0], 0), self.fill, self.padding_mode)
            label = F.pad(label, (self.size[1] - label.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and image1.size[1] < self.size[0]:
            image1 = F.pad(image1, (0, self.size[0] - image1.size[1]), self.fill, self.padding_mode)
            image2 = F.pad(image2, (0, self.size[0] - image2.size[1]), self.fill, self.padding_mode)
            label = F.pad(label, (0, self.size[0] - image1.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(image1, self.size)

        return F.crop(image1, i, j, h, w),F.crop(image2, i, j, h, w), F.crop(label, i, j, h, w)


class TripleCompose(transforms.Compose):
    def __call__(self, image1, image2, label):
        for t in self.transforms:
            image1, image2, label = t(image1, image2, label)
        return image1, image2, label


class TripleRandomHorizontalFilp(transforms.RandomHorizontalFlip):
    def __call__(self, img1, img2, label):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img1), F.hflip(img2), F.hflip(label)
        return img1, img2, label


class TripleRandomVerticalFlip(transforms.RandomVerticalFlip):
    def __call__(self, img1, img2, label):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.vflip(img1), F.vflip(img2), F.vflip(label)
        return img1, img2, label


class TripleToTensor(transforms.ToTensor):
    def __call__(self, pic1, pic2, label):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic1), F.to_tensor(pic2), F.to_tensor(label)
