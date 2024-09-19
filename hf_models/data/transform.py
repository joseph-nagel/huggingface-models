'''Data transformations.'''


class DataTransform:
    '''
    Helper class for applying an image transform to a datasets.Dataset.

    Summary
    -------
    An image transformation (torchvision) can be applied to all
    items from a Hugging Face datasets.Datasets individually.
    Additionally, the image and label dict keys can be renamed.

    Parameters
    ----------
    img_transform : callable
        Transformation applied to the images.
    img_source_key : str
        Key of the original images in a batch dict.
    img_target_key : str
        Key of the transformed images in a batch dict.
    lbl_source_key : str
        Key of the labels in the original batch dict.
    lbl_target_key : str
        Key of the labels in returned batch dict.

    '''

    def __init__(self,
                 img_transform,
                 img_source_key='img',
                 img_target_key='pixel_values',
                 lbl_source_key='label',
                 lbl_target_key='labels'):

        # set image transform
        self.img_transform = img_transform

        # prepare image keys
        if (img_source_key is not None) and (img_target_key is None):
            img_source_key = img_source_key
            img_target_key = img_source_key
        elif None in (img_source_key, img_target_key):
            raise TypeError('Invalid image source/target keys')

        # prepare label keys
        if (lbl_source_key is not None) and (lbl_target_key is None):
            lbl_source_key = lbl_source_key
            lbl_target_key = lbl_source_key
        elif None in (lbl_source_key, lbl_target_key):
            raise TypeError('Invalid image source/target keys')

        # set image and label keys
        img_keys = (img_source_key, img_target_key)
        lbl_keys = (lbl_source_key, lbl_target_key)

        if any([k in img_keys for k in lbl_keys]) or any([k in lbl_keys for k in img_keys]):
            raise ValueError('Image and label keys need to be strictly different')
        else:
            self.img_source_key = img_source_key
            self.img_target_key = img_target_key

            self.lbl_source_key = lbl_source_key
            self.lbl_target_key = lbl_target_key

    def __call__(self, batch_dict):

        # apply transform to images
        source_imgs = batch_dict.pop(self.img_source_key)

        if self.img_transform is not None:
            transformed_imgs = [self.img_transform(img.convert('RGB')) for img in source_imgs]
        else:
            transformed_imgs = source_imgs

        batch_dict[self.img_target_key] = transformed_imgs

        # rename labels
        batch_dict[self.lbl_target_key] = batch_dict.pop(self.lbl_source_key)

        return batch_dict

