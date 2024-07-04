from typing import List
import itertools


imagenet_templates_small = [
    "a photo of a {}",
]


material_templates = [
    "which is made of {}"
]

category_templates_suffix = [
    "and is {} in color",
]


def build_bilevel_templates() -> List[str]:
    try:
        import glide_text2im
        return [
            subcat_template + ' ' + cat_template
            for subcat_template in imagenet_templates_small
            for cat_template in category_templates_suffix
        ]
    except Exception as _:
        return imagenet_templates_small


def build_bilevel_concat_templates() -> List[str]:
    return [subcat_template.replace('{}', '{} {}') for subcat_template in imagenet_templates_small]


def build_bilevel_templates_enforce() -> List[str]:
    return [
        subcat_template + ' ' + cat_template
        for subcat_template in imagenet_templates_small
        for cat_template in category_templates_suffix
    ]

def build_trilevel_templates_enforce() -> List[str]:
    return [
        subcat_template + ' ' + mat_template + ' ' + cat_template
        for subcat_template in imagenet_templates_small
        for mat_template in material_templates
        for cat_template in category_templates_suffix
    ]

def build_variable_level_templates(num_layers=2) -> List[str]:
    # note that num_layers counts the top layer, e.g. 'a photo of a <L> which is a <H>' has num_layers=2
    # TODO: move this to config

    cat_template_perms = list(itertools.product(category_templates_suffix, repeat=num_layers-1))
    cat_template_perms = [' '.join(perm) for perm in cat_template_perms]
    return [
        subcat_template + ' ' + cat_template
        for subcat_template in imagenet_templates_small
        for cat_template in cat_template_perms
    ]