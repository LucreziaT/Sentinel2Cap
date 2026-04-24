import hashlib

_BASE_PROMPT = (
    "Describe this satellite image in one single continuous paragraph, comprising less than 200"
    " words. Do not use bullet points, numbered lists, or section titles. Provide a detailed and"
    " natural description focusing on land cover, structures, spatial layout, and colors. Be"
    " precise about the location of the different elements in the image (e.g. in the upper"
    " portion, in the middle, ...). Use the following format: 'In this image, [description]. To"
    " sum up, [summary].'"
)

_SPECIFIC_PROMPT = (
    "Describe this satellite image in one single continuous paragraph, comprising less than 200"
    " words. Do not use bullet points, numbered lists, or section titles. The image is taken from"
    " the {} satellite and uses the following channels: {}. Use this information to better"
    " understand the image, but do not include it in the description. Provide a detailed"
    " description focusing on land cover, structures, spatial layout, and colors. Be precise about"
    " the location of the different elements in the image (e.g. in the upper portion, in the"
    " middle, ...). Use the following format: 'In this image, [description]. To sum up,"
    " [summary].'"
)

# Prompts dict
PROMPTS = {
    "base": _BASE_PROMPT,
    "specific": _SPECIFIC_PROMPT,
}

# Mapping from dataset to satellite and channels for the specific prompt
COMPLETIONS = {
    "ben": ("Sentinel-2", "B4 (Red), B3 (Green), B2 (Blue)"),
    "s1": ("Sentinel-1", "VV (Red), VH (Green), VV/VH ratio (Blue)"),
    "s2": ("Sentinel-2", "B6 (Red), 8A (Green), B12 (Blue)"),
}


def encode_prompt(prompt: str) -> str:
    """Convert prompt to hash to use as unique filename"""
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


__all__ = ["PROMPTS", "COMPLETIONS", "encode_prompt"]
