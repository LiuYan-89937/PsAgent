"""Centralized tool metadata shared by planner, runtime, and state models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToolMetadata:
    """Normalized metadata for one planner-visible tool."""

    name: str
    status_label: str
    keywords: tuple[str, ...] = ()
    macro: bool = False
    whole_image_only: bool = False


EXISTING_TOOL_METADATA: tuple[ToolMetadata, ...] = (
    ToolMetadata("adjust_exposure", "正在调整曝光", ("曝光", "提亮", "压暗", "变亮", "变暗")),
    ToolMetadata("adjust_highlights_shadows", "正在调整高光和阴影", ("高光", "阴影")),
    ToolMetadata("adjust_contrast", "正在调整对比度", ("对比度", "层次")),
    ToolMetadata("adjust_whites_blacks", "正在调整白场和黑场", ("白场", "黑场", "黑位", "白位")),
    ToolMetadata("adjust_curves", "正在调整曲线", ("曲线", "压高光", "提暗部")),
    ToolMetadata("adjust_clarity", "正在增强清晰度", ("清晰度", "通透", "局部对比")),
    ToolMetadata("adjust_texture", "正在调整纹理", ("纹理", "质感", "细节感")),
    ToolMetadata("adjust_dehaze", "正在去灰雾", ("去灰", "去雾", "空气感", "通透感")),
    ToolMetadata("adjust_color_mixer", "正在调整颜色混合", ("HSL", "色相", "颜色混合", "颜色分离")),
    ToolMetadata("adjust_white_balance", "正在调整白平衡", ("白平衡", "色温", "偏暖", "偏冷")),
    ToolMetadata("adjust_vibrance_saturation", "正在调整色彩", ("饱和度", "色彩", "鲜艳", "自然饱和度")),
    ToolMetadata("crop_and_straighten", "正在裁剪和拉直", ("裁剪", "拉直", "构图"), whole_image_only=True),
    ToolMetadata("denoise", "正在去噪", ("降噪", "去噪")),
    ToolMetadata("sharpen", "正在锐化", ("锐化", "清晰", "更锐")),
)

PRIMITIVE_TOOL_METADATA: tuple[ToolMetadata, ...] = (
    ToolMetadata("remove_heal", "正在智能修复", ("修复", "去除", "移除杂物", "智能去除", "remove")),
    ToolMetadata("blemish_remove", "正在去除瑕疵", ("去痘", "祛痘", "瑕疵", "痘痘", "去瑕疵")),
    ToolMetadata("skin_smooth", "正在柔化皮肤", ("磨皮", "皮肤柔化", "柔肤", "皮肤更细腻")),
    ToolMetadata("point_color", "正在精准调色", ("点颜色", "精准调色", "point color", "只调这个颜色")),
    ToolMetadata("spot_heal", "正在点修复", ("点修复", "小瑕疵修复", "污点修复")),
    ToolMetadata("clone_stamp", "正在克隆修复", ("仿制图章", "克隆修复", "复制修补")),
    ToolMetadata("skin_texture_reduce", "正在减弱皮肤纹理", ("减弱皮肤纹理", "减少皮肤纹理", "皮肤纹理弱一点")),
    ToolMetadata("under_eye_brighten", "正在提亮眼下", ("提亮眼下", "淡化黑眼圈", "眼下亮一点")),
    ToolMetadata("teeth_whiten", "正在美白牙齿", ("牙齿美白", "牙更白", "美白牙齿")),
    ToolMetadata("eye_brighten", "正在提亮眼睛", ("提亮眼睛", "眼睛更亮", "瞳孔增强")),
    ToolMetadata("hair_enhance", "正在增强发丝质感", ("发丝质感", "头发更有质感", "增强头发细节")),
    ToolMetadata("lip_enhance", "正在增强唇部质感", ("唇色", "嘴唇更有气色", "增强唇部")),
    ToolMetadata("reflection_reduce", "正在减弱反射和眩光", ("去反光", "减弱反射", "眩光", "玻璃反光")),
    ToolMetadata("lens_correction", "正在校正镜头畸变", ("镜头校正", "畸变校正"), whole_image_only=True),
    ToolMetadata("remove_chromatic_aberration", "正在去除色差", ("色差", "紫边", "chromatic aberration"), whole_image_only=True),
    ToolMetadata("defringe", "正在去除边缘色散", ("去边色", "defringe", "边缘紫边"), whole_image_only=True),
    ToolMetadata("perspective_correction", "正在校正透视", ("透视校正", "拉正建筑", "keystone"), whole_image_only=True),
    ToolMetadata("auto_upright", "正在自动扶正", ("自动扶正", "upright", "自动拉正"), whole_image_only=True),
    ToolMetadata("vignette", "正在调整暗角", ("暗角", "边缘压暗", "vignette")),
    ToolMetadata("grain", "正在添加颗粒", ("颗粒", "胶片颗粒", "grain")),
    ToolMetadata("moire_reduce", "正在抑制摩尔纹", ("摩尔纹", "moire")),
    ToolMetadata("color_grading", "正在进行色彩分级", ("色彩分级", "调色轮", "color grading")),
    ToolMetadata("apply_lut", "正在应用风格预设", ("LUT", "预设", "风格预设")),
    ToolMetadata("convert_black_white", "正在转换黑白", ("黑白", "black and white", "单色")),
    ToolMetadata("camera_calibration", "正在校准相机色彩", ("camera calibration", "校准颜色", "原色校准")),
    ToolMetadata("background_blur", "正在虚化背景", ("背景虚化", "背景模糊", "虚化背景")),
    ToolMetadata("lens_blur", "正在模拟镜头虚化", ("镜头虚化", "景深", "bokeh")),
    ToolMetadata("glow_highlight", "正在增强高光氛围", ("高光氛围", "glow", "bloom", "发光")),
)

MACRO_TOOL_METADATA: tuple[ToolMetadata, ...] = (
    ToolMetadata("portrait_natural_whitening", "正在进行自然美白", ("自然美白", "美白", "净白"), macro=True),
    ToolMetadata("portrait_skin_clean_tone", "正在净透肤色", ("肤色净透", "皮肤净透", "肤色更干净"), macro=True),
    ToolMetadata("portrait_backlight_repair", "正在修复逆光人像", ("逆光修复", "背光修复", "逆光人像修复"), macro=True),
    ToolMetadata("wedding_dress_protect", "正在保护婚纱高光", ("婚纱保护", "白裙保护", "婚纱细节"), macro=True),
    ToolMetadata("summer_airy_look", "正在营造夏日通透感", ("夏日通透", "空气感", "夏日感", "夏日质感"), macro=True),
    ToolMetadata("portrait_retouch", "正在进行人像精修", ("人像精修", "精修人像", "商业人像"), macro=True),
    ToolMetadata("portrait_hair_detail_boost", "正在增强发丝细节", ("发丝增强", "头发细节", "发丝质感增强"), macro=True),
    ToolMetadata("product_specular_enhance", "正在增强商品高光质感", ("商品高光质感", "玻璃质感", "高光质感增强"), macro=True),
    ToolMetadata("cleanup_skin_blemishes", "正在清理皮肤瑕疵", ("去痘去瑕", "清理瑕疵", "皮肤瑕疵"), macro=True),
    ToolMetadata("cleanup_distracting_objects", "正在清理干扰物", ("去杂物", "清理干扰物", "移除杂物"), macro=True),
    ToolMetadata("remove_passersby", "正在去除路人", ("去路人", "移除路人", "路人清理"), macro=True),
)

ALL_TOOL_METADATA: tuple[ToolMetadata, ...] = (
    EXISTING_TOOL_METADATA + PRIMITIVE_TOOL_METADATA + MACRO_TOOL_METADATA
)

TOOL_METADATA_BY_NAME = {item.name: item for item in ALL_TOOL_METADATA}
ALL_TOOL_NAMES: tuple[str, ...] = tuple(item.name for item in ALL_TOOL_METADATA)
MACRO_TOOL_NAMES = frozenset(item.name for item in MACRO_TOOL_METADATA)
WHOLE_IMAGE_ONLY_TOOL_NAMES = frozenset(item.name for item in ALL_TOOL_METADATA if item.whole_image_only)
PACKAGE_STATUS_LABELS = {item.name: item.status_label for item in ALL_TOOL_METADATA}
PARSE_REQUEST_KEYWORDS = tuple((item.name, item.keywords) for item in ALL_TOOL_METADATA if item.keywords)


def validate_tool_name(name: str) -> str:
    """Validate that a planner-facing tool name is registered."""

    if name not in TOOL_METADATA_BY_NAME:
        raise ValueError(f"Unsupported tool name: {name}")
    return name
