import cv2
import numpy as np

from src.classical.segment import adaptive_mask, clean_mask, colour_distance_mask, mask_to_boxes


def sand(h=256, w=256):
    return np.full((h, w, 3), (170, 200, 220), np.uint8)


def test_colour_mask_finds_dark_bird_on_sand():
    image = sand()
    cv2.circle(image, (128, 128), 20, (30, 30, 30), -1)
    mask = colour_distance_mask(image, 4, 51, 15.0)
    assert mask[128, 128] == 1 and mask[20, 20] == 0


def test_colour_mask_finds_bright_bird_on_dark_water():
    image = np.full((256, 256, 3), (90, 60, 30), np.uint8)
    cv2.circle(image, (128, 128), 20, (245, 245, 245), -1)
    assert colour_distance_mask(image, 4, 51, 15.0)[128, 128] == 1


def test_colour_mask_matches_small_nonsquare_image():
    image = np.full((40, 90, 3), 128, np.uint8)
    assert colour_distance_mask(image, 4, 51, 15.0).shape == (40, 90)


def test_min_distance_floor_overrides_otsu():
    image = sand()
    cv2.circle(image, (128, 128), 20, (150, 185, 205), -1)  # faint: under the floor
    assert colour_distance_mask(image, 4, 51, 40.0).sum() == 0
    assert colour_distance_mask(image, 4, 51, 0.0)[128, 128] == 1


def test_adaptive_mask_finds_both_polarities():
    image = np.full((256, 256, 3), 128, np.uint8)
    cv2.circle(image, (70, 128), 15, (10, 10, 10), -1)
    cv2.circle(image, (190, 128), 15, (250, 250, 250), -1)
    mask = adaptive_mask(image, 101, 15)
    assert mask[128, 70] == 1 and mask[128, 190] == 1 and mask[20, 20] == 0


def test_clean_mask_removes_speck_and_fills_pinhole():
    mask = np.zeros((60, 60), np.uint8)
    mask[10:30, 10:30] = 1
    mask[20, 20] = 0
    mask[50, 50] = 1
    cleaned = clean_mask(mask, 3)
    assert cleaned[20, 20] == 1 and cleaned[50, 50] == 0


def test_clean_mask_kernel_one_is_unchanged():
    mask = np.zeros((10, 10), np.uint8)
    mask[5, 5] = 1
    assert (clean_mask(mask, 1) == mask).all()


def test_mask_to_boxes_uses_component_extent_not_a_square():
    mask = np.zeros((100, 100), np.uint8)
    mask[20:30, 10:50] = 1
    assert mask_to_boxes(mask, 16, 5, 0.25, 0.8).tolist() == [[10, 20, 50, 30]]


def test_mask_to_boxes_applies_area_aspect_and_fill_limits():
    mask = np.zeros((100, 100), np.uint8)
    mask[1:3, 1:3] = 1  # area 4
    mask[10:12, 10:60] = 1  # aspect 25
    cv2.line(mask, (40, 40), (80, 80), 1, 1)  # fill about 0.03
    assert len(mask_to_boxes(mask, 16, 4, 0.25, 0.8)) == 0


def test_full_frame_component_rejected():
    assert len(mask_to_boxes(np.ones((100, 100), np.uint8), 16, 4, 0.25, 0.8)) == 0
