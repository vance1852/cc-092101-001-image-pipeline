import pytest
from image_pipeline.algorithms import core as alg

class TestGrayscale:

    def test_red_luminance(self):
        img = [[[255, 0, 0]]]
        gray = alg.to_grayscale(img)
        assert gray[0][0] == 76

    def test_green_luminance(self):
        img = [[[0, 255, 0]]]
        gray = alg.to_grayscale(img)
        assert gray[0][0] == 149

    def test_blue_luminance(self):
        img = [[[0, 0, 255]]]
        gray = alg.to_grayscale(img)
        assert gray[0][0] == 29

    def test_white_stays_255(self):
        img = [[[255, 255, 255]]]
        gray = alg.to_grayscale(img)
        assert gray[0][0] == 255

    def test_black_stays_0(self):
        img = [[[0, 0, 0]]]
        gray = alg.to_grayscale(img)
        assert gray[0][0] == 0

    def test_passthrough_already_gray(self):
        img = [[10, 20], [30, 40]]
        result = alg.to_grayscale(img)
        assert result == img

    def test_preserves_shape(self):
        img = [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]]
        gray = alg.to_grayscale(img)
        assert len(gray) == 2
        assert len(gray[0]) == 2

class TestBrightness:

    def test_add_positive(self):
        img = [[0, 50], [100, 200]]
        result = alg.adjust_brightness(img, 50)
        assert result == [[50, 100], [150, 250]]

    def test_clamps_at_255(self):
        img = [[200]]
        result = alg.adjust_brightness(img, 100)
        assert result[0][0] == 255

    def test_clamps_at_0(self):
        img = [[50]]
        result = alg.adjust_brightness(img, -100)
        assert result[0][0] == 0

    def test_zero_no_change(self):
        img = [[100, 150], [50, 200]]
        result = alg.adjust_brightness(img, 0)
        assert result == img

class TestContrast:

    def test_contrast_1x_no_change(self):
        img = [[64, 128], [192, 200]]
        result = alg.adjust_contrast(img, 1.0)
        assert result[0][1] == 128

    def test_contrast_double_clamps(self):
        img = [[64, 128], [192, 100]]
        result = alg.adjust_contrast(img, 2.0)
        assert result[0][0] == 0
        assert result[1][0] == 255

    def test_zero_contrast_flat_gray(self):
        img = [[0, 128], [255, 50]]
        result = alg.adjust_contrast(img, 0.0)
        for row in result:
            for v in row:
                assert v == 128

class TestThreshold:

    @pytest.fixture
    def sample_gray(self):
        return [[0, 64, 128, 255], [127, 128, 129, 200]]

    def test_binary_threshold(self, sample_gray):
        result = alg.threshold(sample_gray, 128, mode='binary')
        expected = [[0, 0, 0, 255], [0, 0, 255, 255]]
        assert result == expected

    def test_binary_inv(self, sample_gray):
        result = alg.threshold(sample_gray, 128, mode='binary_inv')
        expected = [[255, 255, 255, 0], [255, 255, 0, 0]]
        assert result == expected

    def test_tozero_mode(self, sample_gray):
        result = alg.threshold(sample_gray, 128, mode='tozero')
        expected = [[0, 0, 0, 255], [0, 0, 129, 200]]
        assert result == expected

    def test_truncate_mode(self, sample_gray):
        result = alg.threshold(sample_gray, 128, mode='truncate')
        expected = [[0, 64, 128, 128], [127, 128, 128, 128]]
        assert result == expected

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            alg.threshold([[100]], 128, mode='invalid_mode')

    def test_rgb_input_converts_to_gray(self):
        img = [[[255, 0, 0], [0, 255, 0]]]
        result = alg.threshold(img, 100)
        assert isinstance(result[0][0], int)

class TestConvolution:

    def test_box_blur_even_size_raises(self):
        img = [[0] * 10 for _ in range(10)]
        with pytest.raises(ValueError, match='odd'):
            alg.box_blur(img, size=4)

    def test_gaussian_negative_sigma_raises(self):
        img = [[0] * 10 for _ in range(10)]
        with pytest.raises(ValueError):
            alg.gaussian_blur(img, size=3, sigma=-1.0)

    def test_sharpen_negative_amount_raises(self):
        img = [[0] * 10 for _ in range(10)]
        with pytest.raises(ValueError):
            alg.sharpen(img, amount=-0.5)

    def test_box_blur_center_unchanged_for_uniform(self):
        img = [[128] * 10 for _ in range(10)]
        result = alg.box_blur(img, size=3)
        assert all((v == 128 for row in result for v in row))

    def test_box_blur_preserves_shape(self):
        img = [[0] * 15 for _ in range(12)]
        result = alg.box_blur(img, size=3)
        assert len(result) == 12
        assert len(result[0]) == 15

    def test_convolve_values_in_range(self):
        from image_pipeline.algorithms.core import generate_checkerboard
        img = generate_checkerboard(16, 16, tile_size=4)
        gray = alg.to_grayscale(img)
        result = alg.box_blur(gray, size=5)
        for row in result:
            for v in row:
                assert 0 <= v <= 255

    def test_even_kernel_raises_validation_error(self):
        img = [[0] * 5 for _ in range(5)]
        bad_kernel = [[1, 2], [3, 4]]
        with pytest.raises(ValueError, match='odd'):
            alg.convolve(img, bad_kernel)

class TestEdgeDetection:

    def test_sobel_x_vertical_edge(self):
        img = [[0, 255, 255], [0, 255, 255], [0, 255, 255]]
        result = alg.sobel_edges(img, direction='x')
        assert any((v > 0 for row in result for v in row))
        assert all((0 <= v <= 255 for row in result for v in row))

    def test_sobel_y_horizontal_edge(self):
        img = [[0, 0, 0], [255, 255, 255], [255, 255, 255]]
        result = alg.sobel_edges(img, direction='y')
        assert any((v > 0 for row in result for v in row))

    def test_sobel_magnitude(self):
        img = [[0, 255], [255, 0]]
        result = alg.sobel_edges(img, direction='both')
        assert all((0 <= v <= 255 for row in result for v in row))

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError):
            alg.sobel_edges([[0]], direction='diagonal')

    def test_prewitt_produces_output(self):
        img = [[0, 0, 255], [0, 0, 255], [0, 0, 255]]
        result = alg.prewitt_edges(img, direction='both')
        assert len(result) == 3
        assert len(result[0]) == 3

class TestCrop:

    @pytest.fixture
    def big_image(self):
        return [[i + j * 10 for i in range(10)] for j in range(10)]

    def test_crop_extracts_correct_region(self, big_image):
        cropped = alg.crop(big_image, x=2, y=3, width=4, height=3)
        assert len(cropped) == 3
        assert len(cropped[0]) == 4
        assert cropped[0][0] == 2 + 3 * 10
        assert cropped[2][3] == 5 + 5 * 10

    def test_crop_width_out_of_bounds_raises(self, big_image):
        with pytest.raises(ValueError, match='exceeds'):
            alg.crop(big_image, x=0, y=0, width=20, height=5)

    def test_crop_negative_origin_raises(self, big_image):
        with pytest.raises(ValueError):
            alg.crop(big_image, x=-1, y=0, width=5, height=5)

    def test_crop_zero_width_raises(self, big_image):
        with pytest.raises(ValueError):
            alg.crop(big_image, x=0, y=0, width=0, height=5)

    def test_crop_xy_plus_dim_exceeds_raises(self, big_image):
        with pytest.raises(ValueError):
            alg.crop(big_image, x=8, y=8, width=5, height=2)

class TestResize:

    def test_resize_nearest_2x_shape(self):
        img = [[10, 20], [30, 40]]
        result = alg.resize(img, scale=2.0, method='nearest')
        assert len(result) == 4
        assert len(result[0]) == 4

    def test_resize_nearest_pixel_expansion(self):
        img = [[10, 20], [30, 40]]
        result = alg.resize(img, scale=2.0, method='nearest')
        assert result[0][0] == 10
        assert result[1][1] == 10
        assert result[2][2] == 40
        assert result[3][3] == 40

    def test_resize_down_shape(self):
        img = [[i for i in range(8)] for _ in range(8)]
        result = alg.resize(img, scale=0.5, method='nearest')
        assert len(result) == 4
        assert len(result[0]) == 4

    def test_negative_scale_raises(self):
        img = [[0] * 5 for _ in range(5)]
        with pytest.raises(ValueError, match='positive'):
            alg.resize(img, scale=-1.0)

    def test_bilinear_scales_correctly(self):
        img = [[0, 255], [255, 0]]
        result = alg.resize(img, scale=2.0, method='bilinear')
        assert len(result) == 4
        assert len(result[0]) == 4

    def test_resize_zero_dimensions_not_allowed(self):
        img = [[1, 2], [3, 4]]
        with pytest.raises(ValueError):
            alg.resize(img, target_width=0, target_height=10)

    def test_resize_with_no_constraints_raises(self):
        img = [[1, 2], [3, 4]]
        with pytest.raises(ValueError):
            alg.resize(img)
