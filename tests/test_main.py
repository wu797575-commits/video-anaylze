import unittest
from io import BytesIO
from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import main


class ImageProcessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.full((8, 10, 3), 128, dtype=np.uint8)

    def test_parse_kernel_accepts_odd_square_rows(self) -> None:
        kernel = main.parse_kernel("1, 1, 1; 1, 1, 1; 1, 1, 1")
        self.assertEqual(kernel.shape, (3, 3))
        self.assertEqual(kernel.sum(), 9)

    def test_parse_kernel_rejects_invalid_shapes_and_weights(self) -> None:
        invalid_inputs = (
            "",
            "1, 2\n3, 4",
            "1, 2\n3, 4\n5, 6",
            "NaN",
            "1000001",
            "\n".join([" ".join(["1"] * 33)] * 33),
        )
        for text in invalid_inputs:
            with self.subTest(text=text[:30]):
                with self.assertRaises(ValueError):
                    main.parse_kernel(text)

    def test_normalized_smoothing_preserves_constant_image(self) -> None:
        kernel = main.parse_kernel("1, 1, 1\n1, 1, 1\n1, 1, 1")
        filtered, normalized = main.filter_image(self.image, kernel, True)
        self.assertTrue(normalized)
        np.testing.assert_array_equal(filtered, self.image)

    def test_zero_sum_kernel_requires_normalization_to_be_disabled(self) -> None:
        kernel = main.parse_kernel("0, -1, 0\n-1, 4, -1\n0, -1, 0")
        with self.assertRaisesRegex(ValueError, "總和為 0"):
            main.filter_image(self.image, kernel, True)
        filtered, normalized = main.filter_image(self.image, kernel, False)
        self.assertFalse(normalized)
        self.assertEqual(filtered.dtype, np.uint8)

    def test_psnr_identity_and_maximum_error(self) -> None:
        self.assertTrue(np.isinf(main.calculate_psnr(self.image, self.image)))
        black = np.zeros((1, 1, 3), dtype=np.uint8)
        white = np.full((1, 1, 3), 255, dtype=np.uint8)
        self.assertAlmostEqual(main.calculate_psnr(black, white), 0.0)

    def test_supported_bitmap_formats_decode_to_rgb(self) -> None:
        sample = Image.fromarray(self.image)
        for image_format in ("PNG", "JPEG", "BMP"):
            with self.subTest(image_format=image_format):
                buffer = BytesIO()
                sample.save(buffer, format=image_format)
                decoded, source_mode, size = main.load_uploaded_image(buffer.getvalue())
                self.assertEqual(decoded.shape, (8, 10, 3))
                self.assertEqual(source_mode, "RGB")
                self.assertEqual(size, (10, 8))

    def test_channel_views_and_histograms_cover_all_pixels(self) -> None:
        for mode in ("Original", "R", "G", "B", "Y", "U", "V"):
            with self.subTest(mode=mode):
                channels = main.get_display_channels(self.image, mode)
                figure = main.draw_histogram(channels)
                try:
                    self.assertEqual(len(figure.axes[0].lines), len(channels))
                    for line in figure.axes[0].lines:
                        self.assertEqual(len(line.get_xdata()), 256)
                        self.assertEqual(np.sum(line.get_ydata()), 80)
                finally:
                    plt.close(figure)

    def test_haar_detector_reports_no_faces_on_blank_image(self) -> None:
        annotated, detections = main.detect_faces(np.zeros((256, 256, 3), dtype=np.uint8))
        self.assertEqual(annotated.shape, (256, 256, 3))
        self.assertEqual(detections, [])

    def test_haar_detector_accepts_scalar_decision_margins(self) -> None:
        detector = Mock()
        detector.empty.return_value = False
        detector.detectMultiScale3.return_value = (
            np.array([[10, 12, 30, 30]]),
            np.array([1]),
            np.array([2.0]),
        )
        image = np.zeros((64, 64, 3), dtype=np.uint8)

        with patch("main.cv2.CascadeClassifier", return_value=detector):
            annotated, detections = main.detect_faces(image)

        self.assertEqual(annotated.shape, image.shape)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0][:4], (10, 12, 30, 30))
        self.assertAlmostEqual(detections[0][4], 1 / (1 + np.exp(-2)))


if __name__ == "__main__":
    unittest.main()
