import io
import os
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from unittest import mock

from predict import (
    build_pdf_output_path,
    build_segment_groups,
    build_segments,
    compute_speed_comparison,
    compute_weighted_average_power,
    export_results_pdf,
    export_speed_chart,
    format_gear_label,
    main,
    mode_speed,
    recommend_gear,
    total_weight,
)


class BuildSegmentGroupsTests(unittest.TestCase):
    def test_groups_segments_to_the_smallest_value_at_or_above_target(self):
        segments = [
            {'dist': 60.0, 'pente': 0.0, 'dt': None},
            {'dist': 30.0, 'pente': 0.0, 'dt': None},
            {'dist': 40.0, 'pente': 0.0, 'dt': None},
            {'dist': 20.0, 'pente': 0.0, 'dt': None},
        ]

        groups = build_segment_groups(segments, target_distance=100.0)

        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]['dist'], 130.0)
        self.assertEqual(groups[1]['dist'], 20.0)

    def test_weighted_average_power_uses_only_positive_slope_groups(self):
        groups = [
            {'dist': 100.0, 'pente': 5.0, 'dt': 100.0},
            {'dist': 100.0, 'pente': -5.0, 'dt': 100.0},
        ]

        total_mass = total_weight(80.0, bike=9.0, accessories=3.0)
        average_power = compute_weighted_average_power(groups, weight=total_mass, min_slope=0.0)

        self.assertAlmostEqual(average_power, 54.4, places=1)

    def test_build_segments_keeps_missing_time_as_none(self):
        points = [
            {'lat': 0.0, 'lon': 0.0, 'ele': 0.0, 'time': None},
            {'lat': 0.0, 'lon': 0.01, 'ele': 0.0, 'time': None},
        ]

        linear_segments = build_segments(points)
        self.assertEqual(len(linear_segments), 1)
        self.assertIsNone(linear_segments[0]['dt'])

    def test_speed_comparison_omits_gpx_speed_and_delta_when_no_time_is_available(self):
        groups = [
            {'dist': 100.0, 'pente': 0.0, 'dt': None},
        ]

        results, total_dist, total_gpx_time, total_est_time, filtered_dist, filtered_gpx_time, filtered_est_time = compute_speed_comparison(
            groups,
            power=200.0,
            cyclist=80.0,
            bike=9.0,
            accessories=3.0,
        )

        self.assertEqual(len(results), 1)
        self.assertIsNone(results[0]['gpx_speed'])
        self.assertIsNone(results[0]['delta'])
        self.assertEqual(total_dist, 100.0)
        self.assertEqual(total_gpx_time, 0.0)
        self.assertEqual(filtered_dist, 100.0)
        self.assertEqual(filtered_gpx_time, 0.0)
        self.assertGreater(results[0]['estimated_speed'], 0.0)
        self.assertGreater(total_est_time, 0.0)
        self.assertGreater(filtered_est_time, 0.0)

    def test_main_prints_help_when_no_mode_is_given(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            with mock.patch('sys.argv', ['predict.py']):
                main()

        help_output = stdout.getvalue()
        self.assertIn('usage:', help_output.lower())
        self.assertIn('power', help_output.lower())

    def test_speed_comparison_uses_gpx_and_estimated_speeds(self):
        groups = [
            {'dist': 100.0, 'pente': 0.0, 'dt': 10.0},
        ]

        results, total_dist, total_gpx_time, total_est_time, filtered_dist, filtered_gpx_time, filtered_est_time = compute_speed_comparison(
            groups,
            power=200.0,
            cyclist=80.0,
            bike=9.0,
            accessories=3.0,
        )

        self.assertEqual(total_dist, 100.0)
        self.assertEqual(total_gpx_time, 10.0)
        self.assertGreater(total_est_time, 0.0)
        self.assertEqual(filtered_dist, 100.0)
        self.assertEqual(filtered_gpx_time, 10.0)
        self.assertGreater(filtered_est_time, 0.0)
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0]['gpx_speed'], 10.0, places=6)
        self.assertGreater(results[0]['estimated_speed'], 0.0)
        self.assertAlmostEqual(results[0]['delta'], results[0]['estimated_speed'] - 10.0, places=6)

    def test_weighted_average_power_clamps_negative_values(self):
        groups = [
            {'dist': 100.0, 'pente': -10.0, 'dt': 100.0},
        ]

        total_mass = total_weight(80.0, bike=9.0, accessories=3.0)
        average_power = compute_weighted_average_power(groups, weight=total_mass, min_slope=0.0)

        self.assertEqual(average_power, 0.0)

    def test_speed_comparison_defaults_cyclist_weight_when_missing(self):
        groups = [
            {'dist': 100.0, 'pente': 0.0, 'dt': 10.0},
        ]

        results, _, _, _, _, _, _ = compute_speed_comparison(
            groups,
            power=200.0,
            cyclist=None,
            bike=9.0,
            accessories=3.0,
        )

        self.assertEqual(len(results), 1)
        self.assertGreater(results[0]['estimated_speed'], 0.0)

    def test_format_gear_label_uses_chainring_x_cog_format(self):
        self.assertEqual(format_gear_label({'chainring': 50, 'cog': 24, 'cadence_rpm': 72.6}), '50 x 24')

    def test_recommend_gear_respects_max_cog_for_biggest_chainring(self):
        recommendation = recommend_gear(
            speed_mps=10.0,
            chainrings=[50, 34],
            cogs=[11, 12, 13, 14, 15, 17, 19, 21, 23, 26],
            rpm_min=60.0,
            rpm_max=80.0,
            max_cog=21,
        )

        self.assertLessEqual(recommendation['cog'], 21)

    def test_pdf_output_path_uses_gpx_name_when_not_provided(self):
        self.assertEqual(build_pdf_output_path('/tmp/example.gpx'), '/tmp/example.pdf')
        self.assertEqual(build_pdf_output_path('/tmp/example.GPX'), '/tmp/example.pdf')

    def test_recommend_gear_prefers_cadence_in_range(self):
        recommendation = recommend_gear(
            speed_mps=10.0,
            chainrings=[50, 34],
            cogs=[14, 16, 18, 20, 22, 24, 26, 28, 32, 36],
            rpm_min=60.0,
            rpm_max=80.0,
        )

        self.assertIsNotNone(recommendation)
        self.assertGreaterEqual(recommendation['cog'], 14)
        self.assertGreaterEqual(recommendation['cadence_rpm'], 60.0)
        self.assertLessEqual(recommendation['cadence_rpm'], 80.0)

    def test_export_helpers_create_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = os.path.join(tmp_dir, 'report.pdf')
            chart_path = os.path.join(tmp_dir, 'chart.png')
            rows = [{'group': 1, 'dist': 100.0, 'pente': 2.0, 'speed': 5.0, 'power': 250.0}]

            export_results_pdf(pdf_path, rows, 'Tableau puissance par groupe', chart_rows=rows)

            self.assertTrue(os.path.exists(pdf_path))
            self.assertFalse(os.path.exists(chart_path))
            self.assertGreater(os.path.getsize(pdf_path), 0)

    def test_mode_speed_omits_gpx_metrics_in_pdf_when_no_timestamps_exist(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            gpx_path = os.path.join(tmp_dir, 'route.gpx')
            with open(gpx_path, 'w', encoding='utf-8') as handle:
                handle.write(
                    '''<?xml version="1.0" encoding="UTF-8"?>
                    <gpx version="1.1" creator="test">
                      <trk>
                        <trkseg>
                          <trkpt lat="48.0" lon="2.0"><ele>100</ele></trkpt>
                          <trkpt lat="48.0001" lon="2.0"><ele>101</ele></trkpt>
                        </trkseg>
                      </trk>
                    </gpx>
                    '''
                )

            args = types.SimpleNamespace(
                gpx=gpx_path,
                cyclist=80.0,
                bike=9.0,
                accessories=3.0,
                power=200.0,
                group_distance=100.0,
                min_slope=-0.5,
                scx=0.40,
                mva=1.2,
                kf=0.01,
                chainrings=[50, 34],
                cogs=[11, 12, 13, 14, 15, 17, 19, 21, 23, 26],
                rpm_range=(60.0, 80.0),
                max_cog=None,
            )

            with redirect_stdout(io.StringIO()):
                mode_speed(args)

            pdf_path = os.path.join(tmp_dir, 'route.pdf')
            self.assertTrue(os.path.exists(pdf_path))
            pdf_text = open(pdf_path, 'rb').read().decode('latin-1', 'ignore')
            self.assertNotIn('Temps GPX', pdf_text)
            self.assertNotIn('Vitesse moyenne GPX', pdf_text)


if __name__ == '__main__':
    unittest.main()
