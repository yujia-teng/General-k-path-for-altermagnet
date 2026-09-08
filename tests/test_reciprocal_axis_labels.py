"""Reciprocal labels retain their visible arrow-tip clearance across redraws."""

from io import BytesIO
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from mpl_toolkits.mplot3d import proj3d

from alterseek.plotting_3d import (
    _ReciprocalAxisLabel, _relayout_labels_for_save, setup_3d_ax,
)


@pytest.mark.parametrize('dpi', [100, 200])
def test_axis_labels_follow_camera_and_keep_clearance(dpi):
    vertices = np.array(list(itertools.product([-1., 1.], repeat=3)))
    fig, ax = setup_3d_ax('test', [vertices], np.eye(3) * 2, np.zeros(3))
    fig.set_dpi(dpi)
    labels = [a for a in ax.texts if isinstance(a, _ReciprocalAxisLabel)]
    assert len(labels) == 3
    try:
        centers = []
        for elev, azim in [(14, 20), (25, 110), (5, 210), (65, 300)]:
            ax.view_init(elev=elev, azim=azim)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for label in labels:
                points = np.vstack([label._start3d, label._tip3d])
                x, y, _ = proj3d.proj_transform(*points.T, ax.get_proj())
                start, tip = ax.transData.transform(np.column_stack([x, y]))
                direction = tip - start
                direction /= np.linalg.norm(direction)
                bbox = label.get_window_extent(renderer)
                corners = np.array(list(itertools.product(
                    [bbox.x0, bbox.x1], [bbox.y0, bbox.y1],
                )))
                clearance = np.min((corners - tip) @ direction)
                assert clearance == pytest.approx(renderer.points_to_pixels(7), abs=0.1)
                assert label.get_rotation() == 0
            centers.append(labels[0].get_window_extent(renderer).get_points().mean(axis=0))
        assert all(np.linalg.norm(a - b) > 1 for a, b in zip(centers, centers[1:]))

        # Export redraws use different renderers and DPI; no GUI event is required.
        ax.set_xlim(-2, 2)
        ax.view_init(elev=0, azim=0)
        for fmt in ['png', 'pdf']:
            output = BytesIO()
            fig.savefig(output, format=fmt, dpi=150, bbox_inches='tight')
            assert output.tell() > 1000
        fig.canvas.draw()
        assert all(np.isfinite(label.get_window_extent().get_points()).all() for label in labels)
    finally:
        plt.close(fig)


def test_save_layout_preserves_axis_labels_and_reduces_point_overlap():
    vertices = np.array(list(itertools.product([-1., 1.], repeat=3)))
    fig, ax = setup_3d_ax('test', [vertices], np.eye(3) * 2, np.zeros(3))
    labels = [a for a in ax.texts if isinstance(a, _ReciprocalAxisLabel)]
    ax.text(0, 0, 0, 'A', fontsize=22)
    ax.text(0, 0, 0, 'B', fontsize=22)
    try:
        fig.canvas.draw()
        before = [label.get_window_extent().get_points().copy() for label in labels]
        _relayout_labels_for_save(fig, ax)
        fig.canvas.draw()
        assert {text.get_text() for text in fig.texts} == {'A', 'B'}
        for label, bounds in zip(labels, before):
            assert label.get_visible()
            np.testing.assert_allclose(label.get_window_extent().get_points(), bounds)
        a, b = [text.get_window_extent() for text in fig.texts]
        overlap = max(0, min(a.x1, b.x1) - max(a.x0, b.x0)) * max(
            0, min(a.y1, b.y1) - max(a.y0, b.y0),
        )
        assert overlap < 0.1 * min(a.width * a.height, b.width * b.height)
    finally:
        plt.close(fig)


@pytest.mark.parametrize('deferred', [False, True])
def test_figure1_saved_labels_adjusted_in_both_save_routes(tmp_path, monkeypatch, deferred):
    from alterseek import compute_centroid_3d as centroid

    saved = []

    def inspect_save(fig, output_path, **kwargs):
        axis_labels = [a for a in fig.axes[0].texts if isinstance(a, _ReciprocalAxisLabel)]
        assert len(axis_labels) == 3
        assert all(label.get_visible() for label in axis_labels)
        point_labels = [a for a in fig.axes[0].texts if not isinstance(a, _ReciprocalAxisLabel)]
        assert point_labels and all(not label.get_visible() for label in point_labels)
        assert {t.get_text() for t in fig.texts} == {t.get_text() for t in point_labels}
        saved.append(output_path)
        return [output_path]

    monkeypatch.setattr(centroid, '_save_figure', inspect_save)
    result = centroid.run(
        str(Path(__file__).parent / 'references/case12_POSCAR'),
        output_dir=str(tmp_path), show_plot=deferred, defer_show=deferred,
        verbose=False,
    )
    if deferred:
        assert not saved
        fig = result['display_figures'][0]
        assert not fig.texts  # Interactive point labels retain their 3D positions.
        fig.axes[0].view_init(elev=35, azim=70)
        fig._alterseek_save_after_show()
    assert len(saved) == 1
