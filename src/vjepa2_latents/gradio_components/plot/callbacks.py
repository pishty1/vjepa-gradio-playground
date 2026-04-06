from __future__ import annotations

import base64
import json
from html import escape
from pathlib import Path
from typing import Any

from plotly.offline import get_plotlyjs

from ...gradio_utils import _format_hint_status, _load_latent_metadata, _log_gradio_step
from ..projection import load_saved_projection
from ..render import load_aligned_source_frames, write_video
from .core import build_projection_figure_from_data
from .helpers import _format_plot_status


def _html_to_iframe(content: str, *, height: int) -> str:
    return (
        "<iframe "
        f"style=\"width:100%;height:{height}px;border:0;\" "
    'sandbox="allow-scripts" '
        f"srcdoc=\"{escape(content, quote=True)}\" "
        'loading="lazy"></iframe>'
    )


def _figure_to_html(figure) -> str:
    figure_html = figure.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={"responsive": True, "displaylogo": False},
    )
    return _html_to_iframe(figure_html, height=760)


def _figure_to_sync_html(figure, source_video_path: str, display_fps: float) -> str:
    figure_json = figure.to_plotly_json()
    layout = figure_json.get("layout", {})
    layout.pop("updatemenus", None)
    layout.pop("sliders", None)
    figure_json["layout"] = layout

    video_path = Path(source_video_path)
    video_data = base64.b64encode(video_path.read_bytes()).decode("ascii")
    frame_delay_ms = max(1, int(round(1000.0 / max(float(display_fps), 1e-6))))
    figure_payload = json.dumps(figure_json)
    video_payload = json.dumps(f"data:video/mp4;base64,{video_data}")
    plotly_js_payload = get_plotlyjs()
    template = """
<div style="background:#111;color:#f3f4f6;padding:8px 0;">
  <div style="display:flex;gap:16px;align-items:flex-start;width:100%;">
    <div style="flex:1 1 48%;min-width:320px;">
      <video id="sync-source-video" controls muted playsinline style="width:100%;height:auto;border-radius:8px;background:#111;">
        <source src=__VIDEO__ type="video/mp4" />
      </video>
    </div>
    <div style="flex:1 1 52%;min-width:320px;">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
        <button id="sync-play" style="padding:6px 14px;">Play</button>
        <button id="sync-pause" style="padding:6px 14px;">Pause</button>
        <div id="sync-time-label" style="margin-left:8px;font-family:system-ui,sans-serif;font-size:14px;"></div>
      </div>
      <div id="sync-plot" style="width:100%;height:760px;"></div>
      <input id="sync-slider" type="range" min="0" max="0" step="1" value="0" style="width:100%;margin-top:8px;" />
    </div>
  </div>
</div>
<script>
__PLOTLY_JS__
</script>
<script>
(function() {
  const figure = __FIGURE__;
  const frameNames = (figure.frames || []).map((frame) => frame.name);
  const fallbackDuration = Math.max(frameNames.length / Math.max(__FPS__, 1e-6), 0.001);
  const video = document.getElementById('sync-source-video');
  const plotDiv = document.getElementById('sync-plot');
  const playButton = document.getElementById('sync-play');
  const pauseButton = document.getElementById('sync-pause');
  const slider = document.getElementById('sync-slider');
  const timeLabel = document.getElementById('sync-time-label');
  let currentIndex = 0;
  let animationFrameId = null;
  let isSeekingProgrammatically = false;

  const getPlaybackDuration = () => {
    if (Number.isFinite(video.duration) && video.duration > 0) {
      return video.duration;
    }
    return fallbackDuration;
  };

  const timeToFrameIndex = (timeSeconds) => {
    if (!frameNames.length) {
      return 0;
    }
    if (frameNames.length === 1) {
      return 0;
    }
    const duration = getPlaybackDuration();
    const normalizedTime = Math.max(0, Math.min(timeSeconds, duration));
    return Math.round((normalizedTime / duration) * (frameNames.length - 1));
  };

  const frameIndexToTime = (index) => {
    if (frameNames.length <= 1) {
      return 0;
    }
    const boundedIndex = Math.max(0, Math.min(index, frameNames.length - 1));
    return (boundedIndex / (frameNames.length - 1)) * getPlaybackDuration();
  };

  const updateLabel = () => {
    timeLabel.textContent = frameNames.length ? ('Time step: ' + frameNames[currentIndex]) : '';
    slider.value = String(currentIndex);
  };

  const stopPlaybackLoop = () => {
    if (animationFrameId !== null) {
      cancelAnimationFrame(animationFrameId);
      animationFrameId = null;
    }
  };

  const showFrame = async (index, syncVideo) => {
    if (!frameNames.length) {
      return;
    }
    currentIndex = Math.max(0, Math.min(index, frameNames.length - 1));
    updateLabel();
    if (syncVideo) {
      const targetTime = frameIndexToTime(currentIndex);
      try {
        isSeekingProgrammatically = true;
        video.currentTime = targetTime;
      } catch (error) {
      } finally {
        setTimeout(() => {
          isSeekingProgrammatically = false;
        }, 0);
      }
    }
    await Plotly.animate(plotDiv, [frameNames[currentIndex]], {
      mode: 'immediate',
      frame: { duration: 0, redraw: true },
      transition: { duration: 0 },
    });
  };

  const syncPlotToVideo = async () => {
    if (!frameNames.length) {
      return;
    }
    const desiredIndex = timeToFrameIndex(video.currentTime || 0);
    if (desiredIndex !== currentIndex) {
      await showFrame(desiredIndex, false);
    }
  };

  const tick = async () => {
    if (video.paused || video.ended) {
      stopPlaybackLoop();
      return;
    }
    await syncPlotToVideo();
    animationFrameId = requestAnimationFrame(() => {
      tick();
    });
  };

  const startPlayback = async () => {
    if (!frameNames.length) {
      return;
    }
    if (currentIndex >= frameNames.length - 1) {
      await showFrame(0, true);
    }
    stopPlaybackLoop();
    await video.play();
    animationFrameId = requestAnimationFrame(() => {
      tick();
    });
  };

  slider.min = '0';
  slider.max = String(Math.max(0, frameNames.length - 1));
  slider.value = '0';
  slider.addEventListener('input', async (event) => {
    video.pause();
    stopPlaybackLoop();
    await showFrame(Number(event.target.value), true);
  });

  playButton.addEventListener('click', async () => {
    if (!frameNames.length) {
      return;
    }
    await startPlayback();
  });

  pauseButton.addEventListener('click', () => {
    video.pause();
    stopPlaybackLoop();
  });

  video.addEventListener('play', () => {
    stopPlaybackLoop();
    animationFrameId = requestAnimationFrame(() => {
      tick();
    });
  });

  video.addEventListener('pause', () => {
    stopPlaybackLoop();
  });

  video.addEventListener('ended', async () => {
    stopPlaybackLoop();
    await showFrame(frameNames.length - 1, false);
  });

  video.addEventListener('seeked', async () => {
    if (isSeekingProgrammatically) {
      return;
    }
    await syncPlotToVideo();
  });

  video.addEventListener('timeupdate', async () => {
    if (video.paused) {
      await syncPlotToVideo();
    }
  });

  const initialize = async () => {
    await Plotly.newPlot(plotDiv, figure.data, figure.layout, { responsive: true, displaylogo: false });
    if (figure.frames && figure.frames.length) {
      Plotly.addFrames(plotDiv, figure.frames);
      slider.max = String(figure.frames.length - 1);
      await showFrame(0, false);
    }
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initialize, { once: true });
  } else {
    initialize();
  }
})();
</script>
"""
    sync_html = (
        template.replace("__PLOTLY_JS__", plotly_js_payload)
        .replace("__VIDEO__", video_payload)
        .replace("__FIGURE__", figure_payload)
        .replace("__FPS__", str(float(display_fps)))
        .replace("__DELAY__", str(frame_delay_ms))
    )
    return _html_to_iframe(sync_html, height=920)


def _build_source_preview_video(
    projection_state: dict[str, Any], projection_metadata: dict[str, Any]
) -> tuple[str, float] | None:
    latent_output_prefix_text = projection_state.get("latent_output_prefix") or projection_metadata.get("latent_output_prefix")
    if not latent_output_prefix_text:
        return None

    latent_output_prefix = Path(latent_output_prefix_text)
    latent_metadata = _load_latent_metadata(latent_output_prefix)
    latent_grid_shape = projection_metadata.get("latent_grid_shape") or latent_metadata.get("latent_grid_shape")
    if latent_grid_shape is None:
        return None

    source_frames, display_fps, _ = load_aligned_source_frames(latent_metadata, latent_grid_shape)
    preview_dir = Path(projection_state["output_prefix"]).parent / "plots"
    preview_path = preview_dir / "aligned_source_preview.mp4"
    write_video(preview_path, source_frames, fps=display_fps)
    return str(preview_path), float(display_fps)


def build_plot_step(
    projection_state: dict[str, Any] | None,
    plot_dimensions: int,
    plot_max_points: int,
    plot_animate_over_time: bool,
    plot_x_component: int,
    plot_y_component: int,
    plot_z_component: int | None,
):
    _log_gradio_step(
        "build_plot",
        f"dimensions={plot_dimensions} animate={bool(plot_animate_over_time)} components=({plot_x_component}, {plot_y_component}, {plot_z_component})",
    )
    if not projection_state or not projection_state.get("output_prefix"):
        return None, _format_hint_status("Plot not ready", "Compute or load a projection before building a plot.")

    projection = projection_state.get("projection")
    coordinates = projection_state.get("coordinates")
    metadata = projection_state.get("metadata")
    if projection is None or coordinates is None or metadata is None:
        projection, coordinates, metadata = load_saved_projection(Path(projection_state["output_prefix"]))

    component_indices = [int(plot_x_component) - 1, int(plot_y_component) - 1]
    if int(plot_dimensions) == 3:
        if plot_z_component is None:
            return None, _format_hint_status("Plot not ready", "Choose a Z component for a 3D plot.")
        component_indices.append(int(plot_z_component) - 1)

    figure = build_projection_figure_from_data(
        projection,
        coordinates,
        method=metadata["method"],
        component_indices=tuple(component_indices),
        component_labels=metadata.get("component_labels"),
        max_points=max(100, int(plot_max_points)),
        animate_over_time=bool(plot_animate_over_time),
    )
    _log_gradio_step("build_plot", f"plotted_components={component_indices}")

    if bool(plot_animate_over_time):
        source_preview = _build_source_preview_video(projection_state, metadata)
        if source_preview is not None:
            source_preview_video, display_fps = source_preview
            plot_html = _figure_to_sync_html(
                figure,
                source_preview_video,
                display_fps,
            )
        else:
            plot_html = _figure_to_html(figure)
    else:
        plot_html = _figure_to_html(figure)

    return plot_html, _format_plot_status(
        metadata["method"],
        component_indices,
        metadata.get("method_label"),
        animated=bool(plot_animate_over_time),
    )
