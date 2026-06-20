import imageio
import numpy as np
from PIL import Image, ImageDraw
import os

_frames = []

def show_frame(frame, title: str = "") -> None:
    """
    Annotate a frame with a text overlay and collect it for GIF saving.

    Parameters
    ----------
    frame : np.ndarray  HxWx3 uint8 RGB image (from env.render())
    title : str         info string drawn at the top of the frame
    """
    if frame is None:
        return

    img = Image.fromarray(frame)

    # Add a black banner at the top for the text
    banner_h = 24
    banner = Image.new("RGB", (img.width, banner_h), color=(20, 20, 20))
    annotated = Image.new("RGB", (img.width, img.height + banner_h))
    annotated.paste(banner, (0, 0))
    annotated.paste(img, (0, banner_h))

    draw = ImageDraw.Draw(annotated)
    draw.text((4, 4), title, fill=(255, 255, 255))

    _frames.append(np.array(annotated))


def save_episode(episode: int, output_dir: str, fps: int = 4) -> None:
    """Save collected frames as a GIF and clear the buffer."""
    if _frames:
        imageio.mimsave(f"output/{output_dir}/episode_{episode}.gif", _frames, fps=fps)
        print(f"Saved output/{output_dir}/episode_{episode}.gif  ({len(_frames)} frames)")
    _frames.clear()





class EpisodeLogger:
    """
    Records a step-by-step text trace of an episode for debugging.
    Use one instance per episode you want to record; call log_step() each
    step, then save() at the end of the episode.
    """

    def __init__(self):
        self._lines = []

    def log_step(self, step, state, action, reward, next_state, totalReturn, info=None,
                 q_values=None, agent_pos=None, energy=None, extra=None):
        self._lines.append(f"--- step {step} ---")
        self._lines.append(f"agent_pos   : {agent_pos}")
        self._lines.append(f"energy      : {energy}")
        self._lines.append(f"action      : {action}")
        self._lines.append(f"reward      : {reward}")
        self._lines.append(f"totalReturn      : {totalReturn}")
        if q_values is not None:
            self._lines.append(f"q_values    : {np.round(np.asarray(q_values), 3).tolist()}")
        if info is not None:
            self._lines.append(f"info        : {info}")
        if extra is not None:
            self._lines.append(f"extra       : {extra}")
        self._lines.append(f"state       : {np.asarray(state).tolist()}")
        self._lines.append(f"next_state  : {np.asarray(next_state).tolist()}")
        self._lines.append("")

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("\n".join(self._lines))
        print(f"Saved log: {path}  ({len(self._lines)} lines)")