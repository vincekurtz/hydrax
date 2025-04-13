"""Utilities for video recording using FFmpeg."""

import os
import subprocess
from datetime import datetime
from typing import Optional, Tuple, Dict, Any


def create_video_writer(
    output_dir: str,
    width: int = 720,
    height: int = 480,
    fps: float = 30.0,
    filename_prefix: str = "video",
    crf: int = 23,
    preset: str = "medium",
) -> Tuple[Optional[subprocess.Popen], Optional[str], Dict[str, Any]]:
    """Create a video writer using FFmpeg.

    Args:
        output_dir: Directory to save the video.
        width: Width of the video in pixels.
        height: Height of the video in pixels.
        fps: Frames per second.
        filename_prefix: Prefix for the video filename.
        crf: Constant Rate Factor for H.264 compression (lower is better quality).
        preset: FFmpeg preset (slower presets give better compression).

    Returns:
        Tuple containing:
        - FFmpeg subprocess (or None if creation failed)
        - Path to the output video file (or None if creation failed)
        - Metadata dictionary with video parameters
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate output path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.mp4")

    # Store metadata for reference
    metadata = {
        "width": width,
        "height": height,
        "fps": fps,
        "crf": crf,
        "preset": preset,
        "path": video_path,
    }

    # Check if FFmpeg is available
    try:
        # Test FFmpeg availability
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        # Set up FFmpeg process
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file
            "-f",
            "rawvideo",  # Input format
            "-vcodec",
            "rawvideo",  # Input codec
            "-s",
            f"{width}x{height}",  # Size of one frame
            "-pix_fmt",
            "rgb24",  # Pixel format
            "-r",
            str(fps),  # Frames per second
            "-i",
            "-",  # Take input from pipe
            "-an",  # No audio
            "-vcodec",
            "h264",  # Output codec
            "-crf",
            str(crf),  # Constant quality factor
            "-preset",
            preset,  # Encoding preset
            "-loglevel",
            "error",  # Suppress output except errors
            video_path,
        ]

        ffmpeg_subprocess = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        print(f"Recording video to {video_path}")
        return ffmpeg_subprocess, video_path, metadata

    except (subprocess.SubprocessError, FileNotFoundError):
        print("Warning: FFmpeg not found. Video recording disabled.")
        return None, None, metadata


def add_frame(ffmpeg_subprocess: subprocess.Popen, frame: bytes) -> bool:
    """Add a frame to the video.

    Args:
        ffmpeg_subprocess: FFmpeg subprocess from create_video_writer.
        frame: Raw RGB frame data.

    Returns:
        True if the frame was added successfully, False otherwise.
    """
    if ffmpeg_subprocess is None or ffmpeg_subprocess.stdin is None:
        return False

    try:
        ffmpeg_subprocess.stdin.write(frame)
        return True
    except (BrokenPipeError, IOError):
        print("Warning: Failed to write frame to video")
        return False


def finalize_video(
    ffmpeg_subprocess: subprocess.Popen, video_path: str
) -> bool:
    """Finalize the video recording.

    Args:
        ffmpeg_subprocess: FFmpeg subprocess from create_video_writer.
        video_path: Path to the output video file.

    Returns:
        True if the video was finalized successfully, False otherwise.
    """
    if ffmpeg_subprocess is None:
        return False

    try:
        ffmpeg_subprocess.stdin.close()
        ffmpeg_subprocess.wait()
        print(f"Video saved to {video_path}")
        return True
    except (subprocess.TimeoutExpired, BrokenPipeError, IOError) as e:
        print(f"Warning: Error finalizing video: {e}")
        # Try to terminate the process if it's still running
        try:
            ffmpeg_subprocess.terminate()
        except:
            pass
        return False
