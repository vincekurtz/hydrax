"""Utilities for video recording using FFmpeg."""

import os
import subprocess
from datetime import datetime
from typing import Optional, Dict, Any


class VideoRecorder:
    """Class for recording videos using FFmpeg."""

    def __init__(
        self,
        output_dir: str,
        width: int = 720,
        height: int = 480,
        fps: float = 30.0,
        filename_prefix: str = "video",
        crf: int = 23,
        preset: str = "medium",
    ):
        """Initialize the video recorder.

        Args:
            output_dir: Directory to save the video.
            width: Width of the video in pixels.
            height: Height of the video in pixels.
            fps: Frames per second.
            filename_prefix: Prefix for the video filename.
            crf: Constant Rate Factor for H.264 compression (lower is better quality).
            preset: FFmpeg preset (slower presets give better compression).
        """
        self.output_dir = output_dir
        self.width = width
        self.height = height
        self.fps = fps
        self.filename_prefix = filename_prefix
        self.crf = crf
        self.preset = preset

        self.ffmpeg_process = None
        self.video_path = None
        self.is_recording = False

    def start(self) -> bool:
        """Start recording the video.

        Returns:
            True if recording started successfully, False otherwise.
        """
        if self.is_recording:
            print("Warning: Recording already in progress")
            return True

        # Ensure output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Generate output path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_path = os.path.join(
            self.output_dir, f"{self.filename_prefix}_{timestamp}.mp4"
        )

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
                f"{self.width}x{self.height}",  # Size of one frame
                "-pix_fmt",
                "rgb24",  # Pixel format
                "-r",
                str(self.fps),  # Frames per second
                "-i",
                "-",  # Take input from pipe
                "-an",  # No audio
                "-vcodec",
                "h264",  # Output codec
                "-crf",
                str(self.crf),  # Constant quality factor
                "-preset",
                self.preset,  # Encoding preset
                "-loglevel",
                "error",  # Suppress output except errors
                self.video_path,
            ]

            self.ffmpeg_process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            self.is_recording = True
            print(f"Recording video to {self.video_path}")
            return True

        except (subprocess.SubprocessError, FileNotFoundError):
            print("Warning: FFmpeg not found. Video recording disabled.")
            self.is_recording = False
            return False

    def add_frame(self, frame: bytes) -> bool:
        """Add a frame to the video.

        Args:
            frame: Raw RGB frame data.

        Returns:
            True if the frame was added successfully, False otherwise.
        """
        if (
            not self.is_recording
            or self.ffmpeg_process is None
            or self.ffmpeg_process.stdin is None
        ):
            return False

        try:
            self.ffmpeg_process.stdin.write(frame)
            return True
        except (BrokenPipeError, IOError):
            print("Warning: Failed to write frame to video")
            self.is_recording = False
            return False

    def stop(self) -> bool:
        """Stop recording and finalize the video.

        Returns:
            True if the video was finalized successfully, False otherwise.
        """
        if not self.is_recording or self.ffmpeg_process is None:
            return False

        try:
            self.ffmpeg_process.stdin.close()
            self.ffmpeg_process.wait()
            print(f"Video saved to {self.video_path}")
            self.is_recording = False
            return True
        except (subprocess.TimeoutExpired, BrokenPipeError, IOError) as e:
            print(f"Warning: Error finalizing video: {e}")
            # Try to terminate the process if it's still running
            try:
                self.ffmpeg_process.terminate()
            except:
                pass
            self.is_recording = False
            return False
