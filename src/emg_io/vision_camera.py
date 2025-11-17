# src/io/vision_camera.py
import asyncio
import time
from types import TracebackType
from typing import AsyncIterator, NamedTuple, Optional, Type

import cv2
import numpy as np


class Frame(NamedTuple):
    ts: float
    img: "np.ndarray"


class VisionCamera:
    _cap: Optional[cv2.VideoCapture]

    def __init__(self, src: int | str = 0, width: int = 640, height: int = 480) -> None:
        self.src, self.width, self.height = src, width, height
        self._cap = None

    async def __aenter__(self) -> "VisionCamera":
        cap = cv2.VideoCapture(self.src)
        # OpenCV's setters take floats
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        self._cap = cap
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        """Async context exit — release the camera safely."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    async def stream(self) -> AsyncIterator[Frame]:
        """Yield frames asynchronously from the opened camera."""
        cap = self._cap
        if cap is None:
            raise RuntimeError("Camera not opened — use 'async with VisionCamera():'")

        loop = asyncio.get_running_loop()
        while True:
            # run blocking read() in a thread pool
            ret, img = await loop.run_in_executor(None, cap.read)
            if not ret or img is None:
                await asyncio.sleep(0.01)
                continue
            yield Frame(ts=time.time(), img=img)
