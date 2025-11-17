# scripts/test_camera_show.py
import asyncio

import cv2

from src.emg_io.vision_camera import VisionCamera


async def main() -> None:
    async with VisionCamera(src=0, width=640, height=480) as cam:
        async for frame in cam.stream():
            img = frame.img
            cv2.imshow("Mac Camera", img)

            # 키 입력 체크 (1ms 대기). 'q' 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())
