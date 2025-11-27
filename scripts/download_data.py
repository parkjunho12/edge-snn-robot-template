# scripts/download_assets.py
import argparse
import urllib.request
import zipfile
from pathlib import Path

# ====== 경로 설정 ======
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "src" / "data"
OUTPUT_DIR = PROJECT_ROOT  # encoding_type 하위 디렉토리로 아티팩트 저장
DOWNLOADS_DIR = PROJECT_ROOT / "downloads"  # zip, 임시 파일 저장용

# ====== GitHub Release / 외부 URL 설정 ======
# 1) 아티팩트(zip) URL: .pth, .pkl, .json 등만 포함된 zip
ARTIFACTS_ZIP_URL = "https://github.com/parkjunho12/edge-snn-robot-template/releases/download/v0.1.16/output.zip"
ARTIFACTS_ZIP_NAME = "emg_artifacts_rate.zip"

# 2) mat 파일용 베이스 URL (형태 예시)
#   → mat 이름만 바꿔서 다양한 파일 받기: S1_D1_T1.mat, S2_D1_T1.mat 등
MAT_BASE_URL = "https://github.com/parkjunho12/edge-snn-robot-template/releases/download/v0.1.16/{mat_name}"
# 필요하면 나중에 HuggingFace, S3 주소 등으로 바꿀 수 있음


def download_file(url: str, dst_dir: Path, filename: str | None = None) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = url.split("/")[-1]

    dst_path = dst_dir / filename

    if dst_path.exists():
        print(f"[OK] {filename} already exists → {dst_path}")
        return dst_path

    print(f"[Downloading] {filename} ...")
    urllib.request.urlretrieve(url, dst_path)
    print(f"[DONE] Saved to: {dst_path}")
    return dst_path


def download_and_extract_artifacts_zip():
    """output/rate/ 안으로 .pth/.pkl/.json zip을 다운로드 + 압축 해제"""
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DOWNLOADS_DIR / ARTIFACTS_ZIP_NAME

    # zip 다운로드
    if zip_path.exists():
        print(f"[OK] Artifacts ZIP already exists → {zip_path}")
    else:
        print(f"[Downloading] Artifacts ZIP...")
        urllib.request.urlretrieve(ARTIFACTS_ZIP_URL, zip_path)
        print(f"[DONE] Saved ZIP to: {zip_path}")

    # 압축 해제

    print(f"[Extracting] {zip_path} → {OUTPUT_DIR}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(OUTPUT_DIR)
    print("[DONE] Extraction complete")

    print("\n[INFO] Extracted artifact files:")
    for p in OUTPUT_DIR.glob("*"):
        print("  →", p.name)
        
    # macOS 잔여 폴더 제거
    macos_junk = OUTPUT_DIR / "__MACOSX"
    if macos_junk.exists():
        import shutil
        shutil.rmtree(macos_junk)
        print("[CLEAN] Removed __MACOSX folder.")
    
    for junk in OUTPUT_DIR.glob(".DS_Store"):
        junk.unlink()
        print("[CLEAN] Removed .DS_Store file.")


def download_mat_file(mat_name: str):
    """단일 .mat 파일을 src/data/로 다운로드"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    url = MAT_BASE_URL.format(mat_name=mat_name)
    print(f"[INFO] MAT URL: {url}")
    dst_path = download_file(url, DATA_DIR, filename=mat_name)

    print(f"[DONE] MAT file ready at: {dst_path}")
    return dst_path


def main():
    parser = argparse.ArgumentParser(
        description="Download EMG artifacts (.pth/.pkl/.json) and/or .mat files."
    )
    parser.add_argument(
        "--artifacts",
        action="store_true",
        help="Download and extract artifacts ZIP (pth/pkl/json into ./output).",
    )
    parser.add_argument(
        "--mat",
        type=str,
        help="Download a specific .mat file into ./src/data (e.g., s1.mat).",
    )

    args = parser.parse_args()

    if not args.artifacts and not args.mat:
        parser.print_help()
        return

    if args.artifacts:
        download_and_extract_artifacts_zip()

    if args.mat:
        download_mat_file(args.mat)


if __name__ == "__main__":
    main()
